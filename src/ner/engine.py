"""固有表現抽出エンジン (UI 非依存)。

テキストから固有表現を抽出する。特定のカテゴリ (ラベル) だけを抜き出す
フィルタリングにも対応する。Streamlit / CLI などの表示層からはこのエンジンを
呼び出すだけにし、エンジン自体は表示・IO に依存しない。

使用例::

    from src.ner import NerEngine

    engine = NerEngine("ja_ginza_electra")
    result = engine.extract("エクスモ社に勤める担当者", labels=["Company"])
    for ent in result.entities:
        print(ent.text, ent.label)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import spacy

from src.ner.preprocess import (
    CHUNK_SEPARATOR,
    _body_from_pieces,
    _prepare_pieces,
)

# 進捗 (ステージ) コールバック型：progress(stage_index, stage_total, label)。各ステージ開始時に
# 1 回呼ぶ。UI 非依存 (UI 側でステージ表示に使う)。
# 注: 1 モデルの nlp.pipe は最速の既定バッチ (全チャンクを 1 バッチ) で処理するため、その処理中の
# サブ進捗は出さない (小バッチにすると transformer が遅くなる＝本末転倒。electra 実測で +8〜29%)。
# 代わりに「どのモデル/段階を実行中か (何段/全何段)」を示す。
ProgressCallback = Callable[[int, int, str], None]

# ja_ginza_electra (torch/thinc/huggingface) 系が出す deprecation 警告を抑制する。
# 推論のたびに大量に出る `torch.cuda.amp.autocast(...)` ほか、初回ロード時の
# huggingface_hub / transformers の deprecation などのノイズ (実害なし・我々のコード起因ではない)。
# サードパーティのモジュールに限定して抑制し、自前コードの警告は残す。
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*",
    category=FutureWarning,
)
for _noisy in ("thinc", "torch", "huggingface_hub", "transformers"):
    warnings.filterwarnings("ignore", category=FutureWarning, module=rf"{_noisy}.*")

# 利用可能な GiNZA モデル (先頭が既定)
AVAILABLE_MODELS: tuple[str, ...] = ("ja_ginza_electra", "ja_ginza")
DEFAULT_MODEL = AVAILABLE_MODELS[0]

# チャンク分割の前処理 (小片化・本文/オフセット構築) は src.ner.preprocess に集約した
# (CHUNK_SEPARATOR / _prepare_pieces / _body_from_pieces / build_body)。NER 非依存＝LLM 検出も共有する。

# nlp.pipe を流すときの「1 ミニバッチあたりの累積文字数」上限＝**メモリ保護**。
# spaCy/thinc は 1 ミニバッチのトークンを 1 つの配列 (Σtokens, width) に載せるため、全チャンクを
# 1 バッチで流すと巨大配列になり OOM する (実測：大きな文書で (218861, 256)=214MiB の確保に失敗)。
# 累積文字数でバッチを区切り、配列を小さく保つ (Doc は独立処理なので分割しても結果は不変)。
#
# 注: これは**速度ノブではない**。electra は内部で strided-span を max_batch_items(=4096) 単位に
# 再バッチするので、ここを増やしても transformer の実バッチは変わらず速度・メモリは動かない
# (実測で確認済み。CPU での高速化は量子化/GPU 側＝docs-dev/insight-memo 参照)。
NLP_PIPE_BATCH_CHARS = 20000


@dataclass(frozen=True)
class Entity:
    """抽出された 1 件の固有表現。"""

    text: str
    label: str
    start: int  # 解析対象テキスト中の開始文字位置
    end: int  # 同・終了文字位置


@dataclass(frozen=True)
class TokenInfo:
    """1 トークンの診断情報 (recall の穴を実データで観察するためのデバッグ用)。

    マスキング目的では「GiNZA の NER が逃した固有名詞を、文脈非依存な
    SudachiPy の品詞 (``tag``) で拾えるか」が要点。両者を並べて観察する。
    """

    text: str  # 表層形
    tag: str  # SudachiPy 品詞 (例: 名詞-固有名詞-人名-姓)。文脈依存が小さい
    pos: str  # UD 品詞 (例: PROPN)
    ent_type: str  # GiNZA の NER ラベル (無ければ "")
    ent_iob: str  # B / I / O (エンティティ境界)
    is_oov: bool  # 語彙外フラグ (モデルのベクトル有無に依存。electra では参考値)
    norm: str  # 正規化表層形


@dataclass(frozen=True)
class AnalyzedToken:
    """全文オフセット付きの 1 トークン (マスキングのパイプラインが使う)。"""

    start: int  # 連結した全文中の開始文字位置
    end: int  # 同・終了文字位置
    surface: str
    tag: str  # SudachiPy 品詞
    pos: str  # UD 品詞


@dataclass(frozen=True)
class Analysis:
    """1 モデルでの解析結果 (全文・トークン列・NER エンティティ)。

    トークンは SudachiPy のトークナイズ (品詞つき)、entities はこのモデルの
    NER ラベル。複数モデルを併用するときは entities を和集合する (トークンは
    同じ SudachiPy なのでどれか 1 つで足りる)。

    平坦化 (``flatten_tables=True``) したときは、``text``/``tokens``/``entities`` は
    すべて**平坦化後テキスト基準**。一方 ``original_text`` は平坦化前の原文 (`|` 入り)、
    ``offset_map[i]`` は ``text`` の i 文字目に対応する ``original_text`` の文字位置
    (挿入文字は -1)。検出は平坦化テキストで行い、マスクは原文へ写して当てるため
    (src.masking) に使う。平坦化しない場合は ``original_text == text``・恒等写像。
    """

    text: str
    tokens: tuple[AnalyzedToken, ...]
    entities: tuple[Entity, ...]
    original_text: str = ""
    offset_map: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExtractionResult:
    """抽出結果。

    Attributes:
        text: 実際に解析対象となったテキスト (前処理を行った場合は前処理後)。
        entities: 抽出された固有表現のタプル。
    """

    text: str
    entities: tuple[Entity, ...]

    @property
    def labels(self) -> list[str]:
        """結果に含まれるラベル (カテゴリ) の一覧 (ソート済み)。"""
        return sorted({ent.label for ent in self.entities})

    def filter(self, labels: Iterable[str]) -> ExtractionResult:
        """指定したカテゴリの固有表現だけを残した結果を返す。"""
        allow = set(labels)
        return ExtractionResult(
            text=self.text,
            entities=tuple(e for e in self.entities if e.label in allow),
        )


class NerEngine:
    """GiNZA を用いた固有表現抽出エンジン。

    モデルは初回の解析時に遅延ロードする (生成自体は軽量)。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    @cached_property
    def nlp(self) -> spacy.language.Language:
        """GiNZA モデル (遅延ロードしてインスタンス内でキャッシュ)。"""
        return spacy.load(self.model_name)

    def available_labels(self) -> list[str]:
        """このモデルが出力しうる全ラベル (カテゴリ) の一覧。"""
        return sorted(self.nlp.get_pipe("ner").labels)

    def extract(
        self,
        text: str,
        *,
        labels: Iterable[str] | None = None,
        flatten_tables: bool = False,
    ) -> ExtractionResult:
        """1 つのテキストから固有表現を抽出する。

        長文 (SudachiPy のトークナイズ上限超) でも落ちないよう、内部で
        バイト数安全なチャンクに分割してから解析する。ファイルや kb-mcp の
        ように元から複数チャンクに分かれている場合は :meth:`extract_chunks`
        を使う (kb-mcp と同じ分割単位で解析でき、結果も揃う)。

        Args:
            text: 解析対象のテキスト。
            labels: 残すカテゴリ (ラベル)。None なら全件。
            flatten_tables: True なら Markdown テーブルを平文化してから解析する。

        Returns:
            ExtractionResult (解析対象テキストと抽出結果)。
        """
        return self.extract_chunks([text], labels=labels, flatten_tables=flatten_tables)

    def extract_chunks(
        self,
        chunks: Iterable[str],
        *,
        labels: Iterable[str] | None = None,
        flatten_tables: bool = False,
    ) -> ExtractionResult:
        """複数チャンクから固有表現を抽出し、1 つの結果にマージする。

        各チャンクを個別に解析し、エンティティの文字位置を「全チャンクを
        :data:`CHUNK_SEPARATOR` で連結したテキスト」基準に補正してまとめる。
        これにより displaCy 表示 (manual モード) がそのまま使える。

        チャンクが SudachiPy の上限 (:data:`SUDACHI_MAX_BYTES`) を超える場合は、
        さらにバイト数安全な小片へ分割してから解析する (保険)。

        Args:
            chunks: 解析対象チャンクの列 (kb-mcp / Splitter の出力など)。
            labels: 残すカテゴリ (ラベル)。None なら全件。
            flatten_tables: True なら各チャンクを平文化してから解析する。

        Returns:
            ExtractionResult (連結した解析テキストと、位置補正済みの抽出結果)。
        """
        # 解析する小片を確定 (バイト数安全分割 → 平文化 → 空片除去)
        pieces = _prepare_pieces(chunks, flatten_tables=flatten_tables)

        # 小片ごとに NER (nlp.pipe でバッチ処理) し、全文基準にオフセット補正
        entities: list[Entity] = []
        offset = 0
        sep_len = len(CHUNK_SEPARATOR)
        for piece, doc in zip(
            pieces, _pipe_in_batches(self.nlp, [p.flat for p in pieces])
        ):
            for ent in doc.ents:
                entities.append(
                    Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char + offset,
                        end=ent.end_char + offset,
                    )
                )
            offset += len(piece.flat) + sep_len

        result = ExtractionResult(
            text=CHUNK_SEPARATOR.join(p.flat for p in pieces),
            entities=tuple(entities),
        )
        if labels is not None:
            result = result.filter(labels)
        return result

    def debug_tokens(
        self,
        chunks: Iterable[str],
        *,
        flatten_tables: bool = False,
    ) -> list[TokenInfo]:
        """各トークンの SudachiPy 品詞と GiNZA NER ラベルを並べて返す (デバッグ用)。

        :meth:`extract_chunks` と**同じ小片分割** (平文化 → バイト数安全分割) を
        通すため、ここで見えるトークンは実際に NER が解析する対象と一致する。

        マスキングの recall の穴 (NER は逃すが SudachiPy は固有名詞・人名として
        割っている語など) を実データで観察するために使う。

        Args:
            chunks: 解析対象チャンクの列。
            flatten_tables: True なら各チャンクを平文化してから解析する。

        Returns:
            空白トークンを除いた :class:`TokenInfo` のリスト (出現順)。
        """
        pieces = _prepare_pieces(chunks, flatten_tables=flatten_tables)
        infos: list[TokenInfo] = []
        for doc in _pipe_in_batches(self.nlp, [p.flat for p in pieces]):
            for tok in doc:
                if tok.is_space:
                    continue
                infos.append(
                    TokenInfo(
                        text=tok.text,
                        tag=tok.tag_,
                        pos=tok.pos_,
                        ent_type=tok.ent_type_,
                        ent_iob=tok.ent_iob_,
                        is_oov=tok.is_oov,
                        norm=tok.norm_,
                    )
                )
        return infos

    def analyze_chunks(
        self,
        chunks: Iterable[str],
        *,
        flatten_tables: bool = False,
    ) -> Analysis:
        """チャンク列を解析し、全文・オフセット付きトークン・NER エンティティを返す。

        :meth:`extract_chunks` と同じ小片分割・オフセット補正を使う。マスキングの
        パイプライン (src.masking) が、SudachiPy 品詞とスパンを使って候補を作るために用いる。

        本文系 (``text``/``original_text``/``offset_map``) は **spaCy 非依存**なので
        :func:`~src.ner.preprocess.build_body` と同じ :func:`~src.ner.preprocess._body_from_pieces`
        で組む (＝LLM 検出層と本文座標を共有する)。本メソッドはそれに
        ``tokens``/``entities`` を doc から足すだけ。
        """
        pieces = _prepare_pieces(chunks, flatten_tables=flatten_tables)
        body = _body_from_pieces(pieces)  # text/original_text/offset_map (spaCy 非依存)
        tokens: list[AnalyzedToken] = []
        entities: list[Entity] = []
        offset = 0  # 平坦化テキスト基準のオフセット (トークン/エンティティ用)
        sep_len = len(CHUNK_SEPARATOR)
        for idx, (piece, doc) in enumerate(
            zip(pieces, _pipe_in_batches(self.nlp, [p.flat for p in pieces]))
        ):
            if idx > 0:  # 小片の区切り (CHUNK_SEPARATOR) の分だけ flat 座標を進める
                offset += sep_len
            for tok in doc:
                if tok.is_space:
                    continue
                start = tok.idx + offset
                tokens.append(
                    AnalyzedToken(
                        start=start,
                        end=start + len(tok.text),
                        surface=tok.text,
                        tag=tok.tag_,
                        pos=tok.pos_,
                    )
                )
            for ent in doc.ents:
                entities.append(
                    Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char + offset,
                        end=ent.end_char + offset,
                    )
                )
            offset += len(piece.flat)
        return Analysis(
            text=body.text,
            tokens=tuple(tokens),
            entities=tuple(entities),
            original_text=body.original_text,
            offset_map=body.offset_map,
        )


def _pipe_in_batches(
    nlp: spacy.language.Language,
    texts: list[str],
    char_budget: int = NLP_PIPE_BATCH_CHARS,
) -> Iterator[Any]:
    """``nlp.pipe`` を累積文字数で小バッチに区切り、Doc を順序どおり yield する。

    全チャンクを 1 バッチで流すと thinc が巨大なトークン配列を確保して OOM する
    (:data:`NLP_PIPE_BATCH_CHARS` 参照)。1 バッチの合計文字数を上限以下に抑えて配列を小さく保つ。
    Doc は独立に解析されるためバッチ分割で結果は変わらない (順序も保つ)。
    1 件で上限を超えるテキストはそれ単独で 1 バッチにする (さらに小さくはできない)。
    """
    batch: list[str] = []
    size = 0
    for t in texts:
        if batch and size + len(t) > char_budget:
            yield from nlp.pipe(batch)
            batch, size = [], 0
        batch.append(t)
        size += len(t)
    if batch:
        yield from nlp.pipe(batch)


# SudachiPy 単体トークナイザ (GiNZA/NER とは独立・激軽 ~0.02s)。初回のみ生成してキャッシュ。
_SUDACHI: Any = None


def _sudachi_tokenizer() -> Any:
    global _SUDACHI
    if _SUDACHI is None:
        from sudachipy import dictionary, tokenizer

        _SUDACHI = (
            dictionary.Dictionary().create(),
            tokenizer.Tokenizer.SplitMode.C,
        )
    return _SUDACHI


def sudachi_analyze_chunks(
    chunks: Iterable[str], *, flatten_tables: bool = False
) -> Analysis:
    """GiNZA NER を回さず **SudachiPy 単体**でトークン化のみ行う軽量解析。

    ``text``/``original_text``/``offset_map`` は :func:`~src.ner.preprocess.build_body` と同一
    (spaCy 非依存)。``tokens`` は Sudachi 形態素 (surface/品詞/オフセット)、``entities`` は空
    (NER を回さない)。LLM-only / ルールベースのみ の経路で **辞書照合用トークン**を得るために使う。
    各小片 (``_prepare_pieces``) はバイト数安全 (≤``SAFE_CHUNK_BYTES``) なので Sudachi 上限に掛からない。
    """
    pieces = _prepare_pieces(chunks, flatten_tables=flatten_tables)
    body = _body_from_pieces(pieces)
    tk, mode = _sudachi_tokenizer()
    tokens: list[AnalyzedToken] = []
    offset = 0  # 平坦化テキスト (body.text) 基準のオフセット
    sep_len = len(CHUNK_SEPARATOR)
    for idx, piece in enumerate(pieces):
        if idx > 0:  # 小片の区切り (CHUNK_SEPARATOR) の分だけ進める
            offset += sep_len
        for m in tk.tokenize(piece.flat, mode):
            tag = "-".join(p for p in m.part_of_speech() if p != "*")
            tokens.append(
                AnalyzedToken(
                    start=m.begin() + offset,
                    end=m.end() + offset,
                    surface=m.surface(),
                    tag=tag,
                    pos="",
                )
            )
        offset += len(piece.flat)
    return Analysis(
        text=body.text,
        tokens=tuple(tokens),
        entities=(),
        original_text=body.original_text,
        offset_map=body.offset_map,
    )
