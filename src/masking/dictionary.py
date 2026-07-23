"""マスク辞書 (社名・商標・人名の登録リスト)。

こちらが用意する固有名詞の名簿。**SudachiPy の内部辞書とは別物**。テキスト中の語を照合して
確定のマスク対象として拾う。大小文字・全角半角は正規化で吸収する。照合には2段ある:

- **完全一致 (既定) **: 登録語が**まるごとの語**として現れたときだけ拾う (:meth:`match`)。
  トークン列として一致し、かつ**前後がラテン英数字・区切り (``-`` ``_``) で連続していない**
  ことを要求する (＝より長い識別子の断片では拾わない。``社A`` を ``社A工場`` で拾わない・
  ``iAS`` を ``iASMap`` で拾わない・``IF-`` を ``IF-X`` で拾わない)。

- **部分一致 (``部分一致: true`` を付けた語だけ) **: 登録語が**他の語の中に境界沿いで現れたら、
  その部分だけ**を拾う (:meth:`partial_matches`)。境界＝トークン境界・区切り記号・camelCase/
  数字境界のいずれでも良い。命中は一致した部分だけをマスクする。
  例 ``iAS``→``iASMap`` の ``iAS`` / ``IF-``→``IF-X`` の ``IF-`` (``[商標]X`` になる) /
  ``CB``→``CBMark`` の ``CB``。境界照合なので ``ECBType`` の ``CB`` (``ECB`` の途中) は拾わない。

  → ユーザーは「まるごと (既定) ／部分一致 (``部分一致: true``)」だけを意識すればよい。
  「区切り入りか camelCase か」という実装の区別は不要 (同じ ``部分一致: true`` で両方効く)。

YAML 形式 (data/mask_dict.yaml。書式は data/mask_dict.sample.yaml 参照)。**全エントリ同一
フォーマット**＝各エントリは canonical / aliases / mask / partial の全キーを持ち、未定義は
明示的に null (partial は false)。セクション・キーは英語で統一::

    Company:
      - canonical: 社A
        aliases: [社A株式会社, ｼｬA]   # 別表記 (無ければ null)
        mask: 〔社A〕                  # 置換後の固定文字列 (無ければ null＝自動採番)
        partial: false                 # 他の語の中でも拾うか (true/false)
        case_sensitive: false          # 大小を区別するか (true/false)
    Trademark:
      - canonical: STS
        aliases: null
        mask: null
        partial: false
        case_sensitive: true           # 略語：大文字の STS だけ拾う (Sts/sts は拾わない)

``case_sensitive: true``＝**大小を区別** (NFKC のみ・casefold しない)。略語 ``STS`` は ``STS`` のみ
一致し ``Sts`` (Status の略かも) /``sts`` は拾わない。既定 ``false`` は従来どおり大小無視。

読み込みは旧形式も後方互換で受ける：セクションの日本語 (社名/商標/人名)、文字列だけの簡潔形、
``partial`` の旧キー ``部分一致`` / ``embed``。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

# YAML のセクション名 → 内部カテゴリ (内部カテゴリ値は日本語のまま。YAML の見た目だけ英語)。
# キーは casefold 済みで引く (英語は大小無視。日本語は casefold で不変)。旧・日本語セクションも受理。
_SECTION_CATEGORY = {
    "company": "社名",
    "trademark": "商標",
    "person": "人名",
    # 旧・日本語セクション (後方互換)
    "社名": "社名",
    "商標": "商標",
    "人名": "人名",
    "社員名": "人名",
}
# 照合する最大トークン/アトム数 (多トークン語の上限。コスト上限を兼ねる)
MAX_MATCH_TOKENS = 12


# 連続する空白 (半角/全角/タブ/改行)。`match` はトークンを空白なしで連結して照合するため、
# 登録語のスペース (例 `Tokyo Electron`) も除去して一致させる。
_WHITESPACE = re.compile(r"\s+")

# 「まるごと一致」の境界ガード用。正規化後 (NFKC+casefold) に**ラテン英数字か連結記号**なら、
# その位置は「まだ識別子が続いている」＝断片なので既定一致では拾わない。CJK (かな/漢字) や
# 空白・句読点・中黒は連結記号でないので境界とみなす (＝`社A` は `社Aです` で拾う・従来どおり)。
_LATIN_CONT_RE = re.compile(r"[a-z0-9_-]")


def normalize(text: str) -> str:
    """照合用の正規化 (NFKC で全角半角統一・casefold で大小無視・**空白除去**)。

    `match` はトークン列を空白なしで連結して辞書キーと突き合わせる。登録語が複数語
    (`Tokyo Electron` / `Nikon Precision Inc`) でも一致するよう、空白を除去して正規化する。
    """
    return _WHITESPACE.sub("", unicodedata.normalize("NFKC", text).casefold())


def normalize_cs(text: str) -> str:
    """大小を**区別する**正規化 (NFKC＋空白除去。casefold しない)。

    `case_sensitive: true` の語に使う。全角は NFKC で半角化されるが大小はそのまま
    (`ＳＴＳ`→`STS`・`Ｓｔｓ`→`Sts`)。`casefold(normalize_cs(x)) == normalize(x)` が成り立つので、
    部分一致のアトムは case 保存で作り、CI 照合時に casefold して使える。
    """
    return _WHITESPACE.sub("", unicodedata.normalize("NFKC", text))


# 識別子をサブワードに割る正規表現 (camelCase/略語/数字。区切り記号 `_-::@.` 等は自然に境界になる)。
# 例：SmashMark→[Smash,Mark] / CBMark→[CB,Mark] / ECBType→[ECB,Type] / HTTPServer→[HTTP,Server]。
_SUBWORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _split_identifier(surface: str) -> list[tuple[int, int]]:
    """識別子のサブワード境界 (start, end) のリストを返す (区切り記号自身は含めない)。"""
    return [(m.start(), m.end()) for m in _SUBWORD_RE.finditer(surface)]


@dataclass(frozen=True)
class DictMatch:
    """辞書一致 (トークンインデックスの半開区間 [start_token, end_token))。"""

    start_token: int
    end_token: int
    canonical: str
    category: str


class MaskDictionary:
    """正規化表層 → (canonical, category) の対応表 (＋任意の置換語)。"""

    def __init__(
        self,
        surface_map: dict[str, tuple[str, str]],
        placeholders: dict[str, str] | None = None,
        partial_map: dict[str, tuple[str, str]] | None = None,
        cs_map: dict[str, tuple[str, str]] | None = None,
        cs_partial_map: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self._map = surface_map
        # canonical → 置換語 (マスク後の伏せ字)。指定が無い canonical は自動採番に従う。
        self._placeholders = placeholders or {}
        # `部分一致: true` の語だけのマップ (他の語の中に境界沿いで内包照合する対象)。
        self._partial_map = partial_map or {}
        # `case_sensitive: true` の語 (大小区別。キーは normalize_cs＝casefold しない)。
        self._cs_map = cs_map or {}
        self._cs_partial_map = cs_partial_map or {}

    @classmethod
    def empty(cls) -> MaskDictionary:
        return cls({})

    @classmethod
    def load(cls, path: str | Path) -> MaskDictionary:
        """YAML を読み込んでマスク辞書を作る。

        各エントリは文字列 (canonical のみ) か、``canonical``/``aliases``/``mask``/``部分一致``
        (旧 ``embed``) を持つ辞書。``mask`` を指定すると置換語を固定できる (未指定なら自動採番)。
        """
        surface_map: dict[str, tuple[str, str]] = {}
        placeholders: dict[str, str] = {}
        partial_map: dict[str, tuple[str, str]] = {}
        cs_map: dict[str, tuple[str, str]] = {}
        cs_partial_map: dict[str, tuple[str, str]] = {}
        for entry in load_entries(path):
            canonical, category = entry["canonical"], entry["category"]
            if not canonical:
                continue
            cs = entry.get("case_sensitive")
            for surface in [canonical, *entry["aliases"]]:
                if cs:  # 大小区別 (casefold しないキー)
                    cs_map[normalize_cs(surface)] = (canonical, category)
                    if entry.get("partial"):
                        cs_partial_map[normalize_cs(surface)] = (canonical, category)
                else:  # 大小無視 (従来)
                    surface_map[normalize(surface)] = (canonical, category)
                    if entry.get("partial"):
                        partial_map[normalize(surface)] = (canonical, category)
            if entry.get("mask"):
                placeholders[canonical] = entry["mask"]
        return cls(surface_map, placeholders, partial_map, cs_map, cs_partial_map)

    def canonical_of(self, surface: str) -> str | None:
        """表層に対応する canonical (代表表記) を返す。辞書に無ければ None。

        別表記 (英語表記↔カタカナ表記・略称・旧称 等) を 1 つの canonical に束ねるのに使う
        ＝マスク後のプレースホルダを表記ゆれによらず統一できる。
        """
        entry = self._map.get(normalize(surface))
        if entry is None and self._cs_map:  # 大小区別エントリも見る
            entry = self._cs_map.get(normalize_cs(surface))
        return entry[0] if entry is not None else None

    def custom_placeholder(self, canonical: str) -> str | None:
        """canonical に対して指定された置換語 (あれば)。無ければ None (自動採番に従う)。"""
        return self._placeholders.get(canonical)

    def is_case_sensitive(self, surface: str) -> bool:
        """その表層が大小区別エントリ (``case_sensitive: true``) か。

        出現展開 (`_expand`) が、この表層を大小区別で広げるか (``STS`` を ``Sts``/``sts`` に
        広げない) を選ぶのに使う。
        """
        return normalize_cs(surface) in self._cs_map

    def __bool__(self) -> bool:
        return bool(self._map or self._cs_map)

    def __len__(self) -> int:
        return len(self._map) + len(self._cs_map)

    def embedded(self, token_surfaces: list[str]) -> list[tuple[int, str, str]]:
        """辞書語を**部分文字列として内包するが、トークン単位では一致しない**トークンを返す (監査用)。

        返り値は ``(token_index, canonical, category)``。境界を見ない素朴な部分文字列照合なので
        ``ECBType`` の ``CB`` 等も拾う＝**監査専用** (実マスクは境界を見る :meth:`partial_matches`)。
        トークン全体一致 (``match`` で拾う分) や雑音になりやすい 1 文字辞書語は除く。
        """
        out: list[tuple[int, str, str]] = []
        seen: set[tuple[int, str]] = set()
        for i, surface in enumerate(token_surfaces):
            ns = normalize(surface)
            if ns in self._map:
                continue  # トークン全体が辞書語＝match() で拾える (漏れではない)
            for key, (canonical, category) in self._map.items():
                if len(key) >= 2 and key in ns and (i, canonical) not in seen:
                    seen.add((i, canonical))
                    out.append((i, canonical, category))
        return out

    def partial_matches(
        self, tokens: Sequence[tuple[str, int, int]]
    ) -> list[tuple[int, int, str, str]]:
        """``部分一致: true`` の辞書語を、**境界沿いの内包照合**で拾う。

        入力 ``tokens`` は ``(surface, text_start, text_end)`` の列 (全文オフセット付き)。
        全トークンを**アトム**に割る：各トークンをサブワード (camelCase/略語/数字) と区切り記号
        (``-`` ``_`` ``·`` 等の 1 文字) に分解し、区切りトークン (``-`` 単独) も 1 アトムにする。
        空白は「切れ目」 (アトム間をまたげない)。登録語の正規化と、連続アトムの正規化連結が
        一致したら、その**アトム区間の全文スパン**を返す。

        - トークンをまたぐ区切り込みの一致に対応 (``IF``/``-``/``X`` で ``IF-``→``IF-X``)。
        - トークン内 camelCase の一致にも対応 (``iASMap``→``iAS`` / ``CBMark``→``CB``)。
        - アトム境界にしか合わないので途中一致はしない (``ECBType`` の ``CB`` は拾わない)。

        返り値 ``(text_start, text_end, canonical, category)``。
        """
        if not self._partial_map and not self._cs_partial_map:
            return []
        atoms = _partial_atoms(
            tokens
        )  # None = 切れ目 (空白)。文字列は case 保存 (normalize_cs)
        out: list[tuple[int, int, str, str]] = []
        n = len(atoms)
        j = 0
        while j < n:
            if atoms[j] is None:
                j += 1
                continue
            acc_cs = ""  # case 保存の連結
            hit_len = 0
            hit: tuple[str, str] | None = None
            for length in range(1, min(MAX_MATCH_TOKENS, n - j) + 1):
                atom = atoms[j + length - 1]
                if atom is None:  # 切れ目をまたがない
                    break
                acc_cs += atom[0]
                ci = acc_cs.casefold()
                if ci in self._partial_map:  # 大小無視エントリ
                    hit_len, hit = length, self._partial_map[ci]
                elif acc_cs in self._cs_partial_map:  # 大小区別エントリ (case 一致のみ)
                    hit_len, hit = length, self._cs_partial_map[acc_cs]
            if hit_len and hit is not None:
                first = atoms[j]
                last = atoms[j + hit_len - 1]
                assert first is not None and last is not None
                out.append((first[1], last[2], hit[0], hit[1]))
                j += hit_len
            else:
                j += 1
        return out

    def cjk_substring_matches(
        self, tokens: Sequence[tuple[str, int, int]]
    ) -> list[tuple[int, int, str, str]]:
        """辞書語を、**単一トークン内部の CJK/かな連なりの部分文字列**として拾う。

        Sudachi は複合カタカナ (``エクスモーションオリジナル``) を 1 トークンに融合することがあり、
        トークン境界単位の :meth:`match` では内部の登録語 (``エクスモーション``) を取りこぼす。
        LLM が拾える語を、確定の砦である辞書が漏らすのは redactor として不可なので、この穴を
        埋める (recall 優先)。一致部分が **CJK/かな (＋長音符) だけ**のときに限る＝ラテン英数の
        断片一致 (``ECBType`` の ``CB``) は従来どおり対象外 (``部分一致: true`` の opt-in のまま)。

        入力 ``tokens`` は ``(surface, text_start, text_end)``。大小無視 (``_map``) ・大小区別
        (``_cs_map``) の両方を見る。返り値 ``(text_start, text_end, canonical, category)``。
        トークン全体が登録語の分は :meth:`match` が拾うので emit しない (重複回避)。
        """
        if not self._map and not self._cs_map:
            return []
        out: list[tuple[int, int, str, str]] = []
        for surface, tstart, _tend in tokens:
            if self._map:
                norm, offs = _norm_chars_with_offsets(surface, normalize)
                _emit_cjk_substrings(norm, offs, tstart, self._map, out)
            if self._cs_map:
                norm_cs, offs_cs = _norm_chars_with_offsets(surface, normalize_cs)
                _emit_cjk_substrings(norm_cs, offs_cs, tstart, self._cs_map, out)
        return out

    def match(
        self,
        token_surfaces: Sequence[str],
        spans: Sequence[tuple[int, int]] | None = None,
    ) -> list[DictMatch]:
        """正規化トークン列に対し、各位置で最長一致の語を拾う (重なりなし・完全一致)。

        一致は**まるごとの語**に限る：一致区間の前後がラテン英数字・連結記号 (``-`` ``_``) で
        連続していると、より長い識別子の断片なので採らない (``IF-`` を ``IF-X`` で拾わない)。
        CJK・空白・句読点・中黒は境界とみなす (``社A`` は ``社Aです`` で拾う＝従来どおり)。
        大小無視 (`_map`) と大小区別 (`_cs_map`＝`STS` は `STS` のみ・`Sts`/`sts` は不一致) を両方見る。

        ``spans`` (各トークンの全文オフセット) を渡すと、隣トークンと**空白で切れている**場合を
        境界と認識する (``LB SONY`` の ``SONY`` を ``LB`` の末尾英字で誤って弾かない)。渡さない
        場合は従来どおり隣接扱い (:func:`_whole_word_boundary` 参照)。
        """
        ci = [normalize(s) for s in token_surfaces]
        cs = [normalize_cs(s) for s in token_surfaces] if self._cs_map else None
        matches: list[DictMatch] = []
        i = 0
        n = len(ci)
        while i < n:
            hit: DictMatch | None = None
            for length in range(min(MAX_MATCH_TOKENS, n - i), 0, -1):
                if not _whole_word_boundary(ci, i, i + length, spans):
                    continue
                key_ci = "".join(ci[i : i + length])
                if key_ci in self._map:  # 大小無視
                    canonical, category = self._map[key_ci]
                    hit = DictMatch(i, i + length, canonical, category)
                    break
                if cs is not None:  # 大小区別 (case 一致のみ)
                    key_cs = "".join(cs[i : i + length])
                    if key_cs in self._cs_map:
                        canonical, category = self._cs_map[key_cs]
                        hit = DictMatch(i, i + length, canonical, category)
                        break
            if hit is not None:
                matches.append(hit)
                i = hit.end_token
            else:
                i += 1
        return matches


def _whole_word_boundary(
    norm: list[str],
    start: int,
    end: int,
    spans: Sequence[tuple[int, int]] | None = None,
) -> bool:
    """トークン区間 [start, end) が「まるごとの語」境界に挟まれているか (完全一致のガード)。

    区間の直前トークンの末尾・直後トークンの先頭がラテン英数字/連結記号なら、より長い
    識別子の断片なので False (``IF-X`` の中の ``IF-``/``IF`` を弾く)。CJK・空白・句読点は
    連結記号でないので True (``社Aです`` の ``社A`` は従来どおり拾う)。

    ``spans`` (各トークンの全文オフセット (start, end)) を渡すと、**隣トークンと実際に隣接
    している (間に空白等が無い) ときだけ**英字連続ガードを効かせる。トークナイズは空白を
    捨てるため、``LB SONY`` は surface 列だと ``["LB","SONY"]`` と連続して見え、``LB`` の末尾
    ``B`` を根拠に ``SONY`` を「識別子の途中」と誤判定して弾いていた (空白＝語境界なのに)。
    オフセットの隙間で「空白で切れている＝境界」を判定し、この取りこぼしを防ぐ
    (``spans`` 無しは従来どおり隣接扱い＝保守的)。
    """
    if start > 0:
        prev = norm[start - 1]
        adjacent = spans is None or spans[start - 1][1] == spans[start][0]
        if adjacent and prev and _LATIN_CONT_RE.match(prev[-1]):
            return False
    if end < len(norm):
        nxt = norm[end]
        adjacent = spans is None or spans[end - 1][1] == spans[end][0]
        if adjacent and nxt and _LATIN_CONT_RE.match(nxt[0]):
            return False
    return True


def _partial_atoms(
    tokens: Sequence[tuple[str, int, int]],
) -> list[tuple[str, int, int] | None]:
    """トークン列を部分一致用アトムに割る。``None`` は切れ目 (空白＝またげない)。

    各アトムは ``(normalize_cs(部分文字列), text_start, text_end)``＝**case 保存** (CI 照合側が
    casefold して使う。大小区別エントリはそのまま比較)。
    - **ラテン英数のサブワード** (camelCase/略語/数字) ＝ ``_split_identifier`` で切る。
    - **CJK/かな等の連なり** (``_split_identifier`` に載らない文字) ＝**トークン内で1アトムに束ねる**。
      CJK には綴り上の下位境界が無く、morpheme 境界＝トークン境界なので、1トークン内の CJK 連は
      分割しない (``補`` が ``補正`` を割らない＝境界安全。トークンをまたぐ ``用/補正/値`` は
      別トークン＝別アトムなので ``補正`` は拾える)。
    - **区切り記号 1 文字** (``-`` ``_`` ``·`` 等) ＝各 1 アトム。区切りトークン (``-`` 単独) もここ。
    - **空白/正規化で消える文字**＝切れ目 (``None``)。
    """
    atoms: list[tuple[str, int, int] | None] = []
    for surface, tstart, _tend in tokens:
        if not surface:
            atoms.append(None)
            continue

        def emit_gap(lo: int, hi: int) -> None:
            k = lo
            while k < hi:
                ch = surface[k]
                if ch.isspace() or not normalize_cs(ch):
                    atoms.append(None)  # 空白/正規化で消える文字＝切れ目
                    k += 1
                elif ch.isalnum():  # CJK/かな/全角英数：連なりを 1 アトムに束ねる
                    j = k + 1
                    while j < hi and surface[j].isalnum() and not surface[j].isspace():
                        j += 1
                    atoms.append((normalize_cs(surface[k:j]), tstart + k, tstart + j))
                    k = j
                else:  # 区切り記号・記号：1 文字 1 アトム (IF- の "-" 等)
                    atoms.append((normalize_cs(ch), tstart + k, tstart + k + 1))
                    k += 1

        prev = 0
        for a, b in _split_identifier(surface):
            emit_gap(prev, a)
            atoms.append((normalize_cs(surface[a:b]), tstart + a, tstart + b))
            prev = b
        emit_gap(prev, len(surface))
    return atoms


def _is_cjk(ch: str) -> bool:
    """かな/漢字 (＋長音符) か。融合トークン内部の部分一致を CJK 連なりだけに限るために使う。

    ラテン英数の断片一致 (``ECBType`` の ``CB`` 等) は誤検出源なので内部照合の対象外にする
    (それは従来どおり ``部分一致: true`` の opt-in)。正規化後 (NFKC) は半角カナも全角に寄るので
    全角の範囲だけ見れば足りる。
    """
    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF  # ひらがな + カタカナ (長音符 ー=0x30FC を含む)
        or 0x3400 <= o <= 0x9FFF  # CJK 統合漢字 (拡張 A 含む)
        or 0xF900 <= o <= 0xFAFF  # CJK 互換漢字
    )


def _norm_chars_with_offsets(
    surface: str, normalizer: Callable[[str], str]
) -> tuple[str, list[int]]:
    """``surface`` を 1 文字ずつ ``normalizer`` で正規化して連結した文字列と、正規化後 i 文字目
    → ``surface`` の元 index の対応表を返す。空白は正規化で消える＝連結にも対応表にも含めない。

    融合トークン内部の部分文字列一致 (:meth:`MaskDictionary.cjk_substring_matches`) が、命中位置を
    元テキストのオフセットへ戻すために使う。かな/漢字は NFKC・casefold で 1:1 (長さ不変) なので、
    対象を CJK に絞る本用途では素直に対応づく。
    """
    chars: list[str] = []
    offs: list[int] = []
    for i, ch in enumerate(surface):
        for nc in normalizer(ch):
            chars.append(nc)
            offs.append(i)
    return "".join(chars), offs


def _emit_cjk_substrings(
    norm: str,
    offs: list[int],
    tstart: int,
    keymap: dict[str, tuple[str, str]],
    out: list[tuple[int, int, str, str]],
) -> None:
    """``norm`` (正規化済みトークン文字列) 中に現れる ``keymap`` の登録語を、**CJK/かなの部分
    文字列**として拾って ``out`` に足す (全文オフセットは ``offs`` で戻す)。

    トークン全体が登録語のとき (``match`` が拾う) は重複回避で emit しない。1 文字語は雑音に
    なりやすいので内部照合しない (``match`` は従来どおりトークン境界で拾う)。
    """
    n = len(norm)
    for key, (canonical, category) in keymap.items():
        if len(key) < 2:
            continue
        start = 0
        while True:
            idx = norm.find(key, start)
            if idx < 0:
                break
            end = idx + len(key)
            whole_token = idx == 0 and end == n  # トークン全体一致は match() の担当
            if not whole_token and all(_is_cjk(norm[k]) for k in range(idx, end)):
                out.append(
                    (
                        tstart + offs[idx],
                        tstart + offs[end - 1] + 1,
                        canonical,
                        category,
                    )
                )
            start = idx + 1


def contains_partial(
    surfaces: Sequence[str], ci_keys: set[str], cs_keys: set[str]
) -> bool:
    """表層列が ``ci_keys`` (大小無視) /``cs_keys`` (大小区別) のいずれかを**境界沿いに内包**するか。

    :meth:`MaskDictionary.partial_matches` と同じアトム分解・境界照合を使う (除外リストと
    辞書で照合仕様を1本化するための共有関数)。除外リストは候補を丸ごと外すため、位置でなく
    「含むか否か」だけを返す。``surfaces`` は候補を覆うトークン列でも、候補表層 1 本でもよい。
    """
    if not ci_keys and not cs_keys:
        return False
    # オフセットは使わない (合成 0 起点)。境界 (None) と case 保存のアトム文字列だけ見る。
    src = [(s, 0, len(s)) for s in surfaces]
    norms = [a[0] if a is not None else None for a in _partial_atoms(src)]
    n = len(norms)
    j = 0
    while j < n:
        if norms[j] is None:
            j += 1
            continue
        acc_cs = ""
        for length in range(1, min(MAX_MATCH_TOKENS, n - j) + 1):
            nm = norms[j + length - 1]
            if nm is None:  # 切れ目をまたがない
                break
            acc_cs += nm
            if acc_cs.casefold() in ci_keys or acc_cs in cs_keys:
                return True
        j += 1
    return False


# 内部カテゴリ → 書き出し時の YAML セクション名 (英語で統一)。読み込みは _SECTION_CATEGORY で吸収。
_CATEGORY_SECTION = {"社名": "Company", "商標": "Trademark", "人名": "Person"}
# 書き出し時のセクション順 (英語)。
_SECTION_ORDER_OUT = ("Company", "Trademark", "Person")


def load_entries(path: str | Path) -> list[dict]:
    """YAML を**構造のまま**読み込む (UI 編集・round-trip 用)。

    返り値は ``{"category", "canonical", "aliases": list[str], "mask": str, "partial": bool}`` の列。
    ``partial`` は YAML キー ``部分一致`` (旧 ``embed``) のどちらからでも読む (後方互換)。
    (:meth:`MaskDictionary.load` はこれを正規化表層マップに畳む)

    旧バグで置換に書かれてしまった文字列 ``"nan"`` は空 (未指定) として読み込む (自己修復。
    再保存すれば YAML からも消える)。
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries: list[dict] = []
    for section, items in raw.items():
        skey = str(section).strip()
        category = _SECTION_CATEGORY.get(skey.casefold(), skey)  # 英語は大小無視
        for item in items or []:
            if isinstance(item, str):
                entries.append(
                    {
                        "category": category,
                        "canonical": item,
                        "aliases": [],
                        "mask": "",
                        "partial": False,
                        "case_sensitive": False,
                    }
                )
            else:
                mask = str(item.get("mask") or "")
                if mask.strip().lower() == "nan":  # 旧バグの掃除
                    mask = ""
                entries.append(
                    {
                        "category": category,
                        "canonical": item.get("canonical") or item.get("name") or "",
                        "aliases": list(item.get("aliases") or []),
                        "mask": mask,
                        # `partial` (新・英語) 優先・`部分一致`／`embed` (旧) も受理
                        "partial": bool(
                            item.get("partial")
                            or item.get("部分一致")
                            or item.get("embed")
                        ),
                        # 大小区別 (既定 false＝大小無視)
                        "case_sensitive": bool(item.get("case_sensitive")),
                    }
                )
    return entries


def sort_key(canonical: str) -> str:
    """辞書エントリの並び順キー (代表表記を NFKC+casefold で正規化した辞書順)。

    照合用 :func:`normalize` と同じ正規化 (大小・全角半角・空白を無視)。件数が増えても
    探しやすいよう、各セクション内を代表表記でソートする。除外リスト側と方針を揃える。
    """
    return normalize(canonical)


def save_entries(path: str | Path, entries: list[dict]) -> None:
    """構造化エントリを YAML に書き出す (UI 保存用)。

    **全エントリを同一フォーマット (フルセット) で書く**：各エントリは必ず
    ``canonical`` / ``aliases`` / ``mask`` / ``partial`` / ``case_sensitive`` の全キーを持つ。
    未定義は明示的に ``aliases: null`` / ``mask: null``、真偽値 (``partial`` / ``case_sensitive``) は
    ``false`` と書く (＝手入力時にどのキーが使えるか一目で分かる)。キーは英語で統一、セクションも
    英語 (Company / Trademark / Person → (その他) )。各セクション内は代表表記の辞書順にソート。
    canonical が空のエントリは捨てる。
    """
    sections: dict[str, list[tuple[str, dict]]] = {}
    for e in entries:
        canonical = (e.get("canonical") or "").strip()
        if not canonical:
            continue
        category = e.get("category") or "社名"
        section = _CATEGORY_SECTION.get(category, category)
        aliases = [a.strip() for a in (e.get("aliases") or []) if a.strip()]
        mask = (e.get("mask") or "").strip()
        # 内部フィールドは `partial`。旧来の `embed` フィールドも受理 (後方互換)。
        partial = bool(e.get("partial") or e.get("embed"))
        case_sensitive = bool(e.get("case_sensitive"))
        # フルセット：未定義は null (真偽値は false)。
        obj = {
            "canonical": canonical,
            "aliases": aliases or None,
            "mask": mask or None,
            "partial": partial,
            "case_sensitive": case_sensitive,
        }
        sections.setdefault(section, []).append((canonical, obj))

    # 各セクション内を代表表記でソートしてから item だけ取り出す。
    sorted_sections: dict[str, list] = {
        section: [item for _, item in sorted(rows, key=lambda r: sort_key(r[0]))]
        for section, rows in sections.items()
    }

    ordered = {
        s: sorted_sections[s] for s in _SECTION_ORDER_OUT if s in sorted_sections
    }
    ordered.update({s: v for s, v in sorted_sections.items() if s not in ordered})
    Path(path).write_text(
        yaml.safe_dump(ordered, allow_unicode=True, sort_keys=False, indent=2),
        encoding="utf-8",
    )
