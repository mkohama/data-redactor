"""NER 解析前のテキスト整形 (UI 非依存)。

現状は Markdown テーブルの平文化のみ。GiNZA は自然文で学習しているため、
`|` を含むテーブル記法のままだとセル内の語をほとんど抽出できない。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# テーブルのセルを連結する区切り文字。読点が抽出精度・可読性ともに無難。
TABLE_CELL_DELIMITER = "、"

_SEPARATOR_CELL = re.compile(r"^:?-{1,}:?$")


def _split_table_row(line: str) -> list[str]:
    """Markdown テーブルの 1 行をセルのリストに分解する。"""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(line: str) -> bool:
    """`| --- | --- |` のような区切り行か判定する。"""
    cells = [c for c in _split_table_row(line) if c != ""]
    return bool(cells) and all(_SEPARATOR_CELL.match(c) for c in cells)


# 1 文字とその「原文での文字位置」 (挿入文字は -1) の組。平坦化の対応表づくりに使う。
_Char = tuple[str, int]


def _cells_with_pos(line: str, line_start: int) -> list[list[_Char]]:
    """テーブル行を `|` で分割し、各セルを (文字, 原文インデックス) 列で返す (前後空白は除去)。

    `_split_table_row` の位置情報つき版。`|` 区切りで分け、各セルの前後空白を落とす。
    (`line.split("|")` 相当。先頭/末尾 `|` 由来の空セルは呼び出し側で除外する)
    """
    cells: list[list[_Char]] = [[]]
    for i, ch in enumerate(line):
        if ch == "|":
            cells.append([])
        else:
            cells[-1].append((ch, line_start + i))
    stripped: list[list[_Char]] = []
    for cell in cells:
        s, e = 0, len(cell)
        while s < e and cell[s][0].isspace():
            s += 1
        while e > s and cell[e - 1][0].isspace():
            e -= 1
        stripped.append(cell[s:e])
    return stripped


def _flatten_line_with_map(
    line: str, line_start: int, delimiter: str
) -> list[_Char] | None:
    """1 行を平坦化し (文字, 原文インデックス) 列で返す。区切り行は None (=削除)。"""
    if "|" not in line:
        return [(ch, line_start + i) for i, ch in enumerate(line)]
    if _is_separator_row(line):
        return None
    cells = [c for c in _cells_with_pos(line, line_start) if c]
    if not cells:
        return []
    seg: list[_Char] = []
    for ci, cell in enumerate(cells):
        if ci > 0:
            seg.append((delimiter, -1))  # 挿入した区切り (原文に対応なし)
        seg.extend(cell)
    seg.append(("。", -1))  # 挿入した句点 (原文に対応なし)
    return seg


def flatten_markdown_tables_with_map(
    text: str, delimiter: str = TABLE_CELL_DELIMITER
) -> tuple[str, list[int]]:
    """:func:`flatten_markdown_tables` と同じ平坦化に、文字位置の対応表を付けて返す。

    返り値 ``(flat, cmap)``：``flat`` は平坦化後テキスト、``cmap[i]`` は ``flat`` の
    i 文字目に対応する**原文 (引数 text) の文字位置**。挿入文字 (区切り `、`/句点 `。`/
    行連結の改行) は ``-1``。検出 (平坦化テキスト) で得たスパンを、`|` 入り原文へ
    逆写像してマスクするために使う (src.masking.apply)。
    """
    out_lines: list[list[_Char]] = []
    line_start = 0
    for line in text.split("\n"):
        seg = _flatten_line_with_map(line, line_start, delimiter)
        if seg is not None:
            out_lines.append(seg)
        line_start += len(line) + 1  # +1 は split で落ちた "\n" の分
    chars: list[_Char] = []
    for j, seg in enumerate(out_lines):
        if j > 0:
            chars.append(("\n", -1))  # 出力行を連結する改行 (原文対応は付けない)
        chars.extend(seg)
    flat = "".join(ch for ch, _ in chars)
    cmap = [pos for _, pos in chars]
    return flat, cmap


def flatten_markdown_tables(text: str, delimiter: str = TABLE_CELL_DELIMITER) -> str:
    """Markdown テーブルの記法を取り除いて NER 向きの平文にする。

    行単位で
    - 区切り行 (`| --- | --- |`) は削除
    - データ行はセルを区切り文字で連結し、末尾に句点を付与
    する。ヘッダー行に依存しないため、テーブルの途中だけを含むチャンクに
    適用しても破綻しない (`| 由利` のような記号混じりの誤抽出を生まない)。

    なお GiNZA の NER は文脈依存が強く、短い語や曖昧な語 (短い英字の社名など) は
    どの整形をしても抽出されないことがある点には注意。
    """
    return flatten_markdown_tables_with_map(text, delimiter)[0]


# 括弧グルー対策で前後に空白を挟む括弧 (半角・全角の丸/角/波括弧)。
# GiNZA/SudachiPy は `語(中身)` を空白なしだと 1 トークンに融合し (例 `姓A(社B)`→
# 「名詞-固有名詞-人名-姓」1 個、`製品(社B)`→「名詞-普通名詞-一般」1 個)、トークン単位照合の
# マスク辞書が中の語 (社B) を拾えず漏れる。括弧に隣接する非空白の境界へ空白を 1 つ挟むと
# 正しく割れる (実測：`姓A (社B)` は 姓A/(/社B/) に分割され 社B=確定)。
# 引用の鉤括弧「」『』〈〉《》は対象外 (既に分割され、対話/書名の体裁を崩さないため)。
_BRACKET_OPEN = "([{（［｛〔【"
_BRACKET_CLOSE = ")]}）］｝〕】"


def pad_brackets_with_map(text: str) -> tuple[str, list[int]]:
    """括弧の融合を防ぐため、括弧に隣接する非空白境界へ空白を挿入する。

    開き括弧の直前 (直前が非空白のとき) と閉じ括弧の直後 (直後が非空白のとき) に半角空白を
    1 つ挟む。トークナイザが括弧で割れるようになり、辞書語 (例 `姓A(社B)` の `社B`) が
    独立トークンになって確定マスクされる。返り値 ``(padded, cmap)``：``cmap[i]`` は ``padded``
    の i 文字目に対応する ``text`` の文字位置 (挿入した空白は ``-1``＝原文対応なし)。
    """
    out: list[str] = []
    cmap: list[int] = []
    n = len(text)
    for i, ch in enumerate(text):
        prev = text[i - 1] if i > 0 else ""
        if ch in _BRACKET_OPEN and prev and not prev.isspace():
            out.append(" ")
            cmap.append(-1)
        out.append(ch)
        cmap.append(i)
        nxt = text[i + 1] if i + 1 < n else ""
        if ch in _BRACKET_CLOSE and nxt and not nxt.isspace():
            out.append(" ")
            cmap.append(-1)
    return "".join(out), cmap


def _compose_maps(outer: list[int], inner: list[int]) -> list[int]:
    """2 段の位置対応表を合成する。

    ``inner[i]`` は最終テキストの i 文字目 → 中間テキストの位置 (挿入は -1)。
    ``outer[j]`` は中間テキストの j 文字目 → 原文の位置 (挿入は -1)。
    返り値 ``out[i]`` は最終テキストの i 文字目 → 原文の位置 (どちらかが -1 なら -1)。
    """
    return [outer[j] if j != -1 else -1 for j in inner]


def prepare_for_ner(text: str, *, flatten_tables: bool = True) -> str:
    """NER 解析前のテキスト整形 (任意でテーブル平文化 → **常に**括弧の前後に空白挿入)。"""
    return prepare_for_ner_with_map(text, flatten_tables=flatten_tables)[0]


def prepare_for_ner_with_map(
    text: str, *, flatten_tables: bool = True
) -> tuple[str, list[int]]:
    """:func:`prepare_for_ner` の対応表つき版 (整形後テキストと原文位置対応表)。

    ``flatten_tables`` のときだけ ① Markdown テーブル平文化を行い、その後 **常に**
    ② 括弧グルー対策の空白挿入を行う。①②の対応表を合成して返す (括弧対策は表の有無に
    かかわらず効かせる＝`姓A(社B)` の埋没を常に防ぐ)。
    """
    if flatten_tables:
        flat1, cmap1 = flatten_markdown_tables_with_map(text)
    else:
        flat1, cmap1 = text, list(range(len(text)))
    flat2, cmap2 = pad_brackets_with_map(flat1)
    return flat2, _compose_maps(cmap1, cmap2)


# --------------------------------------------------------------------------- #
# チャンク → NER 小片 → 本文 (text / original_text / offset_map) の構築
#
# ここは **spaCy 非依存** (SudachiPy/GiNZA を呼ばない純粋な文字列処理)。NER (engine.py) も
# LLM 検出 (src/llm) も、まずこの層で「解析対象の本文」と「原文への位置対応表」を得る。
# build_body() は GiNZA を回さずに本文座標を作れるため、LLM-only 経路の起点になる
# (docs-dev/LLM適用_調査と設計たたき台.md の接続点 (J1) を参照)。
# --------------------------------------------------------------------------- #

# GiNZA が内部で使う SudachiPy のトークナイズ上限 (1 回の解析あたりのバイト数)。
# これを超えると `SudachiError: Input is too long` で落ちる。
SUDACHI_MAX_BYTES = 49149
# 上限に対する安全マージン。通常はチャンク分割 (src.core.document.text_splitter) で
# 既に十分小さくなっているが、巨大な 1 チャンク/1 行が来ても確実に通すための保険。
SAFE_CHUNK_BYTES = 40000

# チャンク結合時の区切り (表示テキスト＝解析テキストの連結に使う)。
CHUNK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class _Piece:
    """NER に渡す 1 小片。平坦化したときは原文との対応表も持つ。"""

    flat: str  # NER へ渡す (平坦化後) テキスト
    orig: str  # 対応する原文 (平坦化前。`|` 入り)
    cmap: tuple[int, ...]  # flat の各文字 → orig の文字位置 (挿入文字は -1)


@dataclass(frozen=True)
class Body:
    """チャンク列から作った「解析対象の本文」一式 (spaCy 非依存)。

    ``text`` は平坦化後・``CHUNK_SEPARATOR`` 連結の本文 (＝マスキングの検出/マージ座標。
    NER の :class:`~src.ner.engine.Analysis` の ``text`` と一致)。``original_text`` は
    平坦化前の原文 (`|` 入り)、``offset_map[i]`` は ``text`` の i 文字目に対応する
    ``original_text`` の文字位置 (挿入文字は -1)。平坦化しない場合も括弧空白挿入があるため
    ``text`` と ``original_text`` は厳密一致とは限らない (``offset_map`` で写す)。
    """

    text: str
    original_text: str
    offset_map: tuple[int, ...]


def _byte_safe_pieces(text: str, max_bytes: int = SAFE_CHUNK_BYTES) -> list[str]:
    """テキストを UTF-8 で ``max_bytes`` 以下の小片に分割する (保険的フォールバック)。

    まず行 (``\\n``) 境界でまとめ、1 行で超える場合のみ文字単位で強制分割する。
    通常はチャンク分割で十分小さいため、ここはほぼ素通りする。
    """
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    pieces: list[str] = []
    buf = ""
    for line in text.split("\n"):
        candidate = f"{buf}\n{line}" if buf else line
        if len(candidate.encode("utf-8")) <= max_bytes:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
            buf = ""
        if len(line.encode("utf-8")) > max_bytes:
            pieces.extend(_hard_split_by_bytes(line, max_bytes))
        else:
            buf = line
    if buf:
        pieces.append(buf)
    return pieces


def _hard_split_by_bytes(text: str, max_bytes: int) -> list[str]:
    """1 行が上限を超える場合に、文字単位でバイト数上限まで詰めて分割する。"""
    pieces: list[str] = []
    buf = ""
    buf_bytes = 0
    for ch in text:
        ch_bytes = len(ch.encode("utf-8"))
        if buf and buf_bytes + ch_bytes > max_bytes:
            pieces.append(buf)
            buf = ""
            buf_bytes = 0
        buf += ch
        buf_bytes += ch_bytes
    if buf:
        pieces.append(buf)
    return pieces


def _prepare_pieces(
    chunks: Iterable[str], *, flatten_tables: bool = False
) -> list[_Piece]:
    """チャンク列を、実際に NER へ渡す小片 (バイト数安全・空片除去済み) に整える。

    平坦化はバイト数安全分割の **後** に各小片へ適用し、原文 (`|` 入り) と平坦化後の
    文字位置対応表 (cmap) を保持する。NER (extract_chunks/debug_tokens/analyze_chunks) と
    LLM 検出の本文構築 (:func:`build_body`) が同じ入力で処理するよう、分割をここに集約する。
    """
    pieces: list[_Piece] = []
    for chunk in chunks:
        for orig in _byte_safe_pieces(chunk):
            # 括弧グルー対策の空白挿入は flatten の有無によらず常に適用する
            # (`姓A(社B)` の埋没＝辞書語の漏れを防ぐ)。flatten 時はテーブル平文化も。
            flat, cmap = prepare_for_ner_with_map(orig, flatten_tables=flatten_tables)
            if flat.strip():
                pieces.append(_Piece(flat, orig, tuple(cmap)))
    return pieces


def _body_from_pieces(pieces: list[_Piece]) -> Body:
    """小片列から本文一式を組む (``NerEngine.analyze_chunks`` と同じオフセット計算)。

    ``text`` は ``CHUNK_SEPARATOR`` で flat を連結。``offset_map`` は ``text`` の各文字 →
    ``original_text`` の文字位置 (挿入は -1)。小片間の区切り (``CHUNK_SEPARATOR``) は
    flat/orig 双方に入り原文側にも実在するため、-1 でなく実位置に対応づける。
    """
    omap: list[int] = []
    orig_parts: list[str] = []
    orig_offset = 0  # 原文基準のオフセット
    sep_len = len(CHUNK_SEPARATOR)
    for idx, piece in enumerate(pieces):
        if idx > 0:  # 小片の区切り (CHUNK_SEPARATOR) は flat/orig 双方に入る
            omap.extend(orig_offset + k for k in range(sep_len))
            orig_offset += sep_len
            orig_parts.append(CHUNK_SEPARATOR)
        omap.extend(orig_offset + c if c != -1 else -1 for c in piece.cmap)
        orig_offset += len(piece.orig)
        orig_parts.append(piece.orig)
    return Body(
        text=CHUNK_SEPARATOR.join(p.flat for p in pieces),
        original_text="".join(orig_parts),
        offset_map=tuple(omap),
    )


def build_body(chunks: Iterable[str], *, flatten_tables: bool = False) -> Body:
    """チャンク列から「解析対象の本文」一式 (text / original_text / offset_map) を作る。

    **spaCy を呼ばない**＝GiNZA を回さずに本文座標が得られる。NER も LLM 検出もこの本文を
    共有する (NER は加えてトークン/エンティティを doc から足す)。詳細は接続点 (J1)。
    """
    return _body_from_pieces(_prepare_pieces(chunks, flatten_tables=flatten_tables))
