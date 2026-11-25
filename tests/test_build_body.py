"""``src.ner.preprocess.build_body`` の特性化テスト（spaCy 非依存・モデル不要）。

build_body は「chunks → 解析対象の本文（text / original_text / offset_map）」を GiNZA を
回さずに作る純関数（接続点 J1）。NER も LLM 検出もこの本文座標を共有するため、ここが
壊れると検出スパン→原文マスクの写しが全部ずれる。挙動を固定して回帰を防ぐ。

不変条件の要：``offset_map[i] != -1`` のとき ``text[i] == original_text[offset_map[i]]``
（挿入文字＝区切り `、`/句点 `。`/連結改行/括弧パディングの空白 は -1）。
"""

from __future__ import annotations

from src.ner.preprocess import (
    CHUNK_SEPARATOR,
    SAFE_CHUNK_BYTES,
    _byte_safe_pieces,
    build_body,
)


def _assert_map_invariant(b) -> None:
    """offset_map が text の各文字を original_text の正しい文字へ写すこと。"""
    assert len(b.offset_map) == len(b.text)
    for i, o in enumerate(b.offset_map):
        if o != -1:
            assert 0 <= o < len(b.original_text)
            assert b.text[i] == b.original_text[o]


def test_single_chunk_is_identity() -> None:
    """平坦化なし・括弧なしの素直な本文は text==original_text・恒等写像。"""
    b = build_body(["田中さんは横浜にいます"], flatten_tables=False)
    _assert_map_invariant(b)
    assert b.text == b.original_text == "田中さんは横浜にいます"
    assert b.offset_map == tuple(range(len(b.text)))


def test_bracket_padding_inserts_space_but_preserves_original() -> None:
    """括弧グルー対策の空白挿入で text は変わるが original は原文のまま・空白は -1。"""
    b = build_body(["姓A(社B)"], flatten_tables=False)
    _assert_map_invariant(b)
    assert b.original_text == "姓A(社B)"
    assert " " in b.text  # `(` の前に空白が挿入される
    assert -1 in b.offset_map  # 挿入空白は原文対応なし
    # 挿入文字を除けば原文が復元できる
    restored = "".join(b.original_text[o] for o in b.offset_map if o != -1)
    assert restored == b.original_text


def test_multi_chunk_joined_by_separator() -> None:
    """複数チャンクは CHUNK_SEPARATOR で連結。区切りは flat/orig 双方に実在（-1 でない）。"""
    b = build_body(["第一章", "第二章"], flatten_tables=False)
    _assert_map_invariant(b)
    assert b.text == "第一章" + CHUNK_SEPARATOR + "第二章"
    assert CHUNK_SEPARATOR in b.original_text
    # 区切り位置の offset_map は -1 でなく実位置（原文にも区切りがあるため）
    sep_start = len("第一章")
    for k in range(len(CHUNK_SEPARATOR)):
        assert b.offset_map[sep_start + k] != -1


def test_flatten_table_drops_separator_row_and_keeps_map() -> None:
    """テーブル平坦化後も offset_map 不変条件が保たれる（挿入の `、`/`。` は -1）。"""
    b = build_body(
        ["| 名前 | 所属 |\n| --- | --- |\n| 田中 | 営業 |"], flatten_tables=True
    )
    _assert_map_invariant(b)
    assert "田中" in b.text and "営業" in b.text
    assert "---" not in b.text  # 区切り行は削除される


def test_blank_chunks_dropped() -> None:
    """空白のみのチャンクは小片化で落ちる（本文に出ない）。"""
    b = build_body(["   ", "本文", "\n\n"], flatten_tables=False)
    _assert_map_invariant(b)
    assert b.text == "本文"


def _long_markdown(min_bytes: int) -> str:
    """空行・字下げした箇条書きを含み、UTF-8 で min_bytes を超える Markdown。"""
    sections = []
    i = 0
    while len("\n".join(sections).encode("utf-8")) <= min_bytes:
        sections.append(
            f"## 項目{i}\n\n田中さんが確認した内容を記す。\n\n  - 手順を確認する\n  - 結果を記録する\n"
        )
        i += 1
    return "\n".join(sections)


def test_byte_safe_pieces_concatenate_back_to_the_input() -> None:
    """小片を何も足さずにつなぐと入力に戻り、各小片は上限以下に収まる。

    以前は分けた位置の改行を捨て、つなぎ直すときに CHUNK_SEPARATOR ("\\n\\n") を
    足していたため、区切りごとに改行が 1 つ (1 行の途中で切ると 2 つ) 増えていた。
    """
    cases = [
        "一行目\n二行目\n\n  - 箇条\n最後の行",  # 行の境目で分かれる
        "末尾に改行がある\n" * 8,  # 最後が改行で終わる
        "改行の無い長い一行" * 10,  # 1 行の途中で切る
        "\n\n\n先頭と途中に空行\n\n\n",  # 空行だけの行
    ]
    for text in cases:
        pieces = _byte_safe_pieces(text, max_bytes=20)
        assert len(pieces) > 1
        assert "".join(pieces) == text
        assert all(len(p.encode("utf-8")) <= 20 for p in pieces)


def test_long_single_chunk_keeps_original_text() -> None:
    """SAFE_CHUNK_BYTES を超える 1 チャンクでも、original_text は入力と完全に一致する。

    /mask の text は 1 チャンクで来るので、ここがずれると masked_text と span が
    呼び出し側の原文とずれる (2026-11-25 の不具合報告)。
    """
    text = _long_markdown(SAFE_CHUNK_BYTES * 2)
    for flatten in (False, True):
        b = build_body([text], flatten_tables=flatten)
        _assert_map_invariant(b)
        assert b.original_text == text
    # 平坦化なし・括弧なしなら text も入力のまま
    assert build_body([text], flatten_tables=False).text == text


def test_long_single_line_keeps_original_text() -> None:
    """改行の無い 1 行が上限を超えても、行の途中に改行を足さない。"""
    text = "前置き\n" + "あ" * 13333 + "SONY" + "い" * 100
    b = build_body([text], flatten_tables=False)
    _assert_map_invariant(b)
    assert b.original_text == text
    assert "SONY" in b.text


def test_long_chunk_among_chunks() -> None:
    """長いチャンクを含んでも、チャンクの間にだけ CHUNK_SEPARATOR が入る。"""
    long_text = _long_markdown(SAFE_CHUNK_BYTES + 1000)
    b = build_body(["前のチャンク", long_text, "後のチャンク"], flatten_tables=False)
    _assert_map_invariant(b)
    assert b.original_text == CHUNK_SEPARATOR.join(
        ["前のチャンク", long_text, "後のチャンク"]
    )


def test_empty_input() -> None:
    """空入力は空の本文を返す（例外を投げない）。"""
    b = build_body([], flatten_tables=False)
    assert b.text == ""
    assert b.original_text == ""
    assert b.offset_map == ()
