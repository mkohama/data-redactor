"""``src.ner.preprocess.build_body`` の特性化テスト（spaCy 非依存・モデル不要）。

build_body は「chunks → 解析対象の本文（text / original_text / offset_map）」を GiNZA を
回さずに作る純関数（接続点 J1）。NER も LLM 検出もこの本文座標を共有するため、ここが
壊れると検出スパン→原文マスクの写しが全部ずれる。挙動を固定して回帰を防ぐ。

不変条件の要：``offset_map[i] != -1`` のとき ``text[i] == original_text[offset_map[i]]``
（挿入文字＝区切り `、`/句点 `。`/連結改行/括弧パディングの空白 は -1）。
"""

from __future__ import annotations

from src.ner.preprocess import CHUNK_SEPARATOR, build_body


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


def test_empty_input() -> None:
    """空入力は空の本文を返す（例外を投げない）。"""
    b = build_body([], flatten_tables=False)
    assert b.text == ""
    assert b.original_text == ""
    assert b.offset_map == ()
