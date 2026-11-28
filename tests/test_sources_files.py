"""テキストのファイル (.txt・.md) を、中身を書き換えずに読むことのテスト。

2026-11-28 の指摘：file で送ると、全角→半角の正規化・.md の見出しの行の欠け・
空行や末尾の改行の変化が起き、masked_text と span が原文と対応しなかった。
.txt・.md は中身をそのまま 1 チャンクにする (src/sources/files.py)。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sources.files import load_chunks_from_file, read_text_as_is

# 指摘の再現に使われた文 (合成)。見出しだけの行・全角の記号と英数字・表・末尾の改行を含む。
SAMPLE = (
    "# 機能Ａ\n\n## 概要\n\n### パッケージ名：サンプル\n\n### 担当：佐藤\n\n"
    "- 条件：「X@TP」 == 1　／　// 説明\n\n"
    "| 名前 | 単位 | 備考 |\n| --- | --- | --- |\n"
    "| 長さ | ｍ | 予備（未使用） |\n| 記号 | － | Ａ＆Ｂ |\n\n"
    "担当は佐藤。連絡先は田中。\n"
)


@pytest.mark.parametrize("ext", [".md", ".txt", ".MD"])
def test_text_file_is_one_unchanged_chunk(tmp_path: Path, ext: str) -> None:
    p = tmp_path / f"sample{ext}"
    p.write_bytes(SAMPLE.encode("utf-8"))
    assert load_chunks_from_file(p) == [SAMPLE]


def test_keeps_crlf_and_bom(tmp_path: Path) -> None:
    """改行コードも BOM もそのまま (テキストモードで開くと \\r\\n が \\n に変わる)。"""
    text = "﻿一行目\r\n二行目\r\n"
    p = tmp_path / "crlf.txt"
    p.write_bytes(text.encode("utf-8"))
    assert read_text_as_is(p) == text


def test_reads_cp932(tmp_path: Path) -> None:
    text = "担当は佐藤（ＳＯＮＹ）。\r\n"
    p = tmp_path / "sjis.txt"
    p.write_bytes(text.encode("cp932"))
    assert read_text_as_is(p) == text
