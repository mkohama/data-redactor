"""ファイルを入力ソースとして扱うアダプタ。

kb-mcp から移植した DocumentLoader でファイルをテキスト化し、同じく kb-mcp の
チャンク分割（SemanticRAGTextSplitter）でチャンク化して返す。

NER 前にチャンク化する理由:
- GiNZA 内部の SudachiPy は 1 回の解析で 49,149 バイトまでしか扱えず、仕様書 1 本
  （数十〜数百 KB）を丸ごと渡すと `SudachiError: Input is too long` で落ちる。
- kb-mcp が RAG 格納時に使うのと同じ分割単位（xlsx≈1000 トークン等）にすることで、
  検索ヒット単位と抽出結果が揃う。
"""

from __future__ import annotations

from pathlib import Path

from src.config import ChunkingConfig
from src.core.document.document_loader import DocumentLoader
from src.core.document.text_splitter import SemanticRAGTextSplitter


def load_chunks_from_file(file_path: str | Path) -> list[str]:
    """ファイルをテキスト化し、kb-mcp と同じ単位でチャンク化して返す。

    PDF / Excel / PowerPoint などは複数の Document（ページ/シート単位）に
    分割されて返る。それを結合せず、ファイルタイプに応じた Splitter で
    チャンク化したテキストのリストを返す。
    """
    docs = DocumentLoader().load_document(file_path)
    chunks = SemanticRAGTextSplitter(ChunkingConfig()).split_documents(docs)
    return [c.page_content for c in chunks]


def load_text_from_file(file_path: str | Path) -> str:
    """ファイルをテキスト化して 1 本のテキストにまとめて返す（後方互換）。

    チャンク境界を区切りで連結するだけ。長文をそのまま NER に渡すと
    SudachiPy の上限で落ちるため、解析には :func:`load_chunks_from_file` を使う。
    """
    return "\n\n".join(load_chunks_from_file(file_path))
