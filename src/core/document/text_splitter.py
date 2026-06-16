"""ファイルタイプを考慮したテキスト分割器（kb-mcp から移植）。

kb-mcp が RAG 格納前に行うチャンク分割と同一ロジック。NER 前に同じ単位で
チャンク化することで、(1) Sudachi のトークナイズ上限（49,149 バイト）を確実に
下回る、(2) kb-mcp の検索ヒット単位と抽出結果が揃う、という利点がある。

デバッグ用ログ出力（`_log_chunking_details`）は data-redactor では不要なため省略した。
"""

from typing import List, Dict
from langchain_core.documents import Document

from src.config import ChunkingConfig
from src.core.document.splitters.base import BaseSplitter
from src.core.document.splitters.default_splitter import DefaultSplitter
from src.core.document.splitters.markdown_splitter import MarkdownSplitter
from src.core.document.splitters.pdf_splitter import PDFSplitter
from src.core.document.splitters.excel_splitter import ExcelSplitter


class SemanticRAGTextSplitter:
    """
    ファイルタイプを考慮したテキスト分割器（ファクトリー/ルーター）

    ファイルタイプに応じて適切な Splitter に処理を委譲する。
    各ファイルタイプの具体的な分割ロジックは splitters/ 配下の専用クラスで実装する。
    """

    def __init__(self, config: ChunkingConfig):
        """初期化"""
        self.config = config

        # ファイルタイプ別の Splitter マッピング
        # (docx も Markdown 形式として扱う)
        self._splitters: Dict[str, BaseSplitter] = {
            "md": MarkdownSplitter(config),
            "docx": MarkdownSplitter(config),
            "pdf": PDFSplitter(config),
            "xlsx": ExcelSplitter(config, "xlsx"),
            "xls": ExcelSplitter(config, "xls"),
            "xlsm": ExcelSplitter(config, "xlsm"),
        }

        # デフォルト Splitter（特殊処理が不要なファイルタイプ用）
        self._default_splitter = DefaultSplitter(config)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        ドキュメントリストをファイルタイプに応じてチャンク化

        Args:
            documents: 分割対象のドキュメントリスト

        Returns:
            チャンク化されたドキュメントのリスト（chunk_indexが付与される）
        """
        all_chunks: List[Document] = []

        for doc in documents:
            # メタデータからファイルタイプを取得
            file_type = doc.metadata.get("file_type", "default").lower()

            # 適切な Splitter を選択
            splitter = self._splitters.get(file_type, self._default_splitter)

            # 分割実行
            doc_chunks = splitter.split(doc)
            all_chunks.extend(doc_chunks)

        # 全ドキュメント群に対して通し番号 (chunk_index) を付与
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_index"] = i

        return all_chunks
