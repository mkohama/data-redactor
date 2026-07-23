"""
DefaultSplitter - デフォルトのテキスト分割器

特殊な処理が不要なファイルタイプ (txt, html, xml等) に使用する
汎用的な RecursiveCharacterTextSplitter のラッパー。
"""

from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.document.splitters.base import BaseSplitter
from src.config import ChunkingConfig


class DefaultSplitter(BaseSplitter):
    """
    デフォルトのテキスト分割器

    RecursiveCharacterTextSplitter を使用して、
    段落、句読点、スペースなどの汎用的な区切り文字で分割する。
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self._splitter_cache: Dict[str, RecursiveCharacterTextSplitter] = {}

    def split(self, document: Document) -> List[Document]:
        """
        ドキュメントを汎用的なルールで分割

        Args:
            document: 分割対象のドキュメント

        Returns:
            分割されたドキュメントのリスト
        """
        file_type = document.metadata.get("file_type", "default").lower()
        splitter = self._get_or_create_splitter(file_type)
        return splitter.split_documents([document])

    def _get_or_create_splitter(self, file_type: str) -> RecursiveCharacterTextSplitter:
        """
        ファイルタイプに応じた splitter を取得 (キャッシュ付き)
        """
        if file_type in self._splitter_cache:
            return self._splitter_cache[file_type]

        # 汎用的な separator 設定
        # fmt: off
        separators = [
            r"\n\n",          # 二重改行
            r"\n",            # 改行
            r"(?<=[。！？])",  # 日本語句読点 (肯定後読み)
            r"\. ",           # 句点 + space
            r"! ",            # 感嘆符 + space
            r"\? ",           # 疑問符 + space
            r" ",             # 単語区切り
            r"",              # 文字単位
        ]
        # fmt: on

        splitter = self.create_recursive_splitter(file_type, separators)

        self._splitter_cache[file_type] = splitter
        return splitter
