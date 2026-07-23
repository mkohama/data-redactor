"""
ExcelSplitter - Excel専用のテキスト分割器

行構造を保持する特殊な separator 設定を使用する。
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.document.splitters.base import BaseSplitter
from src.config import ChunkingConfig


class ExcelSplitter(BaseSplitter):
    """
    Excel専用のテキスト分割器

    テーブル行全体を保持し、セル区切りを考慮した分割を行う。
    """

    def __init__(self, config: ChunkingConfig, file_type: str = "xlsx"):
        super().__init__(config)
        self.file_type = file_type
        self._splitter = None

    def split(self, document: Document) -> List[Document]:
        """
        Excelドキュメントを行構造を保持して分割

        Args:
            document: 分割対象のExcelドキュメント

        Returns:
            分割されたドキュメントのリスト
        """
        splitter = self._get_splitter()
        return splitter.split_documents([document])

    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        """Excel用の Splitter を取得"""
        if self._splitter is None:

            # Excel用の separator (行構造を優先)
            # fmt: off
            separators = [
                r"\n",             # 1. 最優先: 行の区切り (テーブル行全体を保持)
                r"\n\n",           # 2. 段落の区切り
                r"\|",             # 3. セルの区切り
                r"(?<=[。！？、])",  # 4. 日本語句読点
                r" ",              # 5. 単語単位
                r"",               # 6. 文字単位
            ]
            # fmt: on

            self._splitter = self.create_recursive_splitter(self.file_type, separators)

        return self._splitter
