"""
PDFSplitter - PDF専用のテキスト分割器

PDF特有のノイズ (改行の乱れ等) を除去してから分割する。
"""

import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.document.splitters.base import BaseSplitter
from src.config import ChunkingConfig


class PDFSplitter(BaseSplitter):
    """
    PDF専用のテキスト分割器

    PDFから抽出されたテキストの改行ノイズを除去し、段落を復元してから分割する。
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self._splitter = None  # 遅延初期化

    def split(self, document: Document) -> List[Document]:
        """
        PDFドキュメントを前処理してから分割

        Args:
            document: 分割対象のPDFドキュメント

        Returns:
            分割されたドキュメントのリスト
        """
        # PDF特有の前処理
        cleaned_text = self._clean_pdf_text(document.page_content)
        cleaned_doc = Document(page_content=cleaned_text, metadata=document.metadata)

        # 分割実行
        splitter = self._get_splitter()
        return splitter.split_documents([cleaned_doc])

    def _clean_pdf_text(self, text: str) -> str:
        """
        PDFから抽出されたテキストをクリーンアップ

        改行ノイズの除去、段落復元、箇条書きや番号付きリストの統一、余計な空白の削除を行う。
        """
        # 段落復元
        text = self._restore_paragraphs(text)
        # リスト正規化
        text = self._normalize_lists(text)
        # 余分なスペース削除
        text = self._remove_extra_spaces(text)

        return text

    def _restore_paragraphs(self, text: str) -> str:
        """文脈を考慮して段落を復元"""

        # 文末記号 (。！？.!?) で終わらない単一改行はスペースに置換
        text = re.sub(r"(?<![。！？.!?])\n(?!\n)", " ", text)

        # 複数改行は段落として保持 (\n\nに正規化)
        text = re.sub(r"\n{2,}", "\n\n", text)

        return text

    def _normalize_lists(self, text: str) -> str:
        """箇条書きや番号付きリストの構造を統一"""

        # 箇条書きの先頭 (•, -, –) にスペースを挿入
        text = re.sub(r"\n([•\-–])\s*", r"\n\1 ", text)

        # 数字付きリスト (例: 1.1., 2.) を一行にまとめる
        text = re.sub(r"\n(\d+(?:\.\d+)*\.)\s*", r"\n\1 ", text)

        return text

    def _remove_extra_spaces(self, text: str) -> str:
        """連続するスペースや文頭・文末の空白を削除"""

        # 連続スペースは1つに集約
        text = re.sub(r" +", " ", text)

        # 文頭・文末の空白削除
        return text.strip()

    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        """PDF用の Splitter を取得"""
        if self._splitter is None:
            # fmt: off
            separators = [
                r"\n\n\n",       # 複数改行
                r"\n\n",         # 二重改行
                r"(?<=[。！？])", # 日本語句読点 (肯定後読み)
                r"\. ",          # 句点 + space
                r"! ",           # 感嘆符 + space
                r"\? ",          # 疑問符 + space
                r" ",            # 単語区切り
                r"",             # 文字単位
            ]
            # fmt: on

            self._splitter = self.create_recursive_splitter("pdf", separators)

        return self._splitter
