"""
BaseSplitter 抽象基底クラス

全てのファイルタイプ別 Splitter の基底クラスを定義する。
"""

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkingConfig
from src.core.document.splitters.token_utils import tiktoken_len


class BaseSplitter(ABC):
    """
    テキスト分割の抽象基底クラス

    各ファイルタイプに特化した Splitter はこのクラスを継承し、
    split メソッドを実装する必要がある。
    """

    def __init__(self, config: ChunkingConfig):
        """
        初期化

        Args:
            config: チャンク設定
        """
        self.config = config

    @abstractmethod
    def split(self, document: Document) -> List[Document]:
        """
        ドキュメントをチャンクに分割

        Args:
            document: 分割対象のドキュメント

        Returns:
            分割されたドキュメントのリスト
        """
        pass

    def get_file_type(self) -> str:
        """
        このSplitterが対応するファイルタイプを返す

        Returns:
            ファイルタイプ文字列 (例: "md", "pdf")
        """
        # デフォルトはクラス名から推測
        class_name = self.__class__.__name__
        if class_name.endswith("Splitter"):
            return class_name[:-8].lower()
        return "unknown"

    def create_recursive_splitter(
        self, file_type: str, separators: List[str], is_separator_regex: bool = True
    ) -> RecursiveCharacterTextSplitter:
        """
        RecursiveCharacterTextSplitter を作成するヘルパーメソッド

        Args:
            file_type: ファイルタイプ (設定取得用)
            separators: 区切り文字のリスト
            is_separator_regex: 区切り文字を正規表現として扱うか

        Returns:
            設定済みの RecursiveCharacterTextSplitter
        """
        settings = self.config.get_settings(file_type)

        return RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=tiktoken_len,  # トークンベース
            is_separator_regex=is_separator_regex,
            separators=separators,
        )
