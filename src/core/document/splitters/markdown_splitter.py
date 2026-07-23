"""
MarkdownSplitter - Markdown専用のテキスト分割器

MarkdownHeaderTextSplitter を使用してヘッダー構造に基づいた分割を行い、
コンテキストアウェアなマージで重複を排除する。
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.core.document.splitters.base import BaseSplitter
from src.config import ChunkingConfig


class MarkdownSplitter(BaseSplitter):
    """
    Markdown専用のテキスト分割器

    h1〜h9までの見出しレベルで構造的に分割し、チャンクサイズに基づいて適切にマージする。
    同一チャンク内でのヘッダー重複を避けるため、コンテキストアウェアなマージを行う。
    """

    # ヘッダー定義 (h1〜h9)
    HEADERS_TO_SPLIT_ON = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
        ("#######", "h7"),
        ("########", "h8"),
        ("#########", "h9"),
    ]

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self._fallback_splitter = None  # 遅延初期化

    def split(self, document: Document) -> List[Document]:
        """
        Markdownドキュメントをヘッダー構造に基づいて分割

        Args:
            document: 分割対象のMarkdownドキュメント

        Returns:
            分割されたドキュメントのリスト
        """
        text = document.page_content

        # MarkdownHeaderTextSplitterで構造的に分割
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS_TO_SPLIT_ON,
            strip_headers=True,  # メタデータに移して、本文からは削除
        )

        splits = md_splitter.split_text(text)

        final_chunks = []

        # チャンク構築用バッファ
        current_chunk_text = ""
        current_chunk_metadata = document.metadata.copy()

        # 直前のセクションのヘッダーライン (重複排除用)
        current_chunk_last_header_lines: List[str] = []

        # 設定取得
        settings = self.config.get_settings("md")
        chunk_size = settings.chunk_size

        for split in splits:
            # コンテンツが空の場合はスキップ
            if not split.page_content.strip():
                continue

            # このセクションのヘッダー階層をリスト化
            section_header_lines = []
            for tag, key in self.HEADERS_TO_SPLIT_ON:
                if key in split.metadata:
                    section_header_lines.append(f"{tag} {split.metadata[key]}")

            # チャンク構築時のテキスト決定 (重複排除ロジック)
            content_to_append = self._build_content_to_append(
                current_chunk_text,
                section_header_lines,
                current_chunk_last_header_lines,
                split.page_content,
            )

            # サイズ判定
            estimated_size = (
                len(current_chunk_text) + len(content_to_append) + 2
                if current_chunk_text
                else len(content_to_append)
            )

            # サイズに収まるときはバッファに追加
            if estimated_size <= chunk_size:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + content_to_append
                else:
                    current_chunk_text = content_to_append

                current_chunk_metadata.update(split.metadata)
                current_chunk_last_header_lines = section_header_lines

            # サイズオーバーの場合
            else:
                # これまでのバッファでチャンク確定
                if current_chunk_text:
                    final_chunks.append(
                        Document(
                            page_content=current_chunk_text,
                            metadata=current_chunk_metadata.copy(),
                        )
                    )

                # これまでのバッファをリセット
                current_chunk_text = ""
                current_chunk_metadata = document.metadata.copy()
                current_chunk_last_header_lines = []

                # セクションのテキストを取得
                full_section_text = self._build_full_section_text(
                    section_header_lines, split.page_content
                )

                # セクションが大きい場合は再帰的に分割
                if len(full_section_text) > chunk_size:
                    large_doc = Document(
                        page_content=full_section_text,
                        metadata=document.metadata.copy(),
                    )
                    large_doc.metadata.update(split.metadata)

                    sub_chunks = self._get_fallback_splitter().split_documents(
                        [large_doc]
                    )
                    final_chunks.extend(sub_chunks)

                # セクションが小さい場合はバッファの先頭に追加
                else:
                    current_chunk_text = full_section_text
                    current_chunk_metadata.update(split.metadata)
                    current_chunk_last_header_lines = section_header_lines

        # 最後のバッファでチャンク確定(フラッシュ)
        if current_chunk_text:
            final_chunks.append(
                Document(
                    page_content=current_chunk_text,
                    metadata=current_chunk_metadata.copy(),
                )
            )

        return final_chunks

    def _build_content_to_append(
        self,
        current_chunk_text: str,
        section_header_lines: List[str],
        current_chunk_last_header_lines: List[str],
        page_content: str,
    ) -> str:
        """
        追加するコンテンツを構築 (重複排除ロジック)
        """
        if not current_chunk_text:
            # バッファが空 -> 全ヘッダー + 本文
            full_header_text = "\n".join(section_header_lines)
            if full_header_text:
                return f"{full_header_text}\n{page_content}"
            else:
                return page_content
        else:
            # バッファあり -> 共通ヘッダーを除外して追記
            common_prefix_len = 0
            for i in range(
                min(len(current_chunk_last_header_lines), len(section_header_lines))
            ):
                if current_chunk_last_header_lines[i] == section_header_lines[i]:
                    common_prefix_len += 1
                else:
                    break

            unique_headers = section_header_lines[common_prefix_len:]
            unique_header_text = "\n".join(unique_headers)

            if unique_header_text:
                return f"{unique_header_text}\n{page_content}"
            else:
                return page_content

    def _build_full_section_text(
        self, section_header_lines: List[str], page_content: str
    ) -> str:
        """
        フルヘッダー付きセクションテキストを生成
        """
        full_header_text = "\n".join(section_header_lines)
        if full_header_text:
            return f"{full_header_text}\n{page_content}".strip()
        else:
            return page_content.strip()

    def _get_fallback_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        巨大セクション用のフォールバック Splitter を取得
        """
        if self._fallback_splitter is None:

            # Markdown用の separator (ヘッダー以外)
            # fmt: off
            separators = [
                # 見出しレベルは前段で処理

                # テーブルっぽい行の前 (ヘッダー行 + 区切り行のパターン)
                r"\n+(?=\|[^\n]+\|[^\n]*\n\|[-:\s|]+\|)",

                # HTMLテーブルの前
                r"\n+(?=<table[^>]*>)",

                # コードブロックの前 (``` のチャンクができたりするので対象外...`)
                # r"```\n",

                # 階層的番号 (例: "1. ", "1.1. ", "2.3.4. ")
                r"\n+\s*(?=\d+(?:\.\d+)*\.\s*[^#\n]+)",

                # 段落区切り
                r"\n\n+",

                # 行の区切り
                r"\n",

                # 日本語の句読点など(句点/感嘆符/疑問符) - 肯定後読み
                r"(?<=[。！？])",

                # 英語圏の句読点 + スペース (ピリオド、感嘆符、疑問符)
                r"\.\s",
                r"!\s",
                r"\?\s",

                # 単語区切り(連続スペース)
                r"\s+",

                # ※ "文字単位" 分割は安全な別処理で対応 (ここには含めない)
            ]
            # fmt: on

            self._fallback_splitter = self.create_recursive_splitter("md", separators)

        return self._fallback_splitter
