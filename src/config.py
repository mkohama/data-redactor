"""チャンク分割の設定 (kb-mcp から移植・チャンク設定のみ抜粋)。

kb-mcp 本体の `src/config.py` は埋め込み・ストレージ等も含む大きな設定だが、
data-redactor では NER 前のチャンク分割にしか使わないため、`SemanticRAGTextSplitter`
が必要とする `ChunkingConfig` / `FileTypeChunkingSettings` だけを移植している。
値 (chunk_size / chunk_overlap) は kb-mcp と同一にして、RAG 格納時と同じ単位で
チャンクされるようにしている (chunk_size はトークン数＝tiktoken cl100k_base)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FileTypeChunkingSettings:
    """ファイルタイプ別のチャンク設定。"""

    chunk_size: int
    chunk_overlap: int


@dataclass
class ChunkingConfig:
    """テキスト分割の設定 (ファイルタイプ別)。"""

    # ファイルタイプ別の設定 (kb-mcp の値と一致させる)
    filetype_settings: Dict[str, FileTypeChunkingSettings] = field(
        default_factory=lambda: {
            "txt": FileTypeChunkingSettings(chunk_size=1000, chunk_overlap=100),
            "md": FileTypeChunkingSettings(chunk_size=800, chunk_overlap=0),
            "pdf": FileTypeChunkingSettings(chunk_size=1000, chunk_overlap=100),
            "docx": FileTypeChunkingSettings(chunk_size=900, chunk_overlap=50),
            "doc": FileTypeChunkingSettings(chunk_size=900, chunk_overlap=50),
            "xlsx": FileTypeChunkingSettings(chunk_size=1000, chunk_overlap=0),
            "xls": FileTypeChunkingSettings(chunk_size=1000, chunk_overlap=0),
            "xlsm": FileTypeChunkingSettings(chunk_size=1000, chunk_overlap=0),
            "pptx": FileTypeChunkingSettings(chunk_size=800, chunk_overlap=50),
            "ppt": FileTypeChunkingSettings(chunk_size=800, chunk_overlap=50),
            "html": FileTypeChunkingSettings(chunk_size=1000, chunk_overlap=100),
            "xml": FileTypeChunkingSettings(chunk_size=1000, chunk_overlap=100),
            "default": FileTypeChunkingSettings(chunk_size=1000, chunk_overlap=100),
        }
    )

    def get_settings(self, file_type: str) -> FileTypeChunkingSettings:
        """ファイルタイプに応じた設定を取得 (未知タイプは default)。"""
        return self.filetype_settings.get(file_type, self.filetype_settings["default"])
