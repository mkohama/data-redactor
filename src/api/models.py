"""マスキング HTTP API の入出力スキーマ（Pydantic）。設計 §3-1 / §3-2。

wire（HTTP 表現）の形だけを定義する。確信度の ASCII↔日本語変換は :mod:`src.api.enums`、
検出・マスクの実体は :mod:`src.masking` が持つ（ここは「形」に徹する）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.api.enums import (
    DEFAULT_DETECTION,
    DEFAULT_MASK_LEVEL,
)


class FileRef(BaseModel):
    """同梱ファイル part の参照（本体は multipart で別送。ここは拡張子判定用の filename だけ）。"""

    filename: str


class Part(BaseModel):
    """``/mask`` の入力 part（3 種のいずれか＝text / content_hash / file。設計 §3-1）。"""

    id: str
    text: str | None = None
    content_hash: str | None = None
    file: FileRef | None = None


class MaskRequest(BaseModel):
    """``POST /mask`` のリクエスト（application/json。同梱ファイルは multipart のマニフェスト）。"""

    parts: list[Part] | None = None
    # 単一 text は parts:[{id:"_", text}] の糖衣（設計 §3-1）。
    text: str | None = None
    detection: str = DEFAULT_DETECTION
    mask_level: str = DEFAULT_MASK_LEVEL
    flatten_tables: bool = True
    models: list[str] | None = None
    return_pending: bool = True


class MaskedPart(BaseModel):
    """出力 part（入力と同じ id。masked_text は抽出後テキストのマスク結果）。"""

    id: str
    masked_text: str


class Occurrence(BaseModel):
    """プレースホルダ / pending の 1 出現（どのパートのどのスパンか）。"""

    part: str
    span: tuple[int, int]


class MappingEntry(BaseModel):
    """バンドルで共有する対応表 1 件（設計 §3-1 の ``mapping[]``）。"""

    placeholder: str
    category: str
    canonical: str
    surfaces: list[str]
    confidence: str  # wire（ASCII）
    decided_by: str
    occurrences: list[Occurrence]


class PendingEntry(BaseModel):
    """自動マスク閾値未満のレビュー候補（設計 §3-1 の ``pending[]``）。"""

    surface: str
    category: str
    confidence: str  # wire（ASCII）
    occurrences: list[Occurrence]
    votes: dict[str, str]  # {"ner": "地名", "llm": "人名"} 形式（系統別カテゴリ）


class DetectorInfo(BaseModel):
    """このマスクに使った検出器の構成（監査・再現用）。"""

    detection: str
    models: list[str]
    detector_version: str
    mask_level: str


class MaskResponse(BaseModel):
    """``POST /mask`` のレスポンス（設計 §3-1）。"""

    status: str = "unconfirmed"  # confirmed 層は後回し（当面固定）
    masked_parts: list[MaskedPart]
    mapping: list[MappingEntry]
    pending: list[PendingEntry] = Field(default_factory=list)
    detector: DetectorInfo


class UnmaskRequest(BaseModel):
    """``POST /unmask`` のリクエスト（設計 §3-2）。"""

    text: str
    mapping: list[MappingEntry]


class UnmaskResponse(BaseModel):
    """``POST /unmask`` のレスポンス。"""

    restored_text: str
