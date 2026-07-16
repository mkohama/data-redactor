"""確信度の wire(ASCII) ↔ 内部(日本語) 変換と、API で使う列挙値。設計 §1-A。

wire（HTTP 表現）は **ASCII enum を正**とし、エンジン内部は日本語（確定/強/中/弱/微弱/除外）を使う。
API 層はこのモジュール**だけ**で両者を橋渡しする（変換ロジックを各所に散らさない）。

- カテゴリ（人名/社名/商標/地名/連絡先/その他）は wire でも**日本語のまま**（設計 §3-1 の mapping 例）。
- ``mask_level`` は「自動マスクする下限」。指定順位**以上**を自動マスクする（§1-A）。
- ``excluded`` は下限に関係なく**常に対象外**（allowlist で人が「機密でない」と外した語）。
"""

from __future__ import annotations

# 確信度：wire(ASCII) ↔ 内部(日本語)。高→低の順（設計 §1-A の順位表と一致させる）。
_CONFIDENCE_PAIRS: tuple[tuple[str, str], ...] = (
    ("certain", "確定"),
    ("strong", "強"),
    ("medium", "中"),
    ("weak", "弱"),
    ("faint", "微弱"),
    ("excluded", "除外"),
)

CONFIDENCE_ASCII_TO_JP: dict[str, str] = {a: j for a, j in _CONFIDENCE_PAIRS}
CONFIDENCE_JP_TO_ASCII: dict[str, str] = {j: a for a, j in _CONFIDENCE_PAIRS}

# 自動マスクの下限判定に使う順位（大きいほど強い）。excluded は別枠（順位を持たせない）。
_MASKABLE_ORDER: tuple[str, ...] = ("certain", "strong", "medium", "weak", "faint")
_CONFIDENCE_RANK: dict[str, int] = {
    a: len(_MASKABLE_ORDER) - i for i, a in enumerate(_MASKABLE_ORDER)
}

# /config・入力バリデーション用の選択肢。
MASK_LEVELS: tuple[str, ...] = _MASKABLE_ORDER
DETECTION_MODES: tuple[str, ...] = ("ner", "llm", "both")
DEFAULT_MASK_LEVEL = "strong"
DEFAULT_DETECTION = "both"


def confidence_to_wire(jp: str) -> str:
    """内部（日本語）→ wire（ASCII）。未知値はそのまま返す（前方互換）。"""
    return CONFIDENCE_JP_TO_ASCII.get(jp, jp)


def confidence_from_wire(ascii_conf: str) -> str:
    """wire（ASCII）→ 内部（日本語）。未知値はそのまま返す。"""
    return CONFIDENCE_ASCII_TO_JP.get(ascii_conf, ascii_conf)


def confidences_at_or_above(mask_level: str) -> tuple[str, ...]:
    """``mask_level`` 以上（同順位含む）の **内部確信度（日本語）** 集合を返す。

    ``/mask`` の既定選択（auto_selection）に使う。``excluded`` は下限に関係なく常に
    対象外なので含めない。未知の ``mask_level`` は :class:`ValueError`。
    """
    floor = _CONFIDENCE_RANK.get(mask_level)
    if floor is None:
        raise ValueError(f"unknown mask_level: {mask_level!r}")
    return tuple(
        CONFIDENCE_ASCII_TO_JP[a]
        for a in _MASKABLE_ORDER
        if _CONFIDENCE_RANK[a] >= floor
    )
