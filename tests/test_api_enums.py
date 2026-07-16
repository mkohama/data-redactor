"""確信度 wire(ASCII) ↔ 内部(日本語) 変換のテスト（設計 §1-A）。

エンジン内部の確信度集合（``_CONF_RANK``）と wire の対応表が**ズレていない**ことも検査する
（片方だけ確信度を足して取りこぼす事故を防ぐ）。
"""

from __future__ import annotations

import pytest

from src.api.enums import (
    CONFIDENCE_ASCII_TO_JP,
    confidence_from_wire,
    confidence_to_wire,
    confidences_at_or_above,
)
from src.masking.engine import _CONF_RANK


def test_round_trip_all_confidences() -> None:
    for ascii_conf, jp in CONFIDENCE_ASCII_TO_JP.items():
        assert confidence_to_wire(jp) == ascii_conf
        assert confidence_from_wire(ascii_conf) == jp


def test_wire_covers_every_internal_confidence() -> None:
    # エンジンが使う日本語確信度は、すべて wire(ASCII) に対応が無いといけない。
    assert set(_CONF_RANK) == set(CONFIDENCE_ASCII_TO_JP.values())


def test_unknown_value_passes_through() -> None:
    assert confidence_to_wire("未知") == "未知"
    assert confidence_from_wire("unknown") == "unknown"


def test_mask_level_strong_selects_certain_and_strong() -> None:
    # 既定 strong ＝ 現行 AUTO_MASK_CONFIDENCE と同じ（確定/強）。
    assert confidences_at_or_above("strong") == ("確定", "強")


def test_mask_level_medium_includes_medium() -> None:
    assert confidences_at_or_above("medium") == ("確定", "強", "中")


def test_mask_level_excluded_never_selected() -> None:
    # excluded は下限に関係なく常に対象外＝どの mask_level でも選ばれない。
    for level in ("certain", "strong", "medium", "weak", "faint"):
        assert "除外" not in confidences_at_or_above(level)


def test_unknown_mask_level_raises() -> None:
    with pytest.raises(ValueError):
        confidences_at_or_above("excluded")  # excluded は mask_level ではない
