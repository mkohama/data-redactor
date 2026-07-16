"""ライブラリ核の下ごしらえ（設計 §5 step1）のユニットテスト。

- ``unmask``：プレースホルダ → canonical 復元・長い順優先・未知プレースホルダは無変更。
- ``mapping_to_json`` / ``mapping_from_json``：往復で MaskEntry が保たれる。

エンジン全体（GiNZA）を起動しない軽いテスト。マスク適用が canonical/spans を
埋めることは characterization テスト側で担保する。
"""

from __future__ import annotations

from src.masking import MaskEntry, mapping_from_json, mapping_to_json, unmask


def _mapping() -> tuple[MaskEntry, ...]:
    return (
        MaskEntry(
            placeholder="[社1]",
            category="社名",
            surfaces=("社A", "社A株式会社"),
            canonical="社A",
            spans=((3, 5), (20, 22)),
        ),
        MaskEntry(
            placeholder="[社10]",
            category="社名",
            surfaces=("社J",),
            canonical="社J",
            spans=((40, 42),),
        ),
        MaskEntry(
            placeholder="[人物1]",
            category="人名",
            surfaces=("小坂",),
            canonical="小坂",
            spans=((10, 12),),
        ),
    )


def test_unmask_restores_placeholders_to_canonical() -> None:
    text = "[社1]の[人物1]が[社10]を担当。"
    assert unmask(text, _mapping()) == "社Aの小坂が社Jを担当。"


def test_unmask_prefers_longer_placeholder() -> None:
    # [社1] が [社10] の一部を食わないこと（長い順置換）。
    assert unmask("[社10]", _mapping()) == "社J"


def test_unmask_leaves_unknown_placeholder_untouched() -> None:
    # mapping に無い [社99] は LLM の捏造として無変更（安全側）。
    text = "[社1]と[社99]"
    assert unmask(text, _mapping()) == "社Aと[社99]"


def test_unmask_falls_back_to_first_surface_when_no_canonical() -> None:
    m = (MaskEntry(placeholder="[語1]", category="その他", surfaces=("原語",)),)
    assert unmask("これは[語1]です", m) == "これは原語です"


def test_mapping_json_round_trip() -> None:
    mp = _mapping()
    restored = mapping_from_json(mapping_to_json(mp))
    assert restored == mp


def test_mapping_from_json_tolerates_missing_spans() -> None:
    data = [
        {
            "placeholder": "[社1]",
            "category": "社名",
            "surfaces": ["社A"],
            "canonical": "社A",
        }
    ]
    (entry,) = mapping_from_json(data)
    assert entry.placeholder == "[社1]"
    assert entry.canonical == "社A"
    assert entry.spans == ()
    # spans が無くても unmask は動く（span は復元に使わない）。
    assert unmask("[社1]", [entry]) == "社A"
