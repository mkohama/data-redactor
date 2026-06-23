"""LLM 検出結果の型（data-redactor 側の表現）。

pii-masker の ``Entity``/``Match`` から詰め直して、キャッシュ・表示（出口1）・票変換（Stage B）で使う。
pii-masker の型をそのまま外に漏らさない（依存をアダプタ境界で閉じる）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class LlmSpan:
    """LLM が検出し本文に位置特定できた 1 スパン（``Body.text`` 座標）。

    ``start``/``end`` は窓内 locate の結果に ``window_start`` を足した**全文（merge）座標**。
    """

    start: int
    end: int
    ene_type: str  # pii-masker の ENE type（Person / Company / ... / Trademark）
    reason: str | None  # LLM の判断理由（出口1 で表示）
    how: str  # pii-masker Match.how（"exact"/"normalized"/...）由来の可視化用


@dataclass(frozen=True)
class LlmDetection:
    """1 文書（チャンク列）に対する LLM 検出結果（Stage A の出力）。

    ``spans`` は位置特定できた検出、``not_found`` は本文に当てられなかった検出の
    ``(ene_type, text)``（出口1 で「要確認」として見せる）。``model``/``detector_version``
    はキャッシュ鍵・再現性のために保持する（``detector_version`` ＝ pii-masker の版＋窓ポリシー）。
    """

    spans: tuple[LlmSpan, ...]
    not_found: tuple[tuple[str, str], ...]
    model: str
    detector_version: str


def detection_to_json(d: LlmDetection) -> str:
    """:class:`LlmDetection` を JSON 文字列へ（キャッシュ保存用）。"""
    return json.dumps(
        {
            "model": d.model,
            "detector_version": d.detector_version,
            "spans": [[s.start, s.end, s.ene_type, s.reason, s.how] for s in d.spans],
            "not_found": [[t, x] for t, x in d.not_found],
        },
        ensure_ascii=False,
    )


def detection_from_json(s: str) -> LlmDetection:
    """:func:`detection_to_json` の逆。"""
    d = json.loads(s)
    return LlmDetection(
        spans=tuple(
            LlmSpan(start=sp[0], end=sp[1], ene_type=sp[2], reason=sp[3], how=sp[4])
            for sp in d.get("spans", [])
        ),
        not_found=tuple((nf[0], nf[1]) for nf in d.get("not_found", [])),
        model=d.get("model", ""),
        detector_version=d.get("detector_version", ""),
    )


__all__ = [
    "LlmSpan",
    "LlmDetection",
    "detection_to_json",
    "detection_from_json",
]
