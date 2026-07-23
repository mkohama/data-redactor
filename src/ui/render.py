"""UI 専用のマスク候補ハイライト表示（spaCy 非依存）。

設計 B で UI は純クライアント（エンジンを持たない）。マスク候補の色付き表示は、以前は
spaCy の ``displacy.render`` を使っていたが、displacy は ML ではなく単なる HTML 生成器なので、
ここで等価の素 HTML を組んで **UI から spaCy 依存を完全に外す**（＝UI イメージに spaCy/torch/
GiNZA を入れずに済む）。

配色は src/ner/rendering.py（エンジン側の表示）と同じ値を持つ（UI を engine パッケージから
import させないための意図的な小さな重複。カテゴリ配色が変わったら両方直す）。
"""

from __future__ import annotations

import html
from collections.abc import Iterable

# マスキングのカテゴリ → 背景色（src.masking のカテゴリ名に対応。src/ner/rendering.py と同値）。
_PERSON = "#ff8fab"  # 人名: ピンク
_ORG = "#ffab5e"  # 社名・組織: オレンジ
_PRODUCT = "#c792ea"  # 商標・製品: 紫
_LOCATION = "#4fc3b0"  # 地名・住所: ティール
_CONTACT = "#6cb6ff"  # 連絡先: 青
DEFAULT_COLOR = "#dcdcdc"  # 非 PII / 未分類: 淡いグレー

MASKING_CATEGORY_COLORS: dict[str, str] = {
    "人名": _PERSON,
    "社名": _ORG,
    "商標": _PRODUCT,
    "地名": _LOCATION,
    "連絡先": _CONTACT,
    "その他": DEFAULT_COLOR,
}


def render_masking_html(
    text: str,
    spans: Iterable[tuple[int, int, str]],
    *,
    page: bool = False,  # 互換のため受けるが UI はインライン（呼び出し側が div で包む）
) -> str:
    """マスク候補スパンを色付き ``<mark>`` の HTML にする（displacy 相当・spaCy 非依存）。

    Args:
        text:  元テキスト。
        spans: ``(start, end, category)`` のイテラブル。category は
               :data:`MASKING_CATEGORY_COLORS` のキー（未知は既定色）。
    重なるスパンは先勝ちで捨てる（displacy と同じく重なりを描けないため）。HTML エスケープ
    したうえで該当区間だけ ``<mark>`` で包み、カテゴリ名を小さなラベルとして添える。
    """
    ordered = sorted(spans, key=lambda s: s[0])
    parts: list[str] = []
    cursor = 0
    last_end = -1
    for start, end, category in ordered:
        if start < last_end or start < cursor:  # 重なり・逆行は捨てる
            continue
        parts.append(html.escape(text[cursor:start]))
        color = MASKING_CATEGORY_COLORS.get(category, DEFAULT_COLOR)
        label = html.escape(category)
        parts.append(
            f'<mark style="background:{color}; color:#000; padding:0 .3em; '
            f'border-radius:.35em; white-space:nowrap;">{html.escape(text[start:end])}'
            f'<span style="font-size:.7em; font-weight:700; margin-left:.35em; '
            f'text-transform:uppercase;">{label}</span></mark>'
        )
        cursor = end
        last_end = end
    parts.append(html.escape(text[cursor:]))
    inner = f'<div style="line-height:2.2; white-space:pre-wrap; word-break:break-word;">{"".join(parts)}</div>'
    if page:
        return f"<!doctype html><html><head><meta charset='utf-8'></head><body>{inner}</body></html>"
    return inner
