"""LLM 検出アダプタ (薄い層)。

LLM による PII 検出の**本体は pii-masker** (プロンプト・Azure クライアント・detect・locate を所有)。
本パッケージは pii-masker を**依存として呼ぶだけ**のアダプタで、data-redactor 側に閉じるのは：

- :mod:`src.llm.windows`     … 本文 (`build_body` の text) を ~6-8k トークン窓に切る
- :mod:`src.llm.detect_layer`… 窓ループ → ``pii_masker.detect`` + ``locate_all`` → 全文スパンへ
- :mod:`src.llm.schema`      … 我々の型 (``LlmSpan`` / ``LlmDetection``)

設計は docs-dev/LLM適用_調査と設計たたき台.md を参照。
``MaskingEngine`` はこれらを import せず、算出済みの :class:`~src.llm.schema.LlmDetection` を受け取るだけ。

pii-masker は ``[build-system]`` を持たない PoC で pip/uv インストールできないため、git submodule
(``external/pii-masker``) ＋path-injection で取り込む。``_paths`` の **副作用 import** が
``external/pii-masker/src`` を ``sys.path`` に通し、``import pii_masker`` を解決可能にする。
"""

from src.llm import _paths as _paths  # noqa: F401  副作用: pii-masker の path-injection
from src.llm.detect_layer import cached_detect, detect_document
from src.llm.schema import (
    LlmDetection,
    LlmSpan,
    detection_from_json,
    detection_to_json,
)
from src.llm.windows import iter_windows

__all__ = [
    "LlmSpan",
    "LlmDetection",
    "detection_to_json",
    "detection_from_json",
    "iter_windows",
    "detect_document",
    "cached_detect",
]
