"""pii-masker (git submodule, §8 の B2) を import 可能にする path-injection。

**副作用 import**：このモジュールを import すると ``external/pii-masker/src`` を ``sys.path`` に通す
(``import pii_masker`` を解決可能にする)。pii-masker は ``[build-system]`` を持たない PoC で
pip/uv インストールできないため、editable でなく path-injection で取り込む。

submodule 未取得 (``external/pii-masker`` が無い) なら何もしない＝pii-masker を使わず
スタブで開発・テストする環境でも無害。将来 ``src/pii_masker`` として本体へ統合しても、
``src`` を path に置く前提が同じなのでアダプタの import 文は不変 (§8 前方互換)。
"""

from __future__ import annotations

import sys
from pathlib import Path

# repo ルート = src/llm/_paths.py から 3 つ上 (_paths.py → llm → src → ルート)。
_PII_MASKER_SRC = (
    Path(__file__).resolve().parents[2] / "external" / "pii-masker" / "src"
)
if _PII_MASKER_SRC.is_dir():
    _p = str(_PII_MASKER_SRC)
    if _p not in sys.path:
        sys.path.insert(0, _p)
