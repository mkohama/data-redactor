"""マスキング検出 (UI 非依存)。

公開 API:
    MaskingEngine : マスキング検出エンジン (候補生成→確信度→ルーティング→マスク)
    MaskResult / Candidate / MaskEntry : 結果の型
    MaskDictionary : マスク辞書 (社名・商標・人名の登録リスト)

**遅延ロード **：``MaskingEngine`` / ``NerCache`` などエンジン系は spaCy (→torch) を
引くため、パッケージ import 時には**読み込まない**。UI (純クライアント) は ``content_hash`` や
``dict_sort_key`` など軽いシンボルだけを使い、spaCy 無しで ``import src.masking`` できる。
エンジン系は初アクセス時に :func:`__getattr__` が遅延 import する (型検査は TYPE_CHECKING 経由)。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# --- 軽量シンボル (依存ゼロ〜yaml のみ) ＝ eager import。UI はここだけ使う。 ---
from src.masking.allowlist import (
    MaskAllowlist,
    load_allowlist_entries,
    save_allowlist_entries,
)
from src.masking.allowlist import sort_key as allowlist_sort_key
from src.masking.dictionary import (
    DictMatch,
    MaskDictionary,
    load_entries,
    normalize,
    save_entries,
)
from src.masking.dictionary import sort_key as dict_sort_key
from src.masking.hashing import content_hash

if TYPE_CHECKING:
    # 型検査用の宣言 (実行時は下の __getattr__ が遅延 import する)。これらは engine/cache 経由で
    # spaCy を引くので eager import しない＝UI へ spaCy を持ち込まない。
    from src.masking.cache import NerCache
    from src.masking.engine import (
        AUTO_MASK_CONFIDENCE,
        BundleEntry,
        BundleMaskResult,
        Candidate,
        CandidateGroup,
        MaskAnalysis,
        MaskEntry,
        MaskingEngine,
        MaskResult,
        apply_allowlist,
        apply_allowlist_to_analysis,
        mapping_from_json,
        mapping_to_json,
        tally_votes,
        unmask,
        vote_category,
    )

# 遅延ロード対象: シンボル名 → 実体のあるサブモジュール ("cache" / "engine")。
_LAZY: dict[str, str] = {
    "NerCache": "cache",
    "AUTO_MASK_CONFIDENCE": "engine",
    "BundleEntry": "engine",
    "BundleMaskResult": "engine",
    "Candidate": "engine",
    "CandidateGroup": "engine",
    "MaskAnalysis": "engine",
    "MaskEntry": "engine",
    "MaskingEngine": "engine",
    "MaskResult": "engine",
    "apply_allowlist": "engine",
    "apply_allowlist_to_analysis": "engine",
    "mapping_from_json": "engine",
    "mapping_to_json": "engine",
    "tally_votes": "engine",
    "unmask": "engine",
    "vote_category": "engine",
}


def __getattr__(name: str) -> Any:
    """エンジン系シンボルを初アクセス時に遅延 import する (PEP 562)。"""
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"src.masking.{sub}"), name)


__all__ = [
    "MaskingEngine",
    "MaskAnalysis",
    "MaskResult",
    "BundleEntry",
    "BundleMaskResult",
    "Candidate",
    "CandidateGroup",
    "MaskEntry",
    "unmask",
    "mapping_to_json",
    "mapping_from_json",
    "AUTO_MASK_CONFIDENCE",
    "MaskDictionary",
    "DictMatch",
    "normalize",
    "vote_category",
    "tally_votes",
    "load_entries",
    "save_entries",
    "dict_sort_key",
    "MaskAllowlist",
    "load_allowlist_entries",
    "save_allowlist_entries",
    "allowlist_sort_key",
    "apply_allowlist",
    "apply_allowlist_to_analysis",
    "NerCache",
    "content_hash",
]
