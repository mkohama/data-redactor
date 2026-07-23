"""入力ソース (アダプタ)。

固有表現抽出エンジンに渡すテキストを、各種ソースから取得する層。
- files       : ローカルファイル (DocumentLoader 経由)
- kb_mcp      : kb-mcp サーバ
- sample      : 動作確認用の組み込みサンプル文 (SAMPLE_TEXT)

**遅延ロード **：``files`` は DocumentLoader (→ langchain → transformers/torch) を引くため、
パッケージ import 時には読み込まない。UI (純クライアント) は ``SAMPLE_TEXT`` と ``kb_mcp`` だけを使い、
langchain 無しで ``import src.sources`` できる。``load_chunks_from_file`` などは初アクセス時に
:func:`__getattr__` が遅延 import する (テキスト化・チャンク化の所有者はサーバ)。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from src.sources.sample import SAMPLE_TEXT

if TYPE_CHECKING:
    from src.sources.files import load_chunks_from_file, load_text_from_file

_LAZY: dict[str, str] = {
    "load_chunks_from_file": "files",
    "load_text_from_file": "files",
}


def __getattr__(name: str) -> Any:
    """``files`` 系 (langchain 依存) を初アクセス時に遅延 import する (PEP 562)。"""
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"src.sources.{sub}"), name)


__all__ = ["SAMPLE_TEXT", "load_chunks_from_file", "load_text_from_file"]
