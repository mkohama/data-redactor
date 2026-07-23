"""チャンク列の内容ハッシュ (依存ゼロの軽量モジュール)。

``content_hash`` は cache.db の文書キーであり、**サーバ (エンジン所有) と UI (純クライアント) の
両方が同じ値を出す**必要がある。cache.py は NER 層のために ``src.ner.engine`` (spaCy) を import
するので、UI から ``content_hash`` を使うためだけに spaCy を巻き込まないよう、ハッシュ関数だけを
ここ (hashlib のみ・依存ゼロ) に分離する。cache.py はここから再輸出する。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def content_hash(chunks: Iterable[str]) -> str:
    """チャンク列 (解析対象テキスト) の内容ハッシュ。区切りバイトで連結の曖昧性を避ける。"""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
