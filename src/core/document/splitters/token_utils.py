"""
テキスト長計算ユーティリティ

トークンベースの長さ計算機能を提供する。
"""

import tiktoken
from functools import lru_cache


@lru_cache(maxsize=1)
def get_tokenizer() -> tiktoken.Encoding:
    """
    tiktokenエンコーダを初回ロード時にキャッシュ
    """
    return tiktoken.get_encoding("cl100k_base")


def tiktoken_len(text: str) -> int:
    """
    tiktoken (cl100k_base) を使用したトークン数計測
    """
    try:
        tokenizer = get_tokenizer()
    except Exception:
        return len(text)
    return len(tokenizer.encode(text))
