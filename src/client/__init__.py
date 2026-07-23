"""マスキング HTTP API のクライアント (src/api がサーバ、src/client がクライアント)。

検出・マスク・キャッシュの本体はサーバ (src/api) が持つ。
UI も外部アプリも、自分では処理せず、このクライアント越しにサーバへ HTTP で問い合わせる。
公開名をここで re-export する (``from src.client import MaskClient`` で使う)。
"""

from src.client.mask_client import FileBody, Mapping, MaskApiError, MaskClient

__all__ = ["FileBody", "Mapping", "MaskApiError", "MaskClient"]
