"""マスキング HTTP API のクライアント（src/api＝サーバに対する src/client＝クライアント）。

UI（app.py）も外部アプリも、このクライアント越しにサーバへ接続する（設計 B）。
公開名をここで re-export する（``from src.client import MaskClient`` で使う）。
"""

from src.client.mask_client import FileBody, Mapping, MaskApiError, MaskClient

__all__ = ["FileBody", "Mapping", "MaskApiError", "MaskClient"]
