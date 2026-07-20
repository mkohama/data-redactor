"""マスキング HTTP API の Python クライアント（httpx・依存は httpx だけ）。

`data-redactor serve` で起動したサーバ（既定 ``http://127.0.0.1:8000``）に対して
``/health`` / ``/config`` / ``/mask`` / ``/unmask`` を呼ぶ薄いラッパ。**外部アプリが
そのままコピーして使える**よう、このプロジェクトの ``src`` には一切依存しない
（返り値は API の JSON をそのまま dict で返す）。

想定する使い方（設計 [docs-dev/mask-http-api設計.md] §3-1/§3-2）:

    with MaskClient() as client:
        res = client.mask(text="担当は佐藤。SONYと比較。")
        masked = res["masked_parts"][0]["masked_text"]   # LLM に食わせる形
        answer = call_llm(masked)                         # ← ここは各アプリの LLM 呼び出し
        restored = client.unmask(answer, res["mapping"])["restored_text"]

part は 3 種のいずれか（設計 §3-1）:
    - インライン text     : ``{"id": "prompt", "text": "……"}``
    - 取込済み参照        : ``{"id": "fileA", "content_hash": "ab12…"}``
    - 同梱ファイル        : ``{"id": "fileB", "file": {"filename": "見積.xlsx"}}``
                            ＋ ``files={"fileB": <path or (filename, bytes)>}``

同梱ファイルを 1 つでも渡すと multipart/form-data（JSON マニフェスト＋本体）に、
それ以外は application/json に自動で切り替える。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

# バンドル全体で共有する対応表（/mask の mapping[]、/unmask にそのまま渡せる形）。
Mapping = list[dict[str, Any]]
# 同梱ファイル本体の指定：パス、または (ファイル名, バイト列)。
FileBody = str | Path | tuple[str, bytes]


class MaskApiError(RuntimeError):
    """API が 2xx 以外を返したときの例外（ステータスとサーバの detail を持つ）。"""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class MaskClient:
    """マスキング API の同期クライアント（httpx.Client を内包）。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 120.0,
    ) -> None:
        # NER/LLM は重い（初回は特に）ため、既定タイムアウトは長めに取る。
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    # -- ライフサイクル ---------------------------------------------------- #
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MaskClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- 読み取り系 -------------------------------------------------------- #
    def health(self) -> dict[str, Any]:
        """死活・モデルロード状態（``GET /health``）。"""
        return self._get("/health")

    def config(self) -> dict[str, Any]:
        """既定モデル・detector_version・選択肢（``GET /config``）。"""
        return self._get("/config")

    # -- マスク / 復元 ----------------------------------------------------- #
    def mask(
        self,
        *,
        parts: list[dict[str, Any]] | None = None,
        text: str | None = None,
        detection: str = "both",
        mask_level: str = "strong",
        flatten_tables: bool = True,
        models: list[str] | None = None,
        return_pending: bool = True,
        files: dict[str, FileBody] | None = None,
    ) -> dict[str, Any]:
        """parts をマスクし、バンドル全体で共有する対応表を得る（``POST /mask``。設計 §3-1）。

        ``text`` は ``parts:[{id:"_", text}]`` の糖衣。同梱ファイル part を使う場合は
        ``files={part_id: <path or (filename, bytes)>}`` を渡す（multipart になる）。
        """
        manifest: dict[str, Any] = {
            "detection": detection,
            "mask_level": mask_level,
            "flatten_tables": flatten_tables,
            "return_pending": return_pending,
        }
        if parts is not None:
            manifest["parts"] = parts
        if text is not None:
            manifest["text"] = text
        if models is not None:
            manifest["models"] = models

        if not files:
            return self._post_json("/mask", manifest)

        # 同梱ファイルあり＝multipart（JSON マニフェスト文字列 + 本体）。
        upload = {pid: _as_upload(body) for pid, body in files.items()}
        return self._post_multipart("/mask", manifest, upload)

    def unmask(self, text: str, mapping: Mapping) -> dict[str, Any]:
        """LLM 応答テキストを mapping で復元する（``POST /unmask``。設計 §3-2）。

        mapping は ``mask()`` の戻り値の ``["mapping"]`` をそのまま渡せる。
        mapping に無いプレースホルダはサーバ側で無変更（LLM の捏造への安全側）。
        """
        return self._post_json("/unmask", {"text": text, "mapping": mapping})

    # -- 低レベル（HTTP） -------------------------------------------------- #
    def _get(self, path: str) -> dict[str, Any]:
        return self._unwrap(self._client.get(path))

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._unwrap(self._client.post(path, json=payload))

    def _post_multipart(
        self,
        path: str,
        manifest: dict[str, Any],
        upload: dict[str, tuple[str, bytes, str]],
    ) -> dict[str, Any]:
        # サーバは form の "manifest"（JSON 文字列）＋各 part id をキーにした本体を読む。
        return self._unwrap(
            self._client.post(
                path,
                data={"manifest": json.dumps(manifest, ensure_ascii=False)},
                files=upload,
            )
        )

    @staticmethod
    def _unwrap(resp: httpx.Response) -> dict[str, Any]:
        if resp.is_success:
            return resp.json()
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise MaskApiError(resp.status_code, detail)


def _as_upload(body: FileBody) -> tuple[str, bytes, str]:
    """ファイル本体指定を httpx の files 形式 (filename, bytes, content_type) にする。"""
    if isinstance(body, (str, Path)):
        p = Path(body)
        return (p.name, p.read_bytes(), "application/octet-stream")
    filename, data = body  # (ファイル名, バイト列)
    return (filename, data, "application/octet-stream")
