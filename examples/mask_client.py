"""マスキング HTTP API の Python クライアント（httpx・依存は httpx だけ）。

`data-redactor serve` で起動したサーバ（既定 ``http://127.0.0.1:8000``）に対して
``/health`` / ``/config`` / ``/mask`` / ``/unmask`` を呼ぶ薄いラッパ。

**外部アプリがそのままコピーして使える**よう、このプロジェクトの ``src`` には
一切依存しない（返り値は API の JSON をそのまま dict で返す）。

想定する使い方（設計 [docs-dev/mask-http-api設計.md] §3-1/§3-2）:

    with MaskClient() as client:
        res = client.mask(text="担当は佐藤。SONYと比較。")

        # res["masked_parts"] は「入力 part ごとの結果」のリスト。
        # text= のときは part が 1 個なので [0]（1 番目）を取る。
        #
        # masked_text は機密を伏せ字にした本文＝LLM に送ってよい版。
        #   元:     担当は佐藤。SONYと比較。
        #   伏せ字: 担当は[人物1]。[社1]と比較。
        masked = res["masked_parts"][0]["masked_text"]

        # 各アプリの LLM 呼び出し（伏せ字のまま渡す）。
        answer = call_llm(masked)

        # 応答に残ったプレースホルダ（[社1] 等）を元の語に戻す。
        restored = client.unmask(answer, res["mapping"])["restored_text"]

part（＝LLM に渡す入力 1 個）は 3 種。**クライアントでの渡し方**を種別ごとに示す:

    # ① text ― すでに手元にある文字列（プロンプト・コピペ本文）。
    #    単一なら text= の省略記法が使える。
    client.mask(text="担当は佐藤。SONYと比較。")

    #    複数をまとめるなら parts= に id つきで並べる。
    client.mask(
        parts=[
            {"id": "prompt", "text": "2 社を比較して。担当は佐藤。"},
            {"id": "memo", "text": "SONYの評価は高い。"},
        ]
    )

    # ② content_hash ― 事前にサーバへ取り込み済みのファイルを指す（再送不要・速い）。
    client.mask(parts=[{"id": "fileA", "content_hash": "ab12…"}])

    # ③ file ― 手元のファイル（xlsx/pdf/docx…）をその場で送る（サーバがテキスト化）。
    #    part に filename を書き、files= に本体（パス or (名前, bytes)）を渡す。
    client.mask(
        parts=[{"id": "fileB", "file": {"filename": "見積.xlsx"}}],
        files={"fileB": "見積.xlsx"},
    )

複数 part を 1 回で渡すと（＝バンドル）、同じ実体は全 part で同じプレースホルダに揃う
（SONY はどの part でも [社1]）。masked_parts は入力と同じ順・同じ id で返る。

file を 1 つでも渡すと multipart/form-data（JSON マニフェスト＋本体）、
それ以外は application/json で送る（自動判定）。
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
