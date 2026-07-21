"""マスキング HTTP API の Python クライアント（httpx・依存は httpx だけ）。

`data-redactor serve` で起動したサーバ（既定 ``http://127.0.0.1:8000``）に対して
``/health`` / ``/config`` / ``/mask`` / ``/unmask`` を呼ぶ薄いラッパ。

**外部アプリがそのままコピーして使える**よう、このプロジェクトの ``src`` には
一切依存しない（返り値は API の JSON をそのまま dict で返す）。

使い方（設計 [docs-dev/mask-http-api設計.md] §3-1/§3-2）:

    with MaskClient() as client:
        # 入力は parts の一覧。各 part は kind（中身の取得元）と content だけ。
        res = client.mask(parts=[
            {"kind": "text",         "content": "この3ファイルを要約して。担当は佐藤。"},
            {"kind": "file",         "content": "見積.xlsx"},               # パス
            {"kind": "file",         "content": ("議事録.docx", raw_bytes)},# or (名前, バイト列)
            {"kind": "content_hash", "content": "ab12…"},                  # 取込済み参照
        ])

        # masked_parts は入力と同じ順・同じ id の結果。
        # masked_text＝機密を伏せ字にした本文
        #（例 "SONY"→"[社1]"）＝LLM に送ってよい版（原文は渡さない）。
        for mp in res["masked_parts"]:
            print(mp["id"], mp["masked_text"])

        answer = call_llm(res["masked_parts"])   # 各アプリの LLM 呼び出し（伏せ字のまま）

        # (A) LLM の応答を戻す。応答に混ざった全 part 由来のプレースホルダを
        #     共有 mapping でまとめて復元する（unmask は 1 回でよい）。
        restored = client.unmask(answer, res["mapping"])["restored_text"]

        # (B) 渡した文書側を戻したいときも、同じ mapping で戻せる（part ごとに 1 回）。
        #     mapping は 1 個でバンドル全体を覆うので、戻すテキストの数だけ呼ぶだけ。
        restored_parts = [
            client.unmask(mp["masked_text"], res["mapping"])["restored_text"]
            for mp in res["masked_parts"]
        ]

- ``mapping`` はバンドル共有の 1 個。**戻したいテキストの数だけ ``unmask`` を呼ぶ**
  （LLM 応答＝(A) は 1 回、各 part を戻す＝(B) は part の数だけ）。同じ mapping を使い回す。
- mapping に無いプレースホルダは無変更（LLM の捏造・改変への安全側）。
- ``kind`` … ``"text"`` / ``"file"`` / ``"content_hash"``。``content`` は順に、
  文字列 / ファイルのパス（or ``(名前, バイト列)``）/ 取込済み文書のハッシュ。
- 単一テキストは ``client.mask(text="…")`` の省略記法（part 1 個）。
- 複数 part を 1 回で渡すと（＝バンドル）、同じ実体は全 part で同じプレースホルダに揃う
  （SONY はどの part でも ``[社1]``）。unmask も共有の対応表 1 つで戻せる。
- ``id`` は任意（省略時 ``p0``,``p1``…）。結果 ``masked_parts`` と対応づけたいとき付ける。
- HTTP の送り方（JSON か multipart か）はクライアントが内部で振り分ける（呼ぶ側は気にしない）。
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
    ) -> dict[str, Any]:
        """parts をマスクし、バンドル全体で共有する対応表を得る（``POST /mask``。設計 §3-1）。

        ``parts`` は ``{"kind": "text"|"file"|"content_hash", "content": ..., "id": 任意}``
        の並び。``text="…"`` は単一 text part の省略記法。file を含むかどうかで JSON /
        multipart を**内部で振り分ける**（呼ぶ側は HTTP の送り方を意識しない）。
        """
        if text is not None and parts is not None:
            raise ValueError(
                "text と parts は同時に指定できません（text は単一 part の省略記法）"
            )
        if text is not None:
            parts = [{"kind": "text", "id": "_", "content": text}]
        if not parts:
            raise ValueError("parts か text のいずれかを指定してください")

        # 呼ぶ側の {kind, content} 列を wire 形式へ組み直し、ファイル本体を分離する。
        wire_parts, uploads = _build_parts(parts)
        manifest: dict[str, Any] = {
            "parts": wire_parts,
            "detection": detection,
            "mask_level": mask_level,
            "flatten_tables": flatten_tables,
            "return_pending": return_pending,
        }
        if models is not None:
            manifest["models"] = models

        # ファイル本体があれば multipart、無ければ JSON（送り方はここで隠蔽する）。
        if not uploads:
            return self._post_json("/mask", manifest)
        return self._post_multipart("/mask", manifest, uploads)

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


def _build_parts(
    parts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, bytes, str]]]:
    """呼ぶ側の ``{"kind", "content", "id"?}`` 列をサーバの wire 形式に組み直す。

    戻り値は ``(manifest の parts, 送るファイル本体 {id: (name, bytes, content_type)})``。
    ファイル本体があるときだけ multipart になる（判定は :meth:`MaskClient.mask`）。
    """
    wire: list[dict[str, Any]] = []
    uploads: dict[str, tuple[str, bytes, str]] = {}
    for i, p in enumerate(parts):
        kind = p.get("kind")
        pid = str(p.get("id") or f"p{i}")
        content = p.get("content")
        if kind == "text":
            wire.append({"id": pid, "text": content})
        elif kind == "content_hash":
            wire.append({"id": pid, "content_hash": content})
        elif kind == "file":
            if content is None:
                raise ValueError(f"part[{i}] は file ですが content がありません")
            name, blob, ctype = _as_upload(content)
            wire.append({"id": pid, "file": {"filename": name}})
            uploads[pid] = (name, blob, ctype)
        else:
            raise ValueError(
                f"part[{i}] の kind が不正です: {kind!r}"
                "（text / file / content_hash のいずれか）"
            )
    return wire, uploads


def _as_upload(body: FileBody) -> tuple[str, bytes, str]:
    """file part の content（パス or (名前, バイト列)）を (名前, バイト列, MIME) にする。"""
    if isinstance(body, (str, Path)):
        p = Path(body)
        return (p.name, p.read_bytes(), "application/octet-stream")
    filename, data = body  # (ファイル名, バイト列)
    return (filename, data, "application/octet-stream")
