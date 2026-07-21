"""マスキング HTTP API の Python クライアント（依存は httpx だけ）。

data-redactor serve で起動したサーバ（既定 http://127.0.0.1:8000）の
/health /config /mask /unmask を呼ぶための薄いラッパ。
外部アプリがそのままコピーして使えるよう、このプロジェクトの src には一切依存しない。
返り値はサーバの JSON をそのまま dict で返す。


呼べるメソッド（詳しい引数と戻り値は各メソッドの説明を参照）:

    MaskClient(base_url="http://127.0.0.1:8000", timeout=120.0)
        クライアントを作る。with 文で使うと抜けるときに自動でクローズする。

    health() -> dict
        サーバの死活と、NER モデルのロード状態を返す。

    config() -> dict
        既定モデル・detector_version・指定できる選択肢の一覧を返す。

    mask(parts=..., text=..., detection=..., mask_level=...,
         flatten_tables=..., models=..., return_pending=..., refresh=...) -> dict
        入力（parts）をマスクする。戻り値に masked_parts と mapping が入る。

    unmask(text, mapping) -> dict
        mask で得た mapping を使って、伏せ字を元の語に戻す。


入力 parts の書き方:

    part は「マスクしたい入力 1 個」。各 part は kind（中身の取得元）と content で書く。
    kind は次の 2 種:

        text   content は文字列そのもの（プロンプトやコピペ本文）
        file   content はファイルのパス、または ファイル名とバイト列。

    id は任意（省略すると p0, p1, ... と自動採番）。結果の masked_parts と対応づけ
    たいときに付ける。テキスト 1 本だけなら text="..." が省略記法（part 1 個と同じ）。


使い方の例:

    with MaskClient() as client:
        res = client.mask(parts=[
            {"kind": "text", "content": "この3ファイルを要約して。担当は佐藤。"},
            {"kind": "file", "content": "見積.xlsx"},
            {"kind": "file", "content": ("議事録.docx", raw_bytes)},
        ])

        # masked_parts は入力と同じ順・同じ id。masked_text は機密を伏せ字にした本文
        # （例: SONY を [社1] に置換）。これを LLM に渡す（原文は渡さない）。
        for mp in res["masked_parts"]:
            print(mp["id"], mp["masked_text"])

        answer = call_llm(res["masked_parts"])   # 各アプリの LLM 呼び出し

        # (A) LLM の応答を戻す。応答に混ざった全 part 由来のプレースホルダを、
        #     共有 mapping で一度に戻せる（unmask は 1 回でよい）。
        restored = client.unmask(answer, res["mapping"])["restored_text"]

        # (B) 渡した文書側を戻したいときも、同じ mapping で戻せる（part ごとに 1 回）。
        restored_parts = [
            client.unmask(mp["masked_text"], res["mapping"])["restored_text"]
            for mp in res["masked_parts"]
        ]


復元（unmask）の考え方:

    mapping はバンドル全体で 1 個。戻したいテキストの数だけ unmask を呼ぶだけ
    （LLM 応答なら 1 回、各 part を戻すなら part の数だけ）。同じ mapping を使い回す。
    mapping に無いプレースホルダは触らない（LLM が勝手に作った・変えた語への安全側）。

    file を含むかどうかで JSON か multipart かはクライアントが内部で選ぶので、
    呼ぶ側は HTTP の送り方を意識しなくてよい。

設計の詳細は docs-dev/mask-http-api設計.md の 3-1（/mask）・3-2（/unmask）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

# ローカル（ループバック）宛はプロキシを通さない。社内プロキシ環境だと OS の
# プロキシ設定を拾って localhost 宛まで経由してしまい、遅延・タイムアウトの原因になる。
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

# mask の戻り値 mapping（そのまま unmask に渡せる形）。
Mapping = list[dict[str, Any]]
# file part の content に指定できる型：ファイルのパス、または (ファイル名, バイト列)。
FileBody = str | Path | tuple[str, bytes]


class MaskApiError(RuntimeError):
    """API が 2xx 以外を返したときに送出する例外。

    status_code  HTTP ステータス（422 不正入力・未対応拡張子 / 502 LLM 資格情報 /
                 503 モデル未ロード など）。
    detail       サーバが返したエラー内容（JSON の detail、無ければ本文テキスト）。
    """

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class MaskClient:
    """マスキング API の同期クライアント。内部で 1 本の httpx 接続を持つ。

    with 文（コンテキストマネージャ）で使うと、抜けるときに自動でクローズする。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float | None = 300.0,
        use_proxy: bool | None = None,
    ) -> None:
        """接続先・タイムアウト・プロキシ利用を決める。

        base_url   サーバの URL（末尾の / は自動で除く）。既定 http://127.0.0.1:8000。
        timeout    1 リクエストの秒数。NER/LLM は重く（特に detection=both で複数ファイル）
                   時間がかかるので長め（既定 300 秒）。None を渡すと無制限。
        use_proxy  OS/環境のプロキシ設定を使うか。既定 None は「接続先が localhost の
                   ときだけ自動でプロキシを迂回」する（社内プロキシ経由で localhost に
                   つなぎに行って遅延・タイムアウトするのを防ぐ）。remote は環境設定に従う。
                   True で常に使う、False で常に使わない。
        """
        host = (urlparse(base_url).hostname or "").lower()
        trust_env = (host not in _LOCAL_HOSTS) if use_proxy is None else use_proxy
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, trust_env=trust_env
        )

    # -- ライフサイクル ---------------------------------------------------- #
    def close(self) -> None:
        """内部の httpx 接続を閉じる（with 文なら自動で呼ばれる）。"""
        self._client.close()

    def __enter__(self) -> MaskClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- 読み取り系 -------------------------------------------------------- #
    def health(self) -> dict[str, Any]:
        """サーバの死活と、NER モデルのロード状態を返す（GET /health）。

        戻り値: {"status": "ok", "models_ready": bool, "models_loaded": [モデル名, ...]}
        models_ready が false の間は、NER を要求すると mask が 503 になる。
        """
        return self._get("/health")

    def config(self) -> dict[str, Any]:
        """既定値と、指定できる選択肢の一覧を返す（GET /config）。

        戻り値の主なキー:
            models / default_models   ロード済みモデル名。
            llm_model                 LLM 検出に使うモデル名。
            detector_version          検出器の版（キャッシュ鍵に使う識別子）。
            detection_modes           detection に指定できる値 ["ner", "llm", "both"]。
            default_detection         detection の既定。
            mask_levels               mask_level に指定できる値
                                      ["certain", "strong", "medium", "weak", "faint"]。
            default_mask_level        mask_level の既定。
        """
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
        refresh: bool = False,
    ) -> dict[str, Any]:
        """入力（parts）をマスクして、バンドル共有の対応表を得る（POST /mask）。

        引数（すべてキーワード指定）:
            parts           入力の一覧。各要素は {"kind": ..., "content": ..., "id": 任意}。
                            kind は "text" / "file"（書き方はモジュール冒頭の説明を参照）。
            text            テキスト 1 本の省略記法。parts=[{"kind":"text","content":text}]
                            と同じ。parts とは同時に指定できない。
            detection       検出の系統。"ner" / "llm" / "both"。既定 "both"。
                            "llm" と "both" はサーバ側で Azure 資格情報が要る（無いと 502）。
            mask_level      自動マスクする下限。"certain" / "strong" / "medium" / "weak" /
                            "faint"。既定 "strong"。下限以上の確からしさの実体だけを伏せ字に
                            する（下限未満は pending に回る）。
            flatten_tables  表を平文化して検出するか。既定 True。
            models          使うモデルの明示指定（任意）。省略でサーバの既定（両モデル）。
            return_pending  下限未満のレビュー候補（pending）を返すか。既定 True。
            refresh         True でキャッシュを無視して強制再解析（NER/LLM とも）。結果で
                            キャッシュを上書きする。既定 False（あれば再利用＝速い）。

        戻り値（dict）:
            status         当面 "unconfirmed" 固定。
            masked_parts   入力と同じ順・同じ id のリスト。各要素 {"id", "masked_text"}。
                           masked_text が伏せ字済みの本文（LLM に渡すのはこれ）。
            mapping        バンドル共有の対応表。各要素は placeholder / category /
                           canonical / surfaces / confidence / decided_by / occurrences。
                           unmask にそのまま渡す。
            pending        下限未満のレビュー候補（return_pending=True のとき）。
            detector       使った検出器の構成（detection / models / detector_version /
                           mask_level）。

        file を含むかどうかで JSON か multipart かは内部で振り分ける（呼ぶ側は気にしない）。
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
            "refresh": refresh,
        }
        if models is not None:
            manifest["models"] = models

        # ファイル本体があれば multipart、無ければ JSON（送り方はここで隠蔽する）。
        if not uploads:
            return self._post_json("/mask", manifest)
        return self._post_multipart("/mask", manifest, uploads)

    def unmask(self, text: str, mapping: Mapping) -> dict[str, Any]:
        """mapping を使って、伏せ字を元の語に戻す（POST /unmask）。

        引数:
            text     復元したいテキスト（LLM の応答や、各 part の masked_text）。
            mapping  mask の戻り値の ["mapping"] をそのまま渡す。

        戻り値: {"restored_text": 復元後のテキスト}

        mapping に無いプレースホルダは変更しない（LLM が作った・変えた語への安全側）。
        戻したいテキストが複数あるときは、同じ mapping で必要な回数だけ呼ぶ。
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
        # サーバは form の manifest（JSON 文字列）と、各 part id をキーにした本体を読む。
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
    """呼ぶ側の {"kind","content","id"?} 列を、サーバが受け取る形に組み直す。

    kind は "text"（content=文字列）か "file"（content=パス or (名前, バイト列)）。
    戻り値は (parts, uploads)。parts はサーバへ送る JSON 用、uploads は multipart で
    送るファイル本体 {id: (ファイル名, バイト列, MIME)}。uploads が空でなければ
    mask() は multipart で送る。
    """
    wire: list[dict[str, Any]] = []
    uploads: dict[str, tuple[str, bytes, str]] = {}
    for i, p in enumerate(parts):
        kind = p.get("kind")
        pid = str(p.get("id") or f"p{i}")
        content = p.get("content")
        if kind == "text":
            wire.append({"id": pid, "text": content})
        elif kind == "file":
            if content is None:
                raise ValueError(f"part[{i}] は file ですが content がありません")
            name, blob, ctype = _as_upload(content)
            wire.append({"id": pid, "file": {"filename": name}})
            uploads[pid] = (name, blob, ctype)
        else:
            raise ValueError(
                f"part[{i}] の kind が不正です: {kind!r}（text / file のいずれか）"
            )
    return wire, uploads


def _as_upload(body: FileBody) -> tuple[str, bytes, str]:
    """file part の content（パス、または (名前, バイト列)）を (名前, バイト列, MIME) にする。"""
    if isinstance(body, (str, Path)):
        p = Path(body)
        return (p.name, p.read_bytes(), "application/octet-stream")
    filename, data = body  # (ファイル名, バイト列)
    return (filename, data, "application/octet-stream")
