"""マスキング HTTP API（FastAPI）。設計 [docs-dev/mask-http-api設計.md] の B 案・最小面。

エンジン（GiNZA モデル）と ``data/cache.db`` の**所有者をこの 1 プロセスに集約**する
（設計 B：Streamlit も外部アプリも HTTP クライアント）。エンドポイント：

    最小面（M2）:
      GET  /health           死活・モデルロード状態
      GET  /config           既定モデル・detector_version・選択肢・対応拡張子
      POST /mask             parts（text / content_hash 参照 / 同梱ファイル）→ 共有対応表でマスク
      POST /unmask           text＋mapping → 復元テキスト
    全体面（M5・Streamlit クライアント用。stateful）:
      POST   /documents      入力取込 → content_hash 発行（JSON=text / multipart=file）
      GET    /documents      取込済み一覧
      GET    /documents/{h}  メタ＋チャンク
      DELETE /documents/{h}  削除（?layer=ner で NER 層のみ）
      PATCH  /documents/{h}  メタ更新（source_kind）

起動時にモデルを 1 回ロードする（lifespan、エンジン singleton）。残りの全体面（/analyze・
/apply・/draft・/allowlist・/dictionary・/kb）と Streamlit のクライアント化は M5b 以降で足す。

起動：``uv run data-redactor serve``（uvicorn ラッパ）。テストは ``create_app(ctx=...)`` に
軽量な :class:`~src.api.service.ApiContext` を注入して GiNZA の実ロードを避ける。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from src.api.enums import (
    DEFAULT_DETECTION,
    DEFAULT_MASK_LEVEL,
    DETECTION_MODES,
    MASK_LEVELS,
)
from src.api.models import (
    DocumentDetail,
    DocumentIngestRequest,
    DocumentInfo,
    DocumentPatch,
    MaskRequest,
    MaskResponse,
    UnmaskRequest,
    UnmaskResponse,
)
from src.api.service import (
    SUPPORTED_EXTENSIONS,
    ApiContext,
    delete_document,
    get_document,
    ingest_file,
    ingest_text,
    list_documents,
    patch_document,
    run_mask,
    run_unmask,
)
from src.detector import LLM_MODEL, detector_version
from src.masking import MaskAllowlist, MaskDictionary, MaskingEngine, NerCache
from src.ner import AVAILABLE_MODELS

# 既定パス（app.py＝Streamlit と同じ data/ を共有する＝cache.db を 2 つ持たない。設計 B）。
#   Docker/実機では env で差し替える（名前付きボリューム共有・書き込みは api のみ）。
_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CACHE_DB = os.getenv(
    "DATA_REDACTOR_CACHE_DB", str(_ROOT / "data" / "cache.db")
)
_DEFAULT_DICT = os.getenv("DATA_REDACTOR_DICT", str(_ROOT / "data" / "mask_dict.yaml"))
_DEFAULT_ALLOWLIST = os.getenv(
    "DATA_REDACTOR_ALLOWLIST", str(_ROOT / "data" / "mask_allowlist.yaml")
)


def _load_context() -> ApiContext:
    """サーバ資源（エンジン＝モデル1回ロード・cache.db・辞書・除外）を組み立てる。

    モデルロードに失敗しても API 自体は起動する（``models_ready=False``＝NER 要求は 503）。
    ``/unmask`` や既に解析済みの content_hash 参照はモデル無しでも動くため。
    """
    cache = NerCache(_DEFAULT_CACHE_DB)
    dictionary = (
        MaskDictionary.load(_DEFAULT_DICT)
        if Path(_DEFAULT_DICT).exists()
        else MaskDictionary.empty()
    )
    allowlist = (
        MaskAllowlist.load(_DEFAULT_ALLOWLIST)
        if Path(_DEFAULT_ALLOWLIST).exists()
        else MaskAllowlist.empty()
    )
    model_names = tuple(AVAILABLE_MODELS)
    engine = MaskingEngine(dictionary=dictionary, models=list(model_names))
    models_ready = True
    try:
        for e in engine.engines:
            _ = e.nlp  # 起動時に 1 回ロード（以降の解析を高速化）
    except (
        Exception
    ):  # noqa: BLE001 - ロード失敗でも起動は続ける（/health で可視化・NER は 503）
        models_ready = False
    return ApiContext(
        engine=engine,
        cache=cache,
        allowlist=allowlist,
        model_names=model_names,
        models_ready=models_ready,
    )


def create_app(ctx: ApiContext | None = None) -> FastAPI:
    """FastAPI アプリを生成する。``ctx`` を渡すと起動時ロードを省く（テスト用の注入）。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 注入 ctx（テスト）は create_app で即セット済み。未設定（本番）のみ起動時にロードする。
        if getattr(app.state, "ctx", None) is None:
            app.state.ctx = _load_context()
        yield

    app = FastAPI(title="data-redactor mask API", version="0.1.0", lifespan=lifespan)
    # 注入 ctx はこの時点でセットする（TestClient を with 無しで使ってもルートが動くように）。
    app.state.ctx = ctx

    @app.get("/health")
    def health() -> dict:
        c: ApiContext = app.state.ctx
        return {
            "status": "ok",
            "models_ready": c.models_ready,
            "models_loaded": list(c.model_names) if c.models_ready else [],
        }

    @app.get("/config")
    def config() -> dict:
        c: ApiContext = app.state.ctx
        return {
            "models": list(c.model_names),
            "default_models": list(c.model_names),
            "llm_model": LLM_MODEL,
            "detector_version": detector_version(),
            "detection_modes": list(DETECTION_MODES),
            "default_detection": DEFAULT_DETECTION,
            "mask_levels": list(MASK_LEVELS),
            "default_mask_level": DEFAULT_MASK_LEVEL,
            "supported_extensions": list(SUPPORTED_EXTENSIONS),
        }

    @app.post("/mask", response_model=MaskResponse)
    async def mask(request: Request) -> MaskResponse:
        """parts をマスク。JSON（text/content_hash）または multipart（同梱ファイル）で受ける。"""
        ctype = request.headers.get("content-type", "")
        files: dict[str, tuple[str, bytes]] = {}
        try:
            if ctype.startswith("multipart/form-data"):
                form = await request.form()
                manifest = form.get("manifest")
                if not isinstance(manifest, str):
                    raise HTTPException(
                        422,
                        "multipart には JSON 文字列の `manifest` フィールドが必要です",
                    )
                req = MaskRequest.model_validate_json(manifest)
                for key, val in form.multi_items():
                    if key == "manifest" or not isinstance(val, UploadFile):
                        continue
                    files[key] = (val.filename or key, await val.read())
            else:
                req = MaskRequest.model_validate(await request.json())
        except ValidationError as e:
            raise HTTPException(422, e.errors()) from e
        return run_mask(app.state.ctx, req, files)

    @app.post("/unmask", response_model=UnmaskResponse)
    def unmask_endpoint(req: UnmaskRequest) -> UnmaskResponse:
        return run_unmask(req)

    # ----------------------------------------------------------------- #
    # 全体面：/documents 系（設計 §2-B）。取込→content_hash 発行（D1）。
    # ----------------------------------------------------------------- #
    @app.post("/documents", response_model=DocumentInfo)
    async def documents_ingest(request: Request) -> DocumentInfo:
        """入力を取り込み content_hash を発行する。

        JSON（application/json）＝テキスト取込、multipart/form-data＝ファイル取込
        （``file`` にファイル本体、任意で ``source_name``）を content-type で分岐する。
        """
        ctype = request.headers.get("content-type", "")
        try:
            if ctype.startswith("multipart/form-data"):
                form = await request.form()
                upload = form.get("file")
                if not isinstance(upload, UploadFile):
                    raise HTTPException(
                        422,
                        "multipart には `file` フィールド（ファイル本体）が必要です",
                    )
                data = await upload.read()
                name = upload.filename or "upload"
                return ingest_file(app.state.ctx, name, data)
            req = DocumentIngestRequest.model_validate(await request.json())
        except ValidationError as e:
            raise HTTPException(422, e.errors()) from e
        return ingest_text(app.state.ctx, req.text, req.source_name)

    @app.get("/documents", response_model=list[DocumentInfo])
    def documents_list() -> list[DocumentInfo]:
        return list_documents(app.state.ctx)

    @app.get("/documents/{content_hash}", response_model=DocumentDetail)
    def documents_get(content_hash: str) -> DocumentDetail:
        return get_document(app.state.ctx, content_hash)

    @app.delete("/documents/{content_hash}", status_code=204)
    def documents_delete(content_hash: str, layer: str | None = None) -> None:
        """文書を削除。``?layer=ner`` で NER キャッシュだけ破棄（本文・LLM・draft は残す）。"""
        delete_document(app.state.ctx, content_hash, layer)

    @app.patch("/documents/{content_hash}", response_model=DocumentInfo)
    def documents_patch(content_hash: str, patch: DocumentPatch) -> DocumentInfo:
        return patch_document(app.state.ctx, content_hash, patch.source_kind)

    return app


# uvicorn/`data-redactor serve` が参照するモジュールレベルの app（起動時に実資源をロード）。
app = create_app()
