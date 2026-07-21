"""``/mask`` / ``/unmask`` のオーケストレーション（HTTP 非依存の中身）。

part の解決（text / content_hash 参照 / 同梱ファイル）→ 各 part を解析（NER/LLM）→
:meth:`MaskingEngine.mask_parts` で**バンドルで共有する対応表**を作る、までを行う。エラーは
:class:`fastapi.HTTPException` で表現する（設計 §7 のエラー契約：404/422/502/503）。

FastAPI のルーティング（:mod:`src.api.app`）はこの層を呼ぶだけ＝薄い。テストは
:class:`~fastapi.testclient.TestClient` 経由でこの層の分岐を網羅する。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import openai
from fastapi import HTTPException

from src.api.enums import (
    DETECTION_MODES,
    MASK_LEVELS,
    confidence_to_wire,
    confidences_at_or_above,
)
from src.api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ApplyRequest,
    ApplyResponse,
    CandidateGroupEntry,
    DetectorInfo,
    DocumentDetail,
    DocumentInfo,
    DraftBody,
    GroupOccurrence,
    MappingEntry,
    MaskedPart,
    MaskRequest,
    MaskResponse,
    Occurrence,
    Part,
    PendingEntry,
    UnmaskRequest,
    UnmaskResponse,
)
from src.core.document.document_loader import DocumentLoader
from src.detector import detector_version, run_llm_detection
from src.masking import (
    CandidateGroup,
    MaskAllowlist,
    MaskAnalysis,
    MaskingEngine,
    NerCache,
    content_hash,
    mapping_from_json,
    normalize,
    unmask,
    vote_category,
)
from src.masking.cache import DocInfo
from src.masking.engine import BundleMaskResult, Candidate
from src.sources.files import load_chunks_from_file

# NER 系統に属さないチャネル（decided_by / 系統別 votes の判定で除外する）。
_NON_NER_CHANNELS = frozenset({"dict", "session", "regex", "collected", "llm"})
# 系統別 votes 表示のカテゴリ優先度（特別＝人名/社名/商標を上位に）。engine の _CAT_PRIORITY と同順。
_CAT_ORDER = ("人名", "社名", "商標", "連絡先", "地名", "その他")
_CAT_RANK = {c: i for i, c in enumerate(_CAT_ORDER)}


@dataclass
class ApiContext:
    """サーバが 1 プロセスで所有する資源（起動時にロード。設計 B）。"""

    engine: MaskingEngine
    cache: NerCache
    allowlist: MaskAllowlist
    model_names: tuple[str, ...]
    models_ready: bool


def _normalize_parts(req: MaskRequest) -> list[Part]:
    """``text`` 糖衣を parts へ畳み、各 part がちょうど 1 種（text/content_hash/file）か検証する。"""
    if req.parts is not None and req.text is not None:
        raise HTTPException(422, "`parts` と `text` は同時に指定できません")
    parts = req.parts
    if parts is None:
        if req.text is None:
            raise HTTPException(422, "`parts` か `text` のいずれかが必要です")
        parts = [Part(id="_", text=req.text)]
    if not parts:
        raise HTTPException(422, "`parts` が空です")
    ids: set[str] = set()
    for p in parts:
        n = sum(x is not None for x in (p.text, p.content_hash, p.file))
        if n != 1:
            raise HTTPException(
                422,
                f"part {p.id!r} は text / content_hash / file のうち"
                "ちょうど 1 つを指定してください",
            )
        if p.id in ids:
            raise HTTPException(422, f"part id が重複しています: {p.id!r}")
        ids.add(p.id)
    return parts


def _resolve_chunks(
    part: Part, ctx: ApiContext, files: dict[str, tuple[str, bytes]]
) -> list[str]:
    """1 part を解析対象チャンク列にする（text / content_hash / 同梱ファイル）。"""
    if part.text is not None:
        return [part.text]
    if part.content_hash is not None:
        chunks = ctx.cache.get_chunks(part.content_hash)
        if chunks is None:
            raise HTTPException(404, f"未取込の content_hash です: {part.content_hash}")
        return chunks
    assert part.file is not None  # _normalize_parts でちょうど 1 種を保証済み
    body = files.get(part.id)
    if body is None:
        raise HTTPException(
            422, f"part {part.id!r} のファイル本体が multipart で送られていません"
        )
    filename, data = body
    ext = Path(filename).suffix.lower()
    if ext not in DocumentLoader.SUPPORTED_EXTENSIONS:
        raise HTTPException(422, f"未対応の拡張子です: {ext or '（なし）'}")
    # DocumentLoader は拡張子でローダーを選ぶ＝一時ファイルに元の拡張子を付けて渡す。
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / f"upload{ext}"
        tmp.write_bytes(data)
        return load_chunks_from_file(tmp)


def _analyze_part(
    ctx: ApiContext,
    chunks: list[str],
    *,
    detection: str,
    flatten: bool,
    mask_level: str,
    refresh: bool = False,
) -> tuple[MaskAnalysis, list[Candidate], list[CandidateGroup]]:
    """1 part を解析し、(解析結果, 自動選択候補, 実体グループ) を返す。

    ``refresh=True`` はキャッシュを無視して NER/LLM とも強制再解析する（結果で上書き）。
    """
    run_ner = detection in ("ner", "both")
    use_llm = detection in ("llm", "both")
    if run_ner and not ctx.models_ready:
        raise HTTPException(
            503, "NER モデルが未ロードです（サーバ起動直後 or ロード失敗）"
        )

    llm_detection = None
    if use_llm:
        try:
            _, llm_detection = run_llm_detection(
                ctx.cache, chunks, flatten, force=refresh
            )
        except (openai.OpenAIError, ImportError) as e:
            raise HTTPException(
                502, f"LLM 検出に失敗しました（資格情報・接続を確認）: {e}"
            ) from e

    analysis = ctx.engine.analyze(
        chunks,
        flatten_tables=flatten,
        allowlist=ctx.allowlist,
        ner_cache=ctx.cache,
        refresh_cache=refresh,
        llm_detection=llm_detection,
        run_ner=run_ner,
    )
    groups = ctx.engine.group_candidates(analysis.candidates)
    threshold = set(confidences_at_or_above(mask_level))
    selected = [m for g in groups if g.confidence in threshold for m in g.members]
    return analysis, selected, groups


def _system_votes(group: CandidateGroup) -> dict[str, str]:
    """実体グループの票を系統別（ner / llm）カテゴリにまとめる（pending の ``votes`` 表示用）。"""
    ner_cats: list[str] = []
    llm_cats: list[str] = []
    for ch, label in group.votes:
        cat = vote_category(ch, label)
        if cat is None:
            continue
        if ch == "llm":
            llm_cats.append(cat)
        elif ch not in _NON_NER_CHANNELS:
            ner_cats.append(cat)
    out: dict[str, str] = {}
    if ner_cats:
        out["ner"] = min(ner_cats, key=lambda c: _CAT_RANK.get(c, 99))
    if llm_cats:
        out["llm"] = min(llm_cats, key=lambda c: _CAT_RANK.get(c, 99))
    return out


def _build_pending(
    ctx: ApiContext,
    part_results: list[tuple[MaskAnalysis, list[Candidate], list[CandidateGroup]]],
    parts: list[Part],
    mask_level: str,
) -> list[PendingEntry]:
    """自動マスク閾値未満のレビュー候補をバンドル（全パート）で集約する。

    対象＝マスク可能だが ``mask_level`` 未満、かつ ``微弱``（既定非表示）・``除外`` を除いたもの。
    同じ canonical は全パートでまとめ、出現位置は**原文座標**（masked_text と同じ座標系）にする。
    """
    at_or_above = set(confidences_at_or_above(mask_level))
    maskable = set(confidences_at_or_above("faint"))  # 除外は含まれない
    pending_levels = (maskable - at_or_above) - {"微弱"}

    merged: dict[str, PendingEntry] = {}
    order: list[str] = []
    for part_index, (analysis, _sel, groups) in enumerate(part_results):
        for g in groups:
            if g.confidence not in pending_levels:
                continue
            key = normalize(g.surface)
            occ = [
                Occurrence(part=parts[part_index].id, span=(c.start, c.end))
                for c in ctx.engine.original_spans(analysis, g.members, expand=False)
            ]
            if key not in merged:
                order.append(key)
                merged[key] = PendingEntry(
                    surface=g.surface,
                    category=g.category,
                    confidence=confidence_to_wire(g.confidence),
                    occurrences=occ,
                    votes=_system_votes(g),
                )
            else:
                merged[key].occurrences.extend(occ)
    return [merged[k] for k in order]


# --------------------------------------------------------------------------- #
# 共通バリデーション・mapping 整形（/mask と /analyze・/apply で共有）。
# --------------------------------------------------------------------------- #
def _validate_detection(detection: str) -> None:
    if detection not in DETECTION_MODES:
        raise HTTPException(422, f"未知の detection です: {detection!r}")


def _validate_mask_level(mask_level: str) -> None:
    if mask_level not in MASK_LEVELS:
        raise HTTPException(422, f"未知の mask_level です: {mask_level!r}")


def _validate_models(ctx: ApiContext, models: list[str] | None) -> None:
    if models is not None and set(models) != set(ctx.model_names):
        # 起動時ロードの固定モデルのみ（部分指定は未対応。黙って無視しない）。
        raise HTTPException(
            422,
            "このビルドでは models の部分指定に未対応です"
            f"（利用可能: {list(ctx.model_names)}）。省略してください",
        )


def _wire_mapping(bundle: BundleMaskResult, part_ids: list[str]) -> list[MappingEntry]:
    """BundleMaskResult の entries を wire の MappingEntry に整形する（/mask・/apply 共通）。"""
    return [
        MappingEntry(
            placeholder=e.placeholder,
            category=e.category,
            canonical=e.canonical,
            surfaces=list(e.surfaces),
            confidence=confidence_to_wire(e.confidence),
            decided_by=e.decided_by,
            occurrences=[
                Occurrence(part=part_ids[pi], span=(s, en))
                for pi, s, en in e.occurrences
            ],
        )
        for e in bundle.entries
    ]


def run_mask(
    ctx: ApiContext, req: MaskRequest, files: dict[str, tuple[str, bytes]]
) -> MaskResponse:
    """``POST /mask`` の中身。parts をバンドル共有の対応表でマスクして返す（設計 §3-1）。"""
    _validate_detection(req.detection)
    _validate_mask_level(req.mask_level)
    _validate_models(ctx, req.models)

    parts = _normalize_parts(req)
    part_results = [
        _analyze_part(
            ctx,
            _resolve_chunks(p, ctx, files),
            detection=req.detection,
            flatten=req.flatten_tables,
            mask_level=req.mask_level,
            refresh=req.refresh,
        )
        for p in parts
    ]
    bundle = ctx.engine.mask_parts(
        [(analysis, selected) for analysis, selected, _ in part_results]
    )

    masked_parts = [
        MaskedPart(id=p.id, masked_text=text)
        for p, text in zip(parts, bundle.masked_texts)
    ]
    mapping = _wire_mapping(bundle, [p.id for p in parts])
    pending = (
        _build_pending(ctx, part_results, parts, req.mask_level)
        if req.return_pending
        else []
    )
    return MaskResponse(
        status="unconfirmed",
        masked_parts=masked_parts,
        mapping=mapping,
        pending=pending,
        detector=DetectorInfo(
            detection=req.detection,
            models=list(ctx.model_names),
            detector_version=detector_version(),
            mask_level=req.mask_level,
        ),
    )


def run_unmask(req: UnmaskRequest) -> UnmaskResponse:
    """``POST /unmask`` の中身。mapping で LLM 応答テキストを復元する（設計 §3-2）。"""
    entries = mapping_from_json(
        [
            {
                "placeholder": m.placeholder,
                "category": m.category,
                "surfaces": m.surfaces,
                "canonical": m.canonical,
            }
            for m in req.mapping
        ]
    )
    return UnmaskResponse(restored_text=unmask(req.text, entries))


# --------------------------------------------------------------------------- #
# 全体面：/documents 系（設計 §2-B）。既存の NerCache/DocumentLoader を配線するだけ。
# content_hash はサーバが発行して返す（D1）。取込済み文書は /mask の content_hash で参照できる。
# --------------------------------------------------------------------------- #
SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(
    sorted(DocumentLoader.SUPPORTED_EXTENSIONS)
)


def _doc_info(ctx: ApiContext, d: DocInfo) -> DocumentInfo:
    """cache の DocInfo → wire の DocumentInfo（llm_versions を補って返す）。"""
    return DocumentInfo(
        content_hash=d.content_hash,
        source_kind=d.source_kind,
        source_name=d.source_name,
        char_count=d.char_count,
        chunk_count=d.chunk_count,
        created_at=d.created_at,
        ner_models=list(d.models),
        llm_versions=sorted(ctx.cache.llm_versions(d.content_hash)),
    )


def _find_doc(ctx: ApiContext, chash: str) -> DocInfo | None:
    return next(
        (d for d in ctx.cache.list_documents() if d.content_hash == chash), None
    )


def _ingest_chunks(
    ctx: ApiContext, chunks: list[str], source_kind: str, source_name: str
) -> DocumentInfo:
    """チャンク列を記録し DocumentInfo を返す。content_hash はサーバが発行（D1）。"""
    if not chunks or not any(c.strip() for c in chunks):
        raise HTTPException(422, "取り込むテキストが空です")
    chash = content_hash(chunks)
    ctx.cache.record_document(chash, source_kind, source_name, chunks)
    info = _find_doc(ctx, chash)
    if info is None:  # 記録直後に見つからないのは異常（保険）。
        raise HTTPException(500, "取り込み後に文書が見つかりませんでした")
    return _doc_info(ctx, info)


def ingest_text(ctx: ApiContext, text: str, source_name: str) -> DocumentInfo:
    """テキストを 1 チャンクとして取り込む（/mask の text 経路と同じ単位）。"""
    return _ingest_chunks(ctx, [text], "text", source_name)


def ingest_file(ctx: ApiContext, filename: str, data: bytes) -> DocumentInfo:
    """アップロードされたファイルを DocumentLoader でテキスト化・チャンク化して取り込む。"""
    ext = Path(filename).suffix.lower()
    if ext not in DocumentLoader.SUPPORTED_EXTENSIONS:
        raise HTTPException(422, f"未対応の拡張子です: {ext or '（なし）'}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / f"upload{ext}"
        tmp.write_bytes(data)
        chunks = load_chunks_from_file(tmp)
    return _ingest_chunks(ctx, chunks, "file", filename)


def list_documents(ctx: ApiContext) -> list[DocumentInfo]:
    """キャッシュ済み文書の一覧（新しい順は cache の実装に従う）。"""
    return [_doc_info(ctx, d) for d in ctx.cache.list_documents()]


def get_document(ctx: ApiContext, chash: str) -> DocumentDetail:
    """1 文書のメタ＋チャンク本文。未取込は 404。"""
    chunks = ctx.cache.get_chunks(chash)
    info = _find_doc(ctx, chash)
    if chunks is None or info is None:
        raise HTTPException(404, f"未取込の content_hash です: {chash}")
    base = _doc_info(ctx, info)
    return DocumentDetail(**base.model_dump(), chunks=chunks)


def delete_document(ctx: ApiContext, chash: str, layer: str | None) -> None:
    """文書を削除。``layer="ner"`` は NER キャッシュのみ破棄（本文・LLM・draft は残す。D2/D4）。"""
    if layer is None:
        ctx.cache.delete(chash)
    elif layer == "ner":
        ctx.cache.delete_ner(chash)
    else:
        raise HTTPException(422, f"未知の layer です: {layer!r}（ner または省略）")


def patch_document(ctx: ApiContext, chash: str, source_kind: str) -> DocumentInfo:
    """文書メタを更新（現状 source_kind のみ。D3）。未取込は 404。"""
    info = _find_doc(ctx, chash)
    if info is None:
        raise HTTPException(404, f"未取込の content_hash です: {chash}")
    ctx.cache.set_source_kind(chash, source_kind)
    updated = _find_doc(ctx, chash)
    assert updated is not None  # 直前に存在を確認済み
    return _doc_info(ctx, updated)


# --------------------------------------------------------------------------- #
# 全体面：/documents/{hash}/analyze・/apply・/draft（設計 §3-3〜§3-4）。
# analyze/apply は取込済みチャンクを（NER キャッシュ越しに）解析し直す＝ステートレス。
# span は解析（平坦化後）座標で、analyze の occurrences と apply の selection は同座標系。
# --------------------------------------------------------------------------- #
def _require_chunks(ctx: ApiContext, chash: str) -> list[str]:
    chunks = ctx.cache.get_chunks(chash)
    if chunks is None:
        raise HTTPException(404, f"未取込の content_hash です: {chash}")
    return chunks


def _group_votes(group: CandidateGroup) -> dict[str, str]:
    """実体グループの票を {channel: ラベル（重複は ' / ' 連結）} にまとめる（表示用）。"""
    per_channel: dict[str, list[str]] = {}
    for ch, label in group.votes:
        labels = per_channel.setdefault(ch, [])
        if label not in labels:
            labels.append(label)
    return {ch: " / ".join(labels) for ch, labels in per_channel.items()}


def run_analyze(ctx: ApiContext, chash: str, req: AnalyzeRequest) -> AnalyzeResponse:
    """``POST /documents/{hash}/analyze``＝候補一覧＋既定選択を返す（設計 §3-3）。"""
    chunks = _require_chunks(ctx, chash)
    _validate_detection(req.detection)
    _validate_mask_level(req.mask_level)
    _validate_models(ctx, req.models)
    _analysis, selected, groups = _analyze_part(
        ctx,
        chunks,
        detection=req.detection,
        flatten=req.flatten_tables,
        mask_level=req.mask_level,
        refresh=req.refresh,
    )
    group_entries = [
        CandidateGroupEntry(
            surface=g.surface,
            category=g.category,
            confidence=confidence_to_wire(g.confidence),
            count=g.count,
            votes=_group_votes(g),
            occurrences=[
                GroupOccurrence(
                    span=(m.start, m.end),
                    confidence=confidence_to_wire(m.confidence),
                )
                for m in g.members
            ],
        )
        for g in groups
    ]
    auto = list(dict.fromkeys((c.start, c.end) for c in selected))
    return AnalyzeResponse(groups=group_entries, auto_selection=auto)


def run_apply(ctx: ApiContext, chash: str, req: ApplyRequest) -> ApplyResponse:
    """``POST /documents/{hash}/apply``＝選択 span からマスク結果を作る（設計 §3-4）。

    analyze と同じ検出条件で解析し直し（NER はキャッシュ命中で一瞬）、選択に一致する候補だけを
    出現ごと（expand=False）でマスクする。mapping は /mask と同じ形＝そのまま /unmask に渡せる。
    """
    chunks = _require_chunks(ctx, chash)
    _validate_detection(req.detection)
    _validate_models(ctx, req.models)
    analysis, _selected, _groups = _analyze_part(
        ctx,
        chunks,
        detection=req.detection,
        flatten=req.flatten_tables,
        mask_level="strong",  # apply は selection を使うので閾値は無関係（既定でよい）
    )
    sel = {(s, e) for s, e in req.selection}
    selected = [c for c in analysis.candidates if (c.start, c.end) in sel]
    bundle = ctx.engine.mask_parts([(analysis, selected)], expand=False)
    return ApplyResponse(
        masked_text=bundle.masked_texts[0],
        mapping=_wire_mapping(bundle, ["_"]),
    )


def get_document_draft(ctx: ApiContext, chash: str) -> DraftBody:
    """``GET /documents/{hash}/draft``＝手動選択差分を返す（無ければ空）。未取込は 404。"""
    _require_chunks(ctx, chash)
    draft = ctx.cache.get_draft(chash)
    if draft is None:
        return DraftBody()
    added, removed = draft
    return DraftBody(added=sorted(added), removed=sorted(removed))


def save_document_draft(ctx: ApiContext, chash: str, body: DraftBody) -> DraftBody:
    """``PUT /documents/{hash}/draft``＝手動選択差分を保存する。未取込は 404。"""
    _require_chunks(ctx, chash)
    ctx.cache.save_draft(
        chash,
        {(s, e) for s, e in body.added},
        {(s, e) for s, e in body.removed},
    )
    return get_document_draft(ctx, chash)
