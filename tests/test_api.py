"""マスキング HTTP API（最小面）のテスト。設計 §3-1/§3-2・M3。

GiNZA の実ロードを避けるため、軽量な :class:`~src.api.service.ApiContext`（``ja_ginza`` のみ）を
注入して :func:`~src.api.app.create_app` を組み立てる。LLM（pii-masker/Azure）は呼ばない
（detection=``ner`` で完結させる）。網羅する観点：

- エンドポイント（/health・/config・/mask・/unmask）と確信度の wire(ASCII) 変換。
- **parts バンドルの共有対応表**（同じ実体は全パートで同じプレースホルダ）。
- **unmask の安全性**（mapping に無いプレースホルダは無変更＝LLM 捏造への安全側）。
- エラー契約（404 未取込 hash / 422 不正入力・未対応拡張子 / 503 モデル未ロード）。
- content_hash 参照・multipart 同梱ファイル・pending（閾値未満のレビュー候補）。
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app, ner_models_from_env
from src.api.service import ApiContext
from src.masking import (
    MaskAllowlist,
    MaskDictionary,
    MaskingEngine,
    NerCache,
    content_hash,
)
from src.ner.preprocess import SAFE_CHUNK_BYTES


@pytest.fixture(scope="module")
def _engine() -> MaskingEngine:
    """辞書入り・``ja_ginza`` 単体のエンジン（モジュール内で使い回す＝モデルロードは 1 回）。"""
    dictionary = MaskDictionary({"sony": ("SONY", "社名"), "canon": ("Canon", "社名")})
    engine = MaskingEngine(dictionary=dictionary, models=["ja_ginza"])
    _ = engine.engines[0].nlp  # 事前ロード
    return engine


@pytest.fixture
def client(_engine: MaskingEngine, tmp_path: Path) -> TestClient:
    """テスト用クライアント（テストごとに空の cache.db を持つ ctx を注入）。"""
    cache = NerCache(str(tmp_path / "cache.db"))
    ctx = ApiContext(
        engine=_engine,
        cache=cache,
        allowlist=MaskAllowlist.empty(),
        model_names=("ja_ginza",),
        models_ready=True,
        dict_path=str(tmp_path / "mask_dict.yaml"),
        allowlist_path=str(tmp_path / "mask_allowlist.yaml"),
    )
    return TestClient(create_app(ctx))


# --------------------------------------------------------------------------- #
# /health・/config
# --------------------------------------------------------------------------- #
def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_ready"] is True
    assert body["models_loaded"] == ["ja_ginza"]


def test_config(client: TestClient) -> None:
    body = client.get("/config").json()
    assert body["models"] == ["ja_ginza"]
    assert body["llm_model"]
    assert "pii-masker@" in body["detector_version"]
    assert body["detection_modes"] == ["ner", "llm", "both"]
    assert body["default_detection"] == "both"
    assert body["default_mask_level"] == "strong"
    # mask_levels は wire(ASCII) の下限候補（除外は含まない）。
    assert body["mask_levels"] == ["certain", "strong", "medium", "weak", "faint"]


# --------------------------------------------------------------------------- #
# 起動時に読み込む NER モデル (環境変数 DATA_REDACTOR_NER_MODELS)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [None, "", " , "])
def test_ner_models_default_is_all_models(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """未設定・空なら、これまでどおり全モデル (electra と ja_ginza)。"""
    if value is None:
        monkeypatch.delenv("DATA_REDACTOR_NER_MODELS", raising=False)
    else:
        monkeypatch.setenv("DATA_REDACTOR_NER_MODELS", value)
    assert ner_models_from_env() == ("ja_ginza_electra", "ja_ginza")


def test_ner_models_ginza_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_REDACTOR_NER_MODELS", "ja_ginza")
    assert ner_models_from_env() == ("ja_ginza",)


def test_ner_models_keep_canonical_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """指定の順や空白に関係なく AVAILABLE_MODELS の順になる (判定が書き方で変わらない)。"""
    monkeypatch.setenv("DATA_REDACTOR_NER_MODELS", " ja_ginza , ja_ginza_electra ")
    assert ner_models_from_env() == ("ja_ginza_electra", "ja_ginza")


def test_ner_models_unknown_name_stops_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """未知の名前は黙って無視せず、起動を止める。"""
    monkeypatch.setenv("DATA_REDACTOR_NER_MODELS", "ja_ginza,ja_ginza_eletra")
    with pytest.raises(ValueError, match="ja_ginza_eletra"):
        ner_models_from_env()


# --------------------------------------------------------------------------- #
# /mask 本体
# --------------------------------------------------------------------------- #
def test_mask_shared_mapping_across_parts(client: TestClient) -> None:
    """同じ実体は**全パートで同じプレースホルダ**（バンドル共有の対応表）。"""
    r = client.post(
        "/mask",
        json={
            "parts": [
                {"id": "p1", "text": "SONYとCanonの展示会。"},
                {"id": "p2", "text": "Canonの新製品をSONYが評価。"},
            ],
            "detection": "ner",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "unconfirmed"

    # canonical → placeholder の対応が 1 対 1 でバンドル全体に効く。
    ph = {m["canonical"]: m["placeholder"] for m in body["mapping"]}
    assert ph["SONY"] != ph["Canon"]
    for part in body["masked_parts"]:
        assert "SONY" not in part["masked_text"]
        assert "Canon" not in part["masked_text"]
        assert ph["SONY"] in part["masked_text"]
        assert ph["Canon"] in part["masked_text"]

    # SONY エントリは両パートに出現する（part ラベルつき occurrences）。
    sony = next(m for m in body["mapping"] if m["canonical"] == "SONY")
    assert {o["part"] for o in sony["occurrences"]} == {"p1", "p2"}
    # 辞書一致＝確信度は wire で "certain"、decided_by は "dict"。
    assert sony["confidence"] == "certain"
    assert sony["decided_by"] == "dict"


def test_mask_single_text_sugar(client: TestClient) -> None:
    """単一 text は parts:[{id:"_", text}] の糖衣。"""
    r = client.post("/mask", json={"text": "SONYの新製品。", "detection": "ner"})
    assert r.status_code == 200
    body = r.json()
    assert body["masked_parts"][0]["id"] == "_"
    assert "SONY" not in body["masked_parts"][0]["masked_text"]


def test_mask_detector_echo(client: TestClient) -> None:
    body = client.post("/mask", json={"text": "SONY", "detection": "ner"}).json()
    det = body["detector"]
    assert det["detection"] == "ner"
    assert det["models"] == ["ja_ginza"]
    assert det["mask_level"] == "strong"
    assert "pii-masker@" in det["detector_version"]


def test_mask_content_hash_reference(
    client: TestClient, _engine: MaskingEngine
) -> None:
    """先に取り込んだ文書を content_hash で参照してマスクできる。"""
    # cache に文書を登録（/documents 相当。M2 では直接 record_document で用意）。
    chunks = ["SONY製カメラとCanon製レンズ。"]
    chash = content_hash(chunks)
    # client fixture の ctx が持つ cache に登録する。
    client.app.state.ctx.cache.record_document(chash, "text", "sample.txt", chunks)

    r = client.post(
        "/mask",
        json={"parts": [{"id": "f1", "content_hash": chash}], "detection": "ner"},
    )
    assert r.status_code == 200
    masked = r.json()["masked_parts"][0]["masked_text"]
    assert "SONY" not in masked and "Canon" not in masked


# --------------------------------------------------------------------------- #
# /mask のキャッシュの単位
#
# file の part は part 1 つ (バンドル全体ではない) を単位にキャッシュする。
# text の part はキャッシュしない (毎回内容が変わるプロンプトなどを想定。_uses_cache)。
# 呼び出し側には「キャッシュしたい文書は file で、part に分けて送る」と依頼しているので、
# その説明の根拠をテストで固定しておく。
# --------------------------------------------------------------------------- #
_PROMPT = "次の資料を要約して。"
_DOC_A = "SONYの新型センサはCanonを上回る。"
_DOC_B = "Canonのレンズは堅実との評価。"


def _mask_bundle(
    client: TestClient,
    texts: dict[str, str],
    files: dict[str, str],
    detection: str = "ner",
) -> None:
    """text の part と file の part (中身は .txt) を 1 つのバンドルで /mask に投げる。"""
    parts: list[dict] = [{"id": pid, "text": t} for pid, t in texts.items()]
    parts += [{"id": pid, "file": {"filename": f"{pid}.txt"}} for pid in files]
    r = client.post(
        "/mask",
        data={"manifest": json.dumps({"parts": parts, "detection": detection})},
        files={
            pid: (f"{pid}.txt", body.encode(), "text/plain")
            for pid, body in files.items()
        },
    )
    assert r.status_code == 200, r.text


def _count_ner(_engine: MaskingEngine, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """GiNZA を呼ぶ NerEngine.analyze_chunks に渡った本文を記録するようにする。

    所要時間で測るとマシン差で不安定なので、呼び出しの記録で見る。
    """
    ner_engine = _engine.engines[0]
    analyzed: list[str] = []
    original = ner_engine.analyze_chunks

    def counting(chunks, **kwargs):  # type: ignore[no-untyped-def]
        chunks = list(chunks)
        analyzed.append("".join(chunks))
        return original(chunks, **kwargs)

    monkeypatch.setattr(ner_engine, "analyze_chunks", counting)
    return analyzed


def test_mask_file_parts_are_cached_per_part(client: TestClient) -> None:
    """file の part は part ごとに NER キャッシュが入り、文書一覧にも出る。text の part は入らない。"""
    _mask_bundle(client, {"prompt": _PROMPT}, {"docA": _DOC_A, "docB": _DOC_B})

    docs = client.get("/documents").json()
    assert sorted((d["source_kind"], d["source_name"]) for d in docs) == [
        ("api", "docA.txt"),
        ("api", "docB.txt"),
    ]
    # 一覧の hash と NER キャッシュの鍵が一致している (一覧で NER 済みと表示される)。
    assert all(d["ner_models"] == ["ja_ginza"] for d in docs)
    cache = client.app.state.ctx.cache
    assert cache.get(content_hash([_PROMPT]), "ja_ginza", True) is None


def test_mask_text_part_is_reanalyzed_every_time(
    client: TestClient, _engine: MaskingEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """text の part は同じ内容でも毎回解析する。file の part は 2 回目から使い回す。"""
    analyzed = _count_ner(_engine, monkeypatch)

    _mask_bundle(client, {"prompt": _PROMPT}, {"docA": _DOC_A})
    assert len(analyzed) == 2  # 初回は 2 part とも解析

    analyzed.clear()
    _mask_bundle(client, {"prompt": "この資料の課題を3点で。"}, {"docA": _DOC_A})
    assert analyzed == ["この資料の課題を3点で。"]  # docA はキャッシュ命中＝再解析なし

    analyzed.clear()
    _mask_bundle(client, {"prompt": _PROMPT}, {"docA": _DOC_A})
    assert analyzed == [_PROMPT]  # 同じ text でもキャッシュしていないので解析する


def test_mask_file_cache_misses_when_parts_are_concatenated(
    client: TestClient, _engine: MaskingEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """プロンプトと文書を 1 つの file に連結して送ると、変わらない部分もまとめて再解析される。

    呼び出し側へ「part に分けてほしい」と依頼する根拠。連結した全体で 1 つの
    content_hash になるため、プロンプトが 1 文字変わるだけで全体がミスする。
    """
    _mask_bundle(client, {}, {"all": f"{_PROMPT}\n{_DOC_A}"})
    analyzed = _count_ner(_engine, monkeypatch)
    _mask_bundle(client, {}, {"all": f"この資料の課題を3点で。\n{_DOC_A}"})
    assert len(analyzed) == 1  # docA は前回と同じ内容だが、全体が別の鍵になるので再解析


def test_mask_text_part_skips_llm_cache(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM 検出も同じ。file の part は結果をキャッシュし、text の part は毎回検出する。"""
    import src.detector as detector_mod
    import src.llm.detect_layer as detect_layer_mod
    from src.llm.schema import LlmDetection, body_sha256

    detected: list[str] = []

    def fake_detect(text: str, **kwargs):  # type: ignore[no-untyped-def]
        detected.append(text)
        return LlmDetection(
            spans=(),
            not_found=(),
            model="fake",
            detector_version=kwargs["detector_version"],
            body_sha256=body_sha256(text),
        )

    monkeypatch.setattr(detector_mod, "detect_document", fake_detect)
    monkeypatch.setattr(detect_layer_mod, "detect_document", fake_detect)

    _mask_bundle(client, {"prompt": _PROMPT}, {"docA": _DOC_A}, detection="llm")
    assert len(detected) == 2
    detected.clear()
    _mask_bundle(client, {"prompt": _PROMPT}, {"docA": _DOC_A}, detection="llm")
    assert detected == [_PROMPT]  # docA はキャッシュ命中、text は毎回検出

    docs = client.get("/documents").json()
    assert [d["source_name"] for d in docs] == ["docA.txt"]
    assert docs[0]["llm_versions"] != []


def test_mask_multipart_file(client: TestClient) -> None:
    """同梱ファイル（multipart）を DocumentLoader でテキスト化してマスクする。"""
    manifest = '{"parts": [{"id": "fileA", "file": {"filename": "memo.txt"}}], "detection": "ner"}'
    r = client.post(
        "/mask",
        data={"manifest": manifest},
        files={"fileA": ("memo.txt", "SONYとCanonの比較メモ。".encode(), "text/plain")},
    )
    assert r.status_code == 200
    masked = r.json()["masked_parts"][0]["masked_text"]
    assert masked  # 非空
    assert "SONY" not in masked and "Canon" not in masked


def test_mask_pending_below_threshold(client: TestClient) -> None:
    """閾値未満（単系統 NER の人名＝中）は pending に出る（wire enum・votes つき）。"""
    r = client.post(
        "/mask",
        json={"text": "佐藤さんが会議に出席した。", "detection": "ner"},
    )
    assert r.status_code == 200
    pending = r.json()["pending"]
    sato = next(p for p in pending if p["surface"] == "佐藤")
    assert sato["category"] == "人名"
    assert sato["confidence"] == "medium"  # 単系統＝中 → wire で medium
    assert sato["votes"].get("ner") == "人名"
    assert sato["occurrences"] and sato["occurrences"][0]["part"] == "_"


def test_mask_return_pending_false(client: TestClient) -> None:
    body = client.post(
        "/mask",
        json={
            "text": "佐藤さんが会議に出席した。",
            "detection": "ner",
            "return_pending": False,
        },
    ).json()
    assert body["pending"] == []


def test_mask_level_medium_masks_single_system(client: TestClient) -> None:
    """mask_level=medium なら単系統（中）の人名も自動マスクされる（§1-A の下限）。"""
    body = client.post(
        "/mask",
        json={
            "text": "佐藤さんが会議に出席した。",
            "detection": "ner",
            "mask_level": "medium",
        },
    ).json()
    assert "佐藤" not in body["masked_parts"][0]["masked_text"]
    assert any(m["canonical"] == "佐藤" for m in body["mapping"])


# --------------------------------------------------------------------------- #
# /unmask
# --------------------------------------------------------------------------- #
def test_unmask_roundtrip(client: TestClient) -> None:
    m = client.post("/mask", json={"text": "SONYの話。", "detection": "ner"}).json()
    masked = m["masked_parts"][0]["masked_text"]
    u = client.post("/unmask", json={"text": masked, "mapping": m["mapping"]})
    assert u.status_code == 200
    assert u.json()["restored_text"] == "SONYの話。"


def test_unmask_ignores_unknown_placeholder(client: TestClient) -> None:
    """mapping に無いプレースホルダは無変更（LLM の捏造・改変への安全側）。"""
    mapping = [
        {
            "placeholder": "[社1]",
            "category": "社名",
            "canonical": "SONY",
            "surfaces": ["SONY"],
            "confidence": "certain",
            "decided_by": "dict",
            "occurrences": [],
        }
    ]
    u = client.post(
        "/unmask",
        json={"text": "[社1]と[社99]の話。", "mapping": mapping},
    )
    assert u.status_code == 200
    # [社1] は復元、未知の [社99] はそのまま残す。
    assert u.json()["restored_text"] == "SONYと[社99]の話。"


# --------------------------------------------------------------------------- #
# エラー契約
# --------------------------------------------------------------------------- #
def test_mask_404_unknown_content_hash(client: TestClient) -> None:
    r = client.post(
        "/mask",
        json={"parts": [{"id": "x", "content_hash": "deadbeef"}], "detection": "ner"},
    )
    assert r.status_code == 404


def test_mask_422_unknown_detection(client: TestClient) -> None:
    r = client.post("/mask", json={"text": "a", "detection": "bogus"})
    assert r.status_code == 422


def test_mask_422_unknown_mask_level(client: TestClient) -> None:
    r = client.post("/mask", json={"text": "a", "mask_level": "bogus"})
    assert r.status_code == 422


def test_mask_422_bad_part_shape(client: TestClient) -> None:
    # text も content_hash も file も無い part。
    r = client.post("/mask", json={"parts": [{"id": "x"}], "detection": "ner"})
    assert r.status_code == 422


def test_mask_422_parts_and_text_both(client: TestClient) -> None:
    r = client.post(
        "/mask",
        json={"text": "a", "parts": [{"id": "x", "text": "b"}], "detection": "ner"},
    )
    assert r.status_code == 422


def test_mask_422_duplicate_part_id(client: TestClient) -> None:
    r = client.post(
        "/mask",
        json={
            "parts": [{"id": "x", "text": "a"}, {"id": "x", "text": "b"}],
            "detection": "ner",
        },
    )
    assert r.status_code == 422


def test_mask_422_unsupported_extension(client: TestClient) -> None:
    manifest = (
        '{"parts": [{"id": "f", "file": {"filename": "a.exe"}}], "detection": "ner"}'
    )
    r = client.post(
        "/mask",
        data={"manifest": manifest},
        files={"f": ("a.exe", b"\x00\x01", "application/octet-stream")},
    )
    assert r.status_code == 422


def test_mask_422_models_subset_not_supported(client: TestClient) -> None:
    r = client.post(
        "/mask",
        json={"text": "a", "detection": "ner", "models": ["ja_ginza_electra"]},
    )
    assert r.status_code == 422


def test_mask_503_when_models_not_ready(_engine: MaskingEngine, tmp_path: Path) -> None:
    """モデル未ロード時、NER を要求すると 503。"""
    ctx = ApiContext(
        engine=_engine,
        cache=NerCache(str(tmp_path / "cache.db")),
        allowlist=MaskAllowlist.empty(),
        model_names=("ja_ginza",),
        models_ready=False,
    )
    c = TestClient(create_app(ctx))
    r = c.post("/mask", json={"text": "a", "detection": "ner"})
    assert r.status_code == 503


# --------------------------------------------------------------------------- #
# 全体面：/documents 系（M5a）
# --------------------------------------------------------------------------- #
def test_config_supported_extensions(client: TestClient) -> None:
    exts = client.get("/config").json()["supported_extensions"]
    assert ".txt" in exts  # DocumentLoader が対応する拡張子が載る


def test_documents_ingest_text_then_reference(client: TestClient) -> None:
    """テキスト取込→content_hash 発行→その hash を /mask で参照できる（D1）。"""
    r = client.post(
        "/documents", json={"text": "SONYとCanonの比較。", "source_name": "memo"}
    )
    assert r.status_code == 200
    doc = r.json()
    assert doc["content_hash"] and doc["source_kind"] == "text"
    assert doc["source_name"] == "memo" and doc["chunk_count"] == 1

    m = client.post(
        "/mask",
        json={
            "parts": [{"id": "d", "content_hash": doc["content_hash"]}],
            "detection": "ner",
        },
    )
    assert m.status_code == 200
    masked = m.json()["masked_parts"][0]["masked_text"]
    assert "SONY" not in masked and "Canon" not in masked


def test_documents_ingest_file(client: TestClient) -> None:
    r = client.post(
        "/documents",
        files={"file": ("memo.txt", "SONYの比較メモ。".encode(), "text/plain")},
    )
    assert r.status_code == 200
    doc = r.json()
    assert doc["source_kind"] == "file" and doc["source_name"] == "memo.txt"


def test_documents_ingest_unsupported_extension_422(client: TestClient) -> None:
    r = client.post(
        "/documents",
        files={"file": ("a.exe", b"\x00\x01", "application/octet-stream")},
    )
    assert r.status_code == 422


def test_documents_list_and_get_detail(client: TestClient) -> None:
    h = client.post("/documents", json={"text": "SONYの話。"}).json()["content_hash"]
    docs = client.get("/documents").json()
    assert any(d["content_hash"] == h for d in docs)
    assert all("ner_models" in d and "llm_versions" in d for d in docs)

    detail = client.get(f"/documents/{h}").json()
    assert detail["chunks"] == ["SONYの話。"]


def test_documents_get_404(client: TestClient) -> None:
    assert client.get("/documents/deadbeef").status_code == 404


def test_documents_delete(client: TestClient) -> None:
    h = client.post("/documents", json={"text": "SONYの話。"}).json()["content_hash"]
    assert client.delete(f"/documents/{h}").status_code == 204
    assert client.get(f"/documents/{h}").status_code == 404


def test_documents_delete_ner_keeps_document(client: TestClient) -> None:
    """?layer=ner は NER キャッシュだけ破棄し、文書本体は残す。"""
    h = client.post("/documents", json={"text": "佐藤さんが出席した。"}).json()[
        "content_hash"
    ]
    # NER 解析でキャッシュを埋める（content_hash 参照）。
    client.post(
        "/mask",
        json={
            "parts": [{"id": "d", "content_hash": h}],
            "detection": "ner",
            "mask_level": "medium",
        },
    )
    assert client.get(f"/documents/{h}").json()["ner_models"]  # NER 済み
    assert client.delete(f"/documents/{h}?layer=ner").status_code == 204
    after = client.get(f"/documents/{h}").json()
    assert after["ner_models"] == []  # NER は破棄・文書は残る


def test_documents_patch_source_kind(client: TestClient) -> None:
    h = client.post("/documents", json={"text": "SONYの話。"}).json()["content_hash"]
    r = client.patch(f"/documents/{h}", json={"source_kind": "kb"})
    assert r.status_code == 200 and r.json()["source_kind"] == "kb"


# --------------------------------------------------------------------------- #
# /mask に file で送った文書は文書一覧に出る (text は出ない)
# --------------------------------------------------------------------------- #
def test_mask_does_not_record_text_part(client: TestClient) -> None:
    """text の part はキャッシュしないので、文書一覧にも記録しない。"""
    text = "SONYとCanonの比較。"
    client.post("/mask", json={"text": text, "detection": "ner"})
    assert client.get("/documents").json() == []
    cache = client.app.state.ctx.cache
    assert cache.get(content_hash([text]), "ja_ginza", True) is None


def test_mask_records_file_part_with_filename(client: TestClient) -> None:
    manifest = (
        '{"parts": [{"id": "f", "file": {"filename": "memo.txt"}}], "detection": "ner"}'
    )
    client.post(
        "/mask",
        data={"manifest": manifest},
        files={"f": ("memo.txt", "SONYの比較メモ。".encode(), "text/plain")},
    )
    docs = [d for d in client.get("/documents").json() if d["source_kind"] == "api"]
    assert [d["source_name"] for d in docs] == ["memo.txt"]


def test_mask_keeps_meta_of_ingested_document(client: TestClient) -> None:
    """取込済みと同じファイルを /mask しても、取込時の種別・名前は "api" で上書きされない。"""
    body = "SONYの話。".encode()
    h = client.post(
        "/documents",
        data={"source_name": "memo"},
        files={"file": ("memo.txt", body, "text/plain")},
    ).json()["content_hash"]
    manifest = (
        '{"parts": [{"id": "f", "file": {"filename": "memo.txt"}}], "detection": "ner"}'
    )
    client.post(
        "/mask",
        data={"manifest": manifest},
        files={"f": ("memo.txt", body, "text/plain")},
    )
    docs = client.get("/documents").json()
    assert [(d["content_hash"], d["source_kind"], d["source_name"]) for d in docs] == [
        (h, "file", "memo")
    ]


def test_mask_does_not_record_on_error(client: TestClient) -> None:
    """入力エラーで解析しなかったときは記録しない。"""
    manifest = json.dumps(
        {
            "parts": [
                {"id": "a", "file": {"filename": "memo.txt"}},
                {"id": "b", "content_hash": "deadbeef"},
            ],
            "detection": "ner",
        }
    )
    r = client.post(
        "/mask",
        data={"manifest": manifest},
        files={"a": ("memo.txt", "SONYの話。".encode(), "text/plain")},
    )
    assert r.status_code == 404
    assert client.get("/documents").json() == []


# --------------------------------------------------------------------------- #
# 長いテキスト (SAFE_CHUNK_BYTES 超) でも、伏せた文書と位置が原文とずれない
# --------------------------------------------------------------------------- #
def _long_text() -> str:
    """空行と字下げした箇条書きを含み、UTF-8 で SAFE_CHUNK_BYTES を超える Markdown。"""
    sections = []
    i = 0
    while len("\n".join(sections).encode("utf-8")) <= SAFE_CHUNK_BYTES + 2000:
        sections.append(
            f"## 項目{i}\n\nSONY と Canon の比較を記す。\n\n  - 手順を確認する\n"
        )
        i += 1
    return "\n".join(sections)


def _rebuild(original: str, mapping: list[dict]) -> str:
    """原文の occurrences の位置を placeholder に置き換えて組み立て直す。"""
    spans = sorted(
        (
            (o["span"][0], o["span"][1], m["placeholder"])
            for m in mapping
            for o in m["occurrences"]
        ),
        reverse=True,
    )
    for start, end, ph in spans:
        original = original[:start] + ph + original[end:]
    return original


def test_mask_long_text_spans_match_original(client: TestClient) -> None:
    """長いテキストでも、span を原文に当てた結果が masked_text と一致し、復元は原文に戻る。

    以前は SAFE_CHUNK_BYTES ごとに改行が足され、その後ろの span がずれていた
    (2026-11-25 の不具合報告)。
    """
    text = _long_text()
    body = client.post("/mask", json={"text": text, "detection": "ner"}).json()
    masked = body["masked_parts"][0]["masked_text"]
    assert masked.count("\n") == text.count("\n")
    assert _rebuild(text, body["mapping"]) == masked
    restored = client.post(
        "/unmask", json={"text": masked, "mapping": body["mapping"]}
    ).json()["restored_text"]
    assert restored == text


def test_stale_ner_cache_is_reanalyzed(
    client: TestClient, _engine: MaskingEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """キャッシュの解析が今の本文と違う本文に対するもの (修正前に保存したもの) なら使わない。"""
    text = "SONYとCanonの比較。"
    # text の part はキャッシュしないので、取り込んで content_hash で参照する。
    chash = client.post("/documents", json={"text": text}).json()["content_hash"]
    req = {"parts": [{"id": "d", "content_hash": chash}], "detection": "ner"}
    client.post("/mask", json=req)
    cache = client.app.state.ctx.cache
    good = cache.get(chash, "ja_ginza", True)
    assert good is not None
    # 修正前の不具合で改行が足された本文を持つ解析に差し替える。
    cache.put(chash, "ja_ginza", True, replace(good, text="\n" + good.text))

    ner_engine = _engine.engines[0]
    analyzed: list[str] = []
    original = ner_engine.analyze_chunks

    def counting(chunks, **kwargs):  # type: ignore[no-untyped-def]
        chunks = list(chunks)
        analyzed.append("".join(chunks))
        return original(chunks, **kwargs)

    monkeypatch.setattr(ner_engine, "analyze_chunks", counting)
    client.post("/mask", json=req)
    assert analyzed == [text]  # 古い解析は使わず解析し直した
    assert cache.get(chash, "ja_ginza", True).text == good.text  # 正しい解析で上書き
    client.post("/mask", json=req)
    assert analyzed == [text]  # 上書き後はキャッシュを使う


# --------------------------------------------------------------------------- #
# 検出中も他の要求は止まらない・重い検出は 1 件ずつ
# --------------------------------------------------------------------------- #
def _block_ner(
    _engine: MaskingEngine, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], threading.Event, threading.Event]:
    """GiNZA の解析を release がセットされるまで止める。

    戻り値は (解析に入った本文の記録, 解析に入ったら立つ合図, 解析を再開させる合図)。
    """
    ner_engine = _engine.engines[0]
    entered: list[str] = []
    started = threading.Event()
    release = threading.Event()
    original = ner_engine.analyze_chunks

    def blocking(chunks, **kwargs):  # type: ignore[no-untyped-def]
        chunks = list(chunks)
        entered.append("".join(chunks))
        started.set()
        release.wait(timeout=30)
        return original(chunks, **kwargs)

    monkeypatch.setattr(ner_engine, "analyze_chunks", blocking)
    return entered, started, release


def _post_in_thread(client: TestClient, text: str) -> threading.Thread:
    t = threading.Thread(
        target=client.post,
        args=("/mask",),
        kwargs={"json": {"text": text, "detection": "ner"}},
    )
    t.start()
    return t


def test_mask_does_not_block_other_requests(
    client: TestClient, _engine: MaskingEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/mask の解析中でも、文書一覧や /health はすぐ返る。

    以前は async def の中で重い解析を直接呼んでいたため、イベントループが止まり、
    解析が終わるまで他の要求がすべて待たされていた (画面がずっと読み込み中になる)。
    with で TestClient を開くと、全スレッドの要求が同じイベントループで処理される
    (本番の uvicorn と同じ条件になる)。
    """
    _entered, started, release = _block_ner(_engine, monkeypatch)
    with client:
        masker = _post_in_thread(client, "SONYの話。")
        try:
            assert started.wait(timeout=30)
            results: list[int] = []
            other = threading.Thread(
                target=lambda: results.extend(
                    [
                        client.get("/health").status_code,
                        client.get("/documents").status_code,
                    ]
                )
            )
            other.start()
            other.join(timeout=10)
            assert results == [200, 200]  # 解析の終了を待たずに返った
        finally:
            release.set()
            masker.join(timeout=30)


def test_mask_runs_heavy_detection_one_at_a_time(
    client: TestClient, _engine: MaskingEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2 つの /mask が同時に来ても、GiNZA の解析は 1 件ずつ順番に行う。"""
    entered, started, release = _block_ner(_engine, monkeypatch)
    with client:
        first = _post_in_thread(client, "SONYの話。")
        second: threading.Thread | None = None
        try:
            assert started.wait(timeout=30)
            second = _post_in_thread(client, "Canonの話。")
            second.join(timeout=1)
            assert entered == ["SONYの話。"]  # 2 件目は 1 件目の解析が終わるまで待つ
        finally:
            release.set()
            first.join(timeout=30)
            if second is not None:
                second.join(timeout=30)
    assert entered == ["SONYの話。", "Canonの話。"]


# --------------------------------------------------------------------------- #
# 全体面：/analyze・/apply・/draft（M5b）
# --------------------------------------------------------------------------- #
def _ingest(client: TestClient, text: str) -> str:
    return client.post("/documents", json={"text": text}).json()["content_hash"]


def test_analyze_groups_shape(client: TestClient) -> None:
    """analyze は実体グループ（surface/category/confidence/count/votes/occurrences）を返す。"""
    h = _ingest(client, "SONYとSONYの再掲。Canonも。")
    body = client.post(
        f"/documents/{h}/analyze", json={"detection": "ner", "mask_level": "medium"}
    ).json()
    sony = next(g for g in body["groups"] if g["surface"] == "SONY")
    assert sony["category"] == "社名" and sony["confidence"] == "certain"
    assert sony["count"] >= 2 and len(sony["occurrences"]) >= 2
    assert sony["votes"].get("dict")  # 辞書一致の票
    # auto_selection は解析座標の span 集合（辞書一致は既定選択に入る）。
    assert body["auto_selection"]
    # text は解析（平坦化後）座標のテキスト。occurrences の span でその語が切り出せる。
    text = body["text"]
    s, e = sony["occurrences"][0]["span"]
    assert text[s:e] == "SONY"


def test_analyze_apply_unmask_roundtrip(client: TestClient) -> None:
    """analyze の auto_selection をそのまま apply → masked_text、mapping で unmask 復元。"""
    h = _ingest(client, "SONYとCanonの比較。")
    a = client.post(
        f"/documents/{h}/analyze", json={"detection": "ner", "mask_level": "medium"}
    ).json()
    ap = client.post(
        f"/documents/{h}/apply",
        json={"selection": a["auto_selection"], "detection": "ner"},
    )
    assert ap.status_code == 200
    masked = ap.json()["masked_text"]
    assert "SONY" not in masked and "Canon" not in masked

    u = client.post("/unmask", json={"text": masked, "mapping": ap.json()["mapping"]})
    assert u.json()["restored_text"] == "SONYとCanonの比較。"


def test_apply_empty_selection_masks_nothing(client: TestClient) -> None:
    h = _ingest(client, "SONYの話。")
    ap = client.post(
        f"/documents/{h}/apply", json={"selection": [], "detection": "ner"}
    ).json()
    assert ap["masked_text"] == "SONYの話。" and ap["mapping"] == []


def test_analyze_404_unknown_hash(client: TestClient) -> None:
    r = client.post("/documents/deadbeef/analyze", json={"detection": "ner"})
    assert r.status_code == 404


def test_analyze_422_bad_detection(client: TestClient) -> None:
    h = _ingest(client, "a")
    r = client.post(f"/documents/{h}/analyze", json={"detection": "bogus"})
    assert r.status_code == 422


def test_draft_get_empty_then_save_roundtrip(client: TestClient) -> None:
    h = _ingest(client, "SONYの話。")
    assert client.get(f"/documents/{h}/draft").json() == {"added": [], "removed": []}
    client.put(f"/documents/{h}/draft", json={"added": [[0, 3]], "removed": [[5, 7]]})
    d = client.get(f"/documents/{h}/draft").json()
    assert [0, 3] in d["added"] and [5, 7] in d["removed"]


def test_draft_404_unknown_hash(client: TestClient) -> None:
    assert client.get("/documents/deadbeef/draft").status_code == 404


# --------------------------------------------------------------------------- #
# 全体面：/allowlist・/dictionary エディタ（M5c）
# --------------------------------------------------------------------------- #
def test_allowlist_get_empty_then_put_roundtrip(client: TestClient) -> None:
    assert client.get("/allowlist").json() == {"entries": []}
    body = {"entries": [{"surface": "Em_NoYes"}]}
    assert client.put("/allowlist", json=body).status_code == 200
    got = client.get("/allowlist").json()["entries"]
    assert any(e["surface"] == "Em_NoYes" for e in got)


def test_allowlist_put_reloads_and_excludes(client: TestClient) -> None:
    """PUT /allowlist は ctx.allowlist を再ロードし、以降のマスクに即反映（佐藤を除外）。"""
    base = client.post(
        "/mask",
        json={
            "text": "佐藤さんが会議に出席。",
            "detection": "ner",
            "mask_level": "medium",
        },
    ).json()
    assert any(m["canonical"] == "佐藤" for m in base["mapping"])  # 除外前はマスク対象

    client.put("/allowlist", json={"entries": [{"surface": "佐藤"}]})
    after = client.post(
        "/mask",
        json={
            "text": "佐藤さんが会議に出席。",
            "detection": "ner",
            "mask_level": "medium",
        },
    ).json()
    assert not any(m["canonical"] == "佐藤" for m in after["mapping"])  # 除外後は対象外


def test_dictionary_editor_roundtrip_and_reload(client: TestClient) -> None:
    """PUT /dictionary は保存＋engine.dictionary 再ロード。共有エンジンは後で戻す。"""
    eng = client.app.state.ctx.engine
    original = eng.dictionary
    try:
        assert client.get("/dictionary").json() == {"entries": []}
        entries = [{"category": "社名", "canonical": "ニコン", "aliases": ["Nikon"]}]
        assert client.put("/dictionary", json={"entries": entries}).status_code == 200
        got = client.get("/dictionary").json()["entries"]
        assert any(e["canonical"] == "ニコン" for e in got)
        # 再ロード反映：ニコンが辞書一致（certain）でマスクされる。
        m = client.post(
            "/mask", json={"text": "ニコンの製品。", "detection": "ner"}
        ).json()
        assert any(
            e["canonical"] == "ニコン" and e["confidence"] == "certain"
            for e in m["mapping"]
        )
    finally:
        eng.dictionary = original  # 共有エンジンを元に戻す（他テストへの影響を防ぐ）


# --------------------------------------------------------------------------- #
# 取込のソース非依存（2026-07-21 の設計見直し）：サーバは text か file を受け取るだけ。
# kb-mcp 等からの取得はクライアント（UI）の責務＝元ファイルを取得して multipart/file で送る。
# --------------------------------------------------------------------------- #
def test_documents_ingest_requires_text(client: TestClient) -> None:
    """JSON 取込は text 必須（空ボディは 422）。バイナリは multipart で送る。"""
    assert client.post("/documents", json={}).status_code == 422


def test_documents_ingest_file_source_name_override(client: TestClient) -> None:
    """multipart の source_name フォーム値で一覧表示名を上書きできる（kb 文書名の保持用）。"""
    r = client.post(
        "/documents",
        data={"source_name": "見積書（kb）"},
        files={"file": ("upload.txt", "SONYの比較メモ。".encode(), "text/plain")},
    )
    assert r.status_code == 200
    doc = r.json()
    assert doc["source_kind"] == "file" and doc["source_name"] == "見積書（kb）"
