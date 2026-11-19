"""scripts/dump_cache.py の孤児の復元 (--restore-orphans) のテスト。

孤児は、以前の API が /mask の入力を documents に記録していなかったために生まれた行。
復元すると UI のキャッシュ一覧 (NerCache.list_documents＝GET /documents と同じ) に出る。
スクリプトはプロジェクトを import しないので、ファイルパスから読み込んで試す。
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
from types import ModuleType

import pytest

from src.masking import NerCache, content_hash

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dump_cache.py"


@pytest.fixture(scope="module")
def dump_cache() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dump_cache", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _put_ner(db: Path, h: str, original_text: str, created_at: str) -> None:
    """ner 行を直接書く (GiNZA を動かさずに孤児を作るため)。"""
    analysis = {
        "text": original_text,
        "original_text": original_text,
        "offset_map": [],
        "tokens": [],
        "entities": [],
    }
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO ner VALUES (?, ?, ?, ?, ?)",
            (h, "ja_ginza", 1, json.dumps(analysis, ensure_ascii=False), created_at),
        )


def _put_llm(db: Path, h: str, created_at: str) -> None:
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO llm_detection VALUES (?, ?, ?, ?, ?, ?)",
            (h, "m", 1, "v", "{}", created_at),
        )


def test_content_hash_matches_project(dump_cache: ModuleType) -> None:
    """スクリプトが持つハッシュ計算が、プロジェクトの content_hash と一致する。"""
    for chunks in (["SONYの話。"], ["a", "b"], ["ab"], [""], []):
        assert dump_cache._content_hash(chunks) == content_hash(chunks)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """孤児 3 種と、登録済みの文書 1 件が入った cache.db。"""
    path = tmp_path / "cache.db"
    cache = NerCache(str(path))

    # 1. 復元できる孤児：1 チャンクの text 入力。LLM 行のほうが古い。
    text = "佐藤さんの\n会議メモ。" + "あ" * 50
    h1 = content_hash([text])
    _put_ner(path, h1, text, "2026-08-02T10:00:00")
    _put_llm(path, h1, "2026-08-01T09:00:00")

    # 2. 復元できない孤児：複数チャンク。本文はつながった形でしか残らない。
    h2 = content_hash(["一つ目。", "二つ目。"])
    _put_ner(path, h2, "一つ目。二つ目。", "2026-08-03T10:00:00")

    # 3. 復元できない孤児：LLM だけ実行した (ner 行が無い)。
    _put_llm(path, content_hash(["LLMだけ。"]), "2026-08-04T10:00:00")

    # 登録済みの文書 (孤児ではないので触れない)。
    cache.record_document(content_hash(["登録済み。"]), "text", "memo", ["登録済み。"])
    return path


def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def test_restore_dry_run_lists_without_writing(
    dump_cache: ModuleType, db: Path
) -> None:
    conn = _ro(db)  # 確認表示は読み取り専用でも動く
    try:
        report = dump_cache.restore_orphans(conn, execute=False)
    finally:
        conn.close()
    assert "復元できる孤児 1 件" in report
    assert "復元できない孤児 2 件" in report
    assert "--yes" in report
    assert len(NerCache(str(db)).list_documents()) == 1  # 登録済みの 1 件のまま


def test_restore_registers_recoverable_orphans(
    dump_cache: ModuleType, db: Path
) -> None:
    conn = sqlite3.connect(db)
    try:
        report = dump_cache.restore_orphans(conn, execute=True)
    finally:
        conn.close()
    assert "1 件を登録しました" in report

    cache = NerCache(str(db))
    docs = {d.source_kind: d for d in cache.list_documents()}
    assert set(docs) == {"text", "api"}
    assert (docs["text"].source_name, docs["text"].char_count) == ("memo", 5)

    text = "佐藤さんの\n会議メモ。" + "あ" * 50
    restored = docs["api"]
    assert restored.content_hash == content_hash([text])
    # /mask の入力を記録するときと同じ名前の付け方 (改行は空白・40 文字で切る)。
    assert restored.source_name == ("佐藤さんの 会議メモ。" + "あ" * 50)[:40] + "…"
    assert (restored.char_count, restored.chunk_count) == (len(text), 1)
    assert restored.created_at == "2026-08-01T09:00:00"  # 最初にキャッシュされた日時
    # 一覧から NER 済み・LLM 済みと分かる。本文を入力元として再利用できる。
    assert restored.models == ("ja_ginza",) and restored.has_llm
    assert cache.get_chunks(restored.content_hash) == [text]


def test_restore_is_idempotent(dump_cache: ModuleType, db: Path) -> None:
    for _ in range(2):
        conn = sqlite3.connect(db)
        try:
            report = dump_cache.restore_orphans(conn, execute=True)
        finally:
            conn.close()
    assert "復元できる孤児 0 件" in report  # 2 回目は復元済みなので対象なし
    assert len(NerCache(str(db)).list_documents()) == 2
