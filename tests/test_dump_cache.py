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
    # 文書名は本文の先頭 (改行は空白・40 文字で切る)。
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


# --------------------------------------------------------------------------- #
# --vacuum: 削除で空いた領域を取り除いてファイルを縮める (単独でも使える)
# --------------------------------------------------------------------------- #
def _run_main(
    dump_cache: ModuleType, monkeypatch: pytest.MonkeyPatch, *argv: str
) -> int:
    monkeypatch.setattr("sys.argv", ["dump_cache.py", *argv])
    return int(dump_cache.main())


def _bloat_and_delete(db: Path) -> None:
    """大きな文書を入れてから消し、ファイルの中に空き領域を作る (画面で消したときと同じ)。"""
    cache = NerCache(str(db))
    big = ["SONYの話。" * 50_000]
    h = content_hash(big)
    cache.record_document(h, "file", "big.txt", big)
    cache.delete(h)


def test_vacuum_alone_shrinks_file(
    dump_cache: ModuleType,
    db: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bloat_and_delete(db)
    before = db.stat().st_size
    docs_before = NerCache(str(db)).list_documents()
    out = tmp_path / "out.txt"

    assert (
        _run_main(dump_cache, monkeypatch, str(db), "--vacuum", "--out", str(out)) == 0
    )
    assert db.stat().st_size < before
    assert "VACUUM しました" in out.read_text(encoding="utf-8")
    # 縮めるだけで、中身 (文書・孤児) は変わらない。
    assert NerCache(str(db)).list_documents() == docs_before
    with _ro(db) as c:
        assert c.execute("SELECT COUNT(*) FROM ner").fetchone()[0] == 2


def test_vacuum_is_skipped_on_delete_dry_run(
    dump_cache: ModuleType,
    db: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--delete-orphans を --yes 無しで付けたとき (確認表示だけ) は VACUUM もしない。"""
    _bloat_and_delete(db)
    before = db.stat().st_size
    out = tmp_path / "out.txt"
    argv = (str(db), "--delete-orphans", "--vacuum", "--out", str(out))
    assert _run_main(dump_cache, monkeypatch, *argv) == 0
    assert db.stat().st_size == before
    assert "VACUUM しました" not in out.read_text(encoding="utf-8")


def test_vacuum_after_delete_orphans(
    dump_cache: ModuleType,
    db: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "out.txt"
    argv = (str(db), "--delete-orphans", "--yes", "--vacuum", "--out", str(out))
    assert _run_main(dump_cache, monkeypatch, *argv) == 0
    text = out.read_text(encoding="utf-8")
    assert "削除しました" in text and "VACUUM しました" in text
    assert len(NerCache(str(db)).list_documents()) == 1  # 登録済みの文書は残る


@pytest.mark.parametrize("other", [["--restore-orphans"], ["--show", "abc"]])
def test_vacuum_rejects_other_actions(
    dump_cache: ModuleType,
    db: Path,
    monkeypatch: pytest.MonkeyPatch,
    other: list[str],
) -> None:
    assert _run_main(dump_cache, monkeypatch, str(db), "--vacuum", *other) == 2
