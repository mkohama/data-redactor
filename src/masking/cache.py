"""解析キャッシュ（UI 非依存）。

解析は **NER 層（GiNZA 2モデル＝激重。400 チャンクで 7〜10 分）** と **マスキング層（辞書照合・確信度・
クラスタ・除外＝ミリ秒）** に分かれる。**キャッシュするのは NER 層だけ**にし、マスキング層は都度再計算する
（辞書・除外を変えても再 NER 不要。`MaskingEngine.analyze` が利用）。

- キー `(content_hash, model, flatten)` → per-model の :class:`~src.ner.engine.Analysis`。
- `content_hash` はチャンク列（解析対象テキスト）の sha256。内容が同じなら名前が違っても同一視。
- 格納は SQLite（`data/cache.db`）。件数が増えても O(1) 照合・一覧/削除が容易。

確定マスク（人手レビュー後の決定）の保存は別テーブル（将来）。本モジュールはまず NER 層を担う。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# content_hash は軽量モジュールに分離（UI が spaCy 無しで使えるように）。ここは再輸出
# （`as` は明示的再輸出＝未使用 import ではないことを示す）。
from src.masking.hashing import content_hash as content_hash
from src.ner.engine import AnalyzedToken, Analysis, Entity


def analysis_to_dict(a: Analysis) -> dict:
    """:class:`Analysis` を JSON 化可能な dict にする（NER 層キャッシュの値）。"""
    return {
        "text": a.text,
        "original_text": a.original_text,
        "offset_map": list(a.offset_map),
        "tokens": [[t.start, t.end, t.surface, t.tag, t.pos] for t in a.tokens],
        "entities": [[e.text, e.label, e.start, e.end] for e in a.entities],
    }


def analysis_from_dict(d: dict) -> Analysis:
    """:func:`analysis_to_dict` の逆。"""
    return Analysis(
        text=d["text"],
        tokens=tuple(AnalyzedToken(*t) for t in d["tokens"]),
        entities=tuple(
            Entity(text=e[0], label=e[1], start=e[2], end=e[3]) for e in d["entities"]
        ),
        original_text=d.get("original_text", ""),
        offset_map=tuple(d.get("offset_map") or []),
    )


@dataclass(frozen=True)
class DocInfo:
    """キャッシュ済み文書 1 件の表示用メタ情報（キャッシュ一覧で使う）。"""

    content_hash: str
    source_kind: str  # text / file / kb
    source_name: str
    char_count: int
    chunk_count: int
    models: tuple[str, ...]  # NER キャッシュ済みのモデル
    created_at: str
    has_llm: bool = False  # LLM 検出キャッシュ（llm_detection）が 1 件でもあるか


class NerCache:
    """NER 層（per-model Analysis）の SQLite キャッシュ。キー＝(content_hash, model, flatten)。

    あわせて文書メタ（ソース名・チャンク数等）を ``documents`` に持ち、キャッシュ一覧の参照・
    削除に使う（NER 層は content_hash でしか引けないので、人が見て分かる情報を別に記録する）。
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS ner ("
                "content_hash TEXT, model TEXT, flatten INTEGER, "
                "analysis_json TEXT, created_at TEXT, "
                "PRIMARY KEY (content_hash, model, flatten))"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS documents ("
                "content_hash TEXT PRIMARY KEY, source_kind TEXT, source_name TEXT, "
                "char_count INTEGER, chunk_count INTEGER, created_at TEXT, "
                "chunks_json TEXT)"
            )
            # 既存 DB（chunks_json 無し）への移行＝あとから列を足す。
            cols = [r[1] for r in c.execute("PRAGMA table_info(documents)").fetchall()]
            if "chunks_json" not in cols:
                c.execute("ALTER TABLE documents ADD COLUMN chunks_json TEXT")
            # 作業ドラフトの手動選択（auto からの差分）を文書単位で保存する。
            c.execute(
                "CREATE TABLE IF NOT EXISTS mask_draft ("
                "content_hash TEXT PRIMARY KEY, "
                "added_json TEXT, removed_json TEXT, updated_at TEXT)"
            )
            # LLM 検出層（Stage A）のキャッシュ。NER 層と同じ「激重層だけキャッシュ」の思想。
            #   鍵に detector_version（pii-masker 版＋プロンプト＋窓ポリシー）を含める＝改版で自動ミス→再取得。
            #   値は LlmDetection の JSON 文字列（(de)シリアライズは src.llm.schema が持つ＝cache は中身に非依存）。
            c.execute(
                "CREATE TABLE IF NOT EXISTS llm_detection ("
                "content_hash TEXT, model TEXT, flatten INTEGER, detector_version TEXT, "
                "detections_json TEXT, created_at TEXT, "
                "PRIMARY KEY (content_hash, model, flatten, detector_version))"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def record_document(
        self,
        content_hash: str,
        source_kind: str,
        source_name: str,
        chunks: list[str],
    ) -> None:
        """文書メタ＋チャンク本文を記録。同じ content_hash は上書き。

        チャンク本文も保存するのは「キャッシュを入力元に選ぶ」ため（保存チャンクで再解析すると
        content_hash が一致し NER キャッシュにヒット＝一瞬）。
        """
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO documents "
                "(content_hash, source_kind, source_name, char_count, chunk_count, "
                "created_at, chunks_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    content_hash,
                    source_kind,
                    source_name,
                    sum(len(c2) for c2 in chunks),
                    len(chunks),
                    datetime.now().isoformat(timespec="seconds"),
                    json.dumps(list(chunks), ensure_ascii=False),
                ),
            )

    def save_draft(
        self,
        content_hash: str,
        added: set[tuple[int, int]],
        removed: set[tuple[int, int]],
    ) -> None:
        """作業ドラフトの手動選択（auto からの差分）を保存。同じ content_hash は上書き。"""
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO mask_draft "
                "(content_hash, added_json, removed_json, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    content_hash,
                    json.dumps([list(s) for s in added]),
                    json.dumps([list(s) for s in removed]),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def get_draft(
        self, content_hash: str
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
        """保存済みの手動選択差分 (added, removed) を返す。無ければ None。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT added_json, removed_json FROM mask_draft WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        if row is None:
            return None
        added = {(s[0], s[1]) for s in json.loads(row[0] or "[]")}
        removed = {(s[0], s[1]) for s in json.loads(row[1] or "[]")}
        return added, removed

    def get_source(self, content_hash: str) -> tuple[str, str] | None:
        """記録済み文書の ``(source_kind, source_name)`` を返す（無ければ None）。

        キャッシュ入力での再解析時に**元の種別を保つ**ために使う（"cache" で上書きしない）。
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT source_kind, source_name FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return (row[0], row[1]) if row else None

    def set_source_kind(self, content_hash: str, source_kind: str) -> None:
        """文書の種別（出所）を更新する（🗂 一覧での手動修正用）。

        キャッシュ入力での再解析で "cache" に潰れてしまった行を、元の出所（file/kb/text）へ
        人が直すために使う。
        """
        with self._conn() as c:
            c.execute(
                "UPDATE documents SET source_kind = ? WHERE content_hash = ?",
                (source_kind, content_hash),
            )

    def get_chunks(self, content_hash: str) -> list[str] | None:
        """記録済みのチャンク本文を返す（キャッシュを入力元に使う）。無ければ None。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT chunks_json FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def cached_ner_models(self, content_hash: str, flatten: bool) -> set[str]:
        """指定 (content_hash, flatten) で NER キャッシュ済みのモデル名集合。状態表示用。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT model FROM ner WHERE content_hash=? AND flatten=?",
                (content_hash, int(flatten)),
            ).fetchall()
        return {r[0] for r in rows}

    def has_llm(
        self, content_hash: str, model: str, flatten: bool, detector_version: str
    ) -> bool:
        """LLM 検出キャッシュが存在するか（軽い存在チェック。値はデシリアライズしない）。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM llm_detection "
                "WHERE content_hash=? AND model=? AND flatten=? AND detector_version=?",
                (content_hash, model, int(flatten), detector_version),
            ).fetchone()
        return row is not None

    def llm_versions(self, content_hash: str) -> set[str]:
        """この文書に保存済みの LLM 検出 ``detector_version`` 集合（flatten/model 問わず）。

        「キャッシュはあるが現行の detector_version とは違う＝要更新」を一覧で示すための材料。
        現行版が含まれていれば有効、含まれていなければ旧版のみ（再検出で更新）と判断できる。
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT detector_version FROM llm_detection WHERE content_hash=?",
                (content_hash,),
            ).fetchall()
        return {r[0] for r in rows}

    def list_documents(self) -> list[DocInfo]:
        """キャッシュ済み文書の一覧（新しい順）。NER 済みモデルと LLM 済み有無も付ける。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT d.content_hash, d.source_kind, d.source_name, d.char_count, "
                "d.chunk_count, d.created_at, "
                "(SELECT GROUP_CONCAT(DISTINCT model) FROM ner n "
                " WHERE n.content_hash = d.content_hash), "
                "EXISTS(SELECT 1 FROM llm_detection l "
                " WHERE l.content_hash = d.content_hash) "
                "FROM documents d ORDER BY d.created_at DESC"
            ).fetchall()
        return [
            DocInfo(
                content_hash=r[0],
                source_kind=r[1],
                source_name=r[2],
                char_count=r[3],
                chunk_count=r[4],
                created_at=r[5],
                models=tuple((r[6] or "").split(",")) if r[6] else (),
                has_llm=bool(r[7]),
            )
            for r in rows
        ]

    def delete(self, content_hash: str) -> None:
        """1 文書のキャッシュを全層まとめて削除する（文書メタ＋NER＋LLM＋手動選択差分）。

        キャッシュは content_hash をキーに 4 テーブル（documents / ner / llm_detection /
        mask_draft）へ分かれている。全層を消さないと、同一内容を再取込したとき（content_hash は
        内容から決まるので同じ値になる）に旧 LLM 検出や旧ドラフトが孤児として蘇り、
        「削除したのにキャッシュ済み扱い」になる。
        """
        with self._conn() as c:
            c.execute("DELETE FROM ner WHERE content_hash = ?", (content_hash,))
            c.execute(
                "DELETE FROM llm_detection WHERE content_hash = ?", (content_hash,)
            )
            c.execute("DELETE FROM mask_draft WHERE content_hash = ?", (content_hash,))
            c.execute("DELETE FROM documents WHERE content_hash = ?", (content_hash,))

    def delete_ner(self, content_hash: str) -> None:
        """**NER 層だけ**破棄する（文書メタ・チャンク本文は残す）。

        前処理やモデルの改善を反映したいときに使う。次回 :meth:`get` がミスして再解析され、
        新しい NER 結果が :meth:`put` で入る。文書メタを残すので一覧・キャッシュ選択は維持される。
        """
        with self._conn() as c:
            c.execute("DELETE FROM ner WHERE content_hash = ?", (content_hash,))

    def get(self, content_hash: str, model: str, flatten: bool) -> Analysis | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT analysis_json FROM ner "
                "WHERE content_hash=? AND model=? AND flatten=?",
                (content_hash, model, int(flatten)),
            ).fetchone()
        return analysis_from_dict(json.loads(row[0])) if row else None

    def put(
        self, content_hash: str, model: str, flatten: bool, analysis: Analysis
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO ner VALUES (?, ?, ?, ?, ?)",
                (
                    content_hash,
                    model,
                    int(flatten),
                    json.dumps(analysis_to_dict(analysis), ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    # --- LLM 検出層（Stage A）。値は LlmDetection の JSON 文字列（中身は src.llm.schema が定義）。
    #     cache は「激重層の成果を content_hash で引く」storage に徹し、LlmDetection の構造には依存しない。
    def get_llm(
        self, content_hash: str, model: str, flatten: bool, detector_version: str
    ) -> str | None:
        """LLM 検出のキャッシュ（detections_json）を返す。無ければ None。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT detections_json FROM llm_detection "
                "WHERE content_hash=? AND model=? AND flatten=? AND detector_version=?",
                (content_hash, model, int(flatten), detector_version),
            ).fetchone()
        return row[0] if row else None

    def put_llm(
        self,
        content_hash: str,
        model: str,
        flatten: bool,
        detector_version: str,
        detections_json: str,
    ) -> None:
        """LLM 検出（detections_json）を保存。同じ鍵は上書き。"""
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO llm_detection VALUES (?, ?, ?, ?, ?, ?)",
                (
                    content_hash,
                    model,
                    int(flatten),
                    detector_version,
                    detections_json,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
