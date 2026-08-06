"""cache.db にキャッシュされている文書を、孤児も含めて一覧するスクリプト。

なぜ必要か:
  GET /documents（UI のキャッシュ一覧も同じ）は documents テーブルを起点に引くため、
  documents に行が無い文書は出てこない。API の POST /mask は重い層（NER・LLM）の結果を
  ner / llm_detection に保存する一方で record_document を呼ばないので、/mask だけで
  処理した文書は「キャッシュは効いているのに一覧に出ない」状態になる。
  このスクリプトは 4 テーブル（documents / ner / llm_detection / mask_draft）すべてから
  content_hash を集めるので、その取りこぼしを含めた実体が見える。

孤児の本文について:
  documents が無いと chunks_json（チャンク本文）も無いが、ner の analysis_json に解析対象
  テキストが入っているので、そこから本文プレビューを復元する。ner 行も無い（LLM だけ実行した、
  あるいは mask_draft だけ残った）場合は本文を復元できない。

依存は標準ライブラリだけ（sqlite3 / json / argparse）。仮想環境もプロジェクトの import も
不要なので、実機のホストでも API コンテナの中でもそのまま実行できる。

使い方:
    # 開発機（既定で data/cache.db を読む）
    uv run python scripts/dump_cache.py

    # 孤児だけ・ファイルへ書き出す（Windows のコンソールは cp932 で化けるため）
    uv run python scripts/dump_cache.py --orphans-only --out cache_dump.txt

    # 実機の API コンテナの中で実行する（イメージに scripts/ は入っていないので標準入力で渡す）
    docker compose exec -T data-redactor-api python - /app/data/cache.db \
        < scripts/dump_cache.py

    # 機械処理向け（JSON。本文は preview のみ）
    python scripts/dump_cache.py --json

    # 1 エントリの本文を見る（content_hash は先頭一致でよい）。
    # キャッシュ 1 エントリ ＝ /mask の part 1 つなので、その part に何が入っていたかが分かる。
    uv run python scripts/dump_cache.py --show 2b5107017ae1 --out part.txt
    uv run python scripts/dump_cache.py --show 2b5107017ae1 --head 0 --out part.txt  # 全文
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# 集計対象のテーブル。content_hash 列を持つものすべて（documents に無い hash も拾うため）。
_TABLES = ("documents", "ner", "llm_detection", "mask_draft")


def _preview(text: str, width: int) -> str:
    """本文を 1 行のプレビューにする（改行・タブ・連続空白は空白 1 つに潰す）。"""
    flat = " ".join(text.split())
    return flat[:width] + ("…" if len(flat) > width else "")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """テーブルの有無を返す。古い cache.db には llm_detection 等が無いことがある。"""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _body_text(
    conn: sqlite3.Connection, content_hash: str, doc_row: tuple | None
) -> str:
    """1 文書の本文を返す。documents のチャンクが最優先で、無ければ ner の解析結果から復元する。"""
    if doc_row is not None and doc_row[6]:
        return "".join(json.loads(doc_row[6]))
    row = conn.execute(
        "SELECT analysis_json FROM ner WHERE content_hash=? ORDER BY created_at LIMIT 1",
        (content_hash,),
    ).fetchone()
    if row is None:
        return ""
    analysis = json.loads(row[0])
    return analysis.get("original_text") or analysis.get("text") or ""


def collect(conn: sqlite3.Connection, preview_width: int) -> list[dict]:
    """cache.db の全 content_hash を集め、1 文書 1 dict のリストにする（新しい順）。"""
    tables = [t for t in _TABLES if _table_exists(conn, t)]
    hashes: set[str] = set()
    for t in tables:
        hashes |= {r[0] for r in conn.execute(f"SELECT DISTINCT content_hash FROM {t}")}

    docs = {
        r[0]: r
        for r in conn.execute(
            "SELECT content_hash, source_kind, source_name, char_count, chunk_count, "
            "created_at, chunks_json FROM documents"
        )
    }

    rows: list[dict] = []
    for h in hashes:
        d = docs.get(h)
        ner = conn.execute(
            "SELECT model, analysis_json, created_at FROM ner "
            "WHERE content_hash=? ORDER BY created_at",
            (h,),
        ).fetchall()
        llm = (
            conn.execute(
                "SELECT DISTINCT detector_version, created_at FROM llm_detection "
                "WHERE content_hash=? ORDER BY created_at",
                (h,),
            ).fetchall()
            if "llm_detection" in tables
            else []
        )
        has_draft = (
            "mask_draft" in tables
            and conn.execute(
                "SELECT 1 FROM mask_draft WHERE content_hash=?", (h,)
            ).fetchone()
            is not None
        )

        text = _body_text(conn, h, d)
        created = (
            d[5]
            if d is not None
            else min([r[2] for r in ner] + [r[1] for r in llm], default="")
        )
        rows.append(
            {
                "content_hash": h,
                "registered": d is not None,
                "source_kind": d[1] if d is not None else None,
                "source_name": d[2] if d is not None else None,
                "char_count": d[3] if d is not None else len(text),
                "chunk_count": d[4] if d is not None else None,
                "ner_models": sorted({r[0] for r in ner}),
                "llm_versions": sorted({r[0] for r in llm}),
                "has_draft": has_draft,
                "created_at": created,
                "preview": _preview(text, preview_width),
            }
        )

    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return rows


def render(rows: list[dict], db_path: str) -> str:
    """人が読む一覧テキストにする。孤児（documents 未登録）は行頭に ※ を付ける。"""
    n_registered = sum(1 for r in rows if r["registered"])
    out = [
        f"cache.db: {db_path}",
        f"文書 {len(rows)} 件（documents 登録済み {n_registered} / "
        f"未登録 {len(rows) - n_registered}）",
        "",
    ]
    for r in rows:
        mark = "  " if r["registered"] else "※"
        chunks = f"/{r['chunk_count']}chunk" if r["chunk_count"] is not None else ""
        name = r["source_name"] or "(documents 未登録)"
        kind = r["source_kind"] or "-"
        out.append(f"{mark} {r['created_at'] or '(日時不明)'}  {r['content_hash']}")
        out.append(f"    名前 : {name}  [{kind}]  {r['char_count']}字{chunks}")
        out.append(f"    NER  : {', '.join(r['ner_models']) or '（なし）'}")
        out.append(f"    LLM  : {', '.join(r['llm_versions']) or '（なし）'}")
        if r["has_draft"]:
            out.append("    draft: あり（手動選択差分）")
        out.append(
            f"    本文 : {r['preview'] or '（復元不可：ner 行が無く本文が残っていない）'}"
        )
        out.append("")
    out.append(
        "※ = documents に未登録。GET /documents・UI のキャッシュ一覧には出ない。"
    )
    return "\n".join(out) + "\n"


def show(conn: sqlite3.Connection, prefix: str, head: int) -> str:
    """1 文書の本文を全文（または先頭 head 文字）表示する。content_hash は先頭一致で指定する。

    キャッシュ 1 エントリ ＝ /mask の part 1 つなので、本文を見れば「その part に何が入って
    いたか」が分かる。1 エントリの中に文書の見出しが何度も現れるなら、呼び出し側が複数の
    文書を 1 つの part に連結して送っている（＝ファイル単位のキャッシュが効かない）。
    その判定材料として、先頭行と同じ行が本文中に何回現れるかも数える。
    """
    hashes = sorted(
        {
            r[0]
            for t in _TABLES
            if _table_exists(conn, t)
            for r in conn.execute(
                f"SELECT DISTINCT content_hash FROM {t} WHERE content_hash LIKE ?",
                (prefix + "%",),
            )
        }
    )
    if not hashes:
        return f"該当する content_hash がありません: {prefix}\n"
    if len(hashes) > 1:
        listed = "\n".join(f"  {h}" for h in hashes)
        return f"content_hash の指定があいまいです（{len(hashes)} 件一致）:\n{listed}\n"

    h = hashes[0]
    doc = conn.execute(
        "SELECT content_hash, source_kind, source_name, char_count, chunk_count, "
        "created_at, chunks_json FROM documents WHERE content_hash=?",
        (h,),
    ).fetchone()
    text = _body_text(conn, h, doc)

    lines = text.splitlines()
    first = next((line for line in lines if line.strip()), "")
    repeats = sum(1 for line in lines if line == first) if first else 0

    out = [
        f"content_hash : {h}",
        f"documents    : {'登録あり（' + str(doc[2]) + '）' if doc else '未登録'}",
        f"本文         : {len(text)}字 / {len(lines)}行",
        f"先頭行       : {first}",
        f"先頭行と同じ行の出現回数: {repeats}"
        + ("（1 なら 1 文書・2 以上なら複数文書の連結を疑う）" if repeats else ""),
        "-" * 70,
    ]
    body = text if head <= 0 else text[:head]
    out.append(body)
    if head > 0 and len(text) > head:
        out.append(f"…（残り {len(text) - head} 字は省略。全文は --head 0）")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="cache.db の文書を孤児（documents 未登録）も含めて一覧する"
    )
    parser.add_argument(
        "db",
        nargs="?",
        default="data/cache.db",
        help="cache.db のパス（既定 data/cache.db）",
    )
    parser.add_argument(
        "--orphans-only",
        action="store_true",
        help="documents に未登録のものだけ表示する",
    )
    parser.add_argument("--json", action="store_true", help="JSON で出力する")
    parser.add_argument(
        "--show",
        metavar="HASH",
        help="1 文書の本文を表示する（content_hash は先頭一致でよい）。"
        "その part に何が入っていたかを確かめる用",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=2000,
        help="--show で表示する先頭文字数（既定 2000。0 で全文）",
    )
    parser.add_argument(
        "--preview", type=int, default=70, help="本文プレビューの文字数（既定 70）"
    )
    parser.add_argument(
        "--out",
        help="出力先ファイル（UTF-8）。省略すると標準出力"
        "（Windows のコンソールは cp932 で化けるのでファイル推奨）",
    )
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"cache.db が見つかりません: {db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)  # 読み取り専用で開く
    try:
        if args.show:
            text = show(conn, args.show, args.head)
        else:
            rows = collect(conn, args.preview)
            if args.orphans_only:
                rows = [r for r in rows if not r["registered"]]
            text = (
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
                if args.json
                else render(rows, str(db))
            )
    finally:
        conn.close()

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"書き出しました: {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
