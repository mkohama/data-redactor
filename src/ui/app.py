"""機密情報マスキング Streamlit UI（薄い表示層・純クライアント）。

検出・マスク・キャッシュ（GiNZA / cache.db）はすべてサーバ（data-redactor serve）が所有し、
本ファイルは src.client.MaskClient 越しに HTTP で問い合わせる（設計 B）。入力 UI・表示のみを行う。

起動:
    uv run data-redactor ui   （＝ streamlit run src/ui/app.py）
"""

from __future__ import annotations

import html
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from src.client import MaskApiError, MaskClient
from src.core.document.document_loader import DocumentLoader
from src.masking import (
    NerCache,
    allowlist_sort_key,
    content_hash,
    dict_sort_key,
)
from src.detector import detector_version as _detector_version
from src.llm.detect_layer import DEFAULT_MODEL as LLM_MODEL
from src.ner import (
    AVAILABLE_MODELS,
    DEFAULT_COLOR,
    MASKING_CATEGORY_COLORS,
    render_masking_html,
)
from src.sources import SAMPLE_TEXT
from src.sources.kb_mcp import (
    DEFAULT_KB_MCP_URL,
    download_document_sync,
    list_documents_sync,
    suppress_async_generator_errors,
)

suppress_async_generator_errors()

# アップロード可能な拡張子 (DocumentLoader が対応する形式)
SUPPORTED_EXTENSIONS = sorted(e[1:] for e in DocumentLoader.SUPPORTED_EXTENSIONS)

# モデル選択肢（AVAILABLE_MODELS と同順）と説明
MODELS = list(AVAILABLE_MODELS)
MODEL_DESCRIPTIONS = {
    "ja_ginza_electra": "高精度・低速 (ELECTRA / Transformer ベース)",
    "ja_ginza": "軽量・高速 (CNN/Sudachi ベース)",
}


# マスキング API（サーバ）の接続先。UI は純クライアント（設計 B）＝エンジンを内包せず、
# この URL のサーバへ HTTP で問い合わせる。ローカルは data-redactor serve（既定 localhost:8509。
# kb-mcp の既定 8000 と衝突させないため 8509）。Docker は MASK_API_URL=http://api:8509 を渡す。
MASK_API_URL = os.environ.get("MASK_API_URL", "http://127.0.0.1:8509")


@st.cache_resource(show_spinner=False)
def _mask_client(base_url: str) -> MaskClient:
    """マスキング API のクライアントを生成・共有する。

    内部で httpx 接続を 1 本持つので、プロセス内で使い回す（cache_resource）。
    base_url を引数（キャッシュ鍵）にしているので、接続先を変えれば作り直される。
    """
    return MaskClient(base_url)


def _render_connection_status() -> bool:
    """サイドバー上部に API サーバの接続状態を表示し、接続できているかを返す。

    ✅ 接続 OK（NER モデルのロード状態も併記）／❌ 未接続（起動方法を明示）。
    未接続でも UI は落とさず、各操作側でエラーを握ってメッセージを出す前提。
    """
    client = _mask_client(MASK_API_URL)
    try:
        health = client.health()
    except (MaskApiError, httpx.HTTPError) as e:
        st.error(
            f"マスキング API に接続できません（{MASK_API_URL}）。\n\n"
            "別ターミナルで `data-redactor serve` を起動するか、環境変数 "
            "`MASK_API_URL` で接続先を指定してください。"
        )
        st.caption(f"詳細: {e}")
        return False
    if health.get("models_ready"):
        loaded = ", ".join(_short_models(tuple(health.get("models_loaded", []))))
        st.success(f"✅ API 接続 OK（{MASK_API_URL}）")
        st.caption(f"モデル: {loaded or '—'}")
    else:
        st.warning(
            f"⚠ API に接続できましたが、NER モデルがまだロード中です（{MASK_API_URL}）。"
            "しばらく待つか、サーバのログを確認してください。"
        )
    return True


# マスク辞書・除外リスト・キャッシュの既定パス（リポジトリルート直下 data/）。
#   このファイルは src/ui/app.py なので、ルートは親を 3 つ遡る（ui → src → root）。
#   ※ M5e-2 で MaskClient 経由になり、UI が data/ を直接持たなくなったらこれらは消える。
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DICT = _ROOT / "data" / "mask_dict.yaml"
_DEFAULT_ALLOWLIST = _ROOT / "data" / "mask_allowlist.yaml"
_DEFAULT_CACHE_DB = _ROOT / "data" / "cache.db"


@st.cache_resource(show_spinner=False)
def _ner_cache() -> NerCache:
    """NER 層キャッシュ（解析過程の高速化）。SQLite 接続は軽いがインスタンスは共有する。"""
    return NerCache(_DEFAULT_CACHE_DB)


def _short_models(models: tuple[str, ...]) -> str:
    """NER モデル名を短い別名に（一覧表示用）。"""
    names = {"ja_ginza_electra": "electra", "ja_ginza": "ginza"}
    return ", ".join(names.get(m, m) for m in models)


def _kb_doc_label(meta: dict) -> str:
    """kb-mcp 文書メタから表示名を作る。"""
    name = meta.get("title") or meta.get("file_name") or meta.get("id") or "?"
    path = meta.get("file_path") or ""
    return f"{name}　({path})" if path else str(name)


@st.fragment
def _select_table_fragment(
    df: pd.DataFrame,
    ids: list,
    *,
    key: str,
    sel_key: str,
    caption: str,
    column_config: dict | None = None,
) -> None:
    """1 行クリックで単一選択するテーブル（``st.fragment`` で再描画を局所化）。

    行をクリックすると **このテーブルだけ** 再実行され（画面全体は再描画しない＝チラつき/
    スクロール飛びを抑える）。選んだ行の識別子 ``ids[row]`` を ``st.session_state[sel_key]`` に
    書く（未選択は ``None``）。単一選択なので前の選択は自動で置き換わる（チェック残り無し）。
    位置選択だが、呼び出し側が安定順で渡し ``ids`` で解決するので並び替えの取り違えは起きない。
    列見出しクリックで表示の並べ替えも可能（選択には影響しない）。
    """
    st.caption(caption)
    event = st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config=column_config,
        key=key,
    )
    rows = event.selection.rows
    st.session_state[sel_key] = ids[rows[0]] if rows else None


def _llm_cache_status(content_hash: str, flatten: bool, has_llm: bool) -> str:
    """🗂 ピッカー用の LLM キャッシュ状態。``✓``=現行設定で有効 / ``⚠ 要更新``=旧版のみ / ``—``=無し。

    現在の ``(LLM_MODEL, flatten, _detector_version())`` に一致する行があれば「✓」（そのまま使える）。
    LLM 検出履歴はあるが現行設定に一致しなければ「⚠ 要更新」（窓ポリシー等が変わった＝再検出が要る）。
    実際のキャッシュヒット条件（has_llm）と同じ鍵で判定するので、表示と挙動がズレない。
    """
    if not has_llm:
        return "—"
    if _ner_cache().has_llm(content_hash, LLM_MODEL, flatten, _detector_version()):
        return "✓"
    return "⚠ 要更新"


@st.fragment
def _cache_picker_fragment(docs: list, flatten: bool) -> None:
    """🗂 キャッシュ選択の UI 一式（テーブル＋選択依存の操作）を 1 つの fragment にまとめる。

    行クリックは **この fragment だけ** 再実行され、画面全体は再描画しない。
    選択した content_hash を ``st.session_state["cache_sel"]`` に入れる（``render_input`` が読む）。
    ``docs`` は呼び出し側がソース名で安定ソート済み（並び替えで取り違えない）。``flatten`` は現在の平文化
    設定で、LLM 列の「現行設定で有効か（✓）/要更新（⚠）」判定に使う。
    NER のやり直し（キャッシュ無視）は読み込み時でなく **🔍 NER検出 タブ**で行う（パイプライン化に伴う移設）。
    """
    df = pd.DataFrame(
        [
            {
                "ソース": d.source_name,
                "種別": d.source_kind,
                "チャンク": d.chunk_count,
                "文字数": d.char_count,
                "NER": _short_models(d.models) or "—",
                "LLM": _llm_cache_status(d.content_hash, flatten, d.has_llm),
                "解析日時": d.created_at,
            }
            for d in docs
        ]
    )
    st.caption(
        f"キャッシュ済み: {len(docs)} 文書（行をクリックして選択）。"
        "LLM 列: **✓**=現在の設定で有効／**⚠ 要更新**=キャッシュはあるが窓ポリシー等が変わり再検出が必要／**—**=無し。"
    )
    event = st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="cache_pick",
    )
    rows = event.selection.rows
    sel_hash = docs[rows[0]].content_hash if rows else None
    st.session_state["cache_sel"] = sel_hash
    if sel_hash is None:
        st.caption("👆 行をクリックして文書を選択してください。")
        return
    if not _ner_cache().get_chunks(sel_hash):
        st.warning(
            "このキャッシュにはチャンク本文がありません（チャンク保存より前の古いエントリ）。"
            "一度ふつうに解析し直すと、以降ここから選べます。"
        )
        return
    st.caption(
        "保存チャンクを入力に使います。[📥 読み込む] 後、各タブ（NER検出 / LLM検出 / マージ&確信度）で実行します。"
        "NER のやり直し（キャッシュ無視）は NER検出 タブのチェックで行えます。"
    )


def render_input(
    input_mode: str,
    flatten_tables: bool,
) -> tuple[tuple | None, str, str, Callable[[], list[str]] | None]:
    """入力ウィジェットを描画し、解析に必要な情報を返す。

    ``flatten_tables`` は現在の平文化設定で、🗂 キャッシュ選択の LLM 列の有効/要更新判定に渡す。

    重い取込（サーバへの送信・kb からのダウンロード・テキスト化）は**ここでは行わず**、
    ``get_chunks`` 呼び出しに遅延させる（実際に走るのは「読み込む」ボタンが押されたときだけ）。

    text/file/kb は ``get_chunks`` の中で ``MaskClient.ingest_document`` を呼んでサーバへ取り込み
    （テキスト化・チャンク化の所有者はサーバ＝設計 B）、返るチャンクを解析に使う。kb は kb-mcp
    から元ファイルを取得して file として取り込む（cache＝キャッシュ選択は取込済みチャンクを返す）。

    戻り値 ``(input_id, input_kind, source_label, get_chunks)``：
      - ``input_id``  … 入力の同一性を表すハッシュ可能なタプル（署名に使う）。
                        入力未確定なら ``None``（解析不可）。
      - ``input_kind``… ``"text" / "file" / "kb" / "cache"``（平文プレビューの要否判定に使う）。
      - ``source_label``… 結果見出しに出す表示名。
      - ``get_chunks``… 呼ぶとチャンク列を返す callable（未確定なら ``None``）。
    """
    if input_mode.startswith("📄"):
        uploaded_file = st.file_uploader(
            f"対応形式: {', '.join(SUPPORTED_EXTENSIONS)}",
            type=SUPPORTED_EXTENSIONS,
        )
        if uploaded_file is not None:
            input_id = ("file", uploaded_file.name, uploaded_file.size)

            def get_file_chunks(f=uploaded_file) -> list[str]:
                # ファイル本体をサーバへ送り、サーバが DocumentLoader で抽出・チャンク化する
                # （テキスト化・チャンク化の所有者はサーバ＝設計 B）。返るチャンクを解析に使う。
                with st.spinner(
                    "ファイルをサーバへ送信してテキスト化・チャンク化中 ..."
                ):
                    client = _mask_client(MASK_API_URL)
                    info = client.ingest_document(file=(f.name, f.getvalue()))
                    return client.get_document(info["content_hash"])["chunks"]

            return input_id, "file", uploaded_file.name, get_file_chunks
        return None, "file", "", None

    if input_mode.startswith("📚"):
        url = st.text_input(
            "kb-mcp サーバー URL",
            value=st.session_state.get("kb_url", DEFAULT_KB_MCP_URL),
        )
        st.session_state["kb_url"] = url
        if st.button("文書リストを取得"):
            try:
                with st.spinner("文書一覧を取得中 ..."):
                    st.session_state["kb_docs"] = list_documents_sync(url)
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("kb_docs", None)
                st.error(f"kb-mcp への接続/取得に失敗しました: {e}")

        docs = st.session_state.get("kb_docs")
        if docs is None:
            st.info(
                "kb-mcp サーバを起動し（`uv run kb-mcp-server --transport http --port 8000`）、"
                "[文書リストを取得] を押してください。"
            )
            return None, "kb", "", None
        if not docs:
            st.warning("kb-mcp に登録された文書がありません。")
            return None, "kb", "", None

        # 取込済み（＝サーバのキャッシュに file として取り込んだ kb 文書）に「📦」を付ける目安。
        # kb 文書は元ファイルを取得して file 取込するので source_kind は "file"。取込時に
        # source_name へ kb 文書ラベル（_kb_doc_label）を刻むので、それと突き合わせる（案A）。
        # API 未接続なら印だけ出さない（一覧自体は kb-mcp 直取得なので表示できる）。
        try:
            imported = {
                d["source_name"]
                for d in _mask_client(MASK_API_URL).list_documents()
                if d["source_kind"] == "file"
            }
        except (MaskApiError, httpx.HTTPError):
            imported = set()

        # 1 行クリックで選ぶテーブル（fragment で再描画局所化）。📦＝取込済みの目安。
        # チャンク数は kb-mcp の一覧メタ（knowledge://documents の chunk_count）をそのまま表示。
        kb_df = pd.DataFrame(
            [
                {
                    "📦": "✓" if _kb_doc_label(m) in imported else "",
                    "名前": (
                        m.get("title") or m.get("file_name") or m.get("id") or "?"
                    ),
                    "チャンク": m.get("chunk_count", ""),
                    "パス": m.get("file_path") or "",
                }
                for m in docs
            ]
        )
        _select_table_fragment(
            kb_df,
            list(range(len(docs))),
            key="kb_pick",
            sel_key="kb_sel",
            caption=(
                f"kb-mcp 文書: {len(docs)} 件"
                "（行をクリックして選択。📦＝取込済みの目安・チャンク数は kb-mcp の登録値）"
            ),
            column_config={"📦": st.column_config.TextColumn("📦", width="small")},
        )
        sel = st.session_state.get("kb_sel")
        if sel is None or sel >= len(docs):
            return None, "kb", "", None
        meta = docs[sel]
        doc_id = meta.get("id") or meta.get("document_id")
        if not doc_id:
            return None, "kb", "", None

        label = _kb_doc_label(meta)

        def get_kb_chunks(u=url, d=doc_id, name=label) -> list[str]:
            # kb-mcp から「元ファイル」を取得し、それをサーバへ送って file として取り込む
            # （格納チャンクは overlap 付き＝使わない。サーバが overlap 0 で再抽出）。
            # source_name に kb 文書ラベルを刻んで一覧の「📦 取込済み」判定に使う（案A）。
            with st.spinner("kb-mcp から元ファイルを取得してサーバへ送信中 ..."):
                filename, blob = download_document_sync(d, u)
                client = _mask_client(MASK_API_URL)
                info = client.ingest_document(file=(filename, blob), source_name=name)
                return client.get_document(info["content_hash"])["chunks"]

        return ("kb", url, doc_id), "kb", label, get_kb_chunks

    if input_mode.startswith("🗂"):
        docs = _ner_cache().list_documents()
        if not docs:
            st.info(
                "キャッシュがありません。テキスト/ファイル/kb-mcp を解析すると登録され、"
                "ここから入力元に選べるようになります。"
            )
            return None, "cache", "", None
        # 選択 UI 一式（テーブル＋強制再解析チェック＋案内）は 1 つの fragment 内で描く。
        # **ソース名で安定ソート**して渡すので、解析（created_at 更新）で行が動かず選択がズレない。
        docs = sorted(docs, key=lambda doc: doc.source_name)
        _cache_picker_fragment(docs, flatten_tables)

        # fragment が session_state に書いた選択・強制再解析を読んで get_chunks を組む（ここでは
        # 選択依存のウィジェットを描かない＝行クリックで再描画されない外側に widget を置かない）。
        sel_hash = st.session_state.get("cache_sel")
        matches = [x for x in docs if x.content_hash == sel_hash]
        if not matches:
            return None, "cache", "", None
        d = matches[0]
        cached_chunks = _ner_cache().get_chunks(d.content_hash)
        if not cached_chunks:  # 警告は fragment 内で表示済み（古いエントリ）
            return None, "cache", d.source_name, None

        # 読み込みは保存チャンクを返すだけ（NER は回さない）。NER のやり直し（キャッシュ無視）は
        # NER検出 タブのチェックで delete_ner する（パイプライン化に伴い読み込み時の強制再解析は廃止）。
        def get_cache_chunks(c=cached_chunks) -> list[str]:
            return c

        return (("cache", d.content_hash), "cache", d.source_name, get_cache_chunks)

    # テキスト入力（単一チャンクとして扱う。長文でもエンジン側で安全分割される）
    input_text = st.text_area("解析するテキスト", value=SAMPLE_TEXT, height=200)
    if input_text.strip():

        def get_text_chunks(t=input_text) -> list[str]:
            # サーバへ取り込んで cache.db に登録する（一覧・再利用の対象になる）。
            client = _mask_client(MASK_API_URL)
            info = client.ingest_document(text=t, source_name="入力テキスト")
            return client.get_document(info["content_hash"])["chunks"]

        return ("text", input_text), "text", "入力テキスト", get_text_chunks
    return None, "text", "", None


def _dict_signature(dict_path: str) -> tuple[str, float | None]:
    """辞書ファイルの同一性（パス＋更新時刻）。保存で内容が変われば署名がズレる。"""
    p = Path(dict_path)
    try:
        return (str(p), p.stat().st_mtime) if p.exists() else (str(p), None)
    except OSError:
        return (str(p), None)


def _masking_settings_sig(
    models: list[str], flatten_tables: bool, dict_path: str, allowlist_path: str
) -> tuple:
    """マスキングの設定署名（モデル/平文化/辞書 mtime/除外リスト mtime）。

    再解析バナーの判定に使う。除外を「再解析なし」で反映したときに、この署名で stored を
    更新しておけばバナーが誤って出ない（main と同じ式を使うため共通化）。
    """
    return (
        "masking",
        tuple(models),
        flatten_tables,
        _dict_signature(dict_path),
        _dict_signature(allowlist_path),
    )


def _readable_text_block(
    text: str, *, placeholders: dict[str, str] | None = None, height: int = 400
) -> str:
    """読み取り専用テキストを、グレーアウトしない読める div にする HTML を返す。

    `st.text_area(disabled=True)` は背景・文字ともグレーで編集不可カーソルになり読みにくい。
    代わりに通常色・選択可・改行保持の div で表示する。``placeholders``（プレースホルダ→
    カテゴリ）を渡すと、マスク後の伏せ字をカテゴリ色で強調し「どこが変わったか」を見せる。
    """
    escaped = html.escape(text)
    if placeholders:
        pattern = re.compile(
            "|".join(re.escape(p) for p in sorted(placeholders, key=len, reverse=True))
        )

        def _repl(mo: re.Match) -> str:
            ph = mo.group(0)
            color = MASKING_CATEGORY_COLORS.get(placeholders.get(ph, ""), DEFAULT_COLOR)
            return (
                f'<mark style="background:{color}; color:#000; '
                f'padding:0 .15em; border-radius:3px;">{ph}</mark>'
            )

        escaped = pattern.sub(_repl, escaped)
    return (
        f'<div style="height:{height}px; overflow:auto; resize:vertical; '
        "white-space:pre-wrap; word-break:break-word; line-height:1.9; "
        'border:1px solid rgba(128,128,128,0.25); border-radius:6px; padding:0.6em;">'
        f"{escaped}</div>"
    )


def _render_extracted_text(chunks: list[str]) -> None:
    """テキスト化された平文（チャンク連結）を確認用に表示する。

    ファイルや kb-mcp は元がバイナリ/外部なので、何が抽出されたかを目視・ダウンロードできるようにする。
    チャンク境界は ``--- チャンク境界 ---`` で示す（解析単位の確認用）。
    """
    text = "\n\n".join(chunks)
    boundary = "\n\n----- チャンク境界 -----\n\n"
    shown = boundary.join(chunks)
    with st.expander(
        f"📄 テキスト化結果（平文 / {len(chunks)} チャンク・{len(text)} 文字）を確認",
        expanded=False,
    ):
        st.html(_readable_text_block(shown, height=300))
        st.download_button(
            "⬇ 平文をダウンロード",
            text,
            file_name="extracted.txt",
            mime="text/plain",
        )


def _context(text: str, start: int, end: int, width: int = 20) -> str:
    """出現箇所の前後文脈スニペット（対象を《》で囲む）。"""
    s = max(0, start - width)
    e = min(len(text), end + width)
    head = "…" if s > 0 else ""
    tail = "…" if e < len(text) else ""
    return f"{head}{text[s:start]}《{text[start:end]}》{text[end:e]}{tail}"


# 確信度の並び順（確定→強→中→弱→微弱→除外）。文字列順だと崩れるので明示する。
_CONFIDENCE_ORDER = {"確定": 0, "強": 1, "中": 2, "弱": 3, "微弱": 4, "除外": 5}
# 確信度フィルタの選択肢と既定（微弱＝コードらしき誤検出・除外＝allowlist は既定で非表示）。
_CONFIDENCE_LEVELS = ["確定", "強", "中", "弱", "微弱", "除外"]
_CONFIDENCE_DEFAULT = ["確定", "強", "中", "弱"]


def _confidence_label(confidence: str) -> str:
    """並び順の番号を前置した表示用ラベル（例 '1 : 確定'）。

    列ヘッダで文字列ソートしても 確定→強→中→弱 の順になるようにする
    （番号なしだと文字コード順で「中→確定」になってしまう）。1=確定 … 4=弱。
    """
    return f"{_CONFIDENCE_ORDER.get(confidence, 9) + 1} : {confidence}"


# API の確信度は wire（ASCII）。UI は日本語表示なのでここで対応づける（設計 §1-A）。
_WIRE_TO_JP = {
    "certain": "確定",
    "strong": "強",
    "medium": "中",
    "weak": "弱",
    "faint": "微弱",
    "excluded": "除外",
}


def _conf_jp(wire: str) -> str:
    """確信度の wire（ASCII）→ 日本語表示。未知値はそのまま返す。"""
    return _WIRE_TO_JP.get(wire, wire)


def _group_spans(group: dict) -> set[tuple[int, int]]:
    """API の実体グループ（dict）の全出現 span を集合で返す。"""
    return {tuple(o["span"]) for o in group["occurrences"]}


def _sorted_by_confidence(items, *, key, mask_rank=None, conf_key=None):
    """（あれば）マスク状態 → 確信度 降順（確定→強→中→弱）→ 第2キー（表層）昇順で並べる。

    ``conf_key`` は ``it -> 確信度（日本語）`` を返す関数（既定は属性 ``.confidence``）。
    API の dict を並べるときは ``conf_key=lambda g: _conf_jp(g["confidence"])`` を渡す。
    ``mask_rank`` は ``it -> int``（小さいほど上）。指定すると**マスク中の行が先頭**に来る。
    """
    if conf_key is None:

        def conf_key(it: object) -> str:
            return it.confidence  # type: ignore[attr-defined]

    return sorted(
        items,
        key=lambda it: (
            mask_rank(it) if mask_rank is not None else 0,
            _CONFIDENCE_ORDER.get(conf_key(it), 9),
            key(it),
        ),
    )


def _auto_mask_spans(analysis_json: dict) -> set[tuple[int, int]]:
    """既定でマスクする出現の span 集合（共有選択 ``mask_sel`` の初期値）。

    API `/analyze` の ``auto_selection``（mask_level に基づく実体単位の既定選択・案2）をそのまま
    集合にする。ある表層に 確定/強 の出現が 1 つでもあればその表層の全出現が入る（サーバ側で計算済み）。
    """
    return {tuple(s) for s in analysis_json["auto_selection"]}


def _doc_status(client: MaskClient, content_hash_: str) -> dict:
    """サーバの文書メタ（ner_models / llm_versions 等）を取得する。未接続・未取込は空 dict。

    NER/LLM タブと状態ヘッダの「キャッシュ済みか」の判定に使う（設計 B：状態はサーバの
    get_document 由来にし、UI はローカルの cache.db を直接見ない）。取得失敗は空 dict＝
    「キャッシュ無し」扱いにして UI を落とさない（各操作側でエラー表示する前提）。
    """
    try:
        return client.get_document(content_hash_)
    except (MaskApiError, httpx.HTTPError):
        return {}


def _status_has_llm(status: dict) -> bool:
    """get_document の結果に現行版の LLM 検出キャッシュがあるか。

    llm_versions（キャッシュ済み detector_version の一覧）に現行 detector_version を含む項目が
    あれば True。空・未接続などは False（＝LLM を強制しない＝NER のみ集約）。
    """
    dv = _detector_version()
    return any(dv in v for v in status.get("llm_versions", []))


def _llm_available(client: MaskClient, content_hash_: str, flatten: bool) -> bool:
    """この文書に現行版の LLM 検出キャッシュがサーバにあるか（マージの detection 自動判定用）。"""
    return _status_has_llm(_doc_status(client, content_hash_))


def _render_by_entity(groups, confidences, sel, ver, stored):
    """実体ごと：同じ語は文書内の全出現を一括マスク。``confidences`` で表示する確信度を絞る。

    ``groups`` は API `/analyze` の実体グループ（dict）のリスト。
    """

    def _group_mask_rank(g: dict) -> int:
        spans = _group_spans(g)
        if spans <= sel:  # 全出現が選択＝マスク（チェック ON）
            return 0
        return 1 if spans & sel else 2  # 一部選択 / 未選択

    all_groups = _sorted_by_confidence(
        groups,
        key=lambda g: g["surface"],
        mask_rank=_group_mask_rank,
        conf_key=lambda g: _conf_jp(g["confidence"]),
    )
    shown = [g for g in all_groups if _conf_jp(g["confidence"]) in confidences]
    hidden = len(all_groups) - len(shown)
    if not shown:
        # 空だと DataFrame に列が無く data_editor 後の列参照で落ちるので早期に案内して返す。
        st.subheader("マスク候補（0 実体）")
        if all_groups:
            st.info(
                f"確信度フィルタで全 {len(all_groups)} 実体を非表示中です。"
                "『表示する確信度』を広げてください。"
            )
        else:
            st.info(
                "マスク候補が見つかりませんでした（この文書＋現在の検出設定では 0 件）。"
            )
        return [], False, [], False
    st.subheader(f"マスク候補（{len(shown)} 実体）— チェックで選択")
    cap = "チェックは**全出現が選択されているときだけ ON**。チェックした実体は文書内の全出現がマスクされます。"
    if hidden:
        cap += f"（確信度フィルタで {hidden} 実体を非表示）"
    cap += "　※`選択状況` が **⚠一部** の語は、出現ごとビューで一部だけ選択中です。"
    st.caption(cap)
    rows = []
    for g in shown:
        spans = _group_spans(g)
        count = g["count"]
        n_sel = sum(1 for s in spans if s in sel)
        if n_sel == 0:
            status = ""
        elif n_sel < count:
            status = f"⚠ 一部 {n_sel}/{count}"
        else:
            status = f"全 {count}"
        votes = g["votes"]
        rows.append(
            {
                "マスク": n_sel == count,  # 全出現が選択済みのときだけ ON
                "選択状況": status,
                "除外": _conf_jp(g["confidence"]) == "除外",
                "辞書登録": False,
                "確信度": _confidence_label(_conf_jp(g["confidence"])),
                "カテゴリ": g["category"],
                "表層": g["surface"],
                "出現": count,
                "ja_ginza": votes.get("ja_ginza", ""),
                "electra": votes.get("ja_ginza_electra", ""),
                "Sudachi": votes.get("sudachi", ""),
                "LLM": votes.get("llm", ""),
                "辞書": "○" if votes.get("dict") else "",
            }
        )
    table = pd.DataFrame(rows)
    st.caption("チェックしてから **[✅ マスクを反映]** を押すと反映されます。")
    # st.form で囲む：チェックのたびに再実行せず、ボタン押下時だけまとめて適用する
    # （data_editor は編集ごとに rerun し画面が先頭へ飛ぶため。フォームで抑止）。
    # data_editor の鍵に ver を含める＝共有選択 sel が変わったら貼り直し、古い編集状態を残さない。
    with st.form("mask_entity_form"):
        edited = st.data_editor(
            table,
            hide_index=True,
            width="stretch",
            disabled=[
                c for c in table.columns if c not in ("マスク", "除外", "辞書登録")
            ],
            column_config={
                "マスク": st.column_config.CheckboxColumn("マスク"),
                "除外": st.column_config.CheckboxColumn(
                    "除外",
                    help="チェックして [🚫 選択を除外リストへ] を押すと候補外に。",
                ),
                "辞書登録": st.column_config.CheckboxColumn(
                    "辞書登録",
                    help="チェックして [📒 選択を辞書へ登録] を押すと、その語のカテゴリ"
                    "（社名/商標/人名）で辞書に登録＝以後どの文書でも確定マスクになります。",
                ),
            },
            key=f"mask_entity_{ver}",
        )
        col_a, col_b, col_c = st.columns([1, 1, 1])
        applied = col_a.form_submit_button("✅ マスクを反映", type="primary")
        excl = col_b.form_submit_button("🚫 選択を除外リストへ")
        reg = col_c.form_submit_button("📒 選択を辞書へ登録")
    masks = edited["マスク"].tolist()
    excludes = edited["除外"].tolist()
    registers = edited["辞書登録"].tolist()
    if applied:  # **変化したチェックだけ**反映（出現ごとの部分選択を壊さない）
        new_sel = set(sel)
        for g, on, ex in zip(shown, masks, excludes):
            spans = _group_spans(g)
            was_on = spans <= sel  # 表示時のチェック状態（全出現選択済み＝ON）
            if ex or (was_on and not on):  # 除外 or チェックを外した → 全出現を削除
                new_sel -= spans
            elif on and not was_on:  # 新たにチェック → 全出現を追加
                new_sel |= spans
            # 変化なし（was_on == on）→ そのまま（部分選択を保持）
        stored["mask_sel"] = new_sel
        stored["mask_ver"] = ver + 1
        st.rerun()
    to_exclude = [g["surface"] for g, ex in zip(shown, excludes) if ex]
    to_register = [
        (g["surface"], g["category"]) for g, on in zip(shown, registers) if on
    ]
    return to_exclude, excl, to_register, reg


def _render_by_occurrence(groups, confidences, sel, ver, stored, text):
    """出現ごと：各出現を個別にマスク。チェックは共有選択 sel を読み書きする。

    ``groups`` は API の実体グループ（dict）。各グループの occurrences を平坦化して 1 出現 1 行にする。
    出現の確信度は occurrence 側、カテゴリ/表層/votes はグループ側の値を使う。``text`` は文脈表示用。
    """
    occs = [
        {
            "surface": g["surface"],
            "category": g["category"],
            "confidence": _conf_jp(o["confidence"]),
            "votes": g["votes"],
            "span": tuple(o["span"]),
        }
        for g in groups
        for o in g["occurrences"]
    ]
    all_occs = _sorted_by_confidence(
        occs,
        key=lambda o: o["surface"],
        mask_rank=lambda o: 0 if o["span"] in sel else 1,  # マスク中を先頭に
        conf_key=lambda o: o["confidence"],
    )
    cands = [o for o in all_occs if o["confidence"] in confidences]
    hidden = len(all_occs) - len(cands)
    if not cands:
        # 空だと DataFrame に列が無く data_editor 後の列参照で落ちるので早期に案内して返す。
        st.subheader("マスク候補（0 出現）")
        if all_occs:
            st.info(
                f"確信度フィルタで全 {len(all_occs)} 出現を非表示中です。"
                "『表示する確信度』を広げてください。"
            )
        else:
            st.info(
                "マスク候補が見つかりませんでした（この文書＋現在の検出設定では 0 件）。"
            )
        return [], False, [], False
    st.subheader(f"マスク候補（{len(cands)} 出現）— 出現ごとに選択")
    cap = (
        "各出現を個別にマスク（フランク=人名 vs フランクに=気軽に、等を文脈で使い分け）。"
        "**選んだ出現だけ**マスクし、他の出現には広げません。"
    )
    if hidden:
        cap += f"（確信度フィルタで {hidden} 出現を非表示）"
    cap += " チェックしてから **[✅ マスクを反映]** を押すと反映されます。"
    st.caption(cap)
    table = pd.DataFrame(
        [
            {
                "マスク": o["span"] in sel,
                "除外": o["confidence"] == "除外",
                "辞書登録": False,
                "確信度": _confidence_label(o["confidence"]),
                "カテゴリ": o["category"],
                "表層": o["surface"],
                "文脈": _context(text, o["span"][0], o["span"][1]),
                "ja_ginza": o["votes"].get("ja_ginza", ""),
                "electra": o["votes"].get("ja_ginza_electra", ""),
                "Sudachi": o["votes"].get("sudachi", ""),
                "LLM": o["votes"].get("llm", ""),
                "辞書": "○" if o["votes"].get("dict") else "",
            }
            for o in cands
        ]
    )
    with st.form("mask_occurrence_form"):
        edited = st.data_editor(
            table,
            hide_index=True,
            width="stretch",
            disabled=[
                c for c in table.columns if c not in ("マスク", "除外", "辞書登録")
            ],
            column_config={
                "マスク": st.column_config.CheckboxColumn("マスク"),
                "除外": st.column_config.CheckboxColumn(
                    "除外",
                    help="チェックして [🚫 選択を除外リストへ] を押すと候補外に。",
                ),
                "辞書登録": st.column_config.CheckboxColumn(
                    "辞書登録",
                    help="チェックして [📒 選択を辞書へ登録] を押すと、その語のカテゴリ"
                    "（社名/商標/人名）で辞書に登録＝以後どの文書でも確定マスクになります。",
                ),
            },
            key=f"mask_occurrence_{ver}",
        )
        col_a, col_b, col_c = st.columns([1, 1, 1])
        applied = col_a.form_submit_button("✅ マスクを反映", type="primary")
        excl = col_b.form_submit_button("🚫 選択を除外リストへ")
        reg = col_c.form_submit_button("📒 選択を辞書へ登録")
    masks = edited["マスク"].tolist()
    excludes = edited["除外"].tolist()
    registers = edited["辞書登録"].tolist()
    if applied:  # 表示中の出現について sel を更新（ON=その span を追加／OFF=削除）
        new_sel = set(sel)
        for o, on, ex in zip(cands, masks, excludes):
            if on and not ex:
                new_sel.add(o["span"])
            else:
                new_sel.discard(o["span"])
        stored["mask_sel"] = new_sel
        stored["mask_ver"] = ver + 1
        st.rerun()
    to_exclude = [o["surface"] for o, ex in zip(cands, excludes) if ex]
    to_register = [
        (o["surface"], o["category"]) for o, on in zip(cands, registers) if on
    ]
    return to_exclude, excl, to_register, reg


def render_dict_editor() -> None:
    """マスク辞書の確認・追加・編集・保存 UI（独立タブ）。

    辞書ファイルはサーバが所有する（設計 B）。読み書きは API 経由で行い、
    行を編集/追加/削除して保存すると PUT /dictionary でサーバへ書き出す。
    「置換」列に値を入れると、その実体のマスク後の伏せ字を固定できる（空なら自動採番）。
    """
    st.caption(
        "カテゴリ / 代表表記 / 別名（カンマ区切り）/ 置換（任意。空なら `[社1]` 等を自動採番）。"
        "**保存先はサーバ側の辞書ファイル**（機密・git 管理外）。"
    )
    st.caption(
        "📝 **追加**＝一番下の空行に入力。"
        "🗑 **削除**＝左端のチェックを ON → キーボードの **Delete / Backspace** キー"
        "（または表右上のゴミ箱）。いずれも **[💾 辞書を保存] を押すまでサーバには反映されません**。"
    )
    client = _mask_client(MASK_API_URL)
    try:
        entries = client.get_dictionary()["entries"]
    except (MaskApiError, httpx.HTTPError) as e:
        st.error(
            "マスク辞書を取得できません（API 未接続）。サイドバー上部の接続状態を確認してください。"
        )
        st.caption(f"詳細: {e}")
        return
    # 既定でセクション順（社名→商標→人名→その他）＋各内を代表表記の辞書順に表示（保存も同順）。
    _section_rank = {"社名": 0, "商標": 1, "人名": 2}
    entries = sorted(
        entries,
        key=lambda e: (
            _section_rank.get(e["category"], 3),
            e["category"],
            dict_sort_key(e["canonical"]),
        ),
    )
    rows = [
        {
            "カテゴリ": e["category"],
            "代表表記": e["canonical"],
            "別名": ", ".join(e["aliases"]),
            "置換": e["mask"],
            "部分一致": e["partial"],
            "大小区別": e["case_sensitive"],
        }
        for e in entries
    ]
    df = pd.DataFrame(
        rows, columns=["カテゴリ", "代表表記", "別名", "置換", "部分一致", "大小区別"]
    )
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        height=500,
        key="dict_editor",
        column_config={
            "カテゴリ": st.column_config.SelectboxColumn(
                "カテゴリ", options=["社名", "商標", "人名"], default="社名"
            ),
            # 空辞書だと列が float64 になり数値入力欄＝文字を打てないので TextColumn で固定。
            "代表表記": st.column_config.TextColumn("代表表記"),
            "別名": st.column_config.TextColumn("別名", help="カンマ区切り"),
            "置換": st.column_config.TextColumn("置換", help="空なら自動採番"),
            "部分一致": st.column_config.CheckboxColumn(
                "部分一致",
                default=False,
                help="他の語の中（例 SmashMark の Smash、iASMap の iAS、IF-X の IF-）も伏字にする。"
                "境界照合なので ECBType の CB は拾わない。区切り入りでも camelCase でも同じ。",
            ),
            "大小区別": st.column_config.CheckboxColumn(
                "大小区別",
                default=False,
                help="ON にすると大文字・小文字を区別（略語向け）。例 STS は STS だけ拾い "
                "Sts/sts は拾わない。OFF（既定）は大小無視（STS=sts=Sts）。",
            ),
        },
    )
    if st.button("💾 辞書を保存", type="primary", key="dict_save"):

        def cell(value: object) -> str:
            # data_editor の空セルは NaN（float）。`nan or ""` は nan が truthy で
            # すり抜けて "nan" になるので、明示的に空文字へ落とす。
            return "" if pd.isna(value) else str(value).strip()

        new_entries = [
            {
                "category": cell(r["カテゴリ"]) or "社名",
                "canonical": cell(r["代表表記"]),
                "aliases": [a.strip() for a in cell(r["別名"]).split(",") if a.strip()],
                "mask": cell(r["置換"]),
                "partial": bool(r["部分一致"]) if not pd.isna(r["部分一致"]) else False,
                "case_sensitive": (
                    bool(r["大小区別"]) if not pd.isna(r["大小区別"]) else False
                ),
            }
            for _, r in edited.iterrows()
        ]
        kept = [e for e in new_entries if e["canonical"]]
        try:
            client.put_dictionary(kept)
        except (MaskApiError, httpx.HTTPError) as e:
            st.error("マスク辞書を保存できませんでした（API 未接続）。")
            st.caption(f"詳細: {e}")
        else:
            st.success(
                f"サーバに保存しました（{len(kept)} 件）。以降の解析に即反映されます。"
            )


def render_cache_view() -> None:
    """キャッシュ済み文書の一覧・削除（🗂 キャッシュ モード）。

    文書一覧・削除・種別更新はサーバ所有のキャッシュ（cache.db）に対して API 経由で行う。
    各文書は API の DocumentInfo（dict）で、キー content_hash / source_kind / source_name /
    char_count / chunk_count / created_at / ner_models / llm_versions を持つ。
    """
    client = _mask_client(MASK_API_URL)
    try:
        docs = client.list_documents()
    except (MaskApiError, httpx.HTTPError) as e:
        st.error(
            "文書一覧を取得できません（API 未接続）。サイドバー上部の接続状態を確認してください。"
        )
        st.caption(f"詳細: {e}")
        return
    if not docs:
        st.info(
            "キャッシュはまだありません。マスキングで文書を解析すると、NER 結果が自動で"
            "登録され、次回以降の解析が高速になります。"
        )
        return

    st.caption(
        f"キャッシュ済み: {len(docs)} 文書　"
        "（**種別**はプルダウンで修正できます＝再解析で `cache` に潰れた行を元の出所へ。"
        "編集後は [💾 種別の変更を保存] を押す）"
        "　LLM 列: **✓**=現行 detector_version で有効／**⚠ 要更新**=旧版のキャッシュのみ（窓ポリシー等が変わった）／**—**=無し。"
    )
    # 旧方式（kb-mcp の格納チャンクを直接取り込んだ）文書が残っている場合の再取込促し。
    # 新方式では kb 文書は元ファイルを取得して file として取り込む（overlap 0 で再抽出）ため、
    # source_kind=="kb" の行は旧方式の遺物＝content_hash が変わり新経路ではヒットしない。
    if any(d["source_kind"] == "kb" for d in docs):
        st.warning(
            "⚠ 旧方式（kb チャンク）で取り込まれた文書（種別 `kb`）が残っています。"
            "新方式では kb 文書は元ファイルを取得して `file` として取り込みます（別の内容ハッシュに"
            "なります）。🔒 マスキング → 📚 kb-mcp から選択 → 読み込み で取り込み直すとキャッシュが"
            "更新されます。旧エントリが不要なら削除してください。"
        )
    current_ver = _detector_version()
    # 実在する種別（text/file）＋既存データに残る種別の和。旧経路の遺物（"kb"＝旧チャンク取込、
    # "cache"＝再解析で出所が潰れた行）が残っていても表示・修正でき、消えれば選択肢から自然に
    # 消える。SelectboxColumn は現在値が options に無いとエラーになるので和にする。
    kind_options = sorted({"text", "file"} | {d["source_kind"] for d in docs})

    def _llm_col(d: dict) -> str:
        # 管理ビューは flatten 文脈を持たないので detector_version 一致のみで判定する。
        vers = d["llm_versions"]
        if not vers:
            return "—"
        return "✓" if current_ver in vers else "⚠ 要更新"

    df = pd.DataFrame(
        [
            {
                "削除": False,
                "ソース": d["source_name"],
                "種別": d["source_kind"],
                "チャンク": d["chunk_count"],
                "文字数": d["char_count"],
                "NER": _short_models(tuple(d["ner_models"])) or "—",
                "LLM": _llm_col(d),
                "解析日時": d["created_at"],
                "hash": d["content_hash"][:12],
            }
            for d in docs
        ]
    )
    edited = st.data_editor(
        df,
        hide_index=True,
        width="stretch",
        disabled=[c for c in df.columns if c not in ("削除", "種別")],
        column_config={
            "削除": st.column_config.CheckboxColumn("削除"),
            "種別": st.column_config.SelectboxColumn(
                "種別", options=kind_options, required=True
            ),
        },
        key="cache_view",
    )

    # 種別の変更（出所の手動修正）を検出して保存する。
    kind_changes = [
        (d["content_hash"], new_kind)
        for d, new_kind in zip(docs, edited["種別"].tolist())
        if new_kind != d["source_kind"]
    ]
    if kind_changes and st.button(
        f"💾 種別の変更（{len(kind_changes)} 件）を保存", type="primary"
    ):
        try:
            for content_hash_, new_kind in kind_changes:
                client.update_document(content_hash_, source_kind=new_kind)
        except (MaskApiError, httpx.HTTPError) as e:
            st.error("種別の更新に失敗しました（API 未接続）。")
            st.caption(f"詳細: {e}")
            return
        # data_editor の編集状態を破棄してから再描画する。表の行が変わった後も古い編集
        # （チェック行のインデックス）が残ると、行数の減った新しい表へ再適用される際に
        # bool 列へ NaN が入り、pandas 3.x が TypeError を投げるため（静的キーの罠）。
        st.session_state.pop("cache_view", None)
        st.success(f"種別を更新しました（{len(kind_changes)} 件）。")
        st.rerun()

    to_delete = [d for d, on in zip(docs, edited["削除"].tolist()) if on]
    if to_delete and st.button(f"🗑 選択した {len(to_delete)} 件のキャッシュを削除"):
        try:
            for d in to_delete:
                client.delete_document(d["content_hash"])
        except (MaskApiError, httpx.HTTPError) as e:
            st.error("キャッシュの削除に失敗しました（API 未接続）。")
            st.caption(f"詳細: {e}")
            return
        # 削除で行数が減る。古い編集状態を残すと次回描画で bool 列へ NaN が入り落ちるので破棄する。
        st.session_state.pop("cache_view", None)
        st.success(f"{len(to_delete)} 件のキャッシュを削除しました。")
        st.rerun()


def render_allowlist_editor() -> None:
    """除外リストの確認・追加・編集・保存 UI（独立タブ）。

    除外リストファイルはサーバが所有する（設計 B）。読み書きは API 経由で行い、
    行を編集/追加/削除して保存すると PUT /allowlist でサーバへ書き出す。
    1 列（除外語）だけのフラットなリスト。
    """
    st.caption(
        "📝 **追加**＝一番下の空行に語を入力。"
        "🗑 **削除**＝左端のチェックを ON → **Delete / Backspace**（または表右上のゴミ箱）。"
        "いずれも **[💾 除外リストを保存] を押すまでサーバには反映されません**。"
        "**保存先はサーバ側のファイル**（機密・git 管理外）。"
    )
    client = _mask_client(MASK_API_URL)
    try:
        entries = client.get_allowlist()["entries"]
    except (MaskApiError, httpx.HTTPError) as e:
        st.error(
            "除外リストを取得できません（API 未接続）。サイドバー上部の接続状態を確認してください。"
        )
        st.caption(f"詳細: {e}")
        return
    entries = sorted(
        entries, key=lambda e: allowlist_sort_key(e["surface"])
    )  # 既定で辞書順表示（保存も同順）
    # dtype を明示：空（entries=[]）だと既定で float64 列になり data_editor が数値入力欄を出して
    # **文字を打てない**（＝新規登録できない）バグになる。TextColumn/CheckboxColumn でも固定。
    df = pd.DataFrame(
        {
            "除外語": pd.Series([e["surface"] for e in entries], dtype="string"),
            "部分一致": pd.Series([e["partial"] for e in entries], dtype="bool"),
            "大小区別": pd.Series([e["case_sensitive"] for e in entries], dtype="bool"),
        }
    )
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        height=500,
        column_config={
            "除外語": st.column_config.TextColumn("除外語"),
            "部分一致": st.column_config.CheckboxColumn(
                "部分一致",
                default=False,
                help="他の語の中（例 GetFBData の FB、用補正値 の 補正、IF-X の IF-）も除外する。"
                "境界照合なので FBI の FB は拾わない。命中した候補は丸ごと除外。",
            ),
            "大小区別": st.column_config.CheckboxColumn(
                "大小区別",
                default=False,
                help="ON で大文字・小文字を区別（略語向け）。OFF（既定）は大小無視。",
            ),
        },
        key="allowlist_editor",
    )
    if st.button("💾 除外リストを保存", type="primary", key="allowlist_save"):
        kept = [
            {
                "surface": str(s).strip(),
                "partial": bool(p) if not pd.isna(p) else False,
                "case_sensitive": bool(c) if not pd.isna(c) else False,
            }
            for s, p, c in zip(
                edited["除外語"].tolist(),
                edited["部分一致"].tolist(),
                edited["大小区別"].tolist(),
            )
            if not pd.isna(s) and str(s).strip()
        ]
        try:
            client.put_allowlist(kept)
        except (MaskApiError, httpx.HTTPError) as e:
            st.error("除外リストを保存できませんでした（API 未接続）。")
            st.caption(f"詳細: {e}")
        else:
            st.success(
                f"サーバに保存しました（{len(kept)} 件）。以降の解析に即反映されます。"
            )


def render_masking_result(stored: dict) -> None:
    """マスキングの結果表示（サーバの解析 JSON から。候補選択・表示切替は再解析しない）。

    ``stored["analysis_json"]`` は API `/analyze` の応答（groups / auto_selection / text）。
    選択反映は `/apply`、手動選択差分は `/draft`、除外/辞書登録は `/allowlist`・`/dictionary`＋再 `/analyze`。
    """
    models = stored["models"]
    source_label = stored["source_label"]
    analysis = stored["analysis_json"]
    chunks = stored["chunks"]
    flatten = stored.get("flatten", False)
    detection = stored.get("analysis_detection", "both")
    groups = analysis["groups"]
    text = analysis["text"]

    client = _mask_client(MASK_API_URL)
    chash = content_hash(chunks)

    # 再解析（除外/辞書の反映）に使う共通引数。auto 選択は mask_level=strong（確定/強）。
    def _reanalyze() -> dict:
        return client.analyze_document(
            chash,
            detection=detection,
            mask_level="strong",
            flatten_tables=flatten,
            models=models,
        )

    if source_label:
        st.subheader(f"結果: {source_label}")

    # --- マスク単位の切替 ---
    col_unit, col_conf = st.columns([1, 1])
    with col_unit:
        unit = st.radio(
            "マスク単位",
            ["実体ごと（推奨）", "出現ごと（個別に選ぶ）"],
            horizontal=True,
            help="実体ごと=同じ語は文書内の全出現を一括マスク。"
            "出現ごと=各出現を個別に選ぶ（同形異義語＝フランク等の使い分け用）。",
        )
    with col_conf:
        confidences = set(
            st.multiselect(
                "表示する確信度",
                options=_CONFIDENCE_LEVELS,
                default=_CONFIDENCE_DEFAULT,
                help="微弱＝コードらしき誤検出（`Em_NoYes`・`~C02`・`7-410` 等）。既定で非表示。"
                "見たいときは『微弱』を選択（取りこぼし確認用。データは保持されています）。",
            )
        )
    by_entity = unit.startswith("実体")

    # 共有選択（マスクする span 集合）＝ auto（確定/強）＋ 保存済みドラフト（手動の差分）。
    # ドラフトは content_hash 単位でサーバ側 DB に永続化＝**再起動・再解析でも手動選択が消えない**。
    # 実体ごと/出現ごとの両ビューがこの 1 つの集合を読み書きするので、切替で選択が消えない。
    auto = _auto_mask_spans(analysis)
    if "mask_sel" not in stored:
        try:
            draft = client.get_draft(chash)
        except (MaskApiError, httpx.HTTPError):
            draft = {"added": [], "removed": []}  # 取得失敗時は auto だけで開始
        added = {tuple(s) for s in draft.get("added", [])}
        removed = {tuple(s) for s in draft.get("removed", [])}
        stored["mask_sel"] = (auto | added) - removed  # auto ∪ added − removed
        stored["mask_ver"] = 0
    sel = stored["mask_sel"]
    ver = stored["mask_ver"]

    if by_entity:
        to_exclude, excl_clicked, to_register, reg_clicked = _render_by_entity(
            groups, confidences, sel, ver, stored
        )
    else:
        to_exclude, excl_clicked, to_register, reg_clicked = _render_by_occurrence(
            groups, confidences, sel, ver, stored, text
        )

    # 「除外」チェックを除外リスト（サーバ所有）へ追記し、再解析して即反映する。
    if excl_clicked and to_exclude:
        current = client.get_allowlist()["entries"]
        existing = {e["surface"] for e in current}
        # UI から送る除外は完全一致（部分一致=False・大小無視）。細かい指定は除外リストエディタで行う。
        merged = current + [
            {"surface": s, "partial": False, "case_sensitive": False}
            for s in to_exclude
            if s not in existing
        ]
        added_n = len(merged) - len(current)
        client.put_allowlist(merged)  # サーバは PUT 後に allowlist を再ロード
        stored["analysis_json"] = _reanalyze()
        # 除外（confidence=excluded）になった span を共有選択から外す（他の手動選択は保持）。
        excluded = {
            tuple(o["span"])
            for g in stored["analysis_json"]["groups"]
            if _conf_jp(g["confidence"]) == "除外"
            for o in g["occurrences"]
        }
        stored["mask_sel"] = {s for s in sel if s not in excluded}
        stored["mask_ver"] = ver + 1
        st.success(
            f"除外リストに {added_n} 件追加し、再解析して反映しました（計 {len(merged)} 件）。"
        )
        st.rerun()

    # 「辞書登録」チェックを辞書（サーバ所有）へ追記し、再解析して即反映する。辞書一致は確定になり、
    # その語の全出現が自動マスク対象になる（→ 選択に加える）。GiNZA/LLM はサーバのキャッシュ再利用。
    if reg_clicked and to_register:
        current = client.get_dictionary()["entries"]
        existing = {e["canonical"] for e in current}
        registrable = {"社名", "商標", "人名"}
        add: list[dict] = []
        skipped: list[str] = []
        seen: set[str] = set()
        for surface, category in to_register:
            if category not in registrable:
                skipped.append(f"{surface}（{category}）")
                continue
            if surface in existing or surface in seen:
                continue
            seen.add(surface)
            add.append(
                {
                    "category": category,
                    "canonical": surface,
                    "aliases": [],
                    "mask": "",
                    "partial": False,
                    "case_sensitive": False,
                }
            )
        if add:
            client.put_dictionary(current + add)  # サーバは PUT 後に辞書を再ロード
            with st.spinner(
                "辞書を反映して再解析中 ...（GiNZA/LLM はサーバのキャッシュ）"
            ):
                stored["analysis_json"] = _reanalyze()
            # 新たに確定になった語（辞書一致）を共有選択に加える（他の手動選択は保持）。
            stored["mask_sel"] = set(sel) | _auto_mask_spans(stored["analysis_json"])
            stored["mask_ver"] = ver + 1
            msg = f"辞書に {len(add)} 件登録し、再解析して反映しました（確定＝自動マスク）。"
            if skipped:
                msg += (
                    "　※辞書に登録できるカテゴリ（社名/商標/人名）でないため "
                    f"{len(skipped)} 件を除外: {', '.join(skipped)}"
                )
            st.success(msg)
            st.rerun()
        elif skipped:
            st.warning(
                "選択した語は辞書に登録できるカテゴリ（社名/商標/人名）ではありません: "
                + ", ".join(skipped)
                + "。地名/連絡先/その他は 🚫 除外リスト で扱ってください。"
            )
        else:
            st.info("選択した語はすでに辞書に登録済みです。")

    # 手動選択（auto からの差分）をサーバの draft に永続化（変化時のみ）。再起動/再解析で復元される。
    if stored.get("_draft_saved") != stored["mask_sel"]:
        client.save_draft(
            chash,
            added=[list(s) for s in stored["mask_sel"] - auto],
            removed=[list(s) for s in auto - stored["mask_sel"]],
        )
        stored["_draft_saved"] = set(stored["mask_sel"])

    # 共有選択から結果を作る（サーバ /apply。sel の span だけをマスク＝ビュー切替で広がらない）。
    try:
        applied = client.apply_selection(
            chash,
            [list(s) for s in stored["mask_sel"]],
            detection=detection,
            flatten_tables=flatten,
            models=models,
        )
    except MaskApiError as e:
        _show_analyze_error(e, detection)
        return
    except httpx.HTTPError as e:
        st.error(f"マスキング API に接続できません: {e}")
        return
    masked_text = applied["masked_text"]
    mapping = applied["mapping"]

    # 色付き（元文）表示用：選択中 span → カテゴリ（groups から引く。text は解析座標）。
    span_cat = {
        tuple(o["span"]): g["category"] for g in groups for o in g["occurrences"]
    }
    total_occ = sum(g["count"] for g in groups)

    # --- 結果（色付き表示 / マスク済み / 元テキスト） ---
    col_main, col_side = st.columns([3, 1])
    with col_side:
        st.caption(f"モデル: {', '.join(models)}")
        st.metric("マスク（選択中）", f"{len(mapping)} 種")
        st.metric("候補", f"{total_occ} 出現")
        st.metric("チャンク数", f"{len(chunks)} 件")

    with col_main:
        view = st.radio(
            "表示", ["色付き（元文）", "マスク済み", "元テキスト"], horizontal=True
        )
        if view.startswith("色付き"):
            spans = [
                (s[0], s[1], span_cat[s]) for s in stored["mask_sel"] if s in span_cat
            ]
            html = render_masking_html(text, spans)
            st.html(
                '<div style="height:400px; overflow:auto; resize:vertical; '
                "line-height:2.2; border:1px solid rgba(128,128,128,0.25); "
                f'border-radius:6px; padding:0.5em;">{html}</div>'
            )
        elif view.startswith("マスク"):
            placeholders = {m["placeholder"]: m["category"] for m in mapping}
            st.html(_readable_text_block(masked_text, placeholders=placeholders))
        else:
            st.html(_readable_text_block(text))
        st.download_button(
            "⬇ マスク済みテキストをダウンロード",
            masked_text,
            file_name="masked.txt",
            mime="text/plain",
        )

    st.subheader(f"対応表（マスク {len(mapping)} 種）")
    if mapping:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "プレースホルダ": m["placeholder"],
                        "カテゴリ": m["category"],
                        "原語": " / ".join(m["surfaces"]),
                    }
                    for m in mapping
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.write("マスク対象が選択されていません。")


def _render_state_header(stored: dict) -> None:
    """選択ソースのパイプライン状態（冒頭設計図のミニ版）。

    ✅=本セッションで実行済み / 📂=キャッシュ有（表示は一瞬） / ⬜=未。状態はサーバ由来
    （get_document の ner_models / llm_versions、get_draft の手動差分）から導出する（設計 B）。
    """
    chash = content_hash(stored["chunks"])
    client = _mask_client(MASK_API_URL)
    status = _doc_status(client, chash)

    def mark(in_session: bool, cached: bool) -> str:
        return "✅" if in_session else ("📂" if cached else "⬜")

    want = set(stored.get("models", []))
    cached_models = set(status.get("ner_models", []))
    ner = mark("ner_json" in stored, bool(want) and want.issubset(cached_models))
    llm = mark("llm_json" in stored, _status_has_llm(status))
    try:
        draft = client.get_draft(chash)
    except (MaskApiError, httpx.HTTPError):
        draft = {"added": [], "removed": []}
    has_draft = bool(draft.get("added") or draft.get("removed"))
    merge = "✅" if "analysis_json" in stored else "⬜"
    if "analysis_json" in stored and has_draft:
        merge += "（下書きあり）"
    st.caption(
        f"パイプライン状態:　平文 ✅　→　NER検出 {ner}　＋　LLM検出 {llm}　→　"
        f"マージ&確信度 {merge}　→　確定 ⬜"
        "　（✅=本セッション実行 / 📂=キャッシュ有 / ⬜=未）"
    )


def _ner_view_groups(analysis_json: dict) -> list[dict]:
    """`/analyze` の groups のうち NER（GiNZA）由来の票を持つものだけ返す（NER検出タブ用）。

    votes に ja_ginza / ja_ginza_electra いずれかのラベルがあれば NER 由来とみなす
    （辞書・正規表現のみで拾った候補は NER タブには出さない＝GiNZA の効きを見るビュー）。
    """
    return [
        g
        for g in analysis_json["groups"]
        if g["votes"].get("ja_ginza") or g["votes"].get("ja_ginza_electra")
    ]


def _render_group_html(text: str, groups: list[dict], *, height: int = 360) -> None:
    """実体グループ（dict）の全出現を text 上でカテゴリ色ハイライトして表示する。"""
    spans = [
        (o["span"][0], o["span"][1], g["category"])
        for g in groups
        for o in g["occurrences"]
    ]
    html = render_masking_html(text, spans)
    st.html(
        f'<div style="height:{height}px; overflow:auto; resize:vertical; line-height:2.2; '
        "border:1px solid rgba(128,128,128,0.25); border-radius:6px; "
        f'padding:0.5em;">{html}</div>'
    )


def _render_detect_buttons(
    *,
    label: str,
    in_session: bool,
    cached: bool,
    cached_caption: str,
    idle_caption: str,
    key_prefix: str,
    run: Callable[[bool], None],
) -> None:
    """NER/LLM タブ共通：実行済み / キャッシュ済み / 未 の 3 状態でボタンを出し分ける。

    ``run(refresh)`` を押下時に呼ぶ（refresh=True でサーバのキャッシュを無視して再解析）。
    """
    if in_session:
        st.caption(f"{label} 状態: ✅ 実行済み（このセッション）")
        if st.button("🔄 再実行（キャッシュ無視）", key=f"{key_prefix}_rerun"):
            run(True)
    elif cached:
        st.caption(f"{label} 状態: 📂 {cached_caption}")
        c1, c2 = st.columns(2)
        if c1.button(
            "📂 キャッシュの結果を表示", type="primary", key=f"{key_prefix}_show"
        ):
            run(False)
        if c2.button("🔄 再実行（キャッシュ無視）", key=f"{key_prefix}_rerun"):
            run(True)
    else:
        st.caption(f"{label} 状態: ⬜ {idle_caption}")
        if st.button(f"▶ {label} 解析を実行", type="primary", key=f"{key_prefix}_run"):
            run(False)


def _render_ner_tab(stored: dict, flatten_tables: bool) -> None:
    """NER検出タブ（独立経路）：サーバの `/analyze`(detection="ner") で候補を集約して表示する。

    重い GiNZA 解析はサーバ側で実行・キャッシュされる（設計 B＝UI はエンジンを内包しない）。
    キャッシュ状態は get_document の ner_models から判定し、実行 / 表示＋再実行 を出し分ける。
    """
    client = _mask_client(MASK_API_URL)
    chash = content_hash(stored["chunks"])
    models = stored["models"]
    cached_models = set(_doc_status(client, chash).get("ner_models", []))
    want = set(models)
    ner_cached = bool(want) and want.issubset(cached_models)

    def _run(refresh: bool) -> None:
        try:
            t0 = time.perf_counter()
            with st.spinner(
                "サーバで NER 解析中 ...（キャッシュ済みなら一瞬・未解析は GiNZA で重い）"
            ):
                stored["ner_json"] = client.analyze_document(
                    chash,
                    detection="ner",
                    mask_level="strong",
                    flatten_tables=flatten_tables,
                    models=models,
                    refresh=refresh,
                )
            stored["ner_elapsed"] = time.perf_counter() - t0
        except MaskApiError as e:
            _show_analyze_error(e, "ner")
            return
        except httpx.HTTPError as e:
            st.error(f"マスキング API に接続できません: {e}")
            return
        # マージは NER 票込みで作り直す必要があるので無効化（マージタブで再実行を促す）。
        stored.pop("analysis_json", None)
        stored.pop("mask_sel", None)
        stored.pop("_draft_saved", None)

    extra = (
        f"（一部のみキャッシュ: {_short_models(tuple(sorted(cached_models)))}）"
        if cached_models and not ner_cached
        else ""
    )
    _render_detect_buttons(
        label="NER",
        in_session="ner_json" in stored,
        cached=ner_cached,
        cached_caption=f"サーバにキャッシュ済み（{_short_models(tuple(sorted(cached_models)))}）",
        idle_caption=f"未解析{extra}（サーバで GiNZA・重い）",
        key_prefix="ner",
        run=_run,
    )

    result = stored.get("ner_json")
    if not result:
        return
    ner_groups = _ner_view_groups(result)
    if stored.get("ner_elapsed") is not None:
        st.success(
            f"⏱ サーバ解析 {stored['ner_elapsed']:.1f}s（{len(ner_groups)} 実体）",
            icon="✅",
        )
    st.caption(
        f"GiNZA(NER) 由来の候補 {len(ner_groups)} 実体（独立ビュー）。確信度の確定はマージ&確信度タブで。"
    )
    if not ner_groups:
        st.write("NER 由来の候補はありません。")
        return
    _render_group_html(result["text"], ner_groups)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "テキスト": g["surface"],
                    "カテゴリ": g["category"],
                    "確信度": _conf_jp(g["confidence"]),
                    "ja_ginza": g["votes"].get("ja_ginza", ""),
                    "electra": g["votes"].get("ja_ginza_electra", ""),
                }
                for g in ner_groups
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def _render_llm_tab(stored: dict, flatten_tables: bool) -> None:
    """LLM検出タブ（独立経路・出口1）：サーバの `/analyze`(detection="llm") で LLM 検出を表示する。

    LLM 検出（pii-masker/Azure）はサーバ側で実行・キャッシュされる（Azure 認証も serve 側）。
    ⑥b で API 化した際、ENE type / 一致方法 / 理由 / 未特定件数は簡略化した（groups の
    surface / category / confidence / LLM 票のみ表示。必要になれば `/analyze` 応答に追加する）。
    """
    client = _mask_client(MASK_API_URL)
    chash = content_hash(stored["chunks"])
    models = stored["models"]
    llm_cached = _status_has_llm(_doc_status(client, chash))

    def _run(refresh: bool) -> None:
        try:
            t0 = time.perf_counter()
            with st.spinner(
                f"サーバで LLM 検出中 ...（{LLM_MODEL} / pii-masker・Azure）"
            ):
                stored["llm_json"] = client.analyze_document(
                    chash,
                    detection="llm",
                    mask_level="strong",
                    flatten_tables=flatten_tables,
                    models=models,
                    refresh=refresh,
                )
            stored["llm_elapsed"] = time.perf_counter() - t0
        except MaskApiError as e:
            _show_analyze_error(e, "llm")
            return
        except httpx.HTTPError as e:
            st.error(f"マスキング API に接続できません: {e}")
            return
        # マージは LLM 票込みで作り直す必要があるので無効化（マージタブで再実行を促す）。
        stored.pop("analysis_json", None)
        stored.pop("mask_sel", None)
        stored.pop("_draft_saved", None)

    _render_detect_buttons(
        label="LLM",
        in_session="llm_json" in stored,
        cached=llm_cached,
        cached_caption="サーバにキャッシュ済み",
        idle_caption=f"未実行（{LLM_MODEL} / サーバの Azure 認証が必要）",
        key_prefix="llm",
        run=_run,
    )

    result = stored.get("llm_json")
    if not result:
        return
    llm_groups = [g for g in result["groups"] if g["votes"].get("llm")]
    if stored.get("llm_elapsed") is not None:
        st.success(
            f"⏱ LLM 検出 {stored['llm_elapsed']:.1f}s（{len(llm_groups)} 実体）",
            icon="✅",
        )
    st.caption(
        f"LLM 由来の候補 {len(llm_groups)} 実体（出口1・独立ビュー）。確信度の確定はマージ&確信度タブで。"
    )
    if not llm_groups:
        st.write("LLM の検出はありませんでした。")
        return
    _render_group_html(result["text"], llm_groups)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "テキスト": g["surface"],
                    "カテゴリ": g["category"],
                    "確信度": _conf_jp(g["confidence"]),
                    "LLM": g["votes"].get("llm", ""),
                }
                for g in llm_groups
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def _show_analyze_error(e: MaskApiError, detection: str) -> None:
    """`/analyze`（や `/apply`）の MaskApiError をユーザ向けメッセージにして表示する。"""
    code = e.status_code
    if code == 502 and detection in ("llm", "both"):
        st.error(
            "LLM 検出でエラーが発生しました（サーバの Azure 認証・接続の問題）。\n"
            "実機で `az login` 済みか、サーバの .env（RESOURCE_NAME_GPT41_MINI）を確認してください。\n"
            "LLM 抜きで進めるには 🤖 LLM検出 を使わず、マージは NER のみで実行されます。\n"
            f"（詳細: {e.detail}）"
        )
    elif code == 503:
        st.error(
            "サーバの NER モデルがまだロード中です。少し待ってから再実行してください。\n"
            f"（詳細: {e.detail}）"
        )
    elif code == 404:
        st.error(
            "この文書がサーバに見つかりません（未取込）。もう一度『読み込む』からやり直してください。\n"
            f"（詳細: {e.detail}）"
        )
    else:
        st.error(f"解析に失敗しました（HTTP {code}）: {e.detail}")


def _render_merge_tab(stored: dict, flatten_tables: bool) -> None:
    """マージ&確信度タブ（出口2）：サーバの `/analyze` で候補を集約・確信度づけしてレビューする。

    detection は「**現行版の LLM キャッシュがある or LLM検出タブを実行済み**なら both、なければ ner」で
    自動判定する（LLM を勝手に走らせない＝Azure を強制しない・現状の振る舞いを踏襲）。サーバが辞書＋
    正規表現＋NER（＋LLM）を集約する。重い NER/LLM はサーバ側でキャッシュ再利用される。
    """
    client = _mask_client(MASK_API_URL)
    chash = content_hash(stored["chunks"])
    models = stored["models"]

    # LLM を合流するか：サーバに現行版キャッシュがある、または LLM 検出タブを実行済みなら both。
    llm_wanted = _llm_available(client, chash, flatten_tables) or "llm_json" in stored
    detection = "both" if llm_wanted else "ner"

    if "analysis_json" not in stored:
        clicked = st.button("▶ マージ&確信度を実行", type="primary", key="run_merge")
        if not clicked:
            chans = "辞書＋正規表現＋NER" + ("＋LLM" if llm_wanted else "")
            msg = f"『▶ マージ&確信度を実行』で **{chans}** の票を集約し確信度づけします。"
            if not llm_wanted:
                msg += (
                    "\n\nLLM は未実行・未キャッシュのため合流しません（Azure を強制しない）。"
                    "合流したい場合は 🤖 LLM検出 タブで実行してください（実行後は自動で合流します）。"
                )
            msg += (
                "\n\n確信度：1系統（NER か LLM の片方）→中／2系統（NER∧LLM）→強／辞書→確定／"
                "正規表現パターン→強／地名・その他→弱。"
            )
            st.info(msg)
            return
        try:
            with st.spinner(
                "候補を集約・確信度づけ中 ...（NER/LLM はサーバのキャッシュ再利用）"
            ):
                stored["analysis_json"] = client.analyze_document(
                    chash,
                    detection=detection,
                    mask_level="strong",
                    flatten_tables=flatten_tables,
                    models=models,
                )
        except MaskApiError as e:
            _show_analyze_error(e, detection)
            return
        except httpx.HTTPError as e:
            st.error(f"マスキング API に接続できません: {e}")
            return
        stored["analysis_detection"] = detection
        stored["analysis_channels"] = {"ner": True, "llm": detection == "both"}
        stored.pop("mask_sel", None)
        stored.pop("_draft_saved", None)
        stored["mask_ver"] = 0
    used = stored.get("analysis_channels", {"ner": True, "llm": detection == "both"})
    st.caption(
        "合流したチャネル: 辞書＋正規表現"
        + ("＋NER" if used.get("ner") else "")
        + ("＋LLM" if used.get("llm") else "")
        + "　— 確信度＝特別カテゴリを出した系統数（NER∧LLM=強／片方=中）。LLM 単独は『🤖 LLM検出』タブ（出口1）。"
    )
    render_masking_result(stored)


def _render_pipeline(stored: dict, flatten_tables: bool) -> None:
    """1ソース＝1パイプライン：状態ヘッダー＋ステージ選択（§12）。

    ステージは **st.radio（key で選択状態を保持）** で切り替える。`st.tabs` は各ステージの実行ボタンや
    マスク反映が起こす rerun のたびに先頭タブへ戻ってしまうため使わない（選択が保持できる radio にする）。
    NER検出/LLM検出 は対等な独立経路、マージ&確信度 が合流。
    """
    _render_state_header(stored)
    stage = st.radio(
        "ステージ",
        ["📄 平文", "🔍 NER検出", "🤖 LLM検出", "🔒 マージ&確信度"],
        horizontal=True,
        key="pipeline_stage",
        label_visibility="collapsed",
    )
    st.divider()
    if stage.startswith("📄"):
        _render_extracted_text(stored["chunks"])
    elif stage.startswith("🔍"):
        _render_ner_tab(stored, flatten_tables)
    elif stage.startswith("🤖"):
        _render_llm_tab(stored, flatten_tables)
    else:
        _render_merge_tab(stored, flatten_tables)


def main() -> None:
    st.set_page_config(
        page_title="data-redactor — マスキング", page_icon="🔒", layout="wide"
    )

    with st.sidebar:
        _render_connection_status()
        st.divider()

    mode = st.radio(
        "モード",
        ["🔒 マスキング", "📒 マスク辞書", "🚫 除外リスト", "🗂 キャッシュ"],
        horizontal=True,
    )
    cache_mode = mode.startswith("🗂")
    dict_mode = mode.startswith("📒")
    allowlist_mode = mode.startswith("🚫")

    # --- キャッシュ一覧モード（解析済み文書の確認・削除） ---
    if cache_mode:
        with st.sidebar:
            st.header("⚙️ 設定")
        st.title("🗂 キャッシュ")
        st.caption(
            "解析（NER）をキャッシュ済みの文書一覧。再解析は NER をスキップして高速になります。"
            "削除すると次回はフル解析に戻ります。**ローカル専用**（`data/cache.db`・git 管理外）。"
        )
        render_cache_view()
        return
    # --- マスク辞書モード（文書入力なし。辞書の確認・編集・保存だけ） ---
    # 辞書ファイルはサーバ所有（設計 B）＝パスはクライアントで指定しない。
    if dict_mode:
        with st.sidebar:
            st.header("⚙️ 設定")
        st.title("📒 マスク辞書")
        st.caption(
            "マスキングで確定マスクする社名・商標・社員名の名簿。"
            "確認・追加・編集・保存ができます。"
        )
        render_dict_editor()
        return

    # --- 除外リストモード（文書入力なし。除外語の確認・編集・保存だけ） ---
    # 除外リストファイルもサーバ所有（設計 B）＝パスはクライアントで指定しない。
    if allowlist_mode:
        with st.sidebar:
            st.header("⚙️ 設定")
        st.title("🚫 除外リスト")
        st.caption(
            "マスク**しない**語の名簿。NER の誤検出（社内コード・変数名・汎用語・誤検出メール"
            "など）をここに入れると、以後どの文書でも候補が「除外」へ落ちます。"
            "**辞書（名簿）は上書きしません**（recall 安全。連絡先 regex の誤検出は除外可）。"
        )
        render_allowlist_editor()
        return

    # --- サイドバー（マスキングの設定） ---
    with st.sidebar:
        st.header("⚙️ 設定")
        models = st.multiselect(
            "モデル（併用推奨）",
            options=MODELS,
            default=MODELS,
            format_func=lambda m: f"{m}（{MODEL_DESCRIPTIONS.get(m, '')}）",
        )
        dict_path = st.text_input("マスク辞書 (YAML)", value=str(_DEFAULT_DICT))
        allowlist_path = st.text_input(
            "除外リスト (YAML)",
            value=str(_DEFAULT_ALLOWLIST),
            help="マスクしない語の名簿。一致した検出候補を「除外」へ落とす"
            "（辞書＝名簿は守る／連絡先の誤検出は除外可）。🚫 除外リスト タブで編集。",
        )
        flatten_tables = st.toggle(
            "テーブルを平文化して検出",
            value=True,
            help="表の `|` を句読点に直して**検出精度を上げる**処理（検出専用）。"
            "マスク結果は `|` を含む原文のまま＝セル内の語だけが伏せ字になり、"
            "`|` は区切りとして残ります（出力の体裁を保持）。既定 ON（表が無ければ無影響）。",
        )

    st.title("🔒 機密情報マスキング")
    st.caption(
        "テキスト入力 / ファイルアップロード / kb-mcp から選択した文書の"
        "機密情報（人名・社名・商標など）を検出してマスクします。"
    )

    # --- 入力（ここでは描画だけ。解析はボタン押下時のみ） ---
    input_mode = st.radio(
        "入力方法",
        [
            "✏️ テキストを入力",
            "📄 ファイルをアップロード",
            "📚 kb-mcp から選択",
            "🗂 キャッシュから選択",
        ],
        horizontal=True,
    )
    input_id, input_kind, source_label, get_chunks = render_input(
        input_mode, flatten_tables
    )

    # 結果は入力方法ごとに別スロットへ保存する。入力方法を切り替えるとその方法の最後の結果
    # （無ければ案内）が出て、別タブから戻れば元の結果が復元される（テキストで解析→ファイルへ
    # 切替えてもテキストの結果が残り続ける、を防ぐ）。
    slot = f"masking:{input_kind}"

    # 再解析が必要かは「設定署名」と「入力署名」の 2 本で見る。
    #  - 設定署名（モデル/平文化/辞書 mtime）は**入力が無くても**算出できる。辞書を保存して
    #    別タブから戻ると file_uploader はファイルを失う（Streamlit が非描画ウィジェットの
    #    状態を捨てる）ので入力署名は不明になるが、設定署名は比較でき辞書変更を検知できる。
    #  - 入力署名（input_id）は入力が確定しているときだけ比較する。
    settings_sig = _masking_settings_sig(
        models, flatten_tables, dict_path, allowlist_path
    )

    # --- 解析ボタン（テキスト/ファイル/kb-mcp 共通。押したときだけ重い解析が走る） ---
    if not models:
        st.warning("モデルを 1 つ以上選択してください。")
    stored = st.session_state.get(slot)

    # 新しい入力があればそれを解析する（can_fresh）。
    # stored フォールバック（テキスト化済み stored["chunks"] で再解析）は **ファイル入力専用**：
    #   file_uploader だけが別タブ往復で中身を失うため、辞書だけ変えた再解析等で上げ直さずに済む。
    # cache/kb は **選択を fragment 内で行う**ため、行クリックでは外側（このボタン）が再実行されず
    #   選択が反映されない。そこでボタンを選択に依存させず、モデルさえあれば押せるようにし、
    #   未選択のクリックは下のハンドラで案内する（stored への誤フォールバックはしない）。
    models_ok = bool(models)
    can_fresh = get_chunks is not None and models_ok
    can_reuse_stored = input_kind == "file" and stored is not None and models_ok
    # cache/kb は選択を fragment 内で行う＝行クリックでは外側（このボタン）が再実行されず選択が
    # 反映されない。ので選択に依存させず、モデルがあれば押せるようにする（クリック＝本体再実行で
    # 選択が解決される）。未選択のままのクリックは下のハンドラで案内する。
    can_select_list = input_kind in ("cache", "kb") and models_ok
    can_analyze = can_fresh or can_reuse_stored or can_select_list
    # 「読み込み」＝チャンク確定のみ。NER/LLM/マージは各タブで個別に実行する。
    action_label = "📥 読み込む"
    clicked = st.button(action_label, type="primary", disabled=not can_analyze)
    if not can_analyze:  # なぜ押せないかを明示（モデル未選択 / 入力未指定）
        if not models:
            st.caption("⚠ サイドバーでモデルを 1 つ以上選択してください。")
        else:
            st.caption("⚠ 入力（テキスト／ファイル／kb-mcp）を指定すると押せます。")

    # ボタン下の出力（案内 / スピナー / 結果）は 1 つの placeholder に集約する。
    # クリック時にここを描き替えてから解析に入るので、モデルロード等で処理が止まっても
    # 前フレームの「…を押してください」が裏に残って透ける現象が起きない（同一スロットを差し替え）。
    output = st.empty()

    if clicked:
        with (
            output.container()
        ):  # 旧フレームの内容を即座に置換（スピナーをこの位置に出す）
            src_label, in_kind, in_sig = source_label, input_kind, input_id
            if can_fresh:
                try:
                    chunks = get_chunks()  # type: ignore[misc]  # can_fresh で None 除外済み
                except Exception as e:  # noqa: BLE001
                    st.error(f"入力の取得に失敗しました: {e}")
                    chunks = None
            elif can_reuse_stored:
                # ファイル入力で file_uploader が空（別タブ往復でクリア）。同じファイルの
                # テキスト化済みチャンク（stored）を再解析する（file 限定＝別文書の誤解析を防ぐ）。
                src_label = stored["source_label"]  # type: ignore[index]
                in_kind = stored["input_kind"]  # type: ignore[index]
                in_sig = stored["input_sig"]  # type: ignore[index]
                chunks = stored["chunks"]  # type: ignore[index]
            else:
                # cache/kb で未選択のままクリック（選択は一覧の行クリックで行う）。
                st.warning("一覧から行をクリックして文書を選択してください。")
                chunks = None
            if chunks:
                # パイプラインは「読み込み」＝チャンク確定のみ。NER/LLM/マージは各タブで個別に実行する。
                # 文書メタ＋チャンクの記録はサーバが取込（ingest_document）時に行う（設計 B）。
                # text/file/kb は get_chunks 内でサーバへ取り込み済み（cache は既に登録済み）。
                st.session_state[slot] = {
                    "settings_sig": settings_sig,
                    "input_sig": in_sig,
                    "chunks": chunks,
                    "source_label": src_label,
                    "input_kind": in_kind,
                    "flatten": flatten_tables,
                    "kind": "masking",
                    "models": models,
                    "dict_path": dict_path,
                    "allowlist_path": allowlist_path,
                }

    stored = st.session_state.get(slot)  # クリックで更新された可能性があるので取り直す
    if not stored:
        # クリック時はハンドラ側が案内（未選択）やエラーを output に表示済み。上書きしない。
        if not clicked:
            output.info(f"入力を指定して [{action_label}] を押してください。")
        return

    # 解析結果は placeholder の中に描く（クリック時はスピナー表示を結果で置き換える）。
    with output.container():
        # 保存時から設定（辞書/モデル/平文化）か入力が変わっていれば、古い結果を残したまま
        # 再解析を促す。設定は入力が無くても比較できる（辞書保存→別タブ往復で検知できる）。
        # 入力が消えていても stored のチャンクで再解析できるので、ボタンは押せる前提でよい。
        settings_changed = stored.get("settings_sig") != settings_sig
        input_changed = input_id is not None and input_id != stored.get("input_sig")
        if settings_changed or input_changed:
            st.warning(
                f"⚠ 入力／設定が変更されています。最新にするには [{action_label}] を押し直してください"
                "（マスキングは再読み込みで各タブの結果がリセットされます）。"
            )

        # 1ソース＝1パイプライン：平文/NER検出/LLM検出/マージ&確信度 のタブで見せる（§12）。
        #   平文はタブ内に置くので、ここでの inline 表示はしない。
        _render_pipeline(stored, flatten_tables)


if __name__ == "__main__":
    main()
