"""LLM 検出器の版・窓ポリシー・検出対象・共有の LLM 検出実行 (UI 非依存)。

app.py (Streamlit クライアント) と src/api (FastAPI サーバ) の**両方**が、同じ
``detector_version`` と同じ LLM 検出経路を使うための共通層。ここに集約することで
LLM 検出キャッシュの鍵 ``(content_hash, model, flatten, detector_version)`` が UI/API で
一致し、**同じ cache.db を共有**できる (エンジンを 1 プロセスに集約する)。

かつてこれらは app.py (Streamlit) にあったが、Streamlit を import せずに使えないため
FastAPI から参照できなかった。UI 非依存のこのモジュールへ移した (app.py は薄い再輸出に)。

``_DETECTOR_STATIC`` の ``pii-masker@<hash>`` は sync-pii-masker (src/cli.py) が submodule 更新時に
このファイルを正規表現で書き換える (＝LLM 検出キャッシュを自動ミス→再取得させる)。
"""

from __future__ import annotations

import os
from collections.abc import Callable

from src.llm import cached_detect
from src.llm.detect_layer import DEFAULT_MODEL as LLM_MODEL
from src.llm.detect_layer import LlmDetectionCache
from src.llm.schema import LlmDetection
from src.llm.windows import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS
from src.masking.cache import content_hash
from src.ner.preprocess import build_body

# detector_version の**静的部分**：pii-masker のコミット版 (``pii-masker@<hash>``) だけ。
#   submodule 更新時に sync-pii-masker (src/cli.py) がこの hash 文字列を正規表現で置換する。
# 窓ポリシー (win…) はここに書かず、実値 (env or windows.py 既定) から _detector_version() が自動合成する
#   ＝env で変えるだけで detector_version が変わりキャッシュ自動無効化 (コード編集・手動バンプ不要)。
# 旧 ene-vN (type-map 版) は廃止：_ENE_TO_CATEGORY は解析時に毎回当たる後段変換で、LLM 検出キャッシュ
#   (生 ene_type のみ保存) に影響しない＝バンプ不要だったため。変更は次の解析で自動反映される。
_DETECTOR_STATIC = "pii-masker@7a0b0a8"

# LLM 検出の対象プリセット (pii-masker の target)。env ``LLM_DETECT_TARGET`` で切替・既定 "all"。
#   all … 人名/社名/商標の調教済み3種 (高精度。実際にマスクしたいのがこの3種なので既定)。
#   pii … 従来の全type汎用 (地名/連絡先/ID も拾うが精度はやや落ちる・単一プロンプト)。
# 値を変えると detector_version() 経由で **LLM 検出キャッシュが自動で無効化** される (再検出)。
_DETECT_TARGETS = ("all", "pii")


def _env_int(name: str, default: int) -> int:
    """環境変数を int で読む。未設定・不正値なら ``default`` (チューニング用の安全側フォールバック)。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def window_policy() -> tuple[int, int]:
    """LLM 検出の窓ポリシー (max_tokens, overlap)。env で上書き可・既定は windows.py の定数。

    ``LLM_WINDOW_MAX_TOKENS`` / ``LLM_WINDOW_OVERLAP_TOKENS`` を .env に置くだけで調整でき、
    値を変えると detector_version() 経由で **LLM 検出キャッシュが自動で無効化** される (再検出)。
    """
    return (
        _env_int("LLM_WINDOW_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        _env_int("LLM_WINDOW_OVERLAP_TOKENS", DEFAULT_OVERLAP_TOKENS),
    )


def detector_target() -> str:
    """LLM 検出の対象プリセット名 (env ``LLM_DETECT_TARGET``。既定 "all")。

    認識するのは ``all`` / ``pii`` のみ。未設定・不正値は安全側の既定 "all" にフォールバックする
    (_env_int と同じ方針)。target は detector_version() にも織り込むのでキャッシュは target 別に分かれる。
    """
    raw = os.getenv("LLM_DETECT_TARGET", "all")
    return raw if raw in _DETECT_TARGETS else "all"


def detector_version() -> str:
    """LLM 検出器の版 ``pii-masker@<hash>|win<max>ov<ov>|tgt<target>``。

    win… は現在の窓ポリシー (env or 既定)、tgt… は検出対象 (env or 既定 all) から合成する。
    pii-masker@<hash> は _DETECTOR_STATIC (静的・sync-pii-masker が自動書換)。target を版に含めるのは、
    別 target (all↔pii) のキャッシュが ``(content_hash, model, flatten, detector_version)`` で衝突しない
    ようにするため (含めないと target を変えても先勝ちの結果が返り続ける)。type-map (_ENE_TO_CATEGORY) は
    版に含めない (解析時の後段変換で検出キャッシュに影響しないため。変更は次の解析で自動反映)。
    """
    max_tokens, overlap = window_policy()
    return f"{_DETECTOR_STATIC}|win{max_tokens}ov{overlap}|tgt{detector_target()}"


def run_llm_detection(
    cache: LlmDetectionCache,
    chunks: list[str],
    flatten_tables: bool,
    *,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[str, LlmDetection]:
    """LLM 検出を実行 (キャッシュ越し)。本文 ``text`` と ``LlmDetection`` を返す。

    pii-masker を呼ぶ (実機・Azure・``az login`` 前提)。キャッシュは
    ``(content_hash, model, flatten, detector_version)`` で、同一文書は再呼び出ししない。
    ``force=True`` でキャッシュ無視の再検出 (LLM 層のみ。NER キャッシュには触れない)。
    ``progress(i, n)`` は窓ごとの進捗 (キャッシュヒット時は呼ばれない)。
    窓ポリシー (max_tokens/overlap) は env で上書き可 (window_policy)。版にも反映されるので整合する。
    """
    max_tokens, overlap = window_policy()
    body = build_body(chunks, flatten_tables=flatten_tables)
    detection = cached_detect(
        cache,
        content_hash(chunks),
        body.text,
        flatten=flatten_tables,
        detector_version=detector_version(),
        target=detector_target(),
        max_tokens=max_tokens,
        overlap_tokens=overlap,
        progress=progress,
        force=force,
    )
    return body.text, detection


__all__ = [
    "LLM_MODEL",
    "detector_version",
    "detector_target",
    "window_policy",
    "run_llm_detection",
]
