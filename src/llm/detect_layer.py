"""Stage A: LLM 検出層（薄いアダプタ）。

J1 本文 ``text`` を窓に切り、各窓で **pii-masker** の ``detect`` / ``locate_all`` を呼び、
窓内スパンに ``window_start`` を足して全文（merge）座標の :class:`~src.llm.schema.LlmDetection` を作る。
LLM 検出の本体（プロンプト・Azure・locate）は pii-masker 側。ここはオーケストレーションのみ。

実機では ``pii_masker`` を依存（git submodule + path-injection, §8）として呼ぶ。開発・テストでは
``detect_fn`` / ``locate_fn`` を差し替えて pii-masker・Azure・GiNZA 無しで検証できる。
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Protocol

import openai

from src.llm.schema import (
    LlmDetection,
    LlmSpan,
    detection_from_json,
    detection_to_json,
)
from src.llm.windows import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS, iter_windows

# 既定モデル（N 社制約: PII を含むデータは Azure OpenAI gpt-4.1-mini のみ）。
DEFAULT_MODEL = "gpt-4.1-mini"

# --- 止血: 一過性 429（Azure バックエンド混雑）を「窓（複合呼び出し）単位」で吸収する --------
# pii-masker ab2cd68 以降、get_client は SDK の max_retries=5（Retry-After 尊重の指数
# バックオフ・env LLM_MAX_RETRIES）を設定済み＝個別リクエストの 429 は SDK 自身が吸収する。
# ここはその上位の層で、detect（person/company/trademark 等 7〜8 リクエストの複合呼び出し）が
# 丸ごと 429 で落ちたときに窓ごと再試行する（SDK は個別リクエスト単位なのでこの粒度は回せない）。
# 注意: pii-masker 側は「アプリ層で二重のリトライループを重ねるな（重ねると最悪レイテンシが
# 掛け算になるだけ）」を推奨し、持続 429 の恒久ノブは並列度 PII_MASKER_CONCURRENCY を下げること。
# この窓層は複合呼び出しの丸ごと失敗に対する薄い保険として残す（recall 最優先＝検出全体を
# 一過性 429 で落とさない）。既定 5、env LLM_DETECT_MAX_RETRIES で調整可（SDK 側とは別 env）。
LLM_DETECT_MAX_RETRIES = int(os.getenv("LLM_DETECT_MAX_RETRIES", "5"))
# 指数バックオフの基数（秒）。Retry-After ヘッダがあればそちらを優先。
LLM_RETRY_BASE_WAIT = float(os.getenv("LLM_DETECT_RETRY_BASE_WAIT", "2.0"))
# 窓ごとの検出前に挟む待機（秒）。連射のピークを均して 429 を引きにくくする。0 で無効。
LLM_WINDOW_THROTTLE = float(os.getenv("LLM_DETECT_WINDOW_THROTTLE", "0.5"))

# 診断ログ（コンソール＝stderr）。既定 ON。``LLM_DETECT_LOG=0`` で無効化できる。
# 「窓 1/5 で無言のまま進まない」ときに、リトライ待機・所要時間・エラーを可視化するため。
# ログ基盤（logging）はこのリポで未使用なので、依存を増やさず stderr へ直接書く。
_LOG_ENABLED = os.getenv("LLM_DETECT_LOG", "1").strip().lower() not in (
    "0",
    "false",
    "",
)


def _log(msg: str) -> None:
    """LLM 検出の進捗・リトライ・エラーを stderr に出す（``[LLM検出]`` 接頭辞・即 flush）。"""
    if _LOG_ENABLED:
        print(f"[LLM検出] {msg}", file=sys.stderr, flush=True)


def _retry_wait(exc: openai.RateLimitError, attempt: int) -> float:
    """429 の待機秒。Retry-After ヘッダがあれば尊重、無ければ指数バックオフ（2^attempt 倍）。"""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is not None:
        val = headers.get("retry-after")
        if val:
            try:
                return float(val)
            except ValueError:
                pass
    return LLM_RETRY_BASE_WAIT * (2**attempt)


def _call_with_retry(fn: Callable[[], Sequence[Any]]) -> Sequence[Any]:
    """``fn`` を呼び、一過性 429（RateLimitError）は待機して最大 LLM_DETECT_MAX_RETRIES 回まで再試行。

    待機は従来**無言**だったので「進まない」と見えていた。各リトライの待機秒・回数と、
    上限到達（＝これ以上は再試行しない）を stderr にログして可視化する。
    """
    last: openai.RateLimitError | None = None
    for attempt in range(LLM_DETECT_MAX_RETRIES + 1):
        try:
            return fn()
        except openai.RateLimitError as e:  # 一過性。待って再試行。
            last = e
            if attempt < LLM_DETECT_MAX_RETRIES:
                wait = _retry_wait(e, attempt)
                _log(
                    f"429 レート制限。{wait:.0f} 秒待機して再試行"
                    f"（{attempt + 1}/{LLM_DETECT_MAX_RETRIES}）"
                )
                time.sleep(wait)
            else:
                _log(
                    f"429 が {LLM_DETECT_MAX_RETRIES} 回再試行しても解消せず＝この窓は失敗として送出"
                    "（環境変数 LLM_DETECT_MAX_RETRIES / LLM_DETECT_RETRY_BASE_WAIT で調整可）"
                )
    assert last is not None  # ループは最低1回回るので 429 経由なら必ず設定される
    raise last


# pii-masker のオブジェクト（Entity: .ene_type/.text/.reason、Match: .start/.end/.entity/.how）を
# 構造的に受けるため Any で扱う（アダプタ境界。型は pii-masker 管轄）。
DetectFn = Callable[..., Sequence[Any]]
LocateFn = Callable[..., tuple[Sequence[Any], Sequence[Any]]]
# progress(window_index, total_windows)。UI の進捗表示用（任意）。
ProgressFn = Callable[[int, int], None]


def _default_detect(document: str, *, model: str, target: str = "all") -> Sequence[Any]:
    """pii-masker の検出（遅延 import＝実機でのみ pii_masker を要求）。

    ``target`` は pii-masker の検出対象プリセット名（``all``＝人名/社名/商標の調教済み3種、
    ``pii``＝従来の全type汎用）。このアダプタは値を解釈せず pii-masker へ素通しする。
    """
    from pii_masker.detector_llm import detect

    if LLM_WINDOW_THROTTLE > 0:
        time.sleep(LLM_WINDOW_THROTTLE)  # 窓間スロットル（連射のピークを均す）
    return _call_with_retry(lambda: detect(document, model=model, target=target))


def _default_locate(
    body: str, detections: Sequence[Any]
) -> tuple[Sequence[Any], Sequence[Any]]:
    """pii-masker の text→span（遅延 import）。窓 ``body`` 基準のスパンを返す。"""
    from pii_masker.locate import locate_all

    return locate_all(body, list(detections))


def detect_document(
    text: str,
    *,
    detector_version: str,
    model: str = DEFAULT_MODEL,
    target: str = "all",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    detect_fn: DetectFn | None = None,
    locate_fn: LocateFn | None = None,
    progress: ProgressFn | None = None,
) -> LlmDetection:
    """J1 本文 ``text`` に対し LLM 検出を行い :class:`LlmDetection`（全文座標）を返す。

    手順: ``iter_windows`` で窓化 → 各窓 ``w=text[ws:we]`` で ``detect_fn(w)`` →
    ``locate_fn(w, ents)`` で窓内スパン → ``+ws`` で全文座標へ → 全窓を集約し重なりを解消。

    ``detect_fn`` / ``locate_fn`` 未指定時は pii-masker を呼ぶ（実機）。テストでは差し替える。
    ``target`` は pii-masker の検出対象プリセット名で、**既定 detect_fn にだけ**束ねて渡す
    （自前 ``detect_fn`` を渡したときは無視＝テスト/差し替えは target を意識しなくてよい）。
    ``progress(i, n)`` を渡すと各窓の検出前に (窓index, 全窓数) を通知する（UI 進捗表示用）。
    """
    detect_fn = detect_fn or partial(_default_detect, target=target)
    locate_fn = locate_fn or _default_locate

    spans: list[LlmSpan] = []
    not_found: list[tuple[str, str]] = []
    windows = iter_windows(text, max_tokens=max_tokens, overlap=overlap_tokens)
    n = len(windows)
    _log(f"開始: {n} 窓 / model={model} / target={target} / 本文 {len(text)} 文字")
    t_all = time.perf_counter()
    for i, (ws, we) in enumerate(windows):
        if progress is not None:
            progress(i, n)
        window = text[ws:we]
        _log(f"窓 {i + 1}/{n} 検出中… ({len(window)} 文字) — pii-masker 呼び出し")
        t0 = time.perf_counter()
        try:
            entities = detect_fn(window, model=model)
            matches, nf = locate_fn(window, entities)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - 一度ログしてから送出（無言で止めない）
            dt = time.perf_counter() - t0
            _log(
                f"窓 {i + 1}/{n} で {dt:.1f}s 後にエラー: {type(exc).__name__}: {exc}"
                "（この窓で検出が中断＝上位で扱われます）"
            )
            raise
        matches = list(matches)
        nf = list(nf)
        dt = time.perf_counter() - t0
        _log(
            f"窓 {i + 1}/{n} 完了 ({dt:.1f}s, 検出 {len(matches)} 件 / 未locate {len(nf)} 件)"
        )
        for m in matches:
            ent = m.entity
            spans.append(
                LlmSpan(
                    start=m.start + ws,
                    end=m.end + ws,
                    ene_type=ent.ene_type,
                    reason=getattr(ent, "reason", None),
                    how=getattr(m, "how", ""),
                )
            )
        for e in nf:
            not_found.append((e.ene_type, e.text))

    _log(
        f"完了: {n} 窓 / 全 {time.perf_counter() - t_all:.1f}s / スパン {len(spans)} 件"
        f"（重複解消前）"
    )
    return LlmDetection(
        spans=tuple(_resolve_overlaps(spans)),
        not_found=tuple(
            dict.fromkeys(not_found)
        ),  # 重複（窓 overlap 由来）を畳む・順序保持
        model=model,
        detector_version=detector_version,
    )


class LlmDetectionCache(Protocol):
    """``cached_detect`` が必要とする最小インターフェース（src.masking.cache.NerCache が満たす）。

    Protocol にして src.llm → src.masking の直接依存を避ける（鍵に flatten・detector_version を含む）。
    """

    def get_llm(
        self, content_hash: str, model: str, flatten: bool, detector_version: str
    ) -> str | None: ...

    def put_llm(
        self,
        content_hash: str,
        model: str,
        flatten: bool,
        detector_version: str,
        detections_json: str,
    ) -> None: ...


def cached_detect(
    cache: LlmDetectionCache,
    content_hash: str,
    text: str,
    *,
    flatten: bool,
    detector_version: str,
    model: str = DEFAULT_MODEL,
    target: str = "all",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    detect_fn: DetectFn | None = None,
    locate_fn: LocateFn | None = None,
    progress: ProgressFn | None = None,
    force: bool = False,
) -> LlmDetection:
    """LLM 検出をキャッシュ越しに行う（NER 層と同じ「激重層だけキャッシュ」）。

    ``(content_hash, model, flatten, detector_version)`` でヒットすれば pii-masker を呼ばない。
    ``detector_version`` を上げると自動ミス→再取得（プロンプト/窓ポリシー改版の反映）。
    ``target`` は pii-masker の検出対象プリセット名（既定 detect_fn へ素通し）。**target を切り替える側は
    必ず ``detector_version`` にも target を織り込むこと**（版に入れないと別 target 同士でキャッシュが衝突する）。
    ``force=True`` でキャッシュを無視して再検出し上書きする（NER キャッシュには触れない）。
    ``progress(i, n)`` は detect_document へ渡る（キャッシュヒット時は呼ばれない＝即返り）。
    """
    if not force:
        hit = cache.get_llm(content_hash, model, flatten, detector_version)
        if hit is not None:
            return detection_from_json(hit)
    detection = detect_document(
        text,
        detector_version=detector_version,
        model=model,
        target=target,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        detect_fn=detect_fn,
        locate_fn=locate_fn,
        progress=progress,
    )
    cache.put_llm(
        content_hash, model, flatten, detector_version, detection_to_json(detection)
    )
    return detection


def _resolve_overlaps(spans: list[LlmSpan]) -> list[LlmSpan]:
    """窓をまたいだ重複・内包スパンを解消する（pii-masker locate._resolve と同方針）。

    長いスパン優先で確定し、既存の確定スパンに**完全内包される**スパン（重複も含む）は捨てる。
    部分重複（互いに内包しない）は両方残す。窓 overlap で同一実体が2回出ても 1 つに畳まれる。
    """
    ordered = sorted(spans, key=lambda s: (-(s.end - s.start), s.start))
    kept: list[LlmSpan] = []
    for s in ordered:
        if s.start >= s.end:
            continue
        if any(k.start <= s.start and s.end <= k.end for k in kept):
            continue  # 既存の確定スパンに完全内包（重複含む）→ 捨てる
        kept.append(s)
    return sorted(kept, key=lambda s: (s.start, -(s.end - s.start)))


__all__ = ["detect_document", "cached_detect", "LlmDetectionCache", "DEFAULT_MODEL"]
