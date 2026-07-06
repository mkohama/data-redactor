"""現行 ``MaskingEngine.analyze`` の特性化テスト（チャネル分離リファクタの安全網）。

analyze は recall 中核。チャネル分離リファクタ（§13）で挙動を変えていないことを保証するため、
決定的な契約を固定する：辞書→確定／regex(メール)→強／LLM 票の合流（単独→中）／
LLM 識別子の微弱免除／allowlist→除外。GiNZA(ja_ginza) を実走する（このマシンで実行可）。
GiNZA 由来の細部には依存せず、決定的チャネルの振る舞いだけを検証する。
"""

from __future__ import annotations

import pytest

from src.llm.schema import LlmDetection, LlmSpan
from src.masking.allowlist import MaskAllowlist
from src.masking.dictionary import MaskDictionary, normalize
from src.masking.engine import MaskingEngine, _looks_like_code
from src.ner.preprocess import build_body


def test_looks_like_code_strips_edge_quotes() -> None:
    """境界の引用符（全角/スマート含む）を剥がして判定：quote+code は code 扱い・引用符付き実名は守る。

    `"O1234.01`（先頭が全角/スマート引用符）が Person で残る誤検出への対処。中身がコードなら
    落とし、中身が実名（`"ソニー"`）や内部アポストロフィ（`L'Oréal`）は守る。
    """
    for s in ['"O1234.01', "＂O1234.01", "“O1234.01”", "'01234.45'"]:
        assert _looks_like_code(s), s  # quote+code → コード扱い
    for s in ['"ソニー"', "“ソニー”", "L'Oréal"]:
        assert not _looks_like_code(s), s  # 引用符付き実名・内部アポストロフィは守る


@pytest.fixture(scope="module")
def engine() -> MaskingEngine:
    eng = MaskingEngine(dictionary=MaskDictionary.empty(), models=["ja_ginza"])
    for e in eng.engines:
        _ = e.nlp  # モデルを先にロード（テスト時間の安定化）
    return eng


def _find(candidates, surface: str):
    return [c for c in candidates if c.surface == surface]


def _llm_span(chunks: list[str], surface: str, ene_type: str) -> LlmDetection:
    """``surface`` の本文位置に ene_type の LLM 検出を1件持つ LlmDetection を作る。"""
    text = build_body(chunks).text
    i = text.index(surface)
    return LlmDetection(
        spans=(LlmSpan(i, i + len(surface), ene_type, None, "exact"),),
        not_found=(),
        model="m",
        detector_version="v",
    )


def test_dict_is_definite_and_email_is_strong_contact(engine: MaskingEngine) -> None:
    """辞書語→確定（dict 票）／メール→連絡先・強（regex 票・1件まるごと）。"""
    engine.dictionary = MaskDictionary({normalize("ニコン"): ("ニコン", "社名")})
    a = engine.analyze(["ニコンへ連絡: a@example.com まで"])

    nikon = _find(a.candidates, "ニコン")
    assert nikon, "辞書語が候補に出ること"
    assert nikon[0].confidence == "確定"
    assert nikon[0].category == "社名"
    assert any(ch == "dict" for ch, _ in nikon[0].votes)

    email = _find(a.candidates, "a@example.com")
    assert email, "メールが1件まるごと候補に出ること"
    assert email[0].category == "連絡先"
    assert email[0].confidence == "強"
    assert any(ch == "regex" for ch, _ in email[0].votes)


def test_llm_only_person_is_chu(engine: MaskingEngine) -> None:
    """LLM 単独（人名）→ 中（1 チャネル）。確定にも強にもしない。"""
    engine.dictionary = MaskDictionary.empty()
    chunks = ["昨日その人が来訪した"]
    det = _llm_span(chunks, "その人", "Person")
    a = engine.analyze(chunks, llm_detection=det)
    c = _find(a.candidates, "その人")
    assert c, "LLM スパンが候補に出ること"
    assert ("llm", "Person") in c[0].votes
    assert c[0].category == "人名"
    assert c[0].confidence == "中"


def test_llm_identifier_not_demoted_to_bibyaku(engine: MaskingEngine) -> None:
    """LLM 識別子（Employee_ID）はコードらしくても微弱に落とさない（弱で残る＝レビュー可視）。"""
    engine.dictionary = MaskDictionary.empty()
    chunks = ["番号は 7-410 です"]
    det = _llm_span(chunks, "7-410", "Employee_ID")
    a = engine.analyze(chunks, llm_detection=det)
    c = _find(a.candidates, "7-410")
    assert c
    assert c[0].confidence != "微弱"
    assert ("llm", "Employee_ID") in c[0].votes


def test_allowlist_excludes_non_dict_candidate(engine: MaskingEngine) -> None:
    """allowlist 一致の非辞書候補→除外（辞書は守る・検出由来は外せる）。"""
    engine.dictionary = MaskDictionary.empty()
    chunks = ["昨日その人が来訪した"]
    det = _llm_span(chunks, "その人", "Person")
    a = engine.analyze(chunks, llm_detection=det, allowlist=MaskAllowlist(["その人"]))
    c = _find(a.candidates, "その人")
    assert c
    assert c[0].confidence == "除外"


def test_llm_only_path_runs_without_ginza() -> None:
    """run_ner=False（§13 ④）：GiNZA を回さず 辞書＋regex＋LLM だけで候補が出る。

    models=[] のエンジン（NerEngine を 1 つも持たない）で動く＝GiNZA 非依存を実証。
    辞書照合は SudachiPy 単体トークナイズ（§13 ③）で効く。Sudachi 品詞票・NER 票は出ない（A 案）。
    """
    eng = MaskingEngine(
        dictionary=MaskDictionary({normalize("ニコン"): ("ニコン", "社名")}),
        models=[],  # NER エンジンを持たない＝GiNZA は一切ロードしない
    )
    chunks = ["ニコンの田中: a@example.com まで"]
    det = _llm_span(chunks, "田中", "Person")
    a = eng.analyze(chunks, run_ner=False, llm_detection=det)

    nikon = _find(a.candidates, "ニコン")
    assert nikon and nikon[0].confidence == "確定"  # 辞書は Sudachi 単体トークンで効く

    email = _find(a.candidates, "a@example.com")
    assert email and email[0].confidence == "強"  # regex は常時

    tanaka = _find(a.candidates, "田中")
    assert tanaka and tanaka[0].confidence == "中"
    assert ("llm", "Person") in tanaka[0].votes

    # NER 経路の票（sudachi 品詞 / GiNZA）は一切入らない（A 案）。
    assert all(
        ch not in ("sudachi", "ja_ginza", "ja_ginza_electra")
        for c in a.candidates
        for ch, _ in c.votes
    )
    assert a.timings == ()  # モデルを回していない
