"""マスク辞書の照合仕様テスト（完全一致の境界ガード ＋ 部分一致マッチャ）。

ユーザーは「まるごと（既定）／部分一致（``部分一致: true``）」だけを意識する。区切り入り
（``IF-``）でも camelCase（``iAS``）でも、同じ ``部分一致: true`` で他の語の中を拾える
（実装の embed/接頭辞という区別を出さない）＝この統一を固定する。

トークナイズは SudachiPy 実走（このマシンで可）。全文オフセット付きトークンを渡す。
"""

from __future__ import annotations

from src.masking.dictionary import MaskDictionary, load_entries, normalize, save_entries
from src.ner.engine import sudachi_analyze_chunks


def _toks(text: str) -> list[tuple[str, int, int]]:
    a = sudachi_analyze_chunks([text])
    return [(t.surface, t.start, t.end) for t in a.tokens]


def _surfaces(text: str) -> list[str]:
    a = sudachi_analyze_chunks([text])
    return [t.surface for t in a.tokens]


def _whole(pairs: dict[str, tuple[str, str]]) -> MaskDictionary:
    return MaskDictionary(dict(pairs))


def _partial(pairs: dict[str, tuple[str, str]]) -> MaskDictionary:
    return MaskDictionary(dict(pairs), partial_map=dict(pairs))


# --------------------------------------------------------------------------- #
# 完全一致（既定）＝まるごとの語だけ。断片は拾わない（ラテン境界ガード）。
# --------------------------------------------------------------------------- #


def test_whole_match_standalone() -> None:
    d = _whole({normalize("iAS"): ("iAS", "商標")})
    ms = d.match(_surfaces("iASのみ"))
    assert [m.canonical for m in ms] == ["iAS"]


def test_whole_match_not_inside_camel_compound() -> None:
    """既定では iAS は iASMap（1トークン）の中では拾わない。"""
    d = _whole({normalize("iAS"): ("iAS", "商標")})
    assert d.match(_surfaces("iASMapを確認")) == []


def test_whole_match_hyphen_prefix_rejected_by_guard() -> None:
    """IF- は IF-X の断片なので既定では拾わない（右隣が英字＝境界ガード）。"""
    d = _whole({normalize("IF-"): ("IF-", "商標")})
    assert d.match(_surfaces("IF-Xを使用")) == []


def test_whole_match_full_hyphenated_term() -> None:
    """フルのハイフン語 IF-X はまるごと一致する。"""
    d = _whole({normalize("IF-X"): ("IF-X", "商標")})
    ms = d.match(_surfaces("IF-Xを使用"))
    assert [m.canonical for m in ms] == ["IF-X"]


def test_whole_match_cjk_boundary_unchanged() -> None:
    """CJK は連結記号でないので従来どおり：社A は 社Aです で拾う。"""
    d = _whole({normalize("社A"): ("社A", "社名")})
    ms = d.match(_surfaces("社Aです"))
    assert [m.canonical for m in ms] == ["社A"]


# --------------------------------------------------------------------------- #
# 部分一致（部分一致: true）＝他の語の中でも境界沿いに拾う。区切り／camelCase を統一。
# --------------------------------------------------------------------------- #


def test_partial_hyphen_prefix() -> None:
    """IF- 部分一致 → IF-X / IF-Y の「IF-」部分を拾う（[商標]X になる）。"""
    d = _partial({normalize("IF-"): ("IF-", "商標")})
    for text in ("IF-Xを使用", "IF-Yのログ"):
        pm = d.partial_matches(_toks(text))
        assert [text[s:e] for s, e, *_ in pm] == ["IF-"]


def test_partial_hyphen_prefix_leaves_bare_if_safe() -> None:
    """IF- 部分一致でも単独 IF は拾わない（「IF-」を含まない）。"""
    d = _partial({normalize("IF-"): ("IF-", "商標")})
    assert d.partial_matches(_toks("IFは共通部品")) == []


def test_partial_camel_subword() -> None:
    """iAS 部分一致 → iASMap の iAS を拾う（従来 embed）。"""
    d = _partial({normalize("iAS"): ("iAS", "商標")})
    pm = d.partial_matches(_toks("iASMapを確認"))
    assert [("iASMapを確認")[s:e] for s, e, *_ in pm] == ["iAS"]


def test_partial_boundary_safe_not_mid_subword() -> None:
    """CB 部分一致 → ECBType の CB（ECB の途中）は拾わない（境界安全）。"""
    d = _partial({normalize("CB"): ("CB", "商標")})
    assert d.partial_matches(_toks("ECBTypeの確認")) == []


def test_partial_same_flag_for_hyphen_and_camel() -> None:
    """同じ 部分一致: true で、区切り入り(IF-)も camelCase(iAS)も両方効く（統一の要）。"""
    pairs = {normalize("IF-"): ("IF-", "商標"), normalize("iAS"): ("iAS", "商標")}
    d = _partial(pairs)

    hyphen = "IF-X"
    assert [hyphen[s:e] for s, e, *_ in d.partial_matches(_toks(hyphen))] == ["IF-"]

    camel = "iASMap"
    assert [camel[s:e] for s, e, *_ in d.partial_matches(_toks(camel))] == ["iAS"]


def test_partial_cjk_across_morphemes() -> None:
    """補正 部分一致 → 用補正値 の 補正 を形態素境界で拾う。"""
    d = _partial({normalize("補正"): ("補正", "商標")})
    text = "用補正値を見る"
    pm = d.partial_matches(_toks(text))
    assert [text[s:e] for s, e, *_ in pm] == ["補正"]


def test_partial_cjk_boundary_safe_single_char() -> None:
    """補 部分一致 → 補正 の中の 補（形態素の途中）は拾わない（CJK 境界安全）。

    アトムは1トークン内の CJK 連を束ねる（per-char にしない）ので 補 は 補正 を割れない。
    """
    d = _partial({normalize("補"): ("補", "商標")})
    assert d.partial_matches(_toks("補正値の確認")) == []


# --------------------------------------------------------------------------- #
# YAML round-trip: 部分一致キー（新）と embed キー（旧・後方互換）。
# --------------------------------------------------------------------------- #


def test_yaml_reads_partial_key(tmp_path) -> None:
    p = tmp_path / "d.yaml"
    p.write_text("商標:\n  - canonical: iAS\n    部分一致: true\n", encoding="utf-8")
    entries = load_entries(p)
    assert entries[0]["partial"] is True
    d = MaskDictionary.load(p)
    assert d.partial_matches(_toks("iASMap"))  # 部分一致が効く


def test_yaml_reads_legacy_embed_key(tmp_path) -> None:
    """旧 embed キーも読める（後方互換）。"""
    p = tmp_path / "d.yaml"
    p.write_text("商標:\n  - canonical: iAS\n    embed: true\n", encoding="utf-8")
    entries = load_entries(p)
    assert entries[0]["partial"] is True


def test_yaml_writes_full_set_english(tmp_path) -> None:
    """保存は英語キー・英語セクション・フルセット（未定義 null／partial は false）で書く。"""
    p = tmp_path / "d.yaml"
    save_entries(
        p,
        [
            {
                "category": "商標",
                "canonical": "iAS",
                "aliases": [],
                "mask": "",
                "partial": True,
            },
            {
                "category": "社名",
                "canonical": "社B",
                "aliases": [],
                "mask": "",
                "partial": False,
            },
        ],
    )
    text = p.read_text(encoding="utf-8")
    # 英語セクション・英語キー
    assert "Trademark:" in text and "Company:" in text
    assert "partial: true" in text and "partial: false" in text
    assert "部分一致" not in text and "embed:" not in text
    # フルセット：未定義は null（partial なしのエントリでも全キーが並ぶ）
    assert "aliases: null" in text and "mask: null" in text
    # round-trip で戻る
    entries = {e["canonical"]: e for e in load_entries(p)}
    assert entries["iAS"]["partial"] is True and entries["iAS"]["category"] == "商標"
    assert entries["社B"]["partial"] is False and entries["社B"]["category"] == "社名"
