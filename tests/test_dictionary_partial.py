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


def _surf_spans_ner(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """NER（electra/ja_ginza）経路相当：空白トークンを落としたオフセット付き列。

    本番 ``analyze`` は NER エンジンのトークンで辞書照合する。NER は空白トークンを保持しない
    ため、``LB SONY`` は surface 列だと ``["LB","SONY"]`` と連続して見える（オフセットには
    隙間が残る）＝この経路でこそ境界ガードの取りこぼしが起きる。
    """
    toks = [(s, st, e) for (s, st, e) in _toks(text) if s.strip()]
    return [t[0] for t in toks], [(t[1], t[2]) for t in toks]


def test_whole_match_space_separated_latin_neighbors() -> None:
    """空白区切りの隣接ラテン語で取りこぼさない（`LB SONY EGA` の SONY を拾う）。

    NER 経路は空白トークンを落とすため surface 列だと SONY の直前が `LB`（末尾英字）に見え、
    境界ガードが `SONY` を「識別子の途中」と誤判定して弾いていた回帰。オフセットの隙間で
    「空白＝境界」を判定して救う。
    """
    d = _whole({normalize("SONY"): ("SONY", "社名")})
    surfaces, spans = _surf_spans_ner("LB SONY EGA Phase-L")
    assert [m.canonical for m in d.match(surfaces, spans)] == ["SONY"]
    # オフセット無し（従来経路）だと隣接扱いで弾かれる＝この回帰の再現も固定しておく。
    assert d.match(surfaces) == []


def test_whole_match_hyphen_prefix_still_rejected_with_spans() -> None:
    """オフセットを渡しても IF-X 内の IF- は弾く（IF/-/X は空白無しで隣接＝識別子の途中）。"""
    d = _whole({normalize("IF-"): ("IF-", "商標")})
    surfaces, spans = _surf_spans_ner("IF-Xを使用")
    assert d.match(surfaces, spans) == []


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
    # フルセット：未定義は null・真偽値は false（全キーが並ぶ）
    assert "aliases: null" in text and "mask: null" in text
    assert "case_sensitive: false" in text
    # round-trip で戻る
    entries = {e["canonical"]: e for e in load_entries(p)}
    assert entries["iAS"]["partial"] is True and entries["iAS"]["category"] == "商標"
    assert entries["社B"]["partial"] is False and entries["社B"]["category"] == "社名"
    assert entries["iAS"]["case_sensitive"] is False


# --------------------------------------------------------------------------- #
# 大小区別（case_sensitive）: 略語は大文字の出現だけ拾う。
# --------------------------------------------------------------------------- #


def _cs(pairs: dict[str, tuple[str, str]], *, partial: bool = False) -> MaskDictionary:
    from src.masking.dictionary import normalize_cs

    cs = {normalize_cs(s): v for s, v in pairs.items()}
    return MaskDictionary(
        {}, cs_map=dict(cs), cs_partial_map=dict(cs) if partial else None
    )


def test_case_sensitive_whole_uppercase_only() -> None:
    """STS(case_sensitive) は STS のみ一致。Sts / sts は不一致（Status の略かも）。"""
    d = _cs({"STS": ("STS", "商標")})
    assert [m.canonical for m in d.match(_surfaces("STS"))] == ["STS"]
    assert d.match(_surfaces("Sts")) == []
    assert d.match(_surfaces("sts")) == []
    # 全角 ＳＴＳ は NFKC で STS になり一致
    assert [m.canonical for m in d.match(_surfaces("ＳＴＳ"))] == ["STS"]


def test_case_sensitive_partial_uppercase_only() -> None:
    """STS(case_sensitive+partial) は STSMap の STS を拾い、StsMap は拾わない。"""
    d = _cs({"STS": ("STS", "商標")}, partial=True)
    up = "STSMap"
    assert [up[s:e] for s, e, *_ in d.partial_matches(_toks(up))] == ["STS"]
    assert d.partial_matches(_toks("StsMap")) == []


def test_case_insensitive_unaffected() -> None:
    """既定（大小無視）の語は従来どおり大小を吸収する。"""
    d = _whole({normalize("ABC"): ("ABC", "商標")})
    assert [m.canonical for m in d.match(_surfaces("abc"))] == ["ABC"]
    assert [m.canonical for m in d.match(_surfaces("Abc"))] == ["ABC"]


def test_case_sensitive_yaml_roundtrip(tmp_path) -> None:
    p = tmp_path / "d.yaml"
    p.write_text(
        "Trademark:\n  - canonical: STS\n    case_sensitive: true\n", encoding="utf-8"
    )
    assert load_entries(p)[0]["case_sensitive"] is True
    d = MaskDictionary.load(p)
    assert [m.canonical for m in d.match(_surfaces("STS"))] == ["STS"]
    assert d.match(_surfaces("sts")) == []
