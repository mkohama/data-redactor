"""除外リスト（allowlist）。マスク候補から恒久的に外す語の名簿。

マスク辞書（[dictionary.py]）の対。NER の誤検出（社内コード・変数名・汎用語など、人名/社名でない語）を
人が「これは機密でない」と判断したら登録し、以後**どの文書でも**候補を「除外」に落とす（1 文書で
登録→他文書でも効く）。

照合は既定で**正規化文字列の完全一致**（:func:`dictionary.normalize` と同じ NFKC+casefold）。
``部分一致: true``（旧 ``embed``）を付けた語は、辞書の ``部分一致`` と**同じ境界照合**で、他の語の
**中**に境界沿いで現れても命中とみなす（例 ``FB`` → ``GetFBData`` の ``FB``、``補正`` → ``用補正値`` の
``補正``、``IF-`` → ``IF-X`` の ``IF-``）。境界照合なので ``FBI`` の ``FB``（``FB`` を含むより長い1語）や
``ECBType`` の ``CB`` は拾わない。**照合ロジックは辞書と共有**（:func:`dictionary.contains_partial`）＝
トークン/区切り/camelCase/CJK を同じアトム分解で扱う。

辞書との違いは**効き方だけ**：辞書 ``部分一致`` は一致部分だけをマスクするが、除外は命中した
**候補を丸ごと除外**する（スパンは割らない）。ユーザーが意識するのは「完全一致／部分一致」だけで
辞書と共通（embed/接頭辞という実装区別は出さない）。

YAML 形式（``data/mask_allowlist.yaml``）。**全エントリ同一フォーマット**＝各エントリは
surface / partial の全キーを持つ（辞書と統一。キー・セクションは英語）::

    exclude:
      - surface: Em_NoYes        # 完全一致で除外
        partial: false
      - surface: FB              # 部分一致: FB を含む複合語（GetFBData 等）も丸ごと除外
        partial: true

読み込みは旧形式も後方互換で受ける：セクションの日本語（``除外``）、文字列だけの簡潔形、
``partial`` の旧キー ``部分一致`` / ``embed``。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml

from src.masking.dictionary import contains_partial, normalize

# 書き出し時の YAML セクション名（英語）。読み込みは旧・日本語 ``除外`` も受理。
_SECTION_OUT = "exclude"
_SECTION_ALIASES = ("exclude", "除外")

# 連続する空白（半角/全角/タブ/改行）。照合前に 1 個へ畳む。
_WHITESPACE = re.compile(r"\s+")


def _read_partial(item: dict) -> bool:
    """辞書項目から部分一致フラグを読む（``部分一致`` 新・``partial`` 内部・``embed`` 旧を受理）。"""
    return bool(item.get("部分一致") or item.get("partial") or item.get("embed"))


def _match_key(surface: str) -> str:
    """除外照合のキー：前後 strip ＋連続空白を 1 個に畳んでから正規化（NFKC+casefold）。

    複数トークンにまたがる実体の表層は ``text[start:end]``＝原文のトークン間スペースを含むため、
    複数スペースや特殊スペースが混じることがある（HTML 表示では 1 個に潰れて見え、人が打つ
    半角 1 スペースの除外語と完全一致しない）。空白を畳んで取りこぼしを防ぐ。
    """
    return normalize(_WHITESPACE.sub(" ", surface).strip())


class MaskAllowlist:
    """除外語の集合。既定は**完全一致**（:meth:`matches`）。``部分一致`` 語は境界内包も照合する。"""

    def __init__(self, entries: Iterable[str | dict]) -> None:
        self._norm: set[str] = set()  # 完全一致キー（全エントリ）
        self._partial: set[str] = set()  # 境界内包照合キー（部分一致: true のみ）
        for e in entries:
            if isinstance(e, dict):
                s = str(e.get("surface") or "").strip()
                partial = _read_partial(e)
            else:
                s = str(e).strip()
                partial = False
            if not s:
                continue
            self._norm.add(_match_key(s))
            if partial:
                self._partial.add(normalize(s))

    @classmethod
    def empty(cls) -> MaskAllowlist:
        return cls([])

    @classmethod
    def load(cls, path: str | Path) -> MaskAllowlist:
        """YAML を読み込んで除外リストを作る。ファイルが無ければ空。"""
        p = Path(path)
        if not p.exists():
            return cls.empty()
        return cls(load_allowlist_entries(p))

    def matches(
        self, surface: str, token_surfaces: Sequence[str] | None = None
    ) -> bool:
        """除外対象か：**完全一致** or （``部分一致`` 語の）**境界内包一致**。

        ``token_surfaces``（候補を覆う SudachiPy トークンの表層列）を渡すと、区切りをまたぐ
        照合（``IF-`` → ``IF-X``）や形態素境界（``補正`` → ``用補正値``）でも効く。渡さないと
        候補表層 1 本をアトム分解して照合する（camelCase・トークン内区切りは効く）。
        照合ロジックは辞書 :func:`dictionary.contains_partial` と共有。
        """
        if _match_key(surface) in self._norm:
            return True
        if not self._partial:
            return False
        surfaces = list(token_surfaces) if token_surfaces else [surface]
        return contains_partial(surfaces, self._partial)

    def __contains__(self, surface: str) -> bool:
        """完全一致のみ（後方互換）。内包も含めた判定は :meth:`matches`。"""
        return _match_key(surface) in self._norm

    def __bool__(self) -> bool:
        return bool(self._norm)

    def __len__(self) -> int:
        return len(self._norm)


def load_allowlist_entries(path: str | Path) -> list[dict]:
    """YAML を**構造のまま**読み込む（UI 編集・round-trip 用）。

    返り値は ``{"surface": str, "partial": bool}`` の列。文字列だけの項目は ``partial=False``。
    ``partial`` は YAML キー ``部分一致``（旧 ``embed``）のどちらからでも読む（後方互換）。
    旧 ``"nan"`` 等の空相当は捨てる。重複（surface 単位）・空白のみは除く。
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items: list = []
    for key in _SECTION_ALIASES:  # exclude（新）優先・除外（旧）も受理
        if raw.get(key):
            items = raw[key]
            break
    out: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            s = str(item.get("surface") or "").strip()
            partial = _read_partial(item)
        else:
            s = str(item).strip()
            partial = False
        if not s or s.lower() == "nan" or s in seen:
            continue
        seen.add(s)
        out.append({"surface": s, "partial": partial})
    return out


def sort_key(surface: str) -> str:
    """除外語の並び順キー（正規化＝NFKC+casefold で大小・全角半角を無視した辞書順）。

    照合キー（:func:`_match_key`）と同じ正規化を使う。件数が増えても探しやすいよう、
    保存・表示の双方で同じ順序にするため共有する。
    """
    return _match_key(surface)


def save_allowlist_entries(path: str | Path, entries: Iterable[str | dict]) -> None:
    """除外語リストを YAML に書き出す（UI 保存用）。空白除去・重複排除・正規化辞書順にソート。

    **全エントリ同一フォーマット**＝各エントリは ``surface`` / ``partial`` の全キーを持つ
    （辞書 :func:`dictionary.save_entries` と統一。キー・セクションは英語）。``entries`` は
    文字列 or ``{surface, partial|部分一致|embed}`` を混在可（後方互換）。
    """
    kept: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for e in entries:
        if isinstance(e, dict):
            s = str(e.get("surface") or "").strip()
            partial = _read_partial(e)
        else:
            s = str(e).strip()
            partial = False
        if not s or s in seen:
            continue
        seen.add(s)
        kept.append((s, partial))
    kept.sort(key=lambda t: sort_key(t[0]))
    items = [{"surface": s, "partial": partial} for s, partial in kept]
    Path(path).write_text(
        yaml.safe_dump(
            {_SECTION_OUT: items}, allow_unicode=True, sort_keys=False, indent=2
        ),
        encoding="utf-8",
    )
