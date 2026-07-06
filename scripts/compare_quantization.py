"""指定ファイルに対し、NER の **fp32 vs INT8 動的量子化** でマスク対象の検出結果と速度を比較する
診断スクリプト（使い捨て）。

electra(transformer) を INT8 量子化すると CPU 推論が速くなる（実測 ~1.7〜2.4x）が、数値が変わるため
NER 出力（＝マスク候補）がずれ得る。本スクリプトは同一ファイルで fp32/int8 の
(1) NER 速度 (2) 検出実体（表層/カテゴリ/確信度）の差 を並べ、**recall 影響を目視**できるようにする。

注意:
- LLM(Azure) は使わず **NER のみ**（量子化が効くのは electra NER）。本番は NER×LLM の2系統なので、
  ここで中/弱に落ちた人名も LLM 併走で強に戻る可能性がある（本番での確認は別途・要 az login）。
- 「自動マスク対象＝確定/強」の差を最重要として先に出す（中/弱/微弱はレビュー/非表示層）。
- モデル既定は electra + ja_ginza（本番同等）。electra が無い構成では量子化できず終了する。

使い方:
    uv run python scripts/compare_quantization.py <ファイル>
    uv run python scripts/compare_quantization.py <ファイル> --flatten --threads 8 --out diff.csv

注: xlsx など表データは ``--flatten`` を付けると Markdown テーブルを平文化してから NER する
（本番 UI の平文化 ON 相当）。付けないと表のパイプ区切り ``|`` を GiNZA がまたいで拾い、
``'千葉|'`` や ``'高橋 (憲) |'`` のようなゴミ表層が出る。本番の設定に合わせて指定する。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path


def _quantize_electra(nlp: object) -> bool:
    """spacy-transformers パイプ内の torch モジュールを INT8 動的量子化する（成否を返す）。

    transformer pipe の Thinc モデルを ``walk()`` で辿り、shim が持つ ``torch.nn.Module``
    （ElectraModel）に ``quantize_dynamic({Linear}, qint8)`` を当てて差し戻す。
    """
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]

    if "transformer" not in nlp.pipe_names:  # type: ignore[attr-defined]
        return False
    for layer in nlp.get_pipe("transformer").model.walk():  # type: ignore[attr-defined]
        for shim in getattr(layer, "shims", []) or []:
            for attr in dir(shim):
                if attr.startswith("__"):
                    continue
                try:
                    val = getattr(shim, attr)
                except Exception:  # noqa: BLE001 - 属性取得の副作用/例外は無視して次へ
                    continue
                if isinstance(val, nn.Module):
                    q = torch.quantization.quantize_dynamic(
                        val, {nn.Linear}, dtype=torch.qint8
                    )
                    setattr(shim, attr, q)
                    return True
    return False


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows コンソール文字化け対策

    parser = argparse.ArgumentParser(
        description="NER の fp32 vs INT8量子化 でマスク候補と速度を比較する。"
    )
    parser.add_argument("file", type=Path, help="対象ファイル（xlsx/pdf/docx/txt 等）")
    parser.add_argument(
        "--threads", type=int, default=8, help="torch intra-op スレッド数（既定 8）"
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Markdown テーブルを平文化してから NER（xlsx 等の表データ推奨・本番 UI の平文化 ON 相当）",
    )
    parser.add_argument(
        "--out", type=Path, help="候補差分を書き出す（.csv なら CSV、それ以外は JSON）"
    )
    args = parser.parse_args(argv)
    if not args.file.exists():
        raise SystemExit(f"エラー: ファイルが存在しません: {args.file}")

    import torch

    torch.set_num_threads(args.threads)
    from src.masking import MaskDictionary, MaskingEngine
    from src.sources.files import load_chunks_from_file

    chunks = load_chunks_from_file(str(args.file))
    print(
        f"入力: {args.file.name} / チャンク {len(chunks)} / "
        f"{sum(len(c) for c in chunks)}字 / torch {args.threads}スレッド / "
        f"平文化 {'ON' if args.flatten else 'OFF'}"
    )
    eng = MaskingEngine(dictionary=MaskDictionary.empty())  # 既定 electra + ja_ginza
    for e in eng.engines:
        _ = e.nlp  # 事前ロード（計測から除外）

    def run(label: str) -> tuple[set[tuple[str, str, str]], float]:
        t = time.perf_counter()
        analysis = eng.analyze(  # NER 両モデル・LLMなし・キャッシュなし
            chunks, flatten_tables=args.flatten
        )
        wall = time.perf_counter() - t
        groups = eng.group_candidates(analysis.candidates)
        conf = Counter(g.confidence for g in groups)
        print(f"[{label}] NER {wall:5.1f}s / 実体 {len(groups)} / {dict(conf)}")
        return {(g.surface, g.category, g.confidence) for g in groups}, wall

    fp32, w_fp32 = run("fp32")
    if not _quantize_electra(eng.engines[0].nlp):
        raise SystemExit(
            "エラー: electra(transformer) が見つからず量子化できません（ja_ginza のみの構成？）"
        )
    int8, w_int8 = run("int8")

    auto = {"確定", "強"}
    fp32_auto = {(s, c) for s, c, cf in fp32 if cf in auto}
    int8_auto = {(s, c) for s, c, cf in int8 if cf in auto}

    print("\n===== 速度 =====")
    print(f"fp32 {w_fp32:.1f}s → int8 {w_int8:.1f}s  ({w_fp32 / w_int8:.2f}x)")

    print("\n===== 自動マスク対象（確定/強）の差【最重要】 =====")
    print(
        f"fp32 {len(fp32_auto)} / int8 {len(int8_auto)} / 共通 {len(fp32_auto & int8_auto)}"
        f" / 消えた {len(fp32_auto - int8_auto)} / 増えた {len(int8_auto - fp32_auto)}"
    )
    for s, c in sorted(fp32_auto - int8_auto):
        print(f"  消[{c}] {s!r}  ← 自動マスクされなくなる（recall 損）")
    for s, c in sorted(int8_auto - fp32_auto):
        print(f"  増[{c}] {s!r}")

    lost, gained = fp32 - int8, int8 - fp32
    print("\n===== 全候補の差（中/弱/微弱含む）=====")
    print(f"共通 {len(fp32 & int8)} / 消えた {len(lost)} / 増えた {len(gained)}")

    def show(sset: set[tuple[str, str, str]], title: str, cap: int = 80) -> None:
        print(f"--- {title}（{len(sset)}件）---")
        for s, c, cf in sorted(sset, key=lambda x: (x[2], x[1], x[0]))[:cap]:
            print(f"  [{cf}/{c}] {s!r}")
        if len(sset) > cap:
            print(f"  … 他 {len(sset) - cap} 件")

    show(lost, "消えた（fp32→int8 でロス）")
    show(gained, "増えた（int8 で新規）")

    if args.out is not None:
        rows = [
            {"diff": "lost", "surface": s, "category": c, "confidence": cf}
            for s, c, cf in sorted(lost)
        ] + [
            {"diff": "gained", "surface": s, "category": c, "confidence": cf}
            for s, c, cf in sorted(gained)
        ]
        fields = ["diff", "surface", "category", "confidence"]
        if args.out.suffix.lower() == ".csv":
            with args.out.open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            args.out.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"\n差分を書き出しました: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
