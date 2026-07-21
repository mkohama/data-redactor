"""マスキング API の一連（/mask → LLM 呼び出し → /unmask）を実演する最小サンプル。

外部アプリの典型的な使い方をなぞる:

    1. 機密を含む入力（プロンプト＋複数パート）を ``/mask`` でマスクする。
    2. **マスク済みテキスト**を LLM に渡す（ここではオフラインで動くモック LLM）。
    3. LLM の応答を ``/unmask`` で復元する（プレースホルダ→元の語）。

要点は「LLM は伏せ字のまま処理し、バンドル全体で同じ実体は同じプレースホルダ」（設計 §3-1）。
実際の LLM 呼び出し（Azure/OpenAI 等）は :func:`mock_llm` を差し替えるだけ。

事前に別ターミナルでサーバを起動しておく:

    uv run data-redactor serve            # 既定 http://127.0.0.1:8000

実行:

    uv run python examples/roundtrip_demo.py
    uv run python examples/roundtrip_demo.py --detection both   # ← LLM 検出は Azure 必要
    uv run python examples/roundtrip_demo.py --base-url http://127.0.0.1:8001

``--detection both``/``llm`` は検出に LLM（pii-masker/Azure）を使うため ``az login`` 等の
資格情報が要る（未設定だとサーバが 502 を返す）。既定の ``ner`` は辞書＋正規表現＋GiNZA
だけでオフライン完結する。
"""

from __future__ import annotations

import argparse
import re
import sys

import httpx

# `python examples/roundtrip_demo.py` で実行すると examples/ が sys.path[0] になる
# （同ディレクトリの mask_client を直接 import できる）。
from mask_client import MaskApiError, MaskClient, Mapping

# マスク済みテキストに現れるプレースホルダ（[社1] / [人物1] …）を拾う。
_PLACEHOLDER = re.compile(r"\[[^\[\]]+?\d+\]")


def mock_llm(masked_prompt: str) -> str:
    """マスク済みプロンプトを受け取り、応答を返す**ダミーの LLM**。

    実際の LLM はここで自由に生成するが、本サンプルでは「登場したプレースホルダを
    そのまま引用して要約文を組み立てる」ことで、伏せ字を保ったまま処理できることと
    unmask で正しく戻せることを示す。実運用ではこの関数を実 LLM 呼び出しに差し替える。
    """
    seen: list[str] = []
    for ph in _PLACEHOLDER.findall(masked_prompt):
        if ph not in seen:
            seen.append(ph)
    if not seen:
        return "（マスク対象は検出されませんでした）要約: " + masked_prompt.strip()
    joined = "・".join(seen)
    return f"要約: 本資料の関係主体は {joined} です。以降 {seen[0]} を主担当とします。"


def _print_mapping(mapping: Mapping) -> None:
    if not mapping:
        print("  （なし）")
        return
    for m in mapping:
        surfaces = "／".join(m["surfaces"])
        print(
            f"  {m['placeholder']:<8} <- {m['canonical']}"
            f"（{m['category']} / {m['confidence']} / {m['decided_by']}）"
            f"  表記: {surfaces}"
        )


def _print_pending(pending: list[dict]) -> None:
    if not pending:
        print("  （なし）")
        return
    for p in pending:
        votes = ", ".join(f"{k}:{v}" for k, v in p["votes"].items()) or "-"
        print(f"  {p['surface']}（{p['category']} / {p['confidence']} / 票 {votes}）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--detection",
        default="ner",
        choices=["ner", "llm", "both"],
        help="検出系統（llm/both は Azure 資格情報が必要）。既定 ner（オフライン）。",
    )
    ap.add_argument(
        "--mask-level",
        default="medium",
        help="自動マスクの下限。単系統(ner/llm)は中止まりなので既定を medium にしてある。",
    )
    args = ap.parse_args(argv)

    # バンドル＝1 回の LLM 呼び出しにまとめる複数パート。同じ実体は全パートで同じ番号。
    # ここでは全て text だが、file / content_hash を混ぜても同じ 1 本の parts で書ける。
    parts = [
        {
            "kind": "text",
            "id": "prompt",
            "content": "次の 2 社の比較資料を要約して。担当は佐藤。",
        },
        {
            "kind": "text",
            "id": "docA",
            "content": "SONYの新型センサはCanonを上回る感度を示した。",
        },
        {
            "kind": "text",
            "id": "docB",
            "content": "一方でCanonのレンズ設計はSONYより堅実との評価。",
        },
    ]

    try:
        with MaskClient(args.base_url) as client:
            health = client.health()
            print(f"■ /health  {health}")
            if args.detection in ("ner", "both") and not health.get("models_ready"):
                print(
                    "  ! モデル未ロードです（models_ready=false）。"
                    "サーバ起動直後か、ロードに失敗しています。",
                    file=sys.stderr,
                )

            cfg = client.config()
            print(
                f"■ /config  detector_version={cfg['detector_version']} "
                f"models={cfg['models']}"
            )

            print(
                f"\n■ /mask  detection={args.detection} "
                f"mask_level={args.mask_level}  parts={len(parts)}"
            )
            res = client.mask(
                parts=parts,
                detection=args.detection,
                mask_level=args.mask_level,
            )

            print("\n-- マスク済みテキスト（これを LLM に渡す）--")
            for mp in res["masked_parts"]:
                print(f"  [{mp['id']}] {mp['masked_text']}")

            print(
                "\n-- 対応表（バンドル全体で共有・同じ実体は全パートで同じプレースホルダ）--"
            )
            _print_mapping(res["mapping"])

            print("\n-- pending（閾値未満のレビュー候補）--")
            _print_pending(res.get("pending", []))

            # LLM はマスク済みテキストだけを見る（原文は渡さない）。
            masked_bundle = "\n".join(mp["masked_text"] for mp in res["masked_parts"])
            llm_answer = mock_llm(masked_bundle)
            print(f"\n■ (mock) LLM 応答（伏せ字のまま）\n  {llm_answer}")

            restored = client.unmask(llm_answer, res["mapping"])["restored_text"]
            print(f"\n■ /unmask  復元\n  {restored}")
    except httpx.ConnectError as e:  # 接続不能を分かりやすく案内
        print(
            f"\n[接続エラー] サーバに接続できません（{args.base_url}）。"
            "別ターミナルで `uv run data-redactor serve` を起動してください。\n"
            f"  詳細: {e}",
            file=sys.stderr,
        )
        return 2
    except MaskApiError as e:
        print(f"\n[APIエラー] {e}", file=sys.stderr)
        if e.status_code == 502:
            print(
                "  detection=llm/both は LLM 検出（Azure）を使います。"
                "資格情報が無い場合は --detection ner を指定してください。",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
