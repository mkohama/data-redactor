"""マスキング API の一連（/mask → LLM 呼び出し → /unmask）を実演する最小サンプル。

外部アプリの典型的な使い方をなぞる:

    1. 機密を含む入力（プロンプト＋複数パート）を /mask でマスクする。
    2. マスク済みテキストを LLM に渡す（ここではオフラインで動くモック LLM）。
    3. LLM の応答を /unmask で復元する（プレースホルダ→元の語）。

要点は「LLM は伏せ字のまま処理し、バンドル全体で同じ実体は同じプレースホルダ」（設計 3-1）。
実際の LLM 呼び出し（Azure/OpenAI 等）は mock_llm を差し替えるだけ。

事前に別ターミナルでサーバを起動しておく:

    uv run data-redactor serve            # 既定 http://127.0.0.1:8000

実行:

    # テキストだけのバンドル（オフライン完結・手軽）
    uv run python examples/roundtrip_demo.py

    # ファイルの受け渡しを確認する（examples/sample_data の全ファイルを添付）
    uv run python examples/roundtrip_demo.py --files

    # 添付ファイルを明示（複数可）
    uv run python examples/roundtrip_demo.py --files path/to/a.xlsx path/to/b.pdf

    uv run python examples/roundtrip_demo.py --detection both   # ← LLM 検出は Azure 必要

--files を付けると file part（サーバがテキスト化）を含むバンドルを送り、ファイルの
受け渡し・マスク・復元まで確認できる。値を省くと examples/sample_data の中身を使う。
--detection both / llm は検出に LLM（Azure）を使うため資格情報が要る（既定 ner は
辞書＋正規表現＋GiNZA だけでオフライン完結）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import httpx

# python examples/roundtrip_demo.py で実行すると examples/ が sys.path[0] になる
# （同ディレクトリの mask_client を直接 import できる）。
from mask_client import MaskApiError, MaskClient, Mapping

# マスク済みテキストに現れるプレースホルダ（[社1] / [人物1] …）を拾う。
_PLACEHOLDER = re.compile(r"\[[^\[\]]+?\d+\]")
# --files を値なしで指定したときに使うサンプル置き場と対象拡張子。
_SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
_SAMPLE_EXTS = {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx", ".pptx"}


def mock_llm(masked_prompt: str) -> str:
    """マスク済みプロンプトを受け取り、応答を返すダミーの LLM。

    実運用ではこの関数を実 LLM 呼び出しに差し替える。本サンプルでは「登場した
    プレースホルダを引用して要約文を組み立てる」ことで、伏せ字のまま処理できることと
    unmask で正しく戻せることを示す（多いので先頭 6 個までに絞る）。
    """
    seen: list[str] = []
    for ph in _PLACEHOLDER.findall(masked_prompt):
        if ph not in seen:
            seen.append(ph)
    if not seen:
        return "（マスク対象は検出されませんでした）"
    shown = seen[:6]
    tail = f" ほか{len(seen) - len(shown)}件" if len(seen) > len(shown) else ""
    return f"要約: 本資料の関係主体は {'・'.join(shown)}{tail} です。担当は {seen[0]}。"


def _default_sample_files() -> list[str]:
    """examples/sample_data 内の対応拡張子ファイルをソートして返す。"""
    files = sorted(
        str(p) for p in _SAMPLE_DIR.glob("*") if p.suffix.lower() in _SAMPLE_EXTS
    )
    if not files:
        raise SystemExit(f"サンプルファイルがありません: {_SAMPLE_DIR}")
    return files


def _build_parts(files: list[str] | None) -> list[dict]:
    """バンドル（parts）を組み立てる。files 指定時はプロンプト＋各ファイル。"""
    prompt = {
        "kind": "text",
        "id": "prompt",
        "content": "次の資料を要約して。担当は佐藤。",
    }
    if files is None:
        # ファイル無し＝テキストだけのバンドル（オフラインで手軽に流す用）。
        return [
            prompt,
            {
                "kind": "text",
                "id": "docA",
                "content": "SONYの新型センサはCanonを上回る。",
            },
            {
                "kind": "text",
                "id": "docB",
                "content": "一方Canonのレンズは堅実との評価。",
            },
        ]
    paths = files or _default_sample_files()
    # file part の id はファイル名。content にパスを渡すだけ（本体送信はクライアントが担う）。
    return [
        prompt,
        *({"kind": "file", "id": Path(p).name, "content": p} for p in paths),
    ]


def _preview(text: str, limit: int = 240) -> str:
    """長い本文を 1 行に潰して先頭だけ表示する（ファイルは長いので）。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + f" …(+{len(flat) - limit}字)"


def _print_mapping(mapping: Mapping, limit: int = 15) -> None:
    if not mapping:
        print("  （なし）")
        return
    for m in mapping[:limit]:
        surfaces = "／".join(m["surfaces"])
        print(
            f"  {m['placeholder']:<8} <- {m['canonical']}"
            f"（{m['category']} / {m['confidence']} / {m['decided_by']}）  表記: {surfaces}"
        )
    if len(mapping) > limit:
        print(f"  …ほか {len(mapping) - limit} 件")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--files",
        nargs="*",
        default=None,
        metavar="PATH",
        help="添付ファイル。値を省くと examples/sample_data を使う。無指定ならテキストのみ。",
    )
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
    ap.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="1 リクエストの秒数。detection=both で複数ファイルは遅いので長め。0 で無制限。",
    )
    args = ap.parse_args(argv)

    parts = _build_parts(args.files)
    timeout = args.timeout if args.timeout > 0 else None

    try:
        with MaskClient(args.base_url, timeout=timeout) as client:
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

            kinds = ", ".join(f"{p['id']}={p['kind']}" for p in parts)
            print(
                f"\n■ /mask  detection={args.detection} mask_level={args.mask_level}"
                f"  parts={len(parts)}  [{kinds}]"
            )
            res = client.mask(
                parts=parts,
                detection=args.detection,
                mask_level=args.mask_level,
            )

            print("\n-- マスク済みテキスト（これを LLM に渡す。原文は渡さない）--")
            for mp in res["masked_parts"]:
                print(f"  [{mp['id']}] {_preview(mp['masked_text'])}")

            # 漏れチェック：mapping の原文表層が masked_text に残っていないか（＝マスク漏れ）。
            surfaces = {s for m in res["mapping"] for s in m["surfaces"]}
            bundle = "\n".join(mp["masked_text"] for mp in res["masked_parts"])
            leaked = sorted(s for s in surfaces if s and s in bundle)
            print(
                "\n■ 漏れチェック（マスク対象が伏せ字後に残っていないか）: "
                + ("OK（残存なし）" if not leaked else f"NG 残存={leaked}")
            )

            print(
                "\n-- 対応表（バンドル共有・同じ実体は全 part で同じプレースホルダ）--"
            )
            _print_mapping(res["mapping"])

            print("\n-- pending（閾値未満のレビュー候補）--")
            pending = res.get("pending", [])
            print(f"  {len(pending)} 件" if pending else "  （なし）")

            # (A) LLM はマスク済みだけを見る → 応答を共有 mapping で 1 回で復元。
            answer = mock_llm(bundle)
            print(f"\n■ (mock) LLM 応答（伏せ字のまま）\n  {answer}")
            restored = client.unmask(answer, res["mapping"])["restored_text"]
            print(f"■ /unmask (A) 応答を復元\n  {restored}")

            # (B) 渡した各 part も同じ mapping で復元できる（part ごとに 1 回）。
            print("\n■ /unmask (B) 各 part を復元（プレースホルダ数 マスク後→復元後）")
            for mp in res["masked_parts"]:
                back = client.unmask(mp["masked_text"], res["mapping"])["restored_text"]
                n0 = len(_PLACEHOLDER.findall(mp["masked_text"]))
                n1 = len(_PLACEHOLDER.findall(back))
                print(f"  [{mp['id']}] {n0} → {n1}   {_preview(back, 120)}")
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
