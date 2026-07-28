# コマンド一覧 (CLI)

統一コマンド `data-redactor` のサブコマンドです (実体は [src/cli.py](../../src/cli.py)。
`uv run main.py <サブコマンド>` でも同じ)。

ふだんの利用は Web UI が中心なので、ここは主に開発・調査で使うものをまとめています。

## 起動

```powershell
# ふだんの起動。API サーバと UI をまとめて起動する
# API が応答できる (GiNZA ロード完了) 状態になってから UI が開くので、初回の接続エラーが出ない
uv run data-redactor dev

# サーバと UI を別々に動かしたいとき
uv run data-redactor serve     # API だけ (既定 http://127.0.0.1:8509)
uv run data-redactor ui        # UI だけ (別ターミナルで serve が動いている前提)
```

## マスキング (CLI から実行する)

ファイル、または `--text` で渡した文字列をマスクします。
辞書は既定で `data/mask_dict.yaml` を自動で読みます。

```powershell
uv run data-redactor mask report.pdf
uv run data-redactor mask --text "本文をここに貼り付け"
uv run data-redactor mask report.docx --out masked.txt   # マスク済みを書き出し
uv run data-redactor mask report.docx --audit            # 候補の票の分布と確信度 (表層なし＝共有可)
uv run data-redactor mask report.docx --audit-surface    # 監査出力に表層も付ける (機密・共有禁止)
uv run data-redactor mask report.docx --no-flatten       # 表の平文化を切る
```

> LLM 検出は現状 **UI (🤖 LLM検出) のみ**です。
> CLI の `mask` は 辞書＋正規表現＋固有表現抽出で動きます。

## 調査・保守

```powershell
# 固有表現抽出の結果を displaCy の HTML (ner.html) で見る
# --open で既定ブラウザ表示・--serve でサーバ表示
uv run data-redactor ner report.pdf --open

# 各トークンの Sudachi 品詞と NER ラベルを並べて観察する (取りこぼしの原因を見る)
uv run data-redactor debug report.pdf --both-models --all-tokens

# 品質ゲート (ruff + mypy)
uv run data-redactor check

# pii-masker (submodule) の更新に追従する。詳細は llm-detection.md
uv run data-redactor sync-pii-masker
```

---

← [README](../../README.md) に戻る
