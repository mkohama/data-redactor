# 仕組み (検出ロジック・キャッシュ・構成)

中で何が起きているかの説明です。使うだけなら読まなくて構いません。
手を入れるとき、結果に納得がいかないときに読んでください。

## 解析キャッシュ (速度)

解析は **重い層 (GiNZA の 2 モデル・LLM 検出) ** と **軽い層 (辞書照合・確信度づけ) ** に分かれます。
本ツールは **重い層だけをキャッシュ**し、軽い層は毎回計算し直します。

- 固有表現抽出のキーは「内容ハッシュ × モデル × 平文化」、LLM 検出のキーはこれに検出器の版
  (`detector_version`) が加わります。**マスクを確定していなくても、解析した時点で自動保存**されます。
- 同じ文書をもう一度解析すると、重い層を飛ばして一瞬で終わります。
- **辞書・除外リストを変えても、重い層はやり直しになりません** (軽い層だけ計算し直します)。
- 入力方法の **「🗂 キャッシュから選択」** で、保存済み文書をそのまま入力に再利用できます。

> 補足: `src/masking` などの自作モジュールを編集したときは、Streamlit を**再起動**してください
> (`src/ui/app.py` 以外はホットリロードされません)。

---

## 検出ロジックの要点

- **候補を出すチャネル**: マスク辞書と連絡先の正規表現 (常時)、Sudachi 品詞と GiNZA の 2 モデル
  (NER検出 を実行したとき)、LLM (LLM検出 を実行したとき)。**走らせたチャネルだけを集約**します。
- **確信度** (同じカテゴリへ投票した独立チャネルの数で決める):
  - **確定** … 実辞書 (名簿) 一致のみ。自動マスク。
  - **強** … 2 チャネル一致／昇格／連絡先の正規表現一致。自動マスク。
  - **中** … 単独チャネル (LLM 単独など)。要レビュー。
  - **弱** … 地名・その他。要レビュー。
  - **微弱** … コードらしき誤検出 (`Em_NoYes` / `~C02` / `7-410` / 漢字以外の 1 文字 など)。既定で非表示。
    ただし **LLM が識別子 (社員番号/アカウント/IP) と判定したものは免除** (弱で残す＝レビュー可視)。
  - **除外** … 除外リスト一致。既定で非表示。
- **自動マスク対象は 確定／強**。中・弱はレビュー、微弱・除外は確信度フィルタで既定非表示。
- **LLM は「文脈を読む 1 票」**として合流します (LLM だけ→中＝レビュー／固有表現抽出と一致→強)。
  確定は名簿のみで、LLM だけで自動マスクはしません (伏せすぎを避けるため)。
- 伏せ字 (プレースホルダ) は**表記ごと**に振ります (同じ表記は同じ番号、表記が違えば別番号)。
  復元は元の表記に正確に戻ります (辞書で `置換` を指定した語だけは 1 つの固定値に統一)。

---

## アーキテクチャ (エンジンと表示層の分離)

エンジン (UI 非依存) と、表示層 (CLI / Streamlit)・入力アダプタを分離しています。
エンジンはライブラリとして再利用できます。

```
src/
  masking/             ← マスキングエンジン (UI 非依存)
    engine.py            MaskingEngine (候補生成→確信度→マスク適用。analyze(run_ner=...) で NER 任意)
    dictionary.py        MaskDictionary (社名・商標・人名の名簿)
    allowlist.py         MaskAllowlist (除外リスト)
    cache.py             NerCache (NER 層 + LLM 検出層キャッシュ・文書インデックス／SQLite)
  ner/                 ← NER エンジン (UI 非依存)
    engine.py            NerEngine / sudachi_analyze_chunks (GiNZA 抜きの軽量トークナイズ)
    preprocess.py        テーブル平文化＋ build_body (spaCy 非依存の本文/オフセット構築)
    rendering.py         displaCy の色マップ・HTML 生成
  llm/                 ← LLM 検出アダプタ (pii-masker を呼ぶ薄い層。任意)
    detect_layer.py      本文を窓に分けて pii-masker に検出させ、全文中の位置に直す／キャッシュ経由の検出
    windows.py           本文を窓に分割 (既定 15000 トークン・重なり 0。env で調整)
    schema.py            LlmSpan / LlmDetection (保存・復元のための型)
    _paths.py            external/pii-masker/src を sys.path へ通す (submodule を import 可能にする)
  sources/             ← 入力アダプタ (チャンクのリストを返す)
    files.py             ファイル → チャンク (DocumentLoader + Splitter)
    kb_mcp.py            kb-mcp からの取得 (分割済みチャンクをそのまま使う)
  core/document/       ← テキスト変換＋チャンク分割 (kb-mcp から移植)
  config.py            ← ChunkingConfig (チャンクサイズ設定)
  detector.py          ← LLM 検出の版・窓ポリシー・run_llm_detection (UI 非依存の共有層)
  api/                 ← マスキング HTTP API (FastAPI サーバ。エンジンを持つのはここだけ)
  client/              ← MaskClient (api へのクライアント。httpx のみ・src 非依存)
  ui/app.py            ← Streamlit UI (api のクライアント。data-redactor ui で起動)
external/pii-masker/   ← git submodule (LLM 検出の本体。コピーせず参照)
                         https://github.com/eiuske-saeki/pii-masker
main.py                ← 後方互換 CLI シム (実体は src/cli.py)
```

---

## チャンク分割について (長文対策)

GiNZA 内部の SudachiPy は **1 回の解析で 49,149 バイト (≒16,000 文字弱) まで**しか扱えず、
長文を丸ごと渡すと `SudachiError: Input is too long` で落ちます。

そこで解析前に `SemanticRAGTextSplitter`
([src/core/document/text_splitter.py](../../src/core/document/text_splitter.py)) で
**ファイルタイプ別にチャンク分割**し、各チャンクの結果を文字位置補正してマージします。
kb-mcp 経由の文書は格納時に分割済みなので、結合せずそのまま使います。

---

## ファイルのテキスト変換

`DocumentLoader` ([src/core/document/document_loader.py](../../src/core/document/document_loader.py)) が
拡張子ごとに最適なローダーへ振り分けます (kb-mcp から移植)。

| 拡張子 | ローダー | 備考 |
| --- | --- | --- |
| `.txt`, `.md` | CustomTextLoader | UTF-8 / Shift-JIS 等を自動判定 |
| `.pdf` | PdfLoader | pdfminer.six で日本語 PDF の文字化けを回避 |
| `.docx` | WordToMarkdownLoader | 見出し・表・リストを Markdown 化 |
| `.xlsx`, `.xlsm`, `.xls` | ExcelToMarkdownLoader | 各シートを Markdown テーブル化 |
| `.pptx` | PowerPointLoader | スライド・表・ノートを抽出 |
| `.html`, `.xml` | Unstructured*Loader | 別途 `uv add unstructured` が必要 |

---

## 表 (テーブル) の扱い

GiNZA は自然文で学習しているため、Markdown のテーブル記法 (`|` 区切り) をそのまま渡すと、
セル内の語を取りこぼします。

そこで検出のときだけ `|` を句読点に直して平文化し ([src/ner/preprocess.py](../../src/ner/preprocess.py))、
**マスクは `|` 入りの原文に当てて体裁を保持**します (平文化は検出専用の内部処理です)。

---

← [README](../../README.md) に戻る
