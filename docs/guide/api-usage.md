# マスキング HTTP API 呼び出しガイド

LLM に渡す前に機密情報を伏せ字にし、LLM の応答を元に戻すための HTTP API です。
検出・マスク・キャッシュはサーバが持ち、呼ぶ側は HTTP でリクエストするだけです。

- ベース URL (既定): `http://127.0.0.1:8509`
- 起動: `uv run data-redactor serve` (Docker は compose の `data-redactor-api` サービス)
- 認証: なし (社内ネットワーク前提)。LLM 検出だけはサーバ側で Azure 認証が要ります (後述)。

---

## 1. 典型フロー

```
原文 --/mask--> 伏せ字テキスト + 対応表(mapping) --> LLM に伏せ字テキストを渡す
LLM 応答(伏せ字入り) + mapping --/unmask--> 元に戻したテキスト
```

- LLM には **`masked_text` だけ**を渡す (原文は渡さない)。
- 応答の復元には **`/mask` が返した `mapping` をそのまま** `/unmask` に渡す。

---

## 2. エンドポイント一覧

| メソッド | パス | 役割 |
|---|---|---|
| GET | `/health` | 死活とモデルのロード状態 |
| GET | `/config` | 既定値・選択肢・`detector_version`・対応拡張子 |
| POST | `/mask` | 入力をマスクし、伏せ字テキストと対応表を返す (主機能) |
| POST | `/unmask` | 対応表で伏せ字を元に戻す |

文書を content_hash で使い回す一連の API もあります
(`/documents` 取込・一覧・削除、`/documents/{h}/analyze`・`/apply`・`/draft`、
`/allowlist`・`/dictionary`)。

主に UI が使うものです。

詳細は `src/api/app.py` と `src/client/mask_client.py` を参照してください。

本書は主機能 `/mask`・`/unmask` に絞ります。

---

## 3. `POST /mask`

入力 (`parts`) をマスクして、伏せ字テキストと対応表を返します。

### リクエスト (application/json)

| フィールド | 型 | 既定 | 説明 |
|---|---|---|---|
| `text` | string | — | テキスト 1 本。`parts:[{"id":"_","text": text}]` の糖衣。`parts` と同時指定は不可 |
| `parts` | array | — | 入力の一覧 (下記)。`text` と同時指定は不可 |
| `detection` | string | `both` | `ner` / `llm` / `both`。`llm`・`both` はサーバ側 Azure 認証が必要 |
| `mask_level` | string | `strong` | 自動で伏せ字にする下限 (下記) |
| `flatten_tables` | bool | `true` | 表を平文化して検出精度を上げる (マスク結果は原文の記号を保つ) |
| `models` | array | 省略 | 使う NER モデル。**省略する**こと (部分指定は 422。サーバのロード済みを使う) |
| `return_pending` | bool | `true` | 下限未満のレビュー候補 (`pending`) を返すか |
| `refresh` | bool | `false` | `true` で解析キャッシュを無視して再解析し、結果で上書き |

`parts` の各要素 (`kind` は 3 種のいずれか):

| kind | 指定するもの | 例 |
|---|---|---|
| `text` | 文字列そのもの | `{"id":"p1","text":"担当は佐藤。"}` |
| `content_hash` | 取込済み文書のハッシュ | `{"id":"p2","content_hash":"9dea…"}` |
| `file` | ファイル本体 (multipart で別送) | 下記 multipart を参照 |

`id` は任意 (省略で `p0`,`p1`,…)。

結果の `masked_parts` と同じ順・同じ id で返ります。

### リクエスト (multipart/form-data・ファイルを送るとき)

`manifest` (JSON 文字列) に `parts` を書き、各 part の `id` をキーにファイル本体を同送します。
拡張子でローダーを選ぶため **ファイル名 (拡張子) は必須**です。

### 伏せる対象の文字列 (masked_text と span の基準)

`masked_text` は「伏せる対象の文字列」の伏せる語だけを伏せ字に置き換えたものです。
`occurrences` の `span` は、伏せる対象の文字列の上の位置 (伏せ字に置き換える前の位置) です。
伏せる対象の文字列は、part の種類で次のように決まります。

| kind | 伏せる対象の文字列 |
|---|---|
| `text` | 送った文字列そのもの |
| `file` (`.txt`・`.md`) | ファイルの中身そのもの (文字コードを判定して文字列に戻すだけ。全角・半角、改行コード、空行、見出し、末尾の改行も変えない) |
| `file` (それ以外：`.docx`・`.pdf`・`.xlsx` など) | ファイルをテキストに変換した文字列 (下記) |
| `content_hash` | 取り込んだときの文字列 (取り込みの方法は上の `text`・`file` と同じ) |

そのため、`text` と `.txt`・`.md` の `file` では、`masked_text` は原文の伏せる語だけを置き換えたものになり、`span` は原文の位置になります。
同じ内容を `text` で送っても `.txt`・`.md` の `file` で送っても、結果は同じです。

`.docx`・`.pdf`・`.xlsx` などはテキストの原文が無いので、テキストに変換してから伏せます。
変換では全角の英数字・記号が半角になる (NFKC 正規化) などの書き換えが入り、ページ・シートなどの単位で分けた文字列を空行 (`\n\n`) でつないだものになります。
`masked_text` と `span` は、この変換した文字列が基準です。

> 2026-11-28 より前は、`.txt`・`.md` の `file` も変換していました (全角が半角になり、`.md` では一部の見出しの行が消えていました)。

### 解析結果のキャッシュ

検出 (NER・LLM) は重いので、サーバは解析結果を保存して使い回します。
使い回すかどうかは part の種類で決まります。

| kind | キャッシュ | 文書一覧 (`GET /documents`) |
|---|---|---|
| `file` | する (同じ内容のファイルは 2 回目から解析しない) | 種別 `api`・ファイル名で出る |
| `content_hash` | する | 取込時のまま |
| `text` | **しない** (毎回解析する) | 出ない |

キャッシュは part ごとです。
何度も送る文書は `file` で、文書ごとに別の part にして送ってください。
テキストの文書は、`.txt` か `.md` のファイル名で送れば、`text` で送ったときと同じ結果のまま、キャッシュが効きます (上の「伏せる対象の文字列」)。
毎回内容が変わるプロンプトなどは `text` で送ります。

キャッシュの鍵は、伏せる対象の文字列 (上の表) から計算したハッシュです。
ファイル名や拡張子は鍵に入りません。
キャッシュするのは検出の結果 (NER・LLM) だけで、辞書・除外リスト・判定の規則は毎回かけ直します。
そのため、辞書を変えると、キャッシュがあっても次の要求から反映されます。
`detector.detector_version` は検出器 (pii-masker の版・LLM の窓の設定・検出対象) の版で、辞書や判定の規則を変えても変わりません。
呼び出し側で結果を保存して使い回す場合の鍵には向きません。

プロンプトと文書を 1 つの part にまとめると、プロンプトが変わるたびに文書ごと解析し直しになります。

### レスポンス

```jsonc
{
  "status": "unconfirmed",
  "masked_parts": [ { "id": "p1", "masked_text": "担当は[人物1]。" } ],
  "mapping": [
    {
      "placeholder": "[人物1]",         // 伏せ字ラベル
      "category": "人名",               // 人名/社名/商標/地名/連絡先/その他
      "canonical": "佐藤",              // 復元先の語 (unmask はここへ戻す)
      "surfaces": ["佐藤"],             // このプレースホルダが指す表記
      "confidence": "strong",           // wire 値 (下記)
      "decided_by": "consensus",
      "occurrences": [ { "part": "p1", "span": [3, 5] } ]
    }
  ],
  "pending": [ /* 下限未満のレビュー候補 (return_pending=true のとき) */ ],
  "detector": { "detection": "both", "models": ["ja_ginza_electra","ja_ginza"],
                "detector_version": "…", "mask_level": "strong" }
}
```

LLM に渡すのは `masked_parts[].masked_text`。復元には `mapping` を使います。

---

## 4. `POST /unmask`

伏せ字を元に戻します。

- リクエスト: `{"text": "...", "mapping": [...]}`
- レスポンス: `{"restored_text": "..."}`

`mapping` は `/mask` の戻り値をそのまま渡します。

**`mapping` に無いプレースホルダは変更しません** (LLM が勝手に作った語への安全側)。

戻したいテキストの数だけ呼びます (同じ `mapping` を使い回す)。

LLM 応答なら 1 回で十分です。

---

## 5. 値の定義

### detection (検出の系統)
| 値 | 意味 |
|---|---|
| `ner` | GiNZA (固有表現) ＋辞書＋正規表現 |
| `llm` | LLM (pii-masker / Azure) ＋辞書＋正規表現 |
| `both` | 両方を合流 (既定) |

`llm` と `both` はサーバ側の Azure 認証が要ります (未認証は 502)。

### mask_level (自動で伏せ字にする下限)
高い順に `certain` > `strong` > `medium` > `weak` > `faint`。
指定した値**以上**の確からしさを自動で伏せ字にし、未満は `pending` (レビュー候補) へ回ります。
既定 `strong` (= `certain`＋`strong` を自動マスク)。

### confidence (`mapping` / `pending` の wire 値)
`certain` (辞書一致)/ `strong` / `medium` / `weak` / `faint` / `excluded` (除外リストで対象外)。

### category (`mapping` / `pending`。wire でも日本語)
`人名` / `社名` / `商標` / `地名` / `連絡先` / `その他`。

---

## 6. プレースホルダと復元の規則

- プレースホルダは **表記 (文字列) ごと**に 1 つ。
  - 同じ表記の全出現は同じプレースホルダ (例: `SONY` が何回出ても `[社1]`)。
  - 表記が違えば別 (例: `SONY`→`[社1]`、`Sony`→`[社2]`、`ソニー`→`[社3]`)。大小・全半角も区別。
- `/unmask` は各プレースホルダを元の表記へ戻すので、**復元は原文と一致**します。
- 例外: マスク辞書で固定の置換語 (`mask:`) を指定した語は、表記ゆれもその 1 つの値に寄り、
  復元は代表表記に統一されます (表記の違いは保存されません)。

---

## 7. エラー (HTTP ステータス)

| コード | 意味 |
|---|---|
| 404 | 未取込の `content_hash` (`/documents` 系で発生) |
| 422 | 不正な入力・未対応の拡張子・`models` の部分指定 |
| 502 | LLM 検出でサーバ側 Azure 認証／接続に失敗 |
| 503 | サーバの NER モデルがまだロード中 (起動直後)。少し待って再送 |

本文の `detail` に理由が入ります。

---

## 8. 最小の呼び出し例

### Python (推奨: `src/client/mask_client.py` をコピーして使う。依存は httpx だけ)

```python
from src.client import MaskClient   # 外部アプリはこのファイルをコピーして import を調整

with MaskClient("http://127.0.0.1:8509") as client:
    res = client.mask(text="担当は佐藤。SONYと比較。", detection="both", mask_level="strong")
    masked = res["masked_parts"][0]["masked_text"]   # ← LLM にはこれを渡す
    answer = your_llm(masked)                          # 伏せ字のまま処理
    restored = client.unmask(answer, res["mapping"])["restored_text"]
```

複数の入力 (プロンプト＋ファイル) を 1 回でマスクする手順は、
[api-quickstart.md](api-quickstart.md) を参照。

`kind` / `content` の詳しい書き方は `examples/README.md`、
動くデモは `examples/roundtrip_demo.py`。

### curl

```bash
# 死活・構成
curl http://127.0.0.1:8509/health
curl http://127.0.0.1:8509/config

# 単一テキストをマスク (標準は detection=both / mask_level=strong)
curl -X POST http://127.0.0.1:8509/mask \
  -H 'Content-Type: application/json' \
  -d '{"text": "担当は佐藤。SONYと比較。", "detection": "both", "mask_level": "strong"}'

# 復元 (mapping は /mask の戻り値をそのまま)
curl -X POST http://127.0.0.1:8509/unmask \
  -H 'Content-Type: application/json' \
  -d '{"text": "[社1]の話。", "mapping": [ /* /mask の mapping */ ]}'
```

---

参照: `src/client/mask_client.py` (クライアント本体・全メソッドの説明) ／
`examples/README.md` (詳しい例・curl・バンドル) ／`examples/roundtrip_demo.py` (実行デモ)。
