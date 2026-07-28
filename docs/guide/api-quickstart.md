# マスキング API 呼び出し手順 (実機)

実機のマスキング API を呼ぶ手順です。

LLM に渡す前に機密情報を伏せ字にし、LLM の応答を元に戻します。


前提と接続先:

- 接続先 (API): `http://ap-cdv2890dapoc:8509`
- 参考 (UI・ブラウザ): `http://ap-cdv2890dapoc:8508`
- サーバは Docker で稼働 (ネットワーク公開済み)。プロキシ設定は不要。
- 標準のオプションは `detection = both` (NER と LLM を併用) と `mask_level = strong`。
  以下の例はこの標準設定です。

処理の流れ:

```
プロンプト＋ファイル --(1) /mask--> 伏せ字テキスト(複数) + 対応表(mapping)
                    --(2) 伏せ字テキストを LLM に渡す--> LLM 応答(伏せ字入り)
                    --(3) /unmask (応答 + mapping)--> 元に戻したテキスト
```

---

## A. Python で使う (推奨)

クライアントコードは `src/client/mask_client.py`です。  
依存は httpx だけなので、外部アプリはこのファイルをコピーして使えます。

使用方法の詳しい例 (実行できるデモ) は `examples/roundtrip_demo.py` を参照してください。

以下では簡単な手順を示します。


### 手順 0: クライアントを用意

```python
from src.client import MaskClient   # 外部アプリはファイルをコピーして import を調整

client = MaskClient("http://ap-cdv2890dapoc:8509")
```

### 手順 1: プロンプト＋複数ファイルをまとめてマスクする (`/mask`)

`parts` に「プロンプト (テキスト)」と「ファイル」を並べます。  
ファイルの `content` はパス、または ` (ファイル名, バイト列)`。

1 回の呼び出しでまとめて処理し、対応表 (`mapping`) は全体で 1 つです。

```python
res = client.mask(
    parts=[
        # プロンプト
        {"kind": "text", "content": "この2ファイルを要約して。担当は佐藤。"},   
        # ファイル1 (パス)
        {"kind": "file", "content": "見積.xlsx"},                            
        # ファイル2 (名前, バイト列)
        {"kind": "file", "content": ("議事録.docx", raw_bytes)},             
    ],
    detection="both",
    mask_level="strong",
)
```

`res["masked_parts"]` は入力と同じ順・同じ id で返ります (id 省略時は `p0`,`p1`,`p2`)。

各要素の `masked_text` が伏せ字済みのテキストです。

```python
for mp in res["masked_parts"]:
    print(mp["id"], mp["masked_text"])   # ← これらを LLM に渡す (原文は渡さない)
```

- 同じ表記はどのファイルでも同じ番号 (`SONY` は全ファイルで `[社1]`)。
- `res["mapping"]` が全体で共有の対応表。手順 3 の復元で使うので保持する。

### 手順 2: 伏せ字テキストを LLM に渡す

各 `masked_text` (例: `この2ファイルを要約して。担当は[人物1]。`) を、
いつもの LLM 呼び出しに組み立てて渡します。

**原文は渡しません。**

LLM の応答にはプレースホルダ (`[人物1]` 等) が残ります。

```python
answer = your_llm(res["masked_parts"])   # いつもの LLM 呼び出し (伏せ字のまま処理)
```

### 手順 3: 応答を復元する (`/unmask`)

LLM の応答テキストと、手順 1 の `mapping` を渡します。

`mapping` は全体で 1 つなので使い回します。

```python
restored = client.unmask(answer, res["mapping"])["restored_text"]
```

`mapping` に無いプレースホルダは変更しません
(LLM が勝手に作った語への安全側)。

---

## B. curl で試す (疎通確認・簡易チェック)

### 手順 0: 疎通確認

```bash
curl http://ap-cdv2890dapoc:8509/health
```

期待するレスポンス (`models_ready` が `true` なら準備完了。`false` はロード中＝少し待つ):

```json
{"status":"ok","models_ready":true,"models_loaded":["ja_ginza_electra","ja_ginza"]}
```

### 手順 1: プロンプト＋複数ファイルをマスク (multipart/form-data)

`manifest` (JSON 文字列) に `parts` を書き、
各 part の `id` をキーにファイル本体を同送します。

拡張子でローダーを選ぶため、**ファイル名 (拡張子) は必須**です。

```bash
curl -X POST http://ap-cdv2890dapoc:8509/mask \
  -F 'manifest={"parts":[{"id":"prompt","text":"この2ファイルを要約して。担当は佐藤。"},{"id":"f1","file":{"filename":"見積.xlsx"}},{"id":"f2","file":{"filename":"議事録.docx"}}],"detection":"both","mask_level":"strong"};type=application/json' \
  -F 'f1=@./見積.xlsx' \
  -F 'f2=@./議事録.docx'
```

レスポンス (要点):

```jsonc
{
  "masked_parts": [
    { "id": "prompt", "masked_text": "この2ファイルを要約して。担当は[人物1]。" },
    { "id": "f1", "masked_text": "..." },
    { "id": "f2", "masked_text": "..." }
  ],
  "mapping": [ /* プレースホルダ ↔ 原語の対応表。手順 2 でそのまま使う */ ]
}
```

### 手順 2: 復元 (`/unmask`)

LLM の応答テキストと、手順 1 の `mapping` を渡します。

```bash
curl -X POST http://ap-cdv2890dapoc:8509/unmask \
  -H 'Content-Type: application/json' \
  -d '{"text": "[社1]は[人物1]が担当。", "mapping": [ /* 手順 1 の mapping を貼る */ ]}'
```

```json
{"restored_text": "SONYは佐藤が担当。"}
```

テキスト 1 本だけ試すなら、次でも叩けます。

```bash
curl -X POST http://ap-cdv2890dapoc:8509/mask \
  -H 'Content-Type: application/json' \
  -d '{"text": "担当は佐藤。", "detection": "both"}'
```

ファイルの受け渡しは Python の方が簡単です。

---

## 補足

- **プレースホルダは表記ごと**に振られます。
  - 同じ表記は同じ番号 (`SONY` は何回出ても `[社1]`)。
  - 表記が違えば別番号 (`SONY`→`[社1]`、`Sony`→`[社2]`、`ソニー`→`[社3]`)。
  - 復元は元の表記に戻ります。
- `detection="both"` は LLM (サーバ側の Azure) を使うため、文書が大きいと時間がかかります。
  - タイムアウトが要るなら `MaskClient(..., timeout=秒数)` で調整できます (`None` で無制限)。
- エンドポイントや値の詳しい定義は [api-usage.md](api-usage.md) を参照。
