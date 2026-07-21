# マスキング API クライアントサンプル

`data-redactor serve` で立てたマスキング HTTP API を外部アプリから使うための最小サンプル。

- [mask_client.py](mask_client.py) … Python クライアント `MaskClient`（依存は `httpx` だけ・`src` 非依存）。
  そのままコピーして自分のアプリに組み込める。
- [roundtrip_demo.py](roundtrip_demo.py) … 一連の流れ **`/mask` → LLM 呼び出し（モック）→ `/unmask`** の実演。

設計の正本は [../docs-dev/mask-http-api設計.md](../docs-dev/mask-http-api設計.md) §3-1（/mask）・§3-2（/unmask）。

---

## 1. サーバを起動する

別ターミナルで（モデルロードのため初回は少し待つ）:

```powershell
uv run data-redactor serve                 # 既定 http://127.0.0.1:8000
uv run data-redactor serve --port 8001     # ポート変更
```

`GET /health` が `models_ready: true` になれば NER が使えます。

---

## 2. デモを実行する

```powershell
# テキストだけのバンドル（オフライン完結・手軽）
uv run python examples/roundtrip_demo.py

# ファイルの受け渡しを確認する（examples/sample_data の全ファイルを添付）
uv run python examples/roundtrip_demo.py --files

# 添付ファイルを明示（複数可）／ポート変更／LLM 検出も併用
uv run python examples/roundtrip_demo.py --files path/to/a.xlsx path/to/b.pdf
uv run python examples/roundtrip_demo.py --base-url http://127.0.0.1:8001
uv run python examples/roundtrip_demo.py --detection both    # ← Azure 資格情報が必要
uv run python examples/roundtrip_demo.py --files --refresh   # ← キャッシュ無視で強制再解析
```

`--files` を付けると **file part**（サーバがテキスト化）を含むバンドルを送り、ファイルの
受け渡し・マスク・復元まで確認できます（値を省くと `examples/sample_data/` の中身を使用。
実データを含みうるため同ディレクトリは git 管理外）。
`detection=llm`/`both` は検出に LLM（pii-masker/Azure）を使うため `az login` 等が必要です
（未設定だとサーバが 502 を返します）。既定の `ner` は辞書＋正規表現＋GiNZA だけで動きます。

出力イメージ（バンドル内の複数パートで **同じ実体が同じプレースホルダ**に揃う）:

```
-- マスク済みテキスト（これを LLM に渡す）--
  [prompt] 次の資料を要約して。担当は[人物1]。
  [docA] [社1]の新型センサは[社2]を上回る。
  [docB] 一方[社2]のレンズは堅実との評価。

-- 対応表（バンドル共有・同じ実体は全 part で同じプレースホルダ）--
  [人物1]  <- 佐藤（人名 / medium / ner）  表記: 佐藤
  [社1]    <- SONY（社名 / medium / ner）  表記: SONY
  ...
```

`--files` を付けると末尾で **漏れチェック**（マスク対象が伏せ字後に残っていないか）と、
**各 part の復元**（プレースホルダ数 マスク後→復元後）も表示します。

---

## 3. curl だけで叩く

### 死活・構成

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/config
```

### 単一テキストをマスク（application/json）

```bash
curl -X POST http://127.0.0.1:8000/mask \
  -H 'Content-Type: application/json' \
  -d '{"text": "担当は佐藤。SONYと比較。", "detection": "ner", "mask_level": "medium"}'
```

`text` は `parts:[{"id":"_","text":"…"}]` の糖衣。複数パートをまとめる場合:

```bash
curl -X POST http://127.0.0.1:8000/mask \
  -H 'Content-Type: application/json' \
  -d '{
        "parts": [
          {"id": "prompt", "text": "2 社を比較して。担当は佐藤。"},
          {"id": "docA",   "text": "SONYはCanonを上回った。"}
        ],
        "detection": "ner", "mask_level": "medium"
      }'
```

### キャッシュを無視して強制再解析（refresh）

`refresh: true` を付けると解析キャッシュ（NER/LLM）を無視して再解析し、結果でキャッシュを
上書きする。既定は `false`（同じ内容なら再利用＝速い）。

```bash
curl -X POST http://127.0.0.1:8000/mask \
  -H 'Content-Type: application/json' \
  -d '{"text": "担当は佐藤。SONYと比較。", "detection": "ner", "mask_level": "medium", "refresh": true}'
```

### 同梱ファイルをワンショットでマスク（multipart/form-data）

`manifest`（JSON 文字列）＋ part id をキーにしたファイル本体を同送する。
拡張子でローダーを選ぶため、**ファイル名（拡張子）は必須**。

```bash
curl -X POST http://127.0.0.1:8000/mask \
  -F 'manifest={"parts":[{"id":"fileB","file":{"filename":"memo.txt"}}],"detection":"ner"};type=application/json' \
  -F 'fileB=@./memo.txt'
```

### 復元（/unmask）

LLM の応答テキストと、`/mask` が返した `mapping` を渡す。
mapping に無いプレースホルダは無変更（LLM の捏造への安全側）。

```bash
curl -X POST http://127.0.0.1:8000/unmask \
  -H 'Content-Type: application/json' \
  -d '{"text": "[社1]の話。", "mapping": [ /* /mask の mapping をそのまま */ ]}'
```

---

## 4. Python から使う

呼ぶ側は **`parts`（入力の一覧）だけ**を組み立てる。各 part は
`{"kind": "text"|"file", "content": ...}`。
**JSON か multipart かはクライアントが内部で振り分ける**ので、curl 例のような
「manifest とファイル本体を分けて送る」作業は不要（file の本体も `content` に入れるだけ）。

### 最小（単一テキスト）

```python
from mask_client import MaskClient   # examples/ 内。自分のアプリでは相対 import を調整

with MaskClient("http://127.0.0.1:8000") as client:
    res = client.mask(text="担当は佐藤。SONYと比較。", detection="ner", mask_level="medium")
    masked = res["masked_parts"][0]["masked_text"]   # ← これを LLM に渡す

    answer = your_llm(masked)                         # 実 LLM 呼び出し（伏せ字のまま処理）

    restored = client.unmask(answer, res["mapping"])["restored_text"]
```

### プロンプト＋複数ファイルを 1 回で（＝バンドル）

`text=` の代わりに `parts=` に並べる。file の `content` はパス、または `(名前, バイト列)`。

```python
res = client.mask(parts=[
    {"kind": "text", "content": "この3ファイルを要約して。担当は佐藤。"},
    {"kind": "file", "content": "見積.xlsx"},                 # パス
    {"kind": "file", "content": ("議事録.docx", raw_bytes)},  # or (名前, バイト列)
])

for mp in res["masked_parts"]:      # 入力と同じ順・同じ id で返る
    print(mp["id"], mp["masked_text"])
# 同じ会社はどのファイルでも同じ番号（SONY=[社1]）。res["mapping"] が共有の対応表。
```

### 復元（restore）

`mapping` はバンドル共有の 1 個。**戻したいテキストの数だけ `unmask` を呼ぶ**だけ。

```python
# (A) LLM の応答を戻す。全 part 由来のプレースホルダをまとめて復元（1 回でよい）。
restored = client.unmask(answer, res["mapping"])["restored_text"]

# (B) 渡した文書側を戻したいときも、同じ mapping で part ごとに戻せる。
restored_parts = [
    client.unmask(mp["masked_text"], res["mapping"])["restored_text"]
    for mp in res["masked_parts"]
]
```

`id` は任意（省略時 `p0`,`p1`…）。結果を対応づけたいときは `"id": "見積"` のように付ける。

解析結果は内容ハッシュでキャッシュされ、同じ内容の 2 回目以降は速い。キャッシュを無視して
やり直したいときは `client.mask(..., refresh=True)`（結果でキャッシュを上書きする）。

エラーは `MaskApiError`（`.status_code` / `.detail`）で受け取れる。
主なステータス: 422（不正入力・未対応拡張子）/ 502（LLM 資格情報）/ 503（モデル未ロード）。
