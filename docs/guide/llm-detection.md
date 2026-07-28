# LLM 検出のセットアップと運用

LLM による文脈判定を使うための準備と、検出結果のキャッシュを正しく作り直すための運用ルールです。
LLM を使わない場合 (辞書＋正規表現＋固有表現抽出) は、この文書は読まなくて構いません。

## 準備 (submodule の取得と接続先の設定)

LLM 検出の中身は別リポジトリの **[pii-masker](https://github.com/eiuske-saeki/pii-masker)** が持っていて、
本リポジトリは git submodule として取り込んだそれを呼ぶだけです
(取り込み先は [external/pii-masker](../../external/pii-masker))。
そのため `uv sync` だけでは動かず、submodule の取得と接続先の設定が要ります。

**渡す先の LLM は、機密情報を入力してよいものに限ります。**
マスク前の原文を LLM に読ませる工程なので、ここを外すと目的 (漏洩防止) が崩れます。
現在の構成では、社内で承認された Azure OpenAI (`gpt-4.1-mini`) を使います。

```powershell
# 1) pii-masker のソースを external/pii-masker に取得 (git submodule)
git submodule update --init
#    ※新規 clone なら `git clone --recurse-submodules <url>` で 1 と同時に取得できます

# 2) 依存をインストール (openai / azure-identity / pydantic などが入る)
uv sync

# 3) .env に接続先を設定 (.env.example をコピーして実値を入れる)
#    RESOURCE_NAME_GPT41_MINI=<Azure リソース名>
#    ※ pii-masker は呼び出し元 (data-redactor) の .env を読むので、**ここ**に置きます
#    ※ 実際に設定が要るのはこの 1 行だけ。他の項目は既定値で動きます

# 4) 認証 (DefaultAzureCredential が使う)
az login
```

**pii-masker の取り込み方**：pii-masker は `[build-system]` を持たないため pip インストールできません。
そこで `src/llm/_paths.py` が `external/pii-masker/src` を `sys.path` に通して `import pii_masker` を解決し、
submodule が未取得なら自動でスキップします (LLM 無しで動作)。
プロンプト・LLM クライアント・位置特定は pii-masker 側にあり、
data-redactor は薄いアダプタ ([src/llm/](../../src/llm/)) から呼ぶだけなので、
**接続先を変えるなら、まず pii-masker 側を見てください。**

> 対象データが別マシンにある場合、本物の文書と `data/cache.db` は実機にしかありません。
> 開発機でも仕組みの動作確認はできますが、LLM を実際に呼ぶには認証と接続先の設定が要ります。

### detector_version の運用ルール (キャッシュ無効化)

LLM 検出キャッシュは ` (content_hash, model, flatten, detector_version)` をキーにします。
**検出結果に影響する設定を変えたら detector_version を変える**——こうするとキャッシュのキーが一致しなくなり、
自動で再検出されます (変え忘れると古い結果が使い回される＝最大の落とし穴)。

detector_version (例 `pii-masker@9d9942e|win15000ov0|tgtall`) は **3 つの版**を `|` 区切りで持ち、
`src/detector.py` の `detector_version()` が合成します。
**変える契機も方法も、3 つそれぞれで別**です。

| 部分 | 変える契機 | 方法 |
|---|---|---|
| `pii-masker@<hash>` | pii-masker (submodule) を更新したとき | `sync-pii-masker` が新ハッシュに**自動置換** (`src/detector.py` の `_DETECTOR_STATIC`) |
| `win…` | 窓ポリシー (窓の大きさ) を変えたいとき | **環境変数を設定するだけ** (下記)。値から `win…` が自動合成され、キャッシュも自動無効化 |
| `tgt…` | LLM 検出対象 (`all`↔`pii`) を変えたいとき | **環境変数を設定するだけ** (下記)。`tgt<target>` が自動合成され、キャッシュは target 別に分かれる |

> 効いているのは「**文字列全体が前回と変わること**」だけです。
> 変わればキャッシュキーが一致せず、再検出になります。
> `win…` には実際の値 (例 `win6000ov400`) が入るので、どのキャッシュがどの窓の設定で作られたか後から分かります。
> どちらも手で数字を上げる必要はありません (ハッシュは自動、`win…` は環境変数から自動)。

> **type とカテゴリの対応表 (`_ENE_TO_CATEGORY`) は detector_version に含めません。**
> この対応づけは、解析 (マージ) のたびに当てる後段の変換だからです。
> LLM 検出キャッシュが保存するのは、LLM が返した `ene_type` そのものなので影響しません。
> よって `_ENE_TO_CATEGORY` を変えても**版を変える必要はありません**。
> 保存済みの検出結果に新しい対応表を当て直すだけなので、次の解析で自動的に反映され、LLM も呼び直しません。

#### 窓ポリシー (窓の大きさ) の調整 — 環境変数

LLM に本文を渡す前の「窓」分割の大きさは **`.env` の環境変数だけで調整**できます (コード編集・コミット不要)。

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `LLM_WINDOW_MAX_TOKENS` | 15000 | 1 窓の上限トークン数。小さくすると窓が増えて呼び出し回数は増えるが、長い文章でモデルが取りこぼす分は減る。15000 ≒ 散文で 1.1〜1.6 万文字/窓 |
| `LLM_WINDOW_OVERLAP_TOKENS` | 0 | 窓間の重なり (**0=重なり無し**。窓の継ぎ目で先行文脈を次窓へ持ち越したいなら 100〜200。窓化は段落境界で割るので実体は切れない) |

値を変えると detector_version の `win…` が自動で変わり、**LLM 検出キャッシュが無効化＝再検出**されます。
値を元に戻せば、元のキャッシュに再びヒットします。
既定値 (`src/llm/windows.py` の `DEFAULT_MAX_TOKENS` / `DEFAULT_OVERLAP_TOKENS`) はコミット済みのベースラインで、
環境変数はそれを上書きするだけです。

> `win…` は、pii-masker の更新有無に関係なく、こちら都合で変える設定です (下の追従手順とは別物)。
> 逆に pii-masker を更新しても、窓の設定を触っていなければ `win…` は変わりません。

#### LLM 検出対象 (target) の切替 — 環境変数

LLM (pii-masker) に「何を抜かせるか」を **`.env` の環境変数だけで切替**できます (コード編集・コミット不要)。

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `LLM_DETECT_TARGET` | `all` | `all`＝人名・社名・商標に絞って作り込んだ 3 種のプロンプト (よく当たる。実際に伏せたいのがこの 3 種なので既定)。`pii`＝全 type を 1 つのプロンプトで拾う従来方式 (地名・連絡先・ID も拾うが当たりはやや落ちる)。有効な値は `all` / `pii` だけで、それ以外は安全側の `all` になる |

値を変えると detector_version の `tgt…` が自動で変わり、**LLM 検出キャッシュが target 別に分かれて自動で無効化＝再検出**されます。
地名・連絡先・ID の類は固有表現抽出や正規表現でも拾えるので、既定の `all` でも
本来伏せたい人名・社名・商標は取りこぼしません。全 type を LLM からも拾いたいときだけ
`LLM_DETECT_TARGET=pii` にします。

> `tgt…` (env) も `win…` と同じくこちら都合で変える設定で、pii-masker の更新有無とは無関係です。

### pii-masker が更新されたら (追従手順)

pii-masker (submodule) を更新するときの手順。それを取り込み、LLM 検出キャッシュを正しく無効化します。
更新を反映する経路は **2 つ**あり、マシンの役割で使い分けます。

- **開発機 (`.venv` あり・検証まで回す) ** → `sync-pii-masker` (下記)。取り込み＋ENE ドリフト検査＋
  ruff/mypy/pytest まで一括で回し、目視して **commit** します。検証はここに属する作業です。
- **ビルド/配布機 (`.venv` を作りたくない) ** → `make docker-sync-build` ([docker.md](docker.md) 参照)。
  git + perl だけで submodule ポインタと `src/detector.py` のハッシュを書き換え、そのままイメージを再ビルドします。
  ホスト側 `.venv` を一切作らず、検証は開発機のコミット済み状態に委ねます。

機械的な部分は **`sync-pii-masker` サブコマンド**が自動化します。

```powershell
# 追跡ブランチの最新へ (特定のコミット/タグにするなら: data-redactor sync-pii-masker <ref>)
uv run data-redactor sync-pii-masker
```

自動で実行されること:

1. submodule のポインタを更新 (`<ref>` 省略時は追跡ブランチの最新)
2. 新 HEAD の短縮ハッシュを取得
3. `src/detector.py` の `_DETECTOR_STATIC` の `pii-masker@<hash>` を書き換え (= LLM 検出キャッシュが
   ` (content_hash, model, flatten, detector_version)` 不一致で**自動ミス→再取得**になる。ここを忘れると
   検出器が変わっても古いキャッシュが使い回される＝最大の落とし穴)
4. **ENE type ドリフト検査** (pii-masker のプロンプトの型 vs `src/masking/engine.py` の
   `_ENE_TO_CATEGORY`)。マップに無い新しい type は「その他」に落ちて取りこぼしにつながるため警告する
5. submodule の変更点 (`targets.py`＝プロンプト/型 / `detector_llm.py` / `schema.py` / `locate.py` 等) を表示
6. `external/pii-masker` と `src/detector.py` を **stage** (コミットはしない)
7. `ruff` / `mypy` / `pytest` を実行

自動化できない (**人手で確認してからコミット**する) 部分:

- 呼び出し方の変更 (`detect` / `locate_all` の戻り値) → [src/llm/](../../src/llm/) のアダプタを修正
- 新しい ENE type が増えていたら → `_ENE_TO_CATEGORY` に追加
  (版を変える必要はなく、次の解析で自動反映。上の運用ルール)
- 実際に LLM を呼べる環境で 🤖 LLM検出 を回し、件数とカテゴリを目視
- 問題なければ `git commit`

> 窓ポリシー (`win…`) は pii-masker 更新では通常触りません。変えるのは windows.py を編集したときで、
> その手順は上の「detector_version の運用ルール」を参照。

> `--no-update` (更新せず現在の HEAD で検査・検証だけ)、`--skip-tests` (ruff/mypy/pytest を省略) も使えます。

---

---

← [README](../../README.md) に戻る
