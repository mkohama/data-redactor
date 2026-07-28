# Docker で動かす

API と UI をコンテナで起動する手順と、イメージを作るときの決めごとです。

## 起動手順

解析エンジンを持つ **API** と、表示だけを行う **UI** を別コンテナ・別イメージで動かします
(`data-redactor-api` = torch/GiNZA を持つ重いイメージ、`data-redactor-ui` = spaCy/torch を含まない軽いイメージ)。
UI は `MASK_API_URL=http://data-redactor-api:8509` で API を参照します。`make` は `id -u` を使うため
**Git Bash / WSL 等の POSIX シェル**から実行してください。

```bash
# 1) .env を用意 (LLM 検出を使う場合。.env.example をコピーして実値を入れる)
#    必須は RESOURCE_NAME_GPT41_MINI だけ。KB_MCP_URL / LLM_WINDOW_* / LLM_DETECT_TARGET は
#    既定で動くので、変えたいときだけ設定する
cp .env.example .env

# 2) ビルド (★ 通常はこれ)。pii-masker (submodule) の取得・更新と、
#    それに合わせた検出器の版の書き換えまで面倒を見てからビルドする
make docker-sync-build

# 3) 起動 (デタッチ)。初回は api の torch＋ELECTRA 重み読み込みで時間がかかります
#    api (:8509) が応答できるようになってから ui (:8508) が起動します
make docker-up        # UI → http://localhost:8508 (ホスト 8508 → コンテナ 8501)/ API → :8509

make docker-logs      # ログ追従
make docker-down      # 停止・削除
make clean            # コンテナ＋ボリュームごと削除
```

`make docker-sync-build` は submodule を**追跡ブランチの最新**へ動かします。
リポジトリが記録している版のまま (自分で上げたくない) ビルドしたいときは、次の 2 つで済ませます。

```bash
git submodule update --init   # 記録されている版の pii-masker を取得
make docker-up                # ビルドして起動 (docker-up は --build 付き)
```

## `make docker-sync-build` が何をしているか (ビルド/配布機向け・venv 不要)

```bash
# 追跡ブランチの最新を取り込んで再ビルド
make docker-sync-build

# 特定のコミット/タグ/ブランチに固定して取り込む
make docker-sync-build PII_REF=<commit/tag/branch>
```

`uv run data-redactor sync-pii-masker` と違い **git + perl だけ**で動くため、**ホスト側に `.venv` を作りません**
(`uv run` は `.venv` を自動生成するので、ビルド機ではそれを避けたい)。やることは 2 つだけです。

1. `git submodule update --init --remote external/pii-masker`
   (未取得でも取得から始まる。`PII_REF` 指定時はその版へ `checkout`)
2. `src/detector.py` の `_DETECTOR_STATIC` の `pii-masker@<hash>` を新 HEAD に書き換え
   (こうしないと LLM 検出キャッシュが古い結果を返し続けます)

そのまま `make docker-build` に続きます。イメージに効くのはこの 2 点 (`external/pii-masker/src` の中身と
`src/detector.py` のハッシュ) だけで、どちらも venv 不要だからこう割り切れます。

> **ENE ドリフト検査・ruff/mypy/pytest・呼び出し方の変更の目視は回りません** (それらは `.venv` が要る開発機の作業)。
> ビルド機は「開発機で `sync-pii-masker` → 目視 → commit 済み」の状態を取り込む前提で使ってください。
> コミット済みの状態をそのまま反映するだけなら `git pull && make docker-up` で足ります
> (`docker-sync-build` は、まだコミットされていない submodule 最新を取りに行きたいとき用)。
>
> perl を使うのは `sed -i` が CRLF を LF に潰すのを避けるため (Git Bash 同梱の perl は改行を保持する)。

成果物: [docker/Dockerfile.api](../../docker/Dockerfile.api)・[docker/Dockerfile.ui](../../docker/Dockerfile.ui)・
[docker/requirements-ui.txt](../../docker/requirements-ui.txt)・[.dockerignore](../../.dockerignore)・
[compose.yaml](../../compose.yaml)・[Makefile](../../Makefile)。

ポイント (data-redactor 固有):

- **2 イメージに分ける**：`Dockerfile.api` はエンジン (torch/spaCy/GiNZA/pii-masker/Azure CLI) を持つ重い
  イメージ、`Dockerfile.ui` は UI が実際に import する分だけ (`requirements-ui.txt`＝streamlit/pandas/httpx/
  mcp/pyyaml) の軽いイメージで **spaCy/torch/langchain/openai を含まない**。UI イメージのビルド時スモークで
  「UI が engine 抜きで import でき、重依存が混入しない」ことを保証する (新 UI 依存の載せ忘れをビルドで検知)。
  UI 依存は Docker 専用の最小リストで、ローカル開発 (`uv sync`) には影響しない。
- **Python 3.11 固定** (ja-ginza-electra の制約)。イメージは `python:3.11-slim`。
- **`ja_ginza_electra` はビルド時に読み込んでおく** (torch＋ELECTRA 重みをイメージに焼く)。実行時はネット不要で、
  取りこぼしの少ない electra を既定のまま使える。代償にイメージは数 GB。
  軽さを優先するなら別途 `ja_ginza` を既定にすることを検討。
- **pii-masker (submodule) は `external/pii-masker/src` を COPY** し、`PYTHONPATH=/app` の path-injection
  ([src/llm/_paths.py](../../src/llm/_paths.py)) で `import pii_masker` を解決。`.dockerignore` で `external/` は除外しない。
- **機密データ `./data` はボリュームマウント** (`cache.db` / `mask_dict.yaml` / `mask_allowlist.yaml`＝git 管理外)。
  イメージには焼かない (`.dockerignore` で `data/` を除外)。
- **kb-mcp** はコンテナに載せず `.env` の `KB_MCP_URL` で外部接続。ホスト側 kb-mcp に繋ぐなら
  `localhost` ではなく `host.docker.internal` を使う (Linux は `compose.yaml` の `extra_hosts` を有効化)。
- **認証**は Azure CLI 同梱。ホストの `az login` キャッシュを使うなら `compose.yaml` の
  `~/.azure` マウントを有効化する。
- `.dockerignore` は **リポジトリ直下**に置く (build context が `.` なので、Docker が確実に参照する位置)。

---

---

← [README](../../README.md) に戻る
