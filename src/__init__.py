"""src パッケージ初期化。

**最初に .env を読み込む**。各モジュールは import 時に ``os.getenv`` で設定を読む
（例: ``src.ner.engine`` の ``NER_PIPE_BATCH_CHARS``、``src.llm`` の ``LLM_DETECT_*`` /
``LLM_WINDOW_*`` など）。したがって .env は**どの src サブモジュールより先に** os.environ へ
流し込む必要がある。``import src.X`` は必ず本 ``src/__init__.py`` を先に実行するので、
ここに置けば全入口（app.py / cli.py / テスト）で確実に効く。

以前は ``src.sources.kb_mcp`` の ``load_dotenv`` に依存していたが、それは app.py で
engine 等より**後**に import されるため、.env の値が import 時読みに間に合わず効かなかった
（NER_PIPE_BATCH_CHARS を .env に書いても反映されない不具合）。それをここで根治する。

``override=False``＝シェルで明示した環境変数（``$env:VAR=...``）が .env より優先。
python-dotenv 未導入・.env 不在なら no-op。
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ModuleNotFoundError:  # dotenv 未導入でも import は壊さない
    pass
