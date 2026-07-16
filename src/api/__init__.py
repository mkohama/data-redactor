"""マスキング HTTP API（エンジンのサーバ化・B 案）。

設計は [docs-dev/mask-http-api設計.md]。この層は薄いアダプタに徹し、
検出・マスク・キャッシュの実体は :mod:`src.masking` / :mod:`src.ner` / :mod:`src.llm`
に置く（表示層と同じく「エンジンを呼ぶだけ」）。

- :mod:`src.api.enums`  … 確信度の wire(ASCII) ↔ 内部(日本語) 変換・列挙値
"""
