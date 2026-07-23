"""Streamlit UI (src/api＝サーバ・src/client＝クライアントと対の ui パッケージ)。

`data-redactor ui` (= `streamlit run src/ui/app.py`) で起動する表示層。M5e で
エンジン直呼びを src.client.MaskClient 経由へ置換し、UI も API のクライアントにする。
"""
