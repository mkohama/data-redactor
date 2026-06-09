"""入力ソース（アダプタ）。

固有表現抽出エンジンに渡すテキストを、各種ソースから取得する層。
- files       : ローカルファイル（DocumentLoader 経由）
- kb_mcp      : kb-mcp サーバ
- SAMPLE_TEXT : 動作確認用の組み込みサンプル文
"""

from src.sources.files import load_chunks_from_file, load_text_from_file

# 動作確認用サンプル（マスキング検証向け。社名の表記ゆれ＝SONY/Sony/ソニー/canon、
# 同形異義語＝地名「小浜市」と人名「小浜」「荒川」、列挙・誤記・商標 Smash などを含む）。
SAMPLE_TEXT = (
    "昨日、SONY・Nikon・Canon製のカメラの合同展示会が小浜市で開かれた。\n"
    "eXmotion(エクスモ)の永野、末吉、佐伯、小浜は展示会に向かった。\n"
    "末吉はキヤノン製カメラの設定で困っており、サポート窓口に出向いて行った。\n"
    "エクスモーション勤務の佐伯さんは、キヤノン / ニコン / ソニー のサンプル画像を並べて色味を比較した。\n"
    "永野はSony, Nikon, Canonと英文スタイルで列挙したメモを作った。\n"
    "また「canon」と小文字で誤記されたポスターを見て、永野は間違いを指摘した。\n"
    "「Canon」ロゴ入りのケースを持った小浜出身の小浜は、Nikon のブースにも立ち寄った。\n"
    "荒川区在住の荒川さんは、SONY製カメラとCANON製交換レンズの相性を調べていた。\n"
    "Smashマーク入りの限定モデルは、Sony×Canonのコラボ仕様として発表された。"
)

__all__ = ["SAMPLE_TEXT", "load_chunks_from_file", "load_text_from_file"]
