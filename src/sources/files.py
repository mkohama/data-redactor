"""ファイルを入力ソースとして扱うアダプタ。

ファイルを解析対象のチャンクのリストにする。読み方は拡張子で 2 通りに分かれる。

.txt・.md (RAW_TEXT_EXTENSIONS):
  ファイルの文字列をそのまま 1 チャンクにする。何も書き換えない
  (全角→半角の正規化・改行の変更・見出しの組み替えをしない)。
  /mask で返す masked_text と span は、このチャンクの上で作る。テキストのファイルは
  呼び出し側に原文があり、伏せ字の位置を原文の位置に対応させて使うので、原文を
  書き換えると対応が崩れるため。text の part と同じ処理になる。

それ以外 (.docx・.pdf・.xlsx・.pptx など):
  テキストの原文が無いので、kb-mcp から移植した DocumentLoader でテキストに変換し、
  同じく kb-mcp のチャンク分割 (SemanticRAGTextSplitter) でチャンクに分ける。
  変換では NFKC 正規化 (全角→半角など) がかかる。masked_text と span は、変換した
  チャンクを空行でつないだ文字列の上で作る。

長い文書でも、チャンクに分けなくてよい。SudachiPy は 1 回に 49,149 バイトまでしか
解析できないが、解析の手前 (src/ner/preprocess.py の _byte_safe_pieces) で、原文を
変えずに小片に分けて解析し、位置を原文に戻している。
"""

from __future__ import annotations

from pathlib import Path

from src.config import ChunkingConfig
from src.core.document.document_loader import DocumentLoader
from src.core.document.text_splitter import SemanticRAGTextSplitter
from src.utils.text_utils import detect_encoding_and_read

# 中身をそのまま 1 チャンクとして読む拡張子。
RAW_TEXT_EXTENSIONS = frozenset({".txt", ".md"})

# テキストのファイルの文字コードを試す順。
_RAW_ENCODINGS = ("utf-8", "cp932", "euc-jp", "utf-16")


def read_text_as_is(file_path: str | Path) -> str:
    """テキストのファイルを、中身を書き換えずに文字列として読む。

    バイト列を文字コードで文字列に戻すだけで、改行コード (\\r\\n など) も、
    先頭の BOM も、末尾の改行もそのまま残す (テキストモードで開くと \\r\\n が
    \\n に変わるので、バイト列から戻す)。

    文字コードは UTF-8、CP932 (Shift_JIS)、EUC-JP、UTF-16 の順に試す。
    どれでも読めないときは detect_encoding_and_read (文字コードの自動判定) に任せる。
    その場合だけは改行コードが \\n にそろうことがある。

    引数:
      file_path: 読むファイル。
    戻り値:
      ファイルの中身の文字列。
    """
    data = Path(file_path).read_bytes()
    for enc in _RAW_ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    text, _enc = detect_encoding_and_read(file_path)
    return text


def load_chunks_from_file(file_path: str | Path) -> list[str]:
    """ファイルを解析対象のチャンクのリストにして返す。

    .txt・.md は中身をそのまま 1 チャンクにする (read_text_as_is)。
    それ以外は DocumentLoader でテキストに変換し、ファイルの種類に応じた
    分割器でチャンクに分ける (PDF・Excel・PowerPoint はページ・シート単位の
    Document を結合せずに分割する)。読み方の違いと理由はモジュールの説明を参照。

    引数:
      file_path: 読むファイル。拡張子で読み方を選ぶ。
    戻り値:
      チャンクの文字列のリスト。
    """
    if Path(file_path).suffix.lower() in RAW_TEXT_EXTENSIONS:
        return [read_text_as_is(file_path)]
    docs = DocumentLoader().load_document(file_path)
    chunks = SemanticRAGTextSplitter(ChunkingConfig()).split_documents(docs)
    return [c.page_content for c in chunks]


def load_text_from_file(file_path: str | Path) -> str:
    """ファイルをテキスト化して 1 本のテキストにまとめて返す (後方互換)。

    チャンクを空行 ("\\n\\n") でつなぐだけ。.txt・.md は中身そのものになる。
    """
    return "\n\n".join(load_chunks_from_file(file_path))
