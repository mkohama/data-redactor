"""UI の処理中ロック (重い処理の実行中は操作させない) の回帰テスト。

Streamlit を実際に描画して確かめる (streamlit.testing の AppTest。ブラウザは要らない)。
API サーバは要らない。届かない URL を指すので接続は失敗するが、ロックの成立と解除は確かめられる。

確かめること:
  - 何も実行していないときはウィジェットを操作できる。
  - 実行中のフレームでは、モード/入力方法のラジオも各ボタンも操作できない
    (タブやステージを切り替えられると、走っている処理が打ち切られるため)。
  - 実行する場所が画面に無い予約は自動で解除される (押したまま操作不能で固まらない)。
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

APP = "src/ui/app.py"


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """API サーバに依存しない UI (接続先を届かない URL にする)。"""
    monkeypatch.setenv("MASK_API_URL", "http://127.0.0.1:9")
    return AppTest.from_file(APP, default_timeout=120)


def _all_widgets(at: AppTest) -> list:
    kinds = ("radio", "button", "toggle", "text_area", "text_input")
    return [w for kind in kinds for w in getattr(at, kind)]


def _navigation(at: AppTest) -> list:
    """モード/入力方法の切替と設定。ロック中はこれらを操作できてはいけない。

    読み込みボタンは対象にしない。ロックとは別の理由 (サーバに NER モデルが無い) でも
    disabled になるので、ロックの判定に使えない。
    """
    return [*at.radio, *at.toggle, *at.text_area]


def test_idle_widgets_are_operable(app: AppTest) -> None:
    at = app.run()
    assert not at.exception
    widgets = _navigation(at)
    assert widgets, "ウィジェットが 1 つも描かれていない"
    assert all(not w.disabled for w in widgets)


def test_widgets_are_locked_while_running(app: AppTest) -> None:
    # 予約した直後 (実行フレームに入る前) の状態を作ると、ロックした画面の描画が残る。
    # ジョブ ID は今の画面が拾わないものにして、処理そのものは走らせない。
    app.session_state["busy_job"] = {"id": "テスト用", "payload": {}}
    app.session_state["busy_job_started"] = True
    at = app.run()
    assert not at.exception
    widgets = _all_widgets(at)
    assert widgets
    assert all(w.disabled for w in widgets), [
        w.label for w in widgets if not w.disabled
    ]


def test_job_without_a_place_to_run_unlocks(app: AppTest) -> None:
    # 拾い手のいない予約 (入力を切り替えた・文書が消えた等) はロックしたままにしない。
    app.session_state["busy_job"] = {"id": "テスト用", "payload": {}}
    at = app.run()
    assert not at.exception
    assert "busy_job" not in at.session_state
    assert all(not w.disabled for w in _navigation(at))
