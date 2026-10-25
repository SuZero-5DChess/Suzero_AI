from __future__ import annotations

from copy import deepcopy
import http.client
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Callable

import pytest

from suzero_5dchess.env import GameEnv
from suzero_5dchess.game_record import build_record, replay_record
from suzero_5dchess.protocol import VERY_SMALL
from suzero_5dchess import ui_server


Record = dict[str, object]
ServerAddress = tuple[str, int, str]


def _canonical_action(
    player: int,
    candidate: tuple[bool, tuple[int, ...] | None],
) -> dict[str, object]:
    is_submit, move = candidate
    if is_submit:
        return {"type": "submit", "player": player}
    assert move is not None
    return {
        "type": "move",
        "player": player,
        "from": list(move[:4]),
        "to": list(move[4:]),
        "promotion": "queen",
    }


@pytest.fixture(scope="module")
def native_records() -> dict[str, Record]:
    """Build both formats from a short game played by the native rules engine."""
    env = GameEnv(VERY_SMALL, board_limit=25)
    moves: list[dict[str, object]] = []
    semimoves = 0
    turns = 0

    for _ in range(4):
        assert not env.is_terminal
        assert env.actions
        action_id = 0
        action = _canonical_action(env.current_player, env.actions[action_id])
        moves.append(action)
        env.apply_action(action_id)
        if action["type"] == "submit":
            turns += 1
        else:
            semimoves += 1

    result = SimpleNamespace(
        moves=moves,
        semimoves=semimoves,
        turns=turns,
        board_limit=env.board_limit,
        board_count=env.board_count,
        final_player=env.current_player,
        white_outcome=env.white_outcome if env.is_terminal else 0.0,
        terminal_reason=env.terminal_reason if env.is_terminal else "aborted",
        final_fen=env.record_snapshot()["fen"],
    )
    common = {
        "iteration": 2,
        "game_id": 3,
        "backend": "ui-test",
    }
    return {
        "gn5d": build_record(result, VERY_SMALL, compact=True, **common),
        "legacy": build_record(result, VERY_SMALL, compact=False, **common),
    }


@pytest.fixture
def replay_server() -> ServerAddress:
    ui_dir = ui_server._project_ui_dir()
    ui_server._check_ui_dir(ui_dir)
    server = ui_server.ReplayHTTPServer(("127.0.0.1", 0), ui_dir)
    thread = threading.Thread(target=server.serve_forever, name="ui-server-test")
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield str(host), int(port), f"127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


def _request(
    address: ServerAddress,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    host, port, expected_host = address
    request_headers = {"Host": expected_host}
    if headers:
        request_headers.update(headers)
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        response_body = response.read()
        return response.status, dict(response.getheaders()), response_body
    finally:
        connection.close()


def _header_only_post(
    address: ServerAddress,
    *headers: tuple[str, str],
) -> tuple[int, bytes]:
    host, port, expected_host = address
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.putrequest(
            "POST", "/api/replay", skip_host=True, skip_accept_encoding=True
        )
        connection.putheader("Host", expected_host)
        for name, value in headers:
            connection.putheader(name, value)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _resolve_record_action(env: GameEnv, raw_action: object) -> int:
    if raw_action == "submit":
        matches = [index for index, candidate in enumerate(env.actions) if candidate[0]]
    else:
        if isinstance(raw_action, dict):
            if raw_action["type"] == "submit":
                matches = [
                    index for index, candidate in enumerate(env.actions) if candidate[0]
                ]
            else:
                move = tuple(raw_action["from"]) + tuple(raw_action["to"])
                matches = [
                    index
                    for index, (is_submit, candidate_move) in enumerate(env.actions)
                    if not is_submit and candidate_move == move
                ]
        else:
            move = tuple(raw_action)
            matches = [
                index
                for index, (is_submit, candidate_move) in enumerate(env.actions)
                if not is_submit and candidate_move == move
            ]
    assert len(matches) == 1
    return matches[0]


def _expected_frames(record: Record) -> list[dict[str, object]]:
    meta = record["meta"]
    assert isinstance(meta, dict)
    env = GameEnv(
        VERY_SMALL,
        board_limit=int(meta["board_limit"]),
        material_scale=float(meta.get("material_scale", 2.0)),
    )

    def frame() -> dict[str, object]:
        snapshot = env.record_snapshot()
        snapshot["current_player"] = env.current_player
        snapshot["is_terminal"] = env.is_terminal
        return snapshot

    frames = [frame()]
    actions = record["actions"]
    assert isinstance(actions, list)
    for action in actions:
        env.apply_action(_resolve_record_action(env, action))
        frames.append(frame())
    return frames


def _json_body(record: object) -> bytes:
    return json.dumps(record, allow_nan=False, separators=(",", ":")).encode("utf-8")


def test_startup_preflight_finds_repository_ui_and_native_resources() -> None:
    expected = Path(ui_server.__file__).resolve().parents[2] / "5dchess_engine-main" / "ui"
    assert ui_server._project_ui_dir() == expected
    assert expected.is_dir()
    assert (expected / "scripts" / "draw.js").is_file()
    assert (expected / "scripts" / "gn5d-worker.js").is_file()
    assert not (expected.parents[1] / "ui").exists()
    ui_server._check_ui_dir(expected)
    ui_server._preflight_native()


def test_engine_page_uses_native_transport_without_replacing_renderer(
    replay_server: ServerAddress,
) -> None:
    status, _, body = _request(replay_server, "GET", "/")
    assert status == 200
    assert b'canvas id="canvas"' in body
    assert b"scripts/gn5d-worker.js" in body
    assert b"window.nativeReplay = true" in body
    assert b"boardStage" not in body
    status, _, draw = _request(replay_server, "GET", "/scripts/draw.js")
    assert status == 200
    assert draw == (ui_server._project_ui_dir() / "scripts" / "draw.js").read_bytes()


def test_initial_position_comes_from_native_turn_zero(replay_server: ServerAddress) -> None:
    status, _, body = _request(replay_server, "GET", "/api/initial")
    assert status == 200
    response = json.loads(body)
    frames = response.pop("frames")
    assert response["variant"] == "standard_turn_zero"
    assert response["actions"] == []
    assert len(frames) == 1
    final = replay_record(response)
    assert frames[0]["fen"] == final.record_snapshot()["fen"]
    assert frames[0]["size"] == {"x": 8, "y": 8}
    assert len(frames[0]["boards"]) == 2


@pytest.mark.parametrize("record_name", ["gn5d", "legacy"])
def test_replay_callback_emits_independent_verified_frames(
    native_records: dict[str, Record], record_name: str
) -> None:
    record = native_records[record_name]
    expected = _expected_frames(record)
    observed: list[dict[str, object]] = []

    def mutate_after_observing(frame: dict[str, object]) -> None:
        observed.append(deepcopy(frame))
        frame["fen"] = "callback mutation"
        frame["current_player"] = 99
        frame["is_terminal"] = not bool(frame["is_terminal"])
        boards = frame["boards"]
        present = frame["present"]
        status = frame["timeline_status"]
        assert isinstance(boards, list)
        assert isinstance(present, dict)
        assert isinstance(status, dict)
        boards.clear()
        present["t"] = 999
        status["mandatory_timelines"] = [999]

    replayed = replay_record(record, on_frame=mutate_after_observing)
    actions = record["actions"]
    meta = record["meta"]
    initial = record["initial"]
    assert isinstance(actions, list)
    assert isinstance(meta, dict)
    assert len(observed) == len(actions) + 1
    assert observed == expected
    if isinstance(initial, str):
        assert observed[0]["fen"] == initial
        final_fen = replayed.record_snapshot()["fen"]
    else:
        assert isinstance(initial, dict)
        assert observed[0]["fen"] == initial["fen"]
        final_fen = meta["final_fen"]
    assert observed[-1]["fen"] == final_fen
    assert replayed.record_snapshot()["fen"] == final_fen


@pytest.mark.parametrize("record_name", ["gn5d", "legacy"])
def test_replay_rejects_illegal_action_wrong_final_fen_and_result(
    native_records: dict[str, Record], record_name: str
) -> None:
    record = native_records[record_name]

    illegal = deepcopy(record)
    illegal_actions = illegal["actions"]
    assert isinstance(illegal_actions, list)
    first = illegal_actions[0]
    if isinstance(first, dict):
        first["to"] = list(first["from"])
    else:
        assert isinstance(first, list)
        illegal_actions[0] = first[:4] + first[:4]
    with pytest.raises(ValueError, match="not legal"):
        replay_record(illegal)

    if not isinstance(record["initial"], str):
        wrong_fen = deepcopy(record)
        wrong_fen_meta = wrong_fen["meta"]
        wrong_fen_initial = wrong_fen["initial"]
        assert isinstance(wrong_fen_meta, dict)
        assert isinstance(wrong_fen_initial, dict)
        wrong_fen_meta["final_fen"] = wrong_fen_initial["fen"]
        with pytest.raises(
            ValueError, match="final native FEN mismatch|final replay frame"
        ):
            replay_record(wrong_fen)

    wrong_outcome = deepcopy(record)
    wrong_outcome_result = wrong_outcome["result"]
    assert isinstance(wrong_outcome_result, dict)
    wrong_outcome_result["outcome"] = 1.0
    with pytest.raises(ValueError, match="truncated record must have zero outcome"):
        replay_record(wrong_outcome)

    wrong_reason = deepcopy(record)
    wrong_reason_result = wrong_reason["result"]
    assert isinstance(wrong_reason_result, dict)
    wrong_reason_result["terminal_reason"] = "capture_king"
    with pytest.raises(ValueError, match="ends before its claimed terminal"):
        replay_record(wrong_reason)


def test_http_health_static_allowlist_and_traversal_does_not_read_outside_ui(
    replay_server: ServerAddress, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui_dir = ui_server._project_ui_dir()
    original_read_bytes = Path.read_bytes
    reads: list[Path] = []

    def guarded_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        assert resolved.is_relative_to(ui_dir.resolve())
        reads.append(resolved)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    status, headers, body = _request(replay_server, "GET", "/api/health")
    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert json.loads(body) == {"ok": True}

    expected_types = {
        "/": "text/html; charset=utf-8",
        "/index.html": "text/html; charset=utf-8",
        "/scripts/main.js": "text/javascript; charset=utf-8",
        "/scripts/draw.js": "text/javascript; charset=utf-8",
        "/scripts/gn5d-worker.js": "text/javascript; charset=utf-8",
        "/scripts/gn5d-replay.js": "text/javascript; charset=utf-8",
        "/styles/main.css": "text/css; charset=utf-8",
        "/assets/king-white.svg": "image/svg+xml",
        "/icons/lucide.svg": "image/svg+xml",
        "/fonts/PlayfairLining-Regular.woff2": "font/woff2",
    }
    for path, content_type in expected_types.items():
        status, headers, body = _request(replay_server, "GET", path)
        assert status == 200
        assert headers["Content-Type"] == content_type
        assert body

    reads_before_rejections = list(reads)
    for path in (
        "/missing",
        "/../README.md",
        "/%2e%2e/README.md",
        "/index.html?path=../README.md",
        "/scripts/../../README.md",
        "/scripts/%2e%2e/%2e%2e/README.md",
        "/scripts/%5c..%5cREADME.md",
        "/test/worker.js",
        "/app.js",
        "/style.css",
    ):
        status, _, body = _request(replay_server, "GET", path)
        assert status == 404
        assert json.loads(body) == {"error": "not found"}
    assert reads == reads_before_rejections


@pytest.mark.parametrize("record_name", ["gn5d", "legacy"])
def test_http_replay_returns_each_native_frame(
    replay_server: ServerAddress,
    native_records: dict[str, Record],
    record_name: str,
) -> None:
    record = native_records[record_name]
    _, port, expected_host = replay_server
    headers = {"Content-Type": "application/json"}
    if record_name == "gn5d":
        headers["Origin"] = f"http://{expected_host}"
    status, response_headers, body = _request(
        replay_server,
        "POST",
        "/api/replay",
        body=_json_body(record),
        headers=headers,
    )
    assert status == 200
    assert response_headers["Content-Type"] == "application/json; charset=utf-8"
    assert int(response_headers["Content-Length"]) == len(body)
    response = json.loads(body)
    assert response == {
        "variant": record["variant"],
        "initial": record["initial"],
        "result": record["result"],
        "meta": record["meta"],
        "actions": record["actions"],
        "frames": _expected_frames(record),
    }
    assert len(response["frames"]) == len(record["actions"]) + 1
    assert port > 0


def test_http_rejects_bad_json_nan_and_illegal_action(
    replay_server: ServerAddress, native_records: dict[str, Record]
) -> None:
    for body in (b"{", b'{"value":NaN}'):
        status, _, response = _request(
            replay_server,
            "POST",
            "/api/replay",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert "invalid JSON" in json.loads(response)["error"]

    illegal = deepcopy(native_records["gn5d"])
    actions = illegal["actions"]
    assert isinstance(actions, list)
    first = actions[0]
    assert isinstance(first, list)
    actions[0] = first[:4] + first[:4]
    status, _, response = _request(
        replay_server,
        "POST",
        "/api/replay",
        body=_json_body(illegal),
        headers={"Content-Type": "application/json"},
    )
    assert status == 400
    assert "not legal" in json.loads(response)["error"]


def test_http_rejects_cross_origin_and_malicious_host(
    replay_server: ServerAddress,
) -> None:
    status, _, body = _request(
        replay_server,
        "GET",
        "/api/health",
        headers={"Origin": "http://evil.example:80"},
    )
    assert status == 403
    assert "cross-origin" in json.loads(body)["error"]

    status, _, body = _request(
        replay_server,
        "GET",
        "/api/health",
        headers={"Host": "evil.example:1234"},
    )
    assert status == 400
    assert "Host" in json.loads(body)["error"]


def test_http_enforces_body_framing_and_size_limits(
    replay_server: ServerAddress, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, body = _header_only_post(replay_server)
    assert status == 411
    assert "Content-Length" in json.loads(body)["error"]

    status, body = _header_only_post(replay_server, ("Transfer-Encoding", "chunked"))
    assert status == 400
    assert "Transfer-Encoding" in json.loads(body)["error"]

    monkeypatch.setattr(ui_server, "_MAX_REQUEST_BYTES", 8)
    status, _, body = _request(
        replay_server,
        "POST",
        "/api/replay",
        body=b'{"more":"than eight bytes"}',
        headers={"Content-Type": "application/json"},
    )
    assert status == 413
    assert "exceeds" in json.loads(body)["error"]
