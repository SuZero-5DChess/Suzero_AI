"""Minimal ``.gn5d`` v2 notation, legacy ``.5dck`` archives and native replay.

GN5D v2 stores only one opening FEN, eight-integer moves / ``submit``, result
and five metadata fields. Board size, opening boards, present board, timeline
status, semimove count and the final position are reconstructed through native
rules instead of being persisted. Legacy v1 archives, including
``replay.steps`` drawing caches, remain readable.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import uuid

from .env import GameEnv
from .protocol import VariantSpec

FORMAT = "gn5d"
VERSION = 2
LEGACY_FORMAT = "5dc.capture-king"
LEGACY_VERSION = 1
FILE_EXT = ".gn5d"
MAX_RECORDS_PER_CATEGORY = 5000
RULE_CAPTURE_KING = "capture_king"
RULE_STRICT = "strict"

_VARIANTS = frozenset({"very_small", "standard", "standard_turn_zero"})
_STATUS_KEYS = (
    "mandatory_timelines",
    "optional_timelines",
    "unplayable_timelines",
)
_PIECES = frozenset("KQRBNPWUDSYCkqrbnpwudsyc")
_SAFE_INTEGER_MAX = 2**53 - 1
_KNOWN_TERMINAL_REASONS = frozenset(
    {
        "capture_king",
        "material",
        "checkmate",
        "stalemate",
        "no_legal_action",
        "max_game_length",
        "aborted",
    }
)
_TRUNCATION_REASONS = frozenset({"max_game_length", "aborted"})


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected an array")
    return value


def _string(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}: expected a string")
    if nonempty and not value.strip():
        raise ValueError(f"{label}: expected a non-empty string")
    return value


def _integer(value: object, label: str, minimum: int | None = None) -> int:
    if type(value) is not int or abs(value) > _SAFE_INTEGER_MAX:
        raise ValueError(f"{label}: expected a safe integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label}: must be >= {minimum}")
    return value


def _number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label}: expected a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label}: expected a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}: expected a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label}: must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label}: must be <= {maximum}")
    return number


def _exact_keys(
    value: Mapping[str, object], expected: set[str], label: str,
    *, optional: frozenset[str] = frozenset(),
) -> None:
    actual = set(value)
    if expected - actual or actual - expected - optional:
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected - optional, key=repr)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(f"{label}: invalid fields ({', '.join(details)})")


def _board_key(board: Mapping[str, object], label: str = "board") -> tuple[int, int, bool]:
    l = _integer(board.get("l"), f"{label}.l")
    t = _integer(board.get("t"), f"{label}.t", 0)
    c = board.get("c")
    if type(c) is not bool:
        raise ValueError(f"{label}.c: expected a boolean")
    return l, t, c


def _validate_fen(fen: object, sx: int, sy: int, label: str, *, unmoved: bool) -> str:
    text = _string(fen, label)
    ranks = text.split("/")
    if len(ranks) != sy:
        raise ValueError(f"{label}: expected {sy} ranks")
    for rank in ranks:
        width = 0
        previous_piece: str | None = None
        for char in rank:
            if char in "12345678":
                width += int(char)
                previous_piece = None
            elif char in _PIECES:
                width += 1
                previous_piece = char
            elif (
                char == "*"
                and unmoved
                and previous_piece is not None
                and previous_piece in "KkRrPpWw"
            ):
                previous_piece = None
            else:
                raise ValueError(f"{label}: invalid character or unmoved flag {char!r}")
        if width != sx:
            raise ValueError(f"{label}: expected rank width {sx}")
    return text


def _parse_position_fen(
    value: object, sx: int, sy: int, label: str
) -> dict[tuple[int, int, bool], str]:
    text = _string(value, label, nonempty=True)
    boards: dict[tuple[int, int, bool], str] = {}
    for line_number, raw_line in enumerate(text.strip().splitlines()):
        line = raw_line.strip()
        if not (line.startswith("[") and line.endswith("]")):
            raise ValueError(f"{label}: expected [fen:l:t:c]")
        fields = line[1:-1].split(":")
        if len(fields) != 4:
            raise ValueError(f"{label}: expected four fields")
        board_fen, raw_l, raw_t, raw_c = fields
        try:
            l = int(raw_l)
            t = int(raw_t)
        except ValueError as exc:
            raise ValueError(f"{label}: invalid timeline coordinate") from exc
        _integer(l, f"{label}[{line_number}].l")
        _integer(t, f"{label}[{line_number}].t", 0)
        if raw_c not in ("w", "b"):
            raise ValueError(f"{label}: board color must be w or b")
        key = (l, t, raw_c == "b")
        if key in boards:
            raise ValueError(f"{label}: duplicate board key {key}")
        boards[key] = _validate_fen(
            board_fen, sx, sy, f"{label}[{line_number}].fen", unmoved=True
        )
    return boards


def _rank_width(rank: str, label: str) -> int:
    width = 0
    previous_piece: str | None = None
    for char in rank:
        if char in "12345678":
            width += int(char)
            previous_piece = None
        elif char in _PIECES:
            width += 1
            previous_piece = char
        elif char == "*" and previous_piece in "KkRrPpWw":
            previous_piece = None
        else:
            raise ValueError(f"{label}: invalid character or unmoved flag {char!r}")
    return width


def _position_shape(value: object, label: str) -> tuple[str, int, int]:
    """Validate a single-source GN5D FEN and derive its square board size."""

    text = _string(value, label, nonempty=True)
    lines = [raw.strip() for raw in text.strip().splitlines()]
    if not lines:
        raise ValueError(f"{label}: empty position")
    first = lines[0]
    if not (first.startswith("[") and first.endswith("]")):
        raise ValueError(f"{label}: expected [fen:l:t:c]")
    fields = first[1:-1].split(":")
    if len(fields) != 4:
        raise ValueError(f"{label}: expected four fields")
    ranks = fields[0].split("/")
    sy = len(ranks)
    sx = _rank_width(ranks[0], f"{label}.fen")
    if sx != sy:
        raise ValueError(f"{label}: only square boards are supported")
    if not 1 <= sx <= 8:
        raise ValueError(f"{label}: board size must be within 1..8")
    _parse_position_fen(text, sx, sy, label)
    return text, sx, sy


def _validate_present(value: object, label: str) -> dict[str, Any]:
    present = _object(value, label)
    _exact_keys(present, {"t", "c"}, label)
    _integer(present["t"], f"{label}.t", 0)
    if type(present["c"]) is not bool:
        raise ValueError(f"{label}.c: expected a boolean")
    return present


def _validate_status(value: object, label: str) -> dict[str, Any]:
    status = _object(value, label)
    _exact_keys(status, set(_STATUS_KEYS), label)
    for name in _STATUS_KEYS:
        items = _array(status[name], f"{label}.{name}")
        for index, item in enumerate(items):
            _integer(item, f"{label}.{name}[{index}]")
        if len(items) != len(set(items)):
            raise ValueError(f"{label}.{name}: duplicate timeline")
    return status


def _validate_coordinate(value: object, sx: int, sy: int, label: str) -> tuple[int, ...]:
    coordinate = _array(value, label)
    if len(coordinate) != 4:
        raise ValueError(f"{label}: expected [x,y,t,l]")
    x = _integer(coordinate[0], f"{label}.x", 0)
    y = _integer(coordinate[1], f"{label}.y", 0)
    t = _integer(coordinate[2], f"{label}.t", 0)
    l = _integer(coordinate[3], f"{label}.l")
    if x >= sx or y >= sy:
        raise ValueError(f"{label}: coordinate outside the board")
    return x, y, t, l


def _validate_board(
    value: object, sx: int, sy: int, label: str, *, with_fen: bool
) -> tuple[tuple[int, int, bool], str | None]:
    board = _object(value, label)
    expected = {"l", "t", "c", "fen"} if with_fen else {"l", "t", "c"}
    _exact_keys(board, expected, label)
    key = _board_key(board, label)
    fen = (
        _validate_fen(board["fen"], sx, sy, f"{label}.fen", unmoved=False)
        if with_fen
        else None
    )
    return key, fen


def _cache_board_map(
    boards: Sequence[Mapping[str, object]], label: str = "boards"
) -> dict[tuple[int, int, bool], str]:
    result: dict[tuple[int, int, bool], str] = {}
    for index, board in enumerate(boards):
        key = _board_key(board, f"{label}[{index}]")
        if key in result:
            raise ValueError(f"{label}: duplicate board key {key}")
        result[key] = str(board["fen"])
    return result


def _fen_without_unmoved(fen: str) -> str:
    return fen.replace("*", "")


def _normal_fen(fen: str) -> str:
    return fen.rstrip("\r\n")


def _validate_snapshot_shape(snapshot: object, label: str) -> dict[str, Any]:
    """Validate a native snapshot before using it to build an archive."""

    frame = _object(snapshot, label)
    _exact_keys(frame, {"size", "fen", "boards", "present", "timeline_status"}, label)
    size = _object(frame["size"], f"{label}.size")
    _exact_keys(size, {"x", "y"}, f"{label}.size")
    sx = _integer(size["x"], f"{label}.size.x", 1)
    sy = _integer(size["y"], f"{label}.size.y", 1)
    if sx > 8 or sy > 8:
        raise ValueError(f"{label}.size: maximum is 8x8")
    position = _parse_position_fen(frame["fen"], sx, sy, f"{label}.fen")
    boards = _array(frame["boards"], f"{label}.boards")
    if not boards:
        raise ValueError(f"{label}.boards: empty position")
    board_map: dict[tuple[int, int, bool], str] = {}
    for index, board in enumerate(boards):
        key, fen = _validate_board(board, sx, sy, f"{label}.boards[{index}]", with_fen=True)
        if key in board_map:
            raise ValueError(f"{label}.boards: duplicate board key {key}")
        assert fen is not None
        board_map[key] = fen
    expected = {key: _fen_without_unmoved(fen) for key, fen in position.items()}
    if board_map != expected:
        raise ValueError(f"{label}: fen and boards describe different positions")
    _validate_present(frame["present"], f"{label}.present")
    _validate_status(frame["timeline_status"], f"{label}.timeline_status")
    return frame


_V2_FIELDS = frozenset(
    {"format", "version", "rules", "variant", "initial", "actions", "result", "meta"}
)
_V2_META_KEYS = frozenset(
    {"iteration", "game_id", "backend", "board_limit", "material_scale"}
)


def _record_kind(root: Mapping[str, Any]) -> str:
    fmt = root.get("format")
    if fmt == FORMAT:
        if _integer(root.get("version"), "version") != VERSION:
            raise ValueError("unsupported record format/version")
        return "v2"
    if fmt == LEGACY_FORMAT:
        if _integer(root.get("version"), "version") != LEGACY_VERSION:
            raise ValueError("unsupported record format/version")
        return "v1"
    raise ValueError("unsupported record format/version")


def _validate_header(root: Mapping[str, Any], kind: str) -> tuple[int, int]:
    if root["rules"] == RULE_STRICT:
        raise ValueError("strict rules are not supported; only capture_king is supported")
    if root["rules"] != RULE_CAPTURE_KING:
        raise ValueError("unsupported rules; only capture_king is supported")
    variant = _string(root["variant"], "variant")
    if variant not in _VARIANTS:
        raise ValueError(f"unsupported variant {variant!r}")

    if kind == "v2":
        _, sx, sy = _position_shape(root["initial"], "initial")
        return sx, sy
    initial = _validate_snapshot_shape(root["initial"], "initial")
    return initial["size"]["x"], initial["size"]["y"]


def _validate_move_actions(root: Mapping[str, Any], sx: int, sy: int) -> int:
    actions = _array(root["actions"], "actions")
    for index, action in enumerate(actions):
        if action == "submit":
            continue
        move = _array(action, f"actions[{index}]")
        if len(move) != 8:
            raise ValueError(f"actions[{index}]: expected eight integers or submit")
        _validate_coordinate(move[:4], sx, sy, f"actions[{index}].from")
        _validate_coordinate(move[4:], sx, sy, f"actions[{index}].to")
    return sum(action != "submit" for action in actions)


def validate_record(record: object) -> None:
    """Validate GN5D v2 or legacy v1 structure without invoking native code.

    Only :func:`replay_record` establishes legality and verifies the result.
    GN5D v2 stores no derived snapshot or position cache to trust or validate.
    """

    root = _object(record, "record")
    kind = _record_kind(root)
    if kind == "v2":
        _exact_keys(root, set(_V2_FIELDS), "record")
        sx, sy = _validate_header(root, kind)
        _validate_move_actions(root, sx, sy)
        _validate_result(root)
        _validate_meta_v2(root["meta"])
        return

    legacy = "replay" in root
    fields = set(_V2_FIELDS)
    _exact_keys(root, fields | ({"replay"} if legacy else set()), "record")
    sx, sy = _validate_header(root, kind)
    if not legacy:
        move_count = _validate_move_actions(root, sx, sy)
        _validate_result_meta(root, sx, sy, move_count)
        return

    current_boards = _cache_board_map(root["initial"]["boards"])
    present = root["initial"]["present"]
    actions = _array(root["actions"], "actions")
    replay = _object(root["replay"], "replay")
    _exact_keys(replay, {"steps"}, "replay")
    steps = _array(replay["steps"], "replay.steps")
    if len(actions) != len(steps):
        raise ValueError("actions and replay.steps must have equal lengths")

    current_player = int(present["c"])
    move_count = 0
    for index, (raw_action, raw_step) in enumerate(zip(actions, steps)):
        action = _object(raw_action, f"actions[{index}]")
        action_type = action.get("type")
        if action_type == "move":
            _exact_keys(action, {"type", "player", "from", "to", "promotion"}, f"actions[{index}]")
        elif action_type == "submit":
            _exact_keys(action, {"type", "player"}, f"actions[{index}]")
        else:
            raise ValueError(f"actions[{index}].type: expected move or submit")
        player = _integer(action["player"], f"actions[{index}].player")
        if player not in (0, 1):
            raise ValueError(f"actions[{index}].player: expected 0 or 1")
        if player != current_player:
            raise ValueError(f"actions[{index}]: player inconsistent with previous frame")
        if action_type == "move":
            _validate_coordinate(action["from"], sx, sy, f"actions[{index}].from")
            _validate_coordinate(action["to"], sx, sy, f"actions[{index}].to")
            if action["promotion"] != "queen":
                raise ValueError(f"actions[{index}].promotion: v1 requires queen")
            move_count += 1

        step = _object(raw_step, f"replay.steps[{index}]")
        _exact_keys(
            step,
            {"upsert", "remove", "present", "timeline_status"},
            f"replay.steps[{index}]",
        )
        next_present = _validate_present(step["present"], f"replay.steps[{index}].present")
        _validate_status(step["timeline_status"], f"replay.steps[{index}].timeline_status")
        remove = _array(step["remove"], f"replay.steps[{index}].remove")
        upsert = _array(step["upsert"], f"replay.steps[{index}].upsert")
        remove_keys: list[tuple[int, int, bool]] = []
        for board_index, value in enumerate(remove):
            key, _ = _validate_board(
                value,
                sx,
                sy,
                f"replay.steps[{index}].remove[{board_index}]",
                with_fen=False,
            )
            remove_keys.append(key)
        upsert_values: dict[tuple[int, int, bool], str] = {}
        for board_index, value in enumerate(upsert):
            key, fen = _validate_board(
                value,
                sx,
                sy,
                f"replay.steps[{index}].upsert[{board_index}]",
                with_fen=True,
            )
            if key in upsert_values:
                raise ValueError(f"replay.steps[{index}].upsert: duplicate board key {key}")
            assert fen is not None
            upsert_values[key] = fen
        if len(remove_keys) != len(set(remove_keys)):
            raise ValueError(f"replay.steps[{index}].remove: duplicate board key")
        remove_set = set(remove_keys)
        if not remove_set <= current_boards.keys():
            raise ValueError(f"replay.steps[{index}].remove: target does not exist")
        if remove_set & upsert_values.keys():
            raise ValueError(f"replay.steps[{index}]: remove/upsert conflict")
        for key in remove_set:
            del current_boards[key]
        current_boards.update(upsert_values)
        if not current_boards:
            raise ValueError(f"replay.steps[{index}]: position has no boards")
        current_player = int(next_present["c"])

    _validate_result_meta(root, sx, sy, move_count, current_boards)


def _validate_result(root: Mapping[str, Any]) -> None:
    result = _object(root["result"], "result")
    _exact_keys(result, {"outcome", "terminal_reason"}, "result")
    _number(result["outcome"], "result.outcome", minimum=-1.0, maximum=1.0)
    _string(result["terminal_reason"], "result.terminal_reason")


def _validate_meta_v2(value: object) -> None:
    meta = _object(value, "meta")
    _exact_keys(meta, set(_V2_META_KEYS), "meta")
    _integer(meta["iteration"], "meta.iteration", 0)
    _integer(meta["game_id"], "meta.game_id", 0)
    _string(meta["backend"], "meta.backend")
    _integer(meta["board_limit"], "meta.board_limit", 1)
    _number(meta["material_scale"], "meta.material_scale", minimum=0.0)


def _validate_result_meta(
    root: Mapping[str, Any], sx: int, sy: int, move_count: int,
    current_boards: Mapping[tuple[int, int, bool], str] | None = None,
) -> None:
    _validate_result(root)

    meta = _object(root["meta"], "meta")
    _exact_keys(
        meta,
        {
            "iteration",
            "game_id",
            "backend",
            "total_semimoves",
            "board_limit",
        },
        "meta",
        optional=frozenset({"final_fen", "material_scale"}),
    )
    _integer(meta["iteration"], "meta.iteration", 0)
    _integer(meta["game_id"], "meta.game_id", 0)
    _string(meta["backend"], "meta.backend")
    _integer(meta["total_semimoves"], "meta.total_semimoves", 0)
    _integer(meta["board_limit"], "meta.board_limit", 1)
    # Older v1 producers omitted material_scale (their default was 2.0).
    _number(meta.get("material_scale", 2.0), "meta.material_scale", minimum=0.0)
    final_fen = _string(meta.get("final_fen", ""), "meta.final_fen")
    if final_fen:
        final_fen_boards = _parse_position_fen(final_fen, sx, sy, "meta.final_fen")
        final_cache = {key: _fen_without_unmoved(fen) for key, fen in final_fen_boards.items()}
        if current_boards is not None and current_boards != final_cache:
            raise ValueError("meta.final_fen and final replay frame describe different positions")
    if meta["total_semimoves"] != move_count:
        raise ValueError("meta.total_semimoves differs from move action count")


def _copy_board(board: Mapping[str, object], *, with_fen: bool = True) -> dict[str, object]:
    result: dict[str, object] = {
        "l": int(board["l"]),
        "t": int(board["t"]),
        "c": bool(board["c"]),
    }
    if with_fen:
        result["fen"] = str(board["fen"])
    return result


def _copy_present(present: Mapping[str, object]) -> dict[str, object]:
    return {"t": int(present["t"]), "c": bool(present["c"])}


def _copy_status(status: Mapping[str, object]) -> dict[str, list[int]]:
    return {
        name: [int(item) for item in status[name]]  # type: ignore[index]
        for name in _STATUS_KEYS
    }


def _copy_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    size = snapshot["size"]
    assert isinstance(size, Mapping)
    return {
        "size": {"x": int(size["x"]), "y": int(size["y"])},
        "fen": str(snapshot["fen"]),
        "boards": [_copy_board(board) for board in snapshot["boards"]],  # type: ignore[arg-type]
        "present": _copy_present(snapshot["present"]),  # type: ignore[arg-type]
        "timeline_status": _copy_status(snapshot["timeline_status"]),  # type: ignore[arg-type]
    }


def _snapshot(env: GameEnv, label: str) -> dict[str, object]:
    record_snapshot = getattr(env, "record_snapshot", None)
    if not callable(record_snapshot):
        raise RuntimeError(
            "GameEnv.record_snapshot() is unavailable; rebuild the native binding "
            "with the game-record snapshot API"
        )
    raw = record_snapshot()
    frame = _validate_snapshot_shape(raw, label)
    return _copy_snapshot(frame)


def _board_map(frame: Mapping[str, object]) -> dict[tuple[int, int, bool], str]:
    boards = frame["boards"]
    assert isinstance(boards, Sequence)
    return {
        (int(board["l"]), int(board["t"]), bool(board["c"])): str(board["fen"])
        for board in boards  # type: ignore[union-attr]
    }


def _sorted_keys(
    keys: Sequence[tuple[int, int, bool]] | set[tuple[int, int, bool]],
) -> list[tuple[int, int, bool]]:
    return sorted(keys, key=lambda item: (item[0], item[1], int(item[2])))


def _make_delta(previous: Mapping[str, object], current: Mapping[str, object]) -> dict[str, object]:
    old = _board_map(previous)
    new = _board_map(current)
    changed = [key for key, fen in new.items() if old.get(key) != fen]
    return {
        "upsert": [
            {"l": key[0], "t": key[1], "c": key[2], "fen": new[key]}
            for key in _sorted_keys(changed)
        ],
        "remove": [
            {"l": key[0], "t": key[1], "c": key[2]}
            for key in _sorted_keys(set(old) - set(new))
        ],
        "present": _copy_present(current["present"]),  # type: ignore[arg-type]
        "timeline_status": _copy_status(current["timeline_status"]),  # type: ignore[arg-type]
    }


def _canonical_action(raw: object, index: int) -> dict[str, object]:
    action = _object(raw, f"result.moves[{index}]")
    action_type = action.get("type")
    player = _integer(action.get("player"), f"result.moves[{index}].player")
    if player not in (0, 1):
        raise ValueError(f"result.moves[{index}].player: expected 0 or 1")
    if action_type == "submit":
        return {"type": "submit", "player": player}
    if action_type != "move":
        raise ValueError(
            f"result.moves[{index}]: expected canonical move/submit action; "
            "old action-id history cannot be archived"
        )
    source = action.get("from")
    target = action.get("to")
    if not isinstance(source, (list, tuple)) or not isinstance(
        target, (list, tuple)
    ):
        raise ValueError(f"result.moves[{index}]: move coordinates must be sequences")
    if len(source) != 4 or len(target) != 4:
        raise ValueError(f"result.moves[{index}]: move coordinates must contain four integers")
    move = {
        "type": "move",
        "player": player,
        "from": list(source),
        "to": list(target),
        "promotion": action.get("promotion"),
    }
    if move["promotion"] != "queen":
        raise ValueError(f"result.moves[{index}].promotion: v1 requires queen")
    return move


def _resolve_action(env: GameEnv, action: Mapping[str, object], index: int) -> int:
    if env.is_terminal:
        raise ValueError(f"action {index}: action after native terminal state")
    player = int(action["player"])
    if player != env.current_player:
        raise ValueError(
            f"action {index}: recorded player {player} differs from native player {env.current_player}"
        )
    if action["type"] == "submit":
        matches = [
            candidate
            for candidate, (is_submit, _) in enumerate(env.actions)
            if is_submit
        ]
    else:
        move = tuple(action["from"]) + tuple(action["to"])  # type: ignore[arg-type]
        matches = [
            candidate
            for candidate, (is_submit, candidate_move) in enumerate(env.actions)
            if not is_submit and candidate_move == move
        ]
    if not matches:
        raise ValueError(f"action {index}: action is not legal in the native position")
    if len(matches) != 1:
        raise ValueError(f"action {index}: action has multiple native candidates")
    return matches[0]


def _result_attr(result: object, name: str) -> object:
    if not hasattr(result, name):
        raise ValueError(f"result is missing required {name!r}")
    return getattr(result, name)


def _compare_result(result: object, env: GameEnv, moves: int, submits: int) -> None:
    semimoves = _integer(_result_attr(result, "semimoves"), "result.semimoves", 0)
    turns = _integer(_result_attr(result, "turns"), "result.turns", 0)
    final_player = _integer(_result_attr(result, "final_player"), "result.final_player")
    board_count = _integer(_result_attr(result, "board_count"), "result.board_count", 1)
    board_limit = _integer(_result_attr(result, "board_limit"), "result.board_limit", 1)
    if semimoves != moves:
        raise ValueError("result.semimoves differs from canonical move count")
    if turns != submits:
        raise ValueError("result.turns differs from canonical submit count")
    if final_player != env.current_player:
        raise ValueError("result.final_player differs from native replay")
    if board_count != env.board_count:
        raise ValueError("result.board_count differs from native replay")
    if board_limit != env.board_limit:
        raise ValueError("result.board_limit differs from native replay")

    outcome = _number(
        _result_attr(result, "white_outcome"),
        "result.white_outcome",
        minimum=-1.0,
        maximum=1.0,
    )
    reason = _string(_result_attr(result, "terminal_reason"), "result.terminal_reason")
    if env.is_terminal:
        if reason != env.terminal_reason:
            raise ValueError("result.terminal_reason differs from native replay")
        if not math.isclose(outcome, env.white_outcome, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("result.white_outcome differs from native replay")
    else:
        if reason not in _TRUNCATION_REASONS:
            raise ValueError(
                "action history is non-terminal; only max_game_length or aborted "
                "may describe a truncated game"
            )
        if outcome != 0.0:
            raise ValueError("a truncated game must have zero outcome")


def build_record(
    result: object,
    spec: VariantSpec,
    *,
    iteration: int,
    game_id: int,
    backend: str,
    material_scale: float = 2.0,
    compact: bool = True,
) -> dict[str, object]:
    """Build GN5D by native replay; ``compact=False`` builds a legacy archive."""

    iteration = _integer(iteration, "iteration", 0)
    game_id = _integer(game_id, "game_id", 0)
    backend = _string(backend, "backend")
    scale = _number(material_scale, "material_scale", minimum=0.0)
    if spec.name not in _VARIANTS:
        raise ValueError(f"unsupported variant {spec.name!r}")
    board_limit = _integer(_result_attr(result, "board_limit"), "result.board_limit", 1)
    raw_moves = _result_attr(result, "moves")
    if not isinstance(raw_moves, list):
        raise ValueError("result.moves: expected an array of canonical actions")
    actions = [_canonical_action(action, index) for index, action in enumerate(raw_moves)]

    env = GameEnv(spec, board_limit=board_limit, material_scale=scale)
    initial_frame = _snapshot(env, "initial native snapshot")
    size = initial_frame["size"]
    assert isinstance(size, dict)
    for index, action in enumerate(actions):
        if action["type"] == "move":
            _validate_coordinate(
                action["from"], int(size["x"]), int(size["y"]), f"result.moves[{index}].from"
            )
            _validate_coordinate(
                action["to"], int(size["x"]), int(size["y"]), f"result.moves[{index}].to"
            )
    previous = initial_frame
    steps: list[dict[str, object]] = []
    move_count = 0
    submit_count = 0
    for index, action in enumerate(actions):
        action_id = _resolve_action(env, action, index)
        env.apply_action(action_id)
        if not compact:
            current = _snapshot(env, f"native snapshot after action {index}")
            steps.append(_make_delta(previous, current))
            previous = current
        if action["type"] == "move":
            move_count += 1
        else:
            submit_count += 1

    if compact:
        previous = _snapshot(env, "final native snapshot")
    _compare_result(result, env, move_count, submit_count)
    final_fen = _result_attr(result, "final_fen")
    final_fen = _string(final_fen, "result.final_fen", nonempty=True)
    native_final_fen = str(previous["fen"])
    if _normal_fen(final_fen) != _normal_fen(native_final_fen):
        raise ValueError("result.final_fen differs from native replay")

    record: dict[str, object] = {
        "format": FORMAT if compact else LEGACY_FORMAT,
        "version": VERSION if compact else LEGACY_VERSION,
        "rules": RULE_CAPTURE_KING,
        "variant": spec.name,
        "initial": (
            str(initial_frame["fen"]) if compact else _copy_snapshot(initial_frame)
        ),
        "actions": actions,
        "result": {
            "outcome": float(_result_attr(result, "white_outcome")),
            "terminal_reason": str(_result_attr(result, "terminal_reason")),
        },
        "meta": {
            "iteration": iteration,
            "game_id": game_id,
            "backend": backend,
            "board_limit": board_limit,
            "material_scale": scale,
        },
    }
    if compact:
        record["actions"] = [
            "submit" if action["type"] == "submit"
            else list(action["from"]) + list(action["to"])  # type: ignore[arg-type]
            for action in actions
        ]
    else:
        meta = record["meta"]
        assert isinstance(meta, dict)
        meta["total_semimoves"] = move_count
        meta["final_fen"] = final_fen
        record["replay"] = {"steps": steps}
    validate_record(record)
    return record


def _custom_spec(record: Mapping[str, object]) -> VariantSpec:
    initial = record["initial"]
    if isinstance(initial, str):
        fen, sx, sy = _position_shape(initial, "initial")
        label = "initial"
    else:
        assert isinstance(initial, Mapping)
        size = initial["size"]
        assert isinstance(size, Mapping)
        sx, sy = int(size["x"]), int(size["y"])
        if sx != sy:
            raise ValueError(
                "native replay only supports square recorded boards; structural validation succeeded"
            )
        fen = str(initial["fen"])
        label = "initial.fen"
    pgn = f'[Size "{sx}x{sy}"]\n[Board "Custom"]\n[Mode "5D"]\n{fen.rstrip()}\n'
    timelines = _parse_position_fen(fen, sx, sy, label)
    minimum_l = min(key[0] for key in timelines)
    line_shift = max(5, -minimum_l)
    meta = record["meta"]
    assert isinstance(meta, Mapping)
    board_limit = int(meta["board_limit"])
    return VariantSpec(
        name=str(record["variant"]),
        pgn=pgn,
        board_side=sx,
        line_shift=line_shift,
        default_board_limit=board_limit,
        min_board_limit=board_limit,
        max_board_limit=board_limit,
    )


def _frame_cache_equal(actual: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    return (
        _board_map(actual) == _board_map(expected)
        and actual["present"] == expected["present"]
        and actual["timeline_status"] == expected["timeline_status"]
    )


def _delta_equal(actual: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    actual_upsert = _cache_board_map(actual["upsert"], "actual.upsert")  # type: ignore[arg-type]
    expected_upsert = _cache_board_map(expected["upsert"], "expected.upsert")  # type: ignore[arg-type]
    actual_remove = {
        _board_key(board, "actual.remove") for board in actual["remove"]  # type: ignore[union-attr]
    }
    expected_remove = {
        _board_key(board, "expected.remove") for board in expected["remove"]  # type: ignore[union-attr]
    }
    return (
        actual_upsert == expected_upsert
        and actual_remove == expected_remove
        and actual["present"] == expected["present"]
        and actual["timeline_status"] == expected["timeline_status"]
    )


def replay_record(
    record: object,
    verify_cache: bool = True,
    *,
    on_frame: Callable[[dict[str, object]], None] | None = None,
) -> GameEnv:
    """Reconstruct state from actions, checking legality, final FEN and result.

    ``verify_cache`` only controls legacy drawing-cache checks; GN5D actions
    are always replayed and the initial snapshot is always checked.
    """

    validate_record(record)
    root = record
    assert isinstance(root, dict)
    meta = root["meta"]
    assert isinstance(meta, dict)
    env = GameEnv(
        _custom_spec(root),
        board_limit=int(meta["board_limit"]),
        material_scale=float(meta.get("material_scale", 2.0)),
    )
    previous = _snapshot(env, "initial replay snapshot")
    initial = root["initial"]
    if isinstance(initial, str):
        if _normal_fen(str(previous["fen"])) != _normal_fen(initial):
            raise ValueError("initial native FEN mismatch")
    else:
        assert isinstance(initial, Mapping)
        if _normal_fen(str(previous["fen"])) != _normal_fen(str(initial["fen"])):
            raise ValueError("initial native FEN mismatch")
    legacy = "replay" in root
    if not isinstance(initial, str) and (verify_cache or not legacy):
        if not _frame_cache_equal(previous, initial):
            raise ValueError("initial native cache mismatch")

    def emit_frame(snapshot: Mapping[str, object]) -> None:
        if on_frame is not None:
            frame = _copy_snapshot(snapshot)
            frame["current_player"] = env.current_player
            frame["is_terminal"] = env.is_terminal
            on_frame(frame)

    emit_frame(previous)
    actions = root["actions"]
    assert isinstance(actions, list)
    steps = root["replay"]["steps"] if legacy else []
    for index, raw_action in enumerate(actions):
        if legacy:
            action = raw_action
            assert isinstance(action, dict)
        elif raw_action == "submit":
            action = {"type": "submit", "player": env.current_player}
        else:
            action = {
                "type": "move", "player": env.current_player,
                "from": raw_action[:4], "to": raw_action[4:], "promotion": "queen",
            }
        action_id = _resolve_action(env, action, index)
        env.apply_action(action_id)
        if (verify_cache and legacy) or on_frame is not None:
            current = _snapshot(env, f"replay snapshot after action {index}")
            if verify_cache and legacy:
                actual_delta = _make_delta(previous, current)
                expected_delta = steps[index]
                assert isinstance(expected_delta, dict)
                if not _delta_equal(actual_delta, expected_delta):
                    raise ValueError(f"native cache mismatch at step {index}")
            previous = current
            emit_frame(current)

    if not (verify_cache and legacy) and on_frame is None:
        previous = _snapshot(env, "final replay snapshot")
    final_fen = str(meta.get("final_fen", ""))
    if final_fen and _normal_fen(str(previous["fen"])) != _normal_fen(final_fen):
        raise ValueError("final native FEN mismatch")
    result = root["result"]
    assert isinstance(result, dict)
    if env.is_terminal:
        if env.terminal_reason != result["terminal_reason"]:
            raise ValueError("native terminal reason mismatch")
        if not math.isclose(
            env.white_outcome,
            float(result["outcome"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("native terminal outcome mismatch")
    else:
        if result["terminal_reason"] not in _TRUNCATION_REASONS:
            raise ValueError("record ends before its claimed terminal condition")
        if float(result["outcome"]) != 0.0:
            raise ValueError("a truncated record must have zero outcome")
    return env


def _reason_directory(terminal_reason: object) -> str:
    reason = str(terminal_reason).strip()
    return reason if reason in _KNOWN_TERMINAL_REASONS else "unknown"


def _prune_old_records(directory: Path, preserve: Path | None) -> None:
    # ``os.scandir`` is ~80x faster than ``Path.glob`` here: on Windows the
    # directory entry already carries the stat data, so no per-file syscall
    # (and no filter-driver round trip) is needed for 5000 archives.
    archives: list[tuple[int, str, Path]] = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.endswith(FILE_EXT):
                continue
            try:
                if entry.is_file():
                    archives.append(
                        (entry.stat().st_mtime_ns, entry.name, Path(entry.path))
                    )
            except FileNotFoundError:
                continue

    excess = len(archives) - MAX_RECORDS_PER_CATEGORY
    if excess <= 0:
        return
    archives.sort(key=lambda item: (item[0], item[1]))
    for _, _, path in archives:
        if excess <= 0:
            break
        if path == preserve:
            continue
        path.unlink(missing_ok=True)
        excess -= 1


def prune_game_records(log_dir: str | Path) -> None:
    """Trim every ``games/<reason>`` directory down to the retention cap.

    Batch callers save many archives in one pass and should prune once at the
    end instead of rescanning the category directory after every write.
    """
    games_root = Path(log_dir) / "games"
    if not games_root.is_dir():
        return
    with os.scandir(games_root) as entries:
        directories = [Path(entry.path) for entry in entries if entry.is_dir()]
    for directory in directories:
        _prune_old_records(directory, None)


def save_game_record(
    result: object,
    spec: VariantSpec,
    log_dir: str | Path,
    *,
    iteration: int,
    game_id: int,
    backend: str,
    material_scale: float = 2.0,
    prune: bool = True,
) -> Path:
    """Build and exclusively write one archive below ``games/<reason>``.

    ``prune=False`` skips the per-write retention scan; batch callers should
    instead call :func:`prune_game_records` once after writing everything.
    """

    record = build_record(
        result,
        spec,
        iteration=iteration,
        game_id=game_id,
        backend=backend,
        material_scale=material_scale,
    )
    result_data = record["result"]
    assert isinstance(result_data, dict)
    games_root = Path(log_dir) / "games"
    directory = games_root / _reason_directory(result_data["terminal_reason"])
    directory.mkdir(parents=True, exist_ok=True)

    # UUID collisions are fantastically unlikely, but exclusive creation is the
    # actual no-clobber guarantee.  Retry collisions without touching them.
    for _ in range(100):
        path = directory / (
            f"iter{int(iteration):04d}_game{int(game_id):06d}_{uuid.uuid4().hex}{FILE_EXT}"
        )
        created = False
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                created = True
                json.dump(
                    record,
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
        except FileExistsError:
            if created:
                path.unlink(missing_ok=True)
                raise
            continue
        except BaseException:
            if created:
                path.unlink(missing_ok=True)
            raise
        if prune:
            _prune_old_records(directory, path)
        return path
    raise FileExistsError("could not allocate a unique .gn5d archive path")


def load_record(path: str | Path, *, verify_cache: bool = True) -> dict[str, object]:
    """Load GN5D by replaying its actions; also accept legacy cached archives.

    The returned document remains compact. Use :func:`replay_record` for the
    restored native game state. ``verify_cache=False`` skips native replay only
    for legacy archives, never for cache-free GN5D.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    validate_record(record)
    if verify_cache or "replay" not in record:
        replay_record(record, verify_cache=verify_cache)
    assert isinstance(record, dict)
    return record


__all__ = [
    "FORMAT",
    "MAX_RECORDS_PER_CATEGORY",
    "VERSION",
    "build_record",
    "load_record",
    "prune_game_records",
    "replay_record",
    "save_game_record",
    "validate_record",
]
