from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from suzero_5dchess import game_record
from suzero_5dchess.game_record import (
    build_record,
    load_record,
    replay_record,
    save_game_record,
    validate_record,
)
from suzero_5dchess.protocol import VariantSpec, VERY_SMALL


def _record(*, reason: str = "aborted") -> dict[str, object]:
    return {
        "format": "5dc.capture-king",
        "version": 1,
        "rules": "capture_king",
        "variant": "very_small",
        "initial": {
            "size": {"x": 4, "y": 4},
            "fen": "[K*2k*/4/4/4:0:0:w]",
            "boards": [{"l": 0, "t": 0, "c": False, "fen": "K2k/4/4/4"}],
            "present": {"t": 0, "c": False},
            "timeline_status": {
                "mandatory_timelines": [],
                "optional_timelines": [0],
                "unplayable_timelines": [],
            },
        },
        "actions": [],
        "result": {"outcome": 0.0, "terminal_reason": reason},
        "meta": {
            "iteration": 3,
            "game_id": 7,
            "backend": "test",
            "total_semimoves": 0,
            "board_limit": 25,
            "final_fen": "[K*2k*/4/4/4:0:0:w]",
            "material_scale": 2.0,
        },
        "replay": {"steps": []},
    }


def test_validate_record_accepts_closed_v1_schema_and_reference_variants() -> None:
    record = _record()
    validate_record(record)
    for variant in ("very_small", "standard", "standard_turn_zero"):
        candidate = deepcopy(record)
        candidate["variant"] = variant
        validate_record(candidate)


def test_validate_record_rejects_strict_rules_and_unknown_fields() -> None:
    strict = _record()
    strict["rules"] = "strict"
    with pytest.raises(ValueError, match="strict rules are not supported"):
        validate_record(strict)

    extra = _record()
    extra["initial"]["path"] = "../../escape"  # type: ignore[index]
    with pytest.raises(ValueError, match="unexpected"):
        validate_record(extra)


def test_validate_record_rejects_bool_integer_nan_and_bad_unmoved_flag() -> None:
    bad_integer = _record()
    bad_integer["meta"]["iteration"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="safe integer"):
        validate_record(bad_integer)

    bad_number = _record()
    bad_number["result"]["outcome"] = float("nan")  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        validate_record(bad_number)

    bad_flag = _record()
    bad_flag["initial"]["fen"] = "[*K2k/4/4/4:0:0:w]"  # type: ignore[index]
    with pytest.raises(ValueError, match="unmoved flag"):
        validate_record(bad_flag)


def test_validate_record_checks_delta_keys_players_and_final_cache() -> None:
    record = _record()
    record["actions"] = [{"type": "submit", "player": 0}]
    record["replay"] = {
        "steps": [
            {
                "upsert": [],
                "remove": [],
                "present": {"t": 1, "c": True},
                "timeline_status": {
                    "mandatory_timelines": [],
                    "optional_timelines": [0],
                    "unplayable_timelines": [],
                },
            }
        ]
    }
    validate_record(record)

    wrong_player = deepcopy(record)
    wrong_player["actions"][0]["player"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="player inconsistent"):
        validate_record(wrong_player)

    missing_remove = deepcopy(record)
    missing_remove["replay"]["steps"][0]["remove"] = [  # type: ignore[index]
        {"l": 9, "t": 0, "c": False}
    ]
    with pytest.raises(ValueError, match="target does not exist"):
        validate_record(missing_remove)

    wrong_final = deepcopy(record)
    wrong_final["meta"]["final_fen"] = "[4/4/4/4:0:0:w]"  # type: ignore[index]
    with pytest.raises(ValueError, match="different positions"):
        validate_record(wrong_final)


def test_load_record_can_do_structural_validation_only(tmp_path: Path) -> None:
    path = tmp_path / "record.5dck"
    path.write_text(json.dumps(_record()), encoding="utf-8")
    assert load_record(path, verify_cache=False)["format"] == "5dc.capture-king"


def _patch_record_builder(monkeypatch: pytest.MonkeyPatch, record: dict[str, object]) -> None:
    compact = deepcopy(record)
    compact["actions"] = [
        "submit" if action["type"] == "submit" else action["from"] + action["to"]
        for action in compact["actions"]
    ]
    del compact["replay"]
    monkeypatch.setattr(game_record, "build_record", lambda *args, **kwargs: deepcopy(compact))


def test_save_maps_hostile_or_unknown_reasons_to_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index, reason in enumerate(("../escape", "..", "C:\\escape", "", "new_reason")):
        record = _record(reason=reason)
        _patch_record_builder(monkeypatch, record)
        path = save_game_record(
            object(), VERY_SMALL, tmp_path / str(index),
            iteration=1, game_id=index, backend="test",
        )
        assert path.parent == tmp_path / str(index) / "games" / "unknown"
        assert path.resolve().is_relative_to((tmp_path / str(index) / "games").resolve())


def test_save_uses_unique_exclusive_paths_without_clobbering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(reason="capture_king")
    _patch_record_builder(monkeypatch, record)
    first = save_game_record(
        object(), VERY_SMALL, tmp_path,
        iteration=2, game_id=4, backend="test",
    )
    second = save_game_record(
        object(), VERY_SMALL, tmp_path,
        iteration=2, game_id=4, backend="test",
    )
    assert first != second
    assert first.read_text(encoding="utf-8")
    assert second.read_text(encoding="utf-8")

    collision_hex = "a" * 32
    collision = first.parent / f"iter0002_game000004_{collision_hex}.gn5d"
    collision.write_text("do not replace", encoding="utf-8")
    values = iter(
        (SimpleNamespace(hex=collision_hex), SimpleNamespace(hex="b" * 32))
    )
    monkeypatch.setattr(game_record.uuid, "uuid4", lambda: next(values))
    third = save_game_record(
        object(), VERY_SMALL, tmp_path,
        iteration=2, game_id=4, backend="test",
    )
    assert collision.read_text(encoding="utf-8") == "do not replace"
    assert third.name.endswith(f"{'b' * 32}.gn5d")


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_save_removes_only_its_partial_file_when_writer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
) -> None:
    record = _record(reason="material")
    _patch_record_builder(monkeypatch, record)
    existing = tmp_path / "games" / "material" / "existing.gn5d"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep", encoding="utf-8")

    def fail_dump(_record: object, handle: object, **_kwargs: object) -> None:
        handle.write("partial")  # type: ignore[attr-defined]
        raise error_type("write interrupted")

    monkeypatch.setattr(game_record.json, "dump", fail_dump)
    with pytest.raises(error_type, match="write interrupted"):
        save_game_record(
            object(), VERY_SMALL, tmp_path,
            iteration=1, game_id=1, backend="test",
        )
    assert existing.read_text(encoding="utf-8") == "keep"
    assert list(existing.parent.glob("*.gn5d")) == [existing]


def test_save_keeps_only_newest_records_in_each_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(reason="material")
    _patch_record_builder(monkeypatch, record)
    monkeypatch.setattr(game_record, "MAX_RECORDS_PER_CATEGORY", 2)

    category = tmp_path / "games" / "material"
    category.mkdir(parents=True)
    oldest = category / "oldest.gn5d"
    newer = category / "newer.gn5d"
    oldest.write_text("oldest", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    os.utime(oldest, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    other_category = tmp_path / "games" / "aborted" / "other.gn5d"
    other_category.parent.mkdir(parents=True)
    other_category.write_text("other", encoding="utf-8")

    saved = save_game_record(
        object(), VERY_SMALL, tmp_path,
        iteration=3, game_id=7, backend="test",
    )

    assert not oldest.exists()
    assert set(category.glob("*.gn5d")) == {newer, saved}
    assert other_category.read_text(encoding="utf-8") == "other"


def test_batched_saves_prune_once_with_prune_game_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(reason="material")
    _patch_record_builder(monkeypatch, record)
    monkeypatch.setattr(game_record, "MAX_RECORDS_PER_CATEGORY", 2)

    category = tmp_path / "games" / "material"
    category.mkdir(parents=True)
    for index in range(3):
        old = category / f"old{index}.gn5d"
        old.write_text("old", encoding="utf-8")
        os.utime(old, ns=(1_000_000_000 + index, 1_000_000_000 + index))

    for game_id in (1, 2, 3):
        save_game_record(
            object(), VERY_SMALL, tmp_path,
            iteration=3, game_id=game_id, backend="test", prune=False,
        )
    # prune=False must not touch the category, even beyond the cap.
    assert len(list(category.glob("*.gn5d"))) == 6

    game_record.prune_game_records(tmp_path)

    kept = sorted(category.glob("*.gn5d"))
    assert len(kept) == 2
    assert all(path.name.startswith("iter0003_") for path in kept)


def test_prune_game_records_ignores_non_category_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(game_record, "MAX_RECORDS_PER_CATEGORY", 1)
    games_root = tmp_path / "games"
    games_root.mkdir()
    stray = games_root / "stray.gn5d"
    stray.write_text("stray", encoding="utf-8")

    game_record.prune_game_records(tmp_path)

    assert stray.read_text(encoding="utf-8") == "stray"


def _v2_record(*, reason: str = "aborted") -> dict[str, object]:
    return {
        "format": "gn5d",
        "version": 2,
        "rules": "capture_king",
        "variant": "very_small",
        "initial": "[K*2k*/4/4/4:0:0:w]",
        "actions": [],
        "result": {"outcome": 0.0, "terminal_reason": reason},
        "meta": {
            "iteration": 3,
            "game_id": 7,
            "backend": "test",
            "board_limit": 25,
            "material_scale": 2.0,
        },
    }


def test_gn5d_schema_accepts_only_eight_integer_moves_and_submit() -> None:
    record = _v2_record()
    record["actions"] = [[0, 0, 1, -2, 3, 3, 0, -1], "submit", "submit"]
    validate_record(record)
    assert set(record) == {
        "format", "version", "rules", "variant", "initial", "actions", "result", "meta",
    }


@pytest.mark.parametrize("action", [
    None, True, 1, "SUBMIT", "move", [], [0] * 7, [0] * 9,
    [True, 0, 1, 0, 1, 0, 1, 0], [0.0, 0, 1, 0, 1, 0, 1, 0],
    [4, 0, 1, 0, 1, 0, 1, 0], [0, -1, 1, 0, 1, 0, 1, 0],
    [0, 0, -1, 0, 1, 0, 1, 0], [0, 0, 1, 2**53, 1, 0, 1, 0],
    {"type": "submit", "player": 0},
])
def test_gn5d_rejects_malformed_actions(action: object) -> None:
    record = _v2_record()
    record["actions"] = [action]
    with pytest.raises(ValueError):
        validate_record(record)


def test_gn5d_rejects_derived_fields_and_extra_data() -> None:
    record = _v2_record()
    record["meta"]["total_semimoves"] = 0
    with pytest.raises(ValueError, match="unexpected"):
        validate_record(record)

    record = _v2_record()
    record["meta"]["final_fen"] = "[K*2k*/4/4/4:0:0:w]"
    with pytest.raises(ValueError, match="unexpected"):
        validate_record(record)

    record = _v2_record()
    record["initial"] = {"fen": "[K*2k*/4/4/4:0:0:w]"}
    with pytest.raises(ValueError, match="expected a string"):
        validate_record(record)

    record = _v2_record()
    record["frames"] = []
    with pytest.raises(ValueError, match="unexpected"):
        validate_record(record)


def _native_empty_record() -> dict[str, object]:
    env = _native_or_skip()
    result = SimpleNamespace(
        moves=[], semimoves=0, turns=0, board_limit=env.board_limit,
        board_count=env.board_count, final_player=env.current_player,
        white_outcome=0.0, terminal_reason="aborted",
        final_fen=env.record_snapshot()["fen"],
    )
    return build_record(result, VERY_SMALL, iteration=0, game_id=0, backend="test")


@pytest.mark.parametrize("verify_cache", [True, False])
def test_gn5d_loading_always_replays_actions(tmp_path: Path, verify_cache: bool) -> None:
    record = _native_empty_record()
    path = tmp_path / "game.gn5d"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_record(path, verify_cache=verify_cache) == record

    # A stationary move is structurally valid, but never legal.
    record["actions"] = [[0, 0, 1, 0, 0, 0, 1, 0]]
    validate_record(record)
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="not legal"):
        load_record(path, verify_cache=verify_cache)


def test_gn5d_replay_checks_result() -> None:
    record = _native_empty_record()
    wrong_reason = deepcopy(record)
    wrong_reason["result"]["terminal_reason"] = "capture_king"
    with pytest.raises(ValueError, match="before its claimed terminal"):
        replay_record(wrong_reason)
    wrong_outcome = deepcopy(record)
    wrong_outcome["result"]["outcome"] = 1.0
    with pytest.raises(ValueError, match="zero outcome"):
        replay_record(wrong_outcome)


def _native_or_skip(
    spec: VariantSpec = VERY_SMALL,
    *,
    board_limit: int = 25,
    material_scale: float = 2.0,
):
    try:
        from suzero_5dchess.env import GameEnv

        env = GameEnv(
            spec, board_limit=board_limit, material_scale=material_scale
        )
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"native 5D chess binding is unavailable: {exc}")
    snapshot = getattr(env, "record_snapshot", None)
    if not callable(snapshot):
        pytest.skip("native record_snapshot API has not been built yet")
    try:
        snapshot()
    except RuntimeError as exc:
        pytest.skip(f"native record_snapshot API has not been built yet: {exc}")
    return env


def _action_dict(player: int, candidate: tuple[bool, tuple[int, ...] | None]) -> dict[str, object]:
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


def test_build_and_replay_deterministic_native_game() -> None:
    env = _native_or_skip(VERY_SMALL, board_limit=25)
    moves: list[dict[str, object]] = []
    semimoves = 0
    turns = 0
    # This is deliberately direct rules play, not model self-play.
    while not env.is_terminal and len(moves) < 12 and semimoves < 6:
        candidates = env.actions
        if not candidates:
            break
        action_id = 0
        action = _action_dict(env.current_player, candidates[action_id])
        moves.append(action)
        env.apply_action(action_id)
        if action["type"] == "submit":
            turns += 1
        else:
            semimoves += 1

    if env.is_terminal:
        outcome = env.white_outcome
        reason = env.terminal_reason
    else:
        outcome = 0.0
        reason = "max_game_length"
    result = SimpleNamespace(
        moves=moves,
        semimoves=semimoves,
        turns=turns,
        board_limit=env.board_limit,
        board_count=env.board_count,
        final_player=env.current_player,
        white_outcome=outcome,
        terminal_reason=reason,
        final_fen=env.record_snapshot()["fen"],
    )
    record = build_record(
        result,
        VERY_SMALL,
        iteration=1,
        game_id=2,
        backend="python",
        compact=False,
    )
    replayed = replay_record(record)
    assert replayed.record_snapshot()["fen"] == result.final_fen
    assert record["meta"]["total_semimoves"] == semimoves  # type: ignore[index]

    # Reference v1 archives predate our optional material_scale field.
    legacy = deepcopy(record)
    del legacy["meta"]["material_scale"]
    assert replay_record(legacy).record_snapshot()["fen"] == result.final_fen
    del legacy["meta"]["final_fen"]
    assert replay_record(legacy).record_snapshot()["fen"] == result.final_fen

    compact = build_record(result, VERY_SMALL, iteration=1, game_id=2, backend="python")
    assert compact["format"] == "gn5d"
    assert compact["version"] == 2
    assert isinstance(compact["initial"], str)
    assert "replay" not in compact
    assert "total_semimoves" not in compact["meta"]  # type: ignore[operator]
    assert "final_fen" not in compact["meta"]  # type: ignore[operator]
    assert compact["actions"] == [
        "submit" if action["type"] == "submit" else action["from"] + action["to"]
        for action in moves
    ]
    assert replay_record(compact).state_key == replayed.state_key

    corrupted = deepcopy(record)
    corrupted["replay"]["steps"][0]["timeline_status"]["mandatory_timelines"] = [999]
    with pytest.raises(ValueError, match="native cache mismatch"):
        replay_record(corrupted)
    with pytest.raises(ValueError, match="semimoves"):
        build_record(
            SimpleNamespace(**{**vars(result), "semimoves": semimoves + 1}),
            VERY_SMALL, iteration=1, game_id=1, backend="test",
        )


def test_final_unsubmitted_capture_is_preserved(tmp_path: Path) -> None:
    spec = VariantSpec(
        name="very_small",
        pgn=(
            '[Size "4x4"]\n[Board "Custom"]\n[Mode "5D"]\n'
            "[K*k*2/4/4/4:0:0:w]\n"
        ),
        board_side=4,
        line_shift=5,
        default_board_limit=25,
        min_board_limit=25,
        max_board_limit=25,
    )
    try:
        env = _native_or_skip(spec)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"custom capture fixture is unsupported by this native build: {exc}")

    capture_id = None
    for action_id, candidate in enumerate(env.actions):
        if candidate[0]:
            continue
        trial = env.clone()
        trial.apply_action(action_id)
        if trial.is_terminal and trial.terminal_reason == "capture_king":
            capture_id = action_id
            break
    if capture_id is None:
        pytest.skip("custom fixture exposes no immediate royal capture")

    action = _action_dict(env.current_player, env.actions[capture_id])
    env.apply_action(capture_id)
    assert env.is_terminal
    result = SimpleNamespace(
        moves=[action],
        semimoves=1,
        turns=0,
        board_limit=env.board_limit,
        board_count=env.board_count,
        final_player=env.current_player,
        white_outcome=env.white_outcome,
        terminal_reason=env.terminal_reason,
        final_fen=env.record_snapshot()["fen"],
    )
    path = save_game_record(
        result, spec, tmp_path,
        iteration=1, game_id=9, backend="test",
    )
    record = load_record(path)
    assert path.suffix == ".gn5d"
    assert record["actions"] == [action["from"] + action["to"]]
    assert "replay" not in record
    assert replay_record(record).is_terminal
    after_terminal = deepcopy(record)
    after_terminal["actions"].append("submit")
    with pytest.raises(ValueError, match="after native terminal"):
        replay_record(after_terminal)


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0])
def test_material_scale_is_preserved_for_terminal_replay(scale: float) -> None:
    # White has an extra rook, so a wrong scale cannot hide behind a draw.
    spec = VariantSpec(
        name="very_small",
        pgn='[Size "4x4"]\n[Board "Custom"]\n[3k/4/4/KR2:0:1:w]\n',
        board_side=4,
    )
    env = _native_or_skip(spec, board_limit=2, material_scale=scale)
    moves: list[dict[str, object]] = []
    semimoves = 0
    turns = 0
    for _ in range(12):
        if env.is_terminal:
            break
        submit_id = next(
            (index for index, candidate in enumerate(env.actions) if candidate[0]),
            None,
        )
        action_id = submit_id if submit_id is not None else 0
        action = _action_dict(env.current_player, env.actions[action_id])
        moves.append(action)
        env.apply_action(action_id)
        if action["type"] == "submit":
            turns += 1
        else:
            semimoves += 1
    if not env.is_terminal or env.terminal_reason != "material":
        pytest.skip("fixture did not reach the native material terminal")

    result = SimpleNamespace(
        moves=moves,
        semimoves=semimoves,
        turns=turns,
        board_limit=env.board_limit,
        board_count=env.board_count,
        final_player=env.current_player,
        white_outcome=env.white_outcome,
        terminal_reason=env.terminal_reason,
        final_fen=env.record_snapshot()["fen"],
    )
    record = build_record(
        result,
        spec,
        iteration=5,
        game_id=13,
        backend="test",
        material_scale=scale,
    )
    assert record["meta"]["material_scale"] == scale  # type: ignore[index]
    replayed = replay_record(record)
    assert replayed.is_terminal
    assert replayed.white_outcome == pytest.approx(result.white_outcome)
    assert (replayed.white_outcome > 0.0) == (scale > 0.0)
    record["meta"]["material_scale"] = scale + 1.0
    with pytest.raises(ValueError, match="native terminal outcome mismatch"):
        replay_record(record)


def test_branch_move_round_trips_by_numeric_coordinates() -> None:
    spec = VariantSpec(
        name="very_small",
        pgn=(
            '[Size "4x4"]\n[Board "Custom"]\n[Mode "5D"]\n'
            "[K*2k*/4/4/4:0:0:w]\n"
            "[K*2k*/4/4/4:0:0:b]\n"
            "[K*2k*/4/4/R*3:0:1:w]\n"
        ),
        board_side=4,
        line_shift=5,
        default_board_limit=25,
        min_board_limit=25,
        max_board_limit=25,
    )
    env = _native_or_skip(spec)
    branch_id = next(
        (
            action_id
            for action_id, (is_submit, move) in enumerate(env.actions)
            if not is_submit
            and move is not None
            and tuple(move[:4]) == (0, 0, 1, 0)
            and tuple(move[4:]) == (0, 0, 0, 0)
        ),
        None,
    )
    if branch_id is None:
        pytest.skip("native engine does not expose the branch fixture move")

    action = _action_dict(env.current_player, env.actions[branch_id])
    env.apply_action(branch_id)
    result = SimpleNamespace(
        moves=[{**action, "action_id": branch_id, "diagnostic": "ignored"}],
        semimoves=1,
        turns=0,
        board_limit=env.board_limit,
        board_count=env.board_count,
        final_player=env.current_player,
        white_outcome=env.white_outcome if env.is_terminal else 0.0,
        terminal_reason=env.terminal_reason if env.is_terminal else "aborted",
        final_fen=env.record_snapshot()["fen"],
    )
    record = build_record(
        result,
        spec,
        iteration=4,
        game_id=12,
        backend="test",
    )
    archived_action = record["actions"][0]  # type: ignore[index]
    assert archived_action == action["from"] + action["to"]
    replayed = replay_record(record)
    assert replayed.record_snapshot()["fen"] == result.final_fen
    assert replayed.board_count == result.board_count
