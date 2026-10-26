"""Tests for the optional multi-process C++ ONNX self-play backend.

Everything here runs on CPU and without a real ``_5dchess`` extension: the
child binding is faked, and the parent-orchestration tests use small helper
scripts that speak the same request/result directory protocol.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from suzero_5dchess import cpp_selfplay_processes as mp
from suzero_5dchess.config import MctsConfig, SelfPlayConfig
from suzero_5dchess.protocol import VERY_SMALL


# ------------------------------------------------------------------- fixtures


def _raw(
    game_indices: list[int],
    *,
    boards: list[int] | None = None,
    candidates: list[int] | None = None,
    submits: list[bool] | None = None,
    board_code: int = 0,
) -> dict[str, object]:
    """A minimal native ``run_selfplay`` payload with per-sample padding."""

    count = len(game_indices)
    boards = list(boards or [2] * count)
    candidates = list(candidates or [3] * count)
    submits = list(submits or [False] * count)
    assert len(boards) == len(candidates) == len(submits) == count

    max_boards = max(boards)
    max_actions = max(candidates)
    totals = [candidate + int(submit) for candidate, submit in zip(candidates, submits)]
    max_policy = max(totals)

    codes = np.full((count, max_boards, 16), board_code, dtype=np.uint8)
    l_coords = np.zeros((count, max_boards), dtype=np.int64)
    t_coords = np.zeros((count, max_boards), dtype=np.int64)
    policy = np.zeros((count, max_policy), dtype=np.float32)
    for index, total in enumerate(totals):
        policy[index, :total] = 1.0 / total

    game_count = max(game_indices) + 1
    games = [
        {
            "index": index,
            "white_outcome": 0.0,
            "terminal_reason": "draw",
            "semimoves": 1,
            "turns": 1,
            "board_limit": 15,
            "board_count": 1,
            "final_player": 0,
        }
        for index in range(game_count)
    ]
    return {
        "board_codes": codes,
        "l_coords": l_coords,
        "t_coords": t_coords,
        "used_board_counts": np.array(boards, dtype=np.int64),
        "urgency": np.zeros(count, dtype=np.float32),
        "side_to_move": np.zeros(count, dtype=np.int64),
        "actions": np.zeros((count, max_actions, 8), dtype=np.int64),
        "action_counts": np.array(totals, dtype=np.int64),
        "policy": policy,
        "has_submit": np.array(submits, dtype=bool),
        "value": np.zeros(count, dtype=np.float32),
        "player": np.zeros(count, dtype=np.int64),
        "move_count": np.zeros(count, dtype=np.int64),
        "game_index": np.array(game_indices, dtype=np.int64),
        "games": games,
        "stats": {
            "total_seconds": 0.1,
            "setup_seconds": 0.0,
            "batches": 1,
            "rows": 1,
            "max_observed_batch": 1,
            "max_batch_size": 128,
            "server_threads": 1,
        },
    }


class _FakeBinding:
    TREE_POSITION_FORMAT_VERSION = 1

    def __init__(
        self, raw: dict[str, object], *, reject_server_threads: bool = False
    ) -> None:
        self.raw = raw
        self.reject_server_threads = reject_server_threads
        self.calls: list[dict[str, object]] = []

    def run_selfplay(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if self.reject_server_threads and "num_server_threads" in kwargs:
            raise TypeError(
                "run_selfplay() got an unexpected keyword argument "
                "'num_server_threads'"
            )
        progress = kwargs.get("progress")
        if callable(progress):
            games = int(kwargs["num_games"])  # type: ignore[arg-type]
            progress(1, 0)
            progress(games, 0)
        return self.raw


def _make_child(
    root: Path, raw: dict[str, object], name: str
) -> tuple[Path, Path]:
    output_dir = root / name
    output_dir.mkdir()
    games = len(raw["games"])  # type: ignore[arg-type]
    request_path = output_dir / mp.REQUEST_NAME
    request_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "binding_kwargs": {
                    "model_path": str(root / "model.onnx"),
                    "metadata_path": str(root / "model.onnx.json"),
                    "num_games": games,
                    "num_threads": 1,
                    "num_server_threads": 1,
                    "seed": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    return output_dir, request_path


# --------------------------------------------------------------------- child


def test_transport_roundtrip_preserves_mixed_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _raw(
        [0, 0],
        boards=[2, 4],
        candidates=[3, 1],
        submits=[False, True],
        board_code=3,
    )
    output_dir, request = _make_child(tmp_path, raw, "child_00")
    fake = _FakeBinding(raw)
    monkeypatch.setattr(mp, "_load_binding", lambda: fake)

    assert mp.run_worker(request) == 0

    progress = json.loads(
        (output_dir / mp.PROGRESS_NAME).read_text(encoding="utf-8")
    )
    assert progress == {"done": 1, "games": 1}
    payload = json.loads(
        (output_dir / mp.RESULT_NAME).read_text(encoding="utf-8")
    )
    assert payload["status"] == "ok"
    assert payload["samples"] == 2
    assert fake.calls[0]["progress"] is not None

    results = mp._load_child_results(output_dir, 1)
    assert len(results) == 1
    samples = results[0].samples
    assert len(samples) == 2
    assert samples[0].board_codes.shape == (2, 16)
    assert samples[1].board_codes.shape == (4, 16)
    assert samples[0].actions.shape == (3, 8)
    assert samples[1].actions.shape == (1, 8)
    assert samples[0].policy.shape == (3,)
    assert samples[1].policy.shape == (2,)
    assert not samples[0].has_submit
    assert samples[1].has_submit
    assert np.all(samples[0].board_codes == 3)
    assert np.all(samples[1].board_codes == 3)


def test_child_clears_inherited_profile_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _raw([0])
    _, request = _make_child(tmp_path, raw, "child_00")
    monkeypatch.setattr(mp, "_load_binding", lambda: _FakeBinding(raw))
    monkeypatch.setenv(mp.PROFILE_ENV, str(tmp_path / "shared_profile.jsonl"))

    assert mp.run_worker(request) == 0
    assert mp.PROFILE_ENV not in os.environ


def test_child_rejects_duplicate_or_missing_local_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _raw([0])
    raw["games"] = [dict(raw["games"][0], index=1)]  # type: ignore[index]
    output_dir, request = _make_child(tmp_path, raw, "child_00")
    monkeypatch.setattr(mp, "_load_binding", lambda: _FakeBinding(raw))

    assert mp.run_worker(request) == 0
    with pytest.raises(RuntimeError, match="local"):
        mp._load_child_results(output_dir, 1)


# ---------------------------------------------------------------- planning


def test_budget_split_and_deterministic_seed() -> None:
    plan = mp.build_plan(
        processes=2,
        total_workers=5,
        total_server_threads=3,
        games=7,
        seed=123,
    )
    assert [(item.workers, item.games, item.server_threads) for item in plan] == [
        (3, 4, 2),
        (2, 3, 1),
    ]
    assert [item.seed for item in plan] == [
        123,
        mp.process_seed(123, 4),
    ]
    assert plan[1].seed != plan[0].seed
    assert 0 <= plan[1].seed <= mp.UINT64_MASK


@pytest.mark.parametrize("processes,games", [(2, 2), (2, 7), (3, 8), (4, 13)])
@pytest.mark.parametrize("seed", [0, 123, mp.UINT64_MASK])
def test_game_seed_stream_is_unique_and_partition_independent(
    processes: int, games: int, seed: int
) -> None:
    # Match src/5dchess/selfplay_bridge.cpp::seedForGame, including uint64 wrap.
    def native_seed(base: int, index: int) -> int:
        z = (base + mp.SEED_STRIDE * (index + 1)) & mp.UINT64_MASK
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mp.UINT64_MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mp.UINT64_MASK
        return z ^ (z >> 31)

    plan = mp.build_plan(processes, processes, processes, games, seed)
    actual = [
        native_seed(item.seed, local_index)
        for item in plan
        for local_index in range(item.games)
    ]
    assert len(set(actual)) == games
    assert actual == [native_seed(seed, index) for index in range(games)]


def test_games_below_workers_only_activates_min_threads() -> None:
    plan = mp.build_plan(
        processes=2,
        total_workers=5,
        total_server_threads=2,
        games=3,
        seed=0,
    )
    assert sum(item.games for item in plan) == 3
    assert sum(min(item.workers, item.games) for item in plan) == 3
    assert sum(item.workers for item in plan) == 5


@pytest.mark.parametrize(
    "args",
    [
        (2, 1, 2, 2, 0),  # workers < processes
        (2, 2, 1, 2, 0),  # server threads < processes
        (2, 2, 2, 1, 0),  # games < processes
    ],
)
def test_budget_split_rejects_totals_below_processes(
    args: tuple[int, int, int, int, int]
) -> None:
    with pytest.raises(ValueError):
        mp.build_plan(*args)


def test_child_kwargs_do_not_mutate_the_parent_copy() -> None:
    base = {
        "model_path": "model.onnx",
        "num_games": 8,
        "num_threads": 6,
        "seed": 5,
        "provider": "cpu",
    }
    plan = mp.build_plan(2, 6, 2, 8, 5)[0]
    child = mp.child_binding_kwargs(base, plan)

    assert child["num_games"] == plan.games == 4
    assert child["num_threads"] == plan.workers == 3
    assert child["num_server_threads"] == 1
    assert child["seed"] == plan.seed
    assert base["num_games"] == 8
    assert "num_server_threads" not in base


# ------------------------------------------------------- single-process path


def _install_fake_cpp_binding(
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeBinding,
    model_path: Path,
) -> None:
    from suzero_5dchess import cpp_self_play

    monkeypatch.setattr(cpp_self_play, "_5dchess", fake)
    monkeypatch.setattr(cpp_self_play, "_require_binding", lambda: None)
    monkeypatch.setattr(
        cpp_self_play,
        "_export_model",
        lambda _network, _spec, path: path,
    )


def test_default_single_process_does_not_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from suzero_5dchess import cpp_self_play

    fake = _FakeBinding(_raw([0]))
    _install_fake_cpp_binding(monkeypatch, fake, tmp_path / "model.onnx")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("processes=1 must not start worker processes")

    monkeypatch.setattr(mp, "run_self_play_processes", forbidden)
    config = SelfPlayConfig(num_games=1, backend="cpp_onnx")

    results = cpp_self_play.generate_games(
        object(),
        VERY_SMALL,
        MctsConfig(num_simulations=4),
        config,
        num_games=1,
        model_path=tmp_path / "model.onnx",
    )

    assert len(results) == 1
    assert len(fake.calls) == 1
    assert "num_server_threads" not in fake.calls[0]
    assert fake.calls[0]["progress"] is None
    assert fake.calls[0]["num_threads"] == 1


def test_old_extension_still_rejects_non_default_server_threads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from suzero_5dchess import cpp_self_play

    fake = _FakeBinding(_raw([0]), reject_server_threads=True)
    _install_fake_cpp_binding(monkeypatch, fake, tmp_path / "model.onnx")
    config = SelfPlayConfig(
        num_games=1, num_server_threads=2, backend="cpp_onnx"
    )

    with pytest.raises(RuntimeError, match="rebuild"):
        cpp_self_play.generate_games(
            object(),
            VERY_SMALL,
            MctsConfig(num_simulations=4),
            config,
            num_games=1,
            model_path=tmp_path / "model.onnx",
        )


def test_processes_two_delegates_without_progress_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from suzero_5dchess import cpp_self_play

    fake = _FakeBinding(_raw([0]))
    _install_fake_cpp_binding(monkeypatch, fake, tmp_path / "model.onnx")
    monkeypatch.setattr(cpp_self_play, "backend_available", lambda: True)

    captured: dict[str, object] = {}

    def fake_runner(
        kwargs: dict[str, object],
        *,
        processes: int,
        total_server_threads: int,
        progress: object,
    ) -> list[object]:
        captured.update(
            kwargs=dict(kwargs),
            processes=processes,
            total_server_threads=total_server_threads,
            progress=progress,
        )
        return []

    monkeypatch.setattr(mp, "run_self_play_processes", fake_runner)
    advances: list[int] = []

    def advance(done: int) -> None:
        advances.append(done)

    config = SelfPlayConfig(
        num_games=4,
        workers=4,
        num_server_threads=2,
        processes=2,
        backend="cpp_onnx",
    )
    results = cpp_self_play.generate_games(
        object(),
        VERY_SMALL,
        MctsConfig(num_simulations=4),
        config,
        num_games=4,
        model_path=tmp_path / "model.onnx",
        progress=advance,
    )

    assert results == []
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["num_games"] == 4
    assert kwargs["num_threads"] == 4
    assert "progress" not in kwargs
    assert captured["processes"] == 2
    assert captured["total_server_threads"] == 2
    assert captured["progress"] is advance
    assert advances == []


def test_wait_for_children_signals_each_child_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mp, "POLL_SECONDS", 0.0)
    plan = [
        mp.ProcessPlan(index=0, workers=1, games=2, server_threads=1, seed=0),
        mp.ProcessPlan(index=1, workers=1, games=2, server_threads=1, seed=1),
    ]
    child_dirs = [tmp_path / "child_00", tmp_path / "child_01"]
    for child_dir in child_dirs:
        child_dir.mkdir()
        # The parent only signals a child after its first progress line, which
        # is written after the child installed its Python signal handlers.
        (child_dir / mp.PROGRESS_NAME).write_text(
            json.dumps({"done": 0, "games": 2}), encoding="utf-8"
        )
    stop = [False]

    def on_poll(_index: int) -> None:
        stop[0] = True

    procs: list[_FakeProc] = [
        _FakeProc([None, None, 0], on_poll),
        _FakeProc([None, None, 0], on_poll),
    ]
    reported = mp._wait_for_children(
        procs, child_dirs, plan, None, stopped=lambda: stop[0]
    )

    expected = (
        signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
    )
    assert reported == 0
    for proc in procs:
        assert proc.signals == [expected]


def test_wait_for_children_does_not_signal_before_first_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mp, "POLL_SECONDS", 0.0)
    plan = [mp.ProcessPlan(index=0, workers=1, games=2, server_threads=1, seed=0)]
    child_dir = tmp_path / "child_00"
    child_dir.mkdir()
    proc = _FakeProc([None, 0])

    reported = mp._wait_for_children(
        [proc], [child_dir], plan, None, stopped=lambda: True
    )

    # The child exited before writing progress; it must not receive a signal
    # that the default action would turn into an abrupt interpreter kill.
    assert reported == 0
    assert proc.signals == []


def test_stopped_child_may_return_a_short_game_prefix(tmp_path: Path) -> None:
    child = tmp_path / "child_00"
    child.mkdir()
    payload = {
        "status": "ok",
        "streamed": True,
        "stopped": True,
        "requested_games": 3,
        "samples": 2,
        "games": [_streamed_entry(0, 2)],
        "stats": {"rows": 0},
    }
    (child / mp.RESULT_NAME).write_text(json.dumps(payload), encoding="utf-8")

    results = mp._load_child_results(child, 3)
    assert len(results) == 1
    assert results[0].sample_count == 2
    assert results[0].samples == []
    assert results[0].persisted is True


def test_unstopped_child_must_return_every_game(tmp_path: Path) -> None:
    child = tmp_path / "child_00"
    child.mkdir()
    payload = {
        "status": "ok",
        "streamed": True,
        "samples": 2,
        "games": [_streamed_entry(0, 2)],
        "stats": {},
    }
    (child / mp.RESULT_NAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected at most|expected 3"):
        mp._load_child_results(child, 3)


# -------------------------------------------------------------- result merge


def test_two_children_keep_their_own_game_zero_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw0 = _raw([0, 0], board_code=0)
    raw1 = _raw([0, 0], boards=[3, 3], candidates=[2, 2], board_code=1)
    out0, req0 = _make_child(tmp_path, raw0, "child_00")
    out1, req1 = _make_child(tmp_path, raw1, "child_01")

    for request, raw in ((req0, raw0), (req1, raw1)):
        fake = _FakeBinding(raw)
        monkeypatch.setattr(mp, "_load_binding", lambda fake=fake: fake)
        assert mp.run_worker(request) == 0

    merged = mp._load_child_results(out0, 1) + mp._load_child_results(out1, 1)
    assert len(merged) == 2
    assert len(merged[0].samples) == 2
    assert len(merged[1].samples) == 2
    assert all(np.all(sample.board_codes == 0) for sample in merged[0].samples)
    assert all(np.all(sample.board_codes == 1) for sample in merged[1].samples)


def test_local_game_indices_are_kept_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _raw([0, 1, 0])
    out, request = _make_child(tmp_path, raw, "child_00")
    monkeypatch.setattr(mp, "_load_binding", lambda: _FakeBinding(raw))

    assert mp.run_worker(request) == 0
    results = mp._load_child_results(out, 2)
    assert [len(result.samples) for result in results] == [2, 1]


def _streamed_entry(index: int, samples: int) -> dict[str, object]:
    return {
        "index": index,
        "white_outcome": 0.0,
        "terminal_reason": "draw",
        "semimoves": 1,
        "turns": 1,
        "board_limit": 15,
        "board_count": 1,
        "final_player": 0,
        "samples": samples,
        "tree_samples": 0,
        "moves": [],
        "final_fen": "",
    }


def test_streamed_child_results_need_no_sample_transport(tmp_path: Path) -> None:
    child = tmp_path / "child_00"
    child.mkdir()
    payload = {
        "status": "ok",
        "streamed": True,
        "samples": 3,
        "games": [_streamed_entry(0, 2), _streamed_entry(1, 1)],
        "stats": {"rows": 3},
    }
    (child / mp.RESULT_NAME).write_text(json.dumps(payload), encoding="utf-8")

    assert not (child / mp.ARRAYS_NAME).exists()
    results = mp._load_child_results(child, 2)
    assert [result.sample_count for result in results] == [2, 1]
    assert [len(result.samples) for result in results] == [0, 0]
    assert all(result.persisted for result in results)


def test_child_streams_finished_games_straight_into_the_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from suzero_5dchess.game_store import GameStore

    def stream_payload(count: int) -> dict[str, object]:
        raw = _raw([0] * count)
        q = int(raw["action_counts"][0])
        moves = np.array(
            [
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 2, 1, 0, 0],
                [1, 0, 0, 0, 2, 2, 0, 0],
            ],
            dtype=np.int64,
        )[:q]
        raw["actions"] = np.repeat(moves[None, :, :], count, axis=0)
        return raw

    payloads = [stream_payload(1), stream_payload(2)]
    output_dir, request = _make_child(tmp_path, payloads[0], "child_00")
    store_dir = tmp_path / "replay_store"
    store_dir.mkdir()
    request_payload = json.loads(request.read_text(encoding="utf-8"))
    request_payload["binding_kwargs"]["num_games"] = 2
    request_payload["replay_store"] = {
        "directory": str(store_dir.resolve()),
        "variant": VERY_SMALL.name,
        "iteration": 4,
        "game_id_start": 7,
    }
    request.write_text(json.dumps(request_payload), encoding="utf-8")

    class StreamingBinding:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def run_selfplay(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(kwargs)
            callback = kwargs["on_game"]
            assert callable(callback)
            for index, payload in enumerate(payloads):
                callback(index + 1, index, payload)
            return {
                "stats": {"rows": 3},
                "games": [
                    dict(
                        payload["games"][0],
                        index=index,
                        samples=int(payload["used_board_counts"].shape[0]),
                    )
                    for index, payload in enumerate(payloads)
                ],
            }

    binding = StreamingBinding()
    monkeypatch.setattr(mp, "_load_binding", lambda: binding)

    assert mp.run_worker(request) == 0
    assert binding.calls[0]["stream_games"] is True

    # No per-child blobs cross back to the parent.
    assert not (output_dir / mp.ARRAYS_NAME).exists()
    result_payload = json.loads(
        (output_dir / mp.RESULT_NAME).read_text(encoding="utf-8")
    )
    assert result_payload["streamed"] is True
    assert result_payload["samples"] == 3

    store = GameStore(store_dir, VERY_SMALL, 16)
    store.scan()
    assert [entry.game_id for entry in store.games] == [7, 8]
    assert [entry.count for entry in store.games] == [1, 2]
    assert len(store) == 3

    results = mp._load_child_results(output_dir, 2)
    assert [result.sample_count for result in results] == [1, 2]
    assert all(result.persisted for result in results)
    assert all(result.samples == [] for result in results)


# ------------------------------------------------------------------ progress


class _FakeProc:
    """Minimal ``Popen`` replacement that reports progress while polling."""

    def __init__(self, states: list[int | None], on_poll=None) -> None:
        self.states = states
        self.on_poll = on_poll
        self.calls = 0
        self.returncode: int | None = None
        self.signals: list[int] = []

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)

    def poll(self) -> int | None:
        index = self.calls
        self.calls += 1
        if self.on_poll is not None:
            self.on_poll(index)
        code = self.states[index] if index < len(self.states) else self.states[-1]
        self.returncode = code
        return code

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_progress_aggregation_is_monotonic_and_ends_at_total(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mp, "POLL_SECONDS", 0.0)
    plan = [
        mp.ProcessPlan(index=0, workers=1, games=2, server_threads=1, seed=0),
        mp.ProcessPlan(index=1, workers=1, games=2, server_threads=1, seed=1),
    ]
    child_dirs = [tmp_path / "child_00", tmp_path / "child_01"]
    for child_dir, done in zip(child_dirs, (1, 0)):
        child_dir.mkdir()
        mp._atomic_json(
            child_dir / mp.PROGRESS_NAME, {"done": done, "games": 2}
        )

    wrote_final = [False]

    def on_poll(_index: int) -> None:
        if wrote_final[0]:
            return
        wrote_final[0] = True
        for child_dir in child_dirs:
            mp._atomic_json(
                child_dir / mp.PROGRESS_NAME, {"done": 2, "games": 2}
            )

    procs = [_FakeProc([None, 0], on_poll), _FakeProc([None, 0], on_poll)]
    seen: list[int] = []

    reported = mp._wait_for_children(
        procs, child_dirs, plan, lambda done: seen.append(done)
    )

    assert reported == 4
    assert seen == [1, 4]
    assert seen[-1] == sum(item.games for item in plan)


def test_progress_write_failures_are_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def locked(*args: object, **kwargs: object) -> None:
        raise PermissionError("destination is briefly locked")

    monkeypatch.setattr(mp, "_atomic_json", locked)
    # Must not raise: progress is auxiliary and the parent fills the total.
    mp._write_progress(tmp_path / "progress.json", {"done": 1, "games": 2})


def test_concurrent_native_progress_callbacks_share_one_atomic_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The native bridge calls ``progress`` from several game threads.

    File I/O releases the GIL, so without the child lock two callbacks could
    rewrite/replace the same ``.tmp`` file.  The recorder rejects overlapping
    writes and checks that ``done`` is clamped to the monotonic maximum.
    """

    raw = _raw([0, 1, 2, 3])
    output_dir, request = _make_child(tmp_path, raw, "child_00")

    original = mp._write_progress
    guard = threading.Lock()
    active = 0
    overlaps: list[str] = []
    seen: list[int] = []

    def recording(path: Path, payload: object) -> None:
        nonlocal active
        with guard:
            active += 1
            if active > 1:
                overlaps.append(path.name)
            seen.append(int(payload["done"]))  # type: ignore[index]
        try:
            time.sleep(0.01)
            original(path, payload)  # type: ignore[arg-type]
        finally:
            with guard:
                active -= 1

    class ConcurrentBinding:
        def run_selfplay(self, **kwargs: object) -> dict[str, object]:
            callback = kwargs["progress"]
            assert callable(callback)
            barrier = threading.Barrier(8)

            def report(done: int) -> None:
                barrier.wait()
                callback(done, 0)  # type: ignore[operator]

            threads = [
                threading.Thread(target=report, args=(done,))
                for done in (1, 2, 3, 4, 4, 3, 2, 1)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return raw

    monkeypatch.setattr(mp, "_write_progress", recording)
    monkeypatch.setattr(mp, "_load_binding", lambda: ConcurrentBinding())

    assert mp.run_worker(request) == 0
    assert overlaps == []
    assert seen[0] == 0
    assert seen[-1] == 4
    assert seen == sorted(seen)
    assert json.loads(
        (output_dir / mp.PROGRESS_NAME).read_text(encoding="utf-8")
    ) == {"done": 4, "games": 4}
    assert not list(output_dir.glob(f"{mp.PROGRESS_NAME}.*.tmp"))


def test_child_env_clears_profile_and_prepends_pythonpath(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(mp.PROFILE_ENV, "shared.jsonl")
    monkeypatch.setenv("PYTHONPATH", "existing")
    monkeypatch.setenv("PYTHONFAULTHANDLER", "0")
    env = mp._child_env(tmp_path)

    assert mp.PROFILE_ENV not in env
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path)
    assert env["PYTHONPATH"].endswith("existing")
    assert env["PYTHONFAULTHANDLER"] == "1"


def test_multiprocess_log_uses_parent_wall_not_child_time_sums(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from suzero_5dchess.self_play import SelfPlayResult

    def result() -> SelfPlayResult:
        return SelfPlayResult(
            samples=[object()],  # type: ignore[list-item]
            white_outcome=0.0,
            terminal_reason="draw",
            semimoves=1,
            turns=1,
            board_limit=15,
            board_count=1,
            final_player=0,
        )

    plan = [
        mp.ProcessPlan(index=0, workers=1, games=1, server_threads=1, seed=0),
        mp.ProcessPlan(index=1, workers=1, games=1, server_threads=1, seed=1),
    ]
    # Deliberately disjoint child clocks: neither total_seconds nor
    # setup_seconds may become the throughput denominator.
    stats = [
        {"rows": 10, "total_seconds": 100.0, "setup_seconds": 50.0},
        {"rows": 20, "total_seconds": 200.0, "setup_seconds": 25.0},
    ]
    with caplog.at_level(logging.INFO, logger=mp.logger.name):
        mp._log_summary(
            plan,
            total_workers=2,
            total_server_threads=2,
            games=2,
            results=[result(), result()],
            child_stats=stats,
            wall_seconds=2.0,
        )

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "parent end-to-end" in text
    assert "1.0 samples/s" in text
    assert "15.0 backend_rows/s" in text
    assert "per_process_seconds=[50.0, 25.0]" in text
    assert "max_seconds=50.00" in text
    # The child totals (300s combined) must not appear as a denominator.
    assert "300.0 seconds" not in text


# ------------------------------------------------------- real child processes


_HELPER_TEMPLATE = """
import json
import os
import sys
import time

request_path = sys.argv[-1]
with open(request_path, encoding="utf-8") as handle:
    request = json.load(handle)
output_dir = request["output_dir"]
name = os.path.basename(output_dir.rstrip("\\/"))
signal_dir = os.environ["SELFPLAY_TEST_SIGNAL_DIR"]
os.makedirs(signal_dir, exist_ok=True)

{body}
"""

_FAIL_HELPER = """
if name.endswith("00"):
    time.sleep(1.0)
    sys.stderr.write("boom from child zero\\n")
    sys.stderr.flush()
    sys.exit(3)

heartbeat = os.path.join(signal_dir, "child_01.heartbeat")
while True:
    with open(heartbeat, "w", encoding="utf-8") as handle:
        handle.write(str(time.time()))
    time.sleep(0.05)
"""

_COOPERATIVE_HELPER = """
games = int(request["binding_kwargs"]["num_games"])
heartbeat = os.path.join(signal_dir, name + ".heartbeat")
with open(heartbeat, "w", encoding="utf-8") as handle:
    handle.write(str(time.time()))

progress_path = os.path.join(output_dir, "progress.json")
tmp_path = progress_path + "." + str(os.getpid()) + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as handle:
    json.dump({"done": games, "games": games}, handle)
os.replace(tmp_path, progress_path)

while True:
    with open(heartbeat, "w", encoding="utf-8") as handle:
        handle.write(str(time.time()))
    time.sleep(0.05)
"""

_BASE_KWARGS = {
    "model_path": "unused.onnx",
    "metadata_path": "unused.onnx.json",
    "provider": "cpu",
    "num_games": 2,
    "num_threads": 2,
    "max_batch_size": 4,
}


def _write_helper(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "helper_worker.py"
    path.write_text(
        _HELPER_TEMPLATE.format(body=body),
        encoding="utf-8",
    )
    return path


def _install_helper(
    monkeypatch: pytest.MonkeyPatch, helper: Path, signal_dir: Path
) -> None:
    monkeypatch.setenv("SELFPLAY_TEST_SIGNAL_DIR", str(signal_dir))
    monkeypatch.setattr(
        mp,
        "worker_command",
        lambda request: [sys.executable, str(helper), "--worker", str(request)],
    )


def _ipc_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    work = tmp_path / "ipc"
    work.mkdir()
    monkeypatch.setattr(mp.tempfile, "mkdtemp", lambda prefix=None: str(work))
    return work


def _heartbeat_is_stopped(path: Path) -> bool:
    before = path.read_text(encoding="utf-8")
    time.sleep(0.4)
    return path.read_text(encoding="utf-8") == before


def test_first_nonzero_exit_cleans_siblings_and_reports_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _write_helper(tmp_path, _FAIL_HELPER)
    signal_dir = tmp_path / "signals"
    _install_helper(monkeypatch, helper, signal_dir)
    work = _ipc_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(mp, "POLL_SECONDS", 0.02)

    started = time.monotonic()
    with pytest.raises(RuntimeError) as excinfo:
        mp.run_self_play_processes(
            _BASE_KWARGS, processes=2, total_server_threads=2
        )
    elapsed = time.monotonic() - started

    message = str(excinfo.value)
    assert "exited with code 3" in message
    assert "boom from child zero" in message
    assert "diagnostics directory" in message
    assert str(work.resolve()) in message
    assert elapsed < 8.0
    assert work.exists()
    for name in ("child_00", "child_01"):
        child_dir = work / name
        assert (child_dir / mp.REQUEST_NAME).exists()
        assert (child_dir / mp.STDERR_NAME).exists()
    heartbeat = signal_dir / "child_01.heartbeat"
    assert heartbeat.exists()
    assert _heartbeat_is_stopped(heartbeat)


def test_success_cleans_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _write_helper(tmp_path, "sys.exit(0)")
    signal_dir = tmp_path / "signals"
    _install_helper(monkeypatch, helper, signal_dir)
    work = _ipc_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        mp,
        "_load_child_results",
        lambda _child_dir, expected_games: [object()] * expected_games,
    )
    monkeypatch.setattr(mp, "_log_summary", lambda *args, **kwargs: None)

    results = mp.run_self_play_processes(
        _BASE_KWARGS, processes=2, total_server_threads=2
    )

    assert len(results) == 2
    assert not work.exists()


def test_keyboard_interrupt_cleans_children_and_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _write_helper(tmp_path, _COOPERATIVE_HELPER)
    signal_dir = tmp_path / "signals"
    _install_helper(monkeypatch, helper, signal_dir)
    work = _ipc_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(mp, "POLL_SECONDS", 0.02)

    def abort(done: int) -> None:
        # Both children write a heartbeat before their progress file, so
        # waiting for the full total guarantees both heartbeats exist.
        if done >= 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        mp.run_self_play_processes(
            _BASE_KWARGS,
            processes=2,
            total_server_threads=2,
            progress=abort,
        )

    assert not work.exists()
    for name in ("child_00.heartbeat", "child_01.heartbeat"):
        heartbeat = signal_dir / name
        assert heartbeat.exists()
        assert _heartbeat_is_stopped(heartbeat)
