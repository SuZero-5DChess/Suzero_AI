"""CPU-only tests for the multiprocess self-play worker.

No GPU, ONNX Runtime, torch or real ``_5dchess`` binding is used: each test
loads ``worker.py`` with an importlib spec and replaces ``import_binding`` with
an in-process fake that writes plausible native profile lines.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

WORKER_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "multiprocess_selfplay"
    / "worker.py"
)

REQUIRED_STATS = (
    "server_threads",
    "max_batch_size",
    "requests",
    "batches",
    "rows",
    "max_observed_batch",
    "total_seconds",
    "setup_seconds",
)

TIMING_FIELDS = (
    "prepare_ns",
    "encode_ns",
    "allocate_ns",
    "pack_ns",
    "lock_wait_ns",
    "run_ns",
    "decode_ns",
)


def load_worker():
    name = "multiprocess_selfplay_worker_under_test"
    spec = importlib.util.spec_from_file_location(name, WORKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {WORKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def worker():
    return load_worker()


class _FakeArray:
    def __init__(self, rows: int) -> None:
        self.shape = (rows, 12, 16)


def test_flat_result_counts_samples_not_boards(worker):
    class Counts:
        shape = (2,)

    assert worker._extract_samples({
        "board_planes": _FakeArray(17), "used_board_counts": Counts()
    }) == 2


def _stats() -> dict:
    return {
        "server_threads": 2,
        "max_batch_size": 4,
        "requests": 5,
        "batches": 3,
        "rows": 5,
        "max_observed_batch": 2,
        "total_seconds": 1.5,
        "setup_seconds": 0.25,
        "extra_field_must_be_dropped": 999,
    }


def _games() -> list[dict]:
    return [
        {"terminal_reason": "draw", "moves": 7},
        {"terminal_reason": "mate", "moves": 3},
    ]


def _profile_lines() -> list[dict]:
    return [
        {
            "backend_calls": 2,
            "backend_rows": 3,
            "inference_rows": 3,
            "forced_rows": 0,
            "ort_runs": 2,
            "used_boards": 2,
            "board_slots": 4,
            "used_actions": 3,
            "action_slots": 5,
            "prepare_ns": 10,
            "encode_ns": 11,
            "allocate_ns": 12,
            "pack_ns": 13,
            "lock_wait_ns": 14,
            "run_ns": 15,
            "decode_ns": 16,
            "batch_histogram": {"1": 1, "2": 1},
        },
        {
            "backend_calls": 1,
            "backend_rows": 2,
            "inference_rows": 1,
            "forced_rows": 1,
            "ort_runs": 1,
            "used_boards": 1,
            "board_slots": 4,
            "used_actions": 2,
            "action_slots": 5,
            "prepare_ns": 20,
            "encode_ns": 21,
            "allocate_ns": 22,
            "pack_ns": 23,
            "lock_wait_ns": 24,
            "run_ns": 25,
            "decode_ns": 26,
            "batch_histogram": {"1": 1},
        },
    ]


class _FakeBinding:
    TREE_POSITION_FORMAT_VERSION = 1

    def __init__(
        self,
        profile_lines: list[dict],
        *,
        stats: dict | None = None,
        games: list[dict] | None = None,
        samples: int = 5,
    ) -> None:
        self.profile_lines = profile_lines
        self.stats = stats if stats is not None else _stats()
        self.games = games if games is not None else _games()
        self.samples = samples
        self.called = 0
        self.seen_profile_env: str | None = None

    def run_selfplay(self, *, num_server_threads: int, **kwargs: object) -> dict:
        self.called += 1
        self.seen_profile_env = os.environ.get("SUZERO_ONNX_PROFILE_PATH")
        profile_path = Path(self.seen_profile_env or "")
        profile_path.write_text(
            "".join(json.dumps(line) + "\n" for line in self.profile_lines),
            encoding="utf-8",
        )
        return {
            "stats": self.stats,
            "games": self.games,
            "board_planes": _FakeArray(self.samples),
        }


def _make_request(
    tmp_path: Path,
    *,
    timeout: float = 1.0,
    start: bool = False,
    binding_kwargs: dict | None = None,
    profile_lines: list[dict] | None = None,
    stats: dict | None = None,
    games: list[dict] | None = None,
    samples: int = 5,
) -> tuple[Path, Path, Path, _FakeBinding]:
    output_dir = tmp_path / "worker"
    output_dir.mkdir()
    start_file = tmp_path / "start.flag"
    if start:
        start_file.write_text("go", encoding="utf-8")

    kwargs = {
        "num_server_threads": 2,
        "max_batch_size": 4,
        "num_games": 2,
        "seed": 20261023,
    }
    if binding_kwargs is not None:
        kwargs.update(binding_kwargs)

    request_path = output_dir / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "start_file": str(start_file.resolve()),
                "barrier_timeout_seconds": timeout,
                "binding_kwargs": kwargs,
            }
        ),
        encoding="utf-8",
    )
    return output_dir, start_file, request_path, _FakeBinding(
        profile_lines if profile_lines is not None else _profile_lines(),
        stats=stats,
        games=games,
        samples=samples,
    )


def test_game_signature_is_order_independent(worker):
    games = _games()
    assert worker.game_signature(games) == worker.game_signature(
        list(reversed(games))
    )


def test_success_summary(worker, tmp_path):
    output_dir, _, request_path, binding = _make_request(tmp_path, start=True)
    worker.import_binding = lambda: binding

    exit_code = worker.run_request(request_path)

    assert exit_code == 0
    assert binding.called == 1
    assert Path(binding.seen_profile_env or "") == output_dir / "profile.jsonl"

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["pid"] == os.getpid()
    assert result["binding_kwargs"] == request["binding_kwargs"]
    assert set(result["stats"]) == set(REQUIRED_STATS)
    assert result["stats"]["server_threads"] == 2
    assert result["samples"] == 5
    assert result["game_count"] == 2
    assert result["call_end_epoch"] >= result["call_start_epoch"]
    assert result["api_wall_seconds"] >= 0
    assert result["run_cpu_seconds"] >= 0

    expected_signature = hashlib.sha256(
        json.dumps(
            sorted(
                _games(),
                key=lambda game: json.dumps(
                    game,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert result["game_signature"] == expected_signature

    summary = result["native_profile_summary"]
    assert summary["line_count"] == 2
    assert summary["backend_calls"] == 3
    assert summary["backend_rows"] == 5
    assert summary["inference_rows"] == 4
    assert summary["forced_rows"] == 1
    assert summary["ort_runs"] == 3
    assert summary["used_boards"] == 3
    assert summary["board_slots"] == 8
    assert summary["used_actions"] == 5
    assert summary["action_slots"] == 10
    assert summary["batch_histogram"] == {"1": 2, "2": 1}
    assert summary["prepare_ns"] == 30
    assert summary["encode_ns"] == 32
    assert summary["allocate_ns"] == 34
    assert summary["pack_ns"] == 36
    assert summary["lock_wait_ns"] == 38
    assert summary["run_ns"] == 40
    assert summary["decode_ns"] == 42
    assert summary["ort_avg_batch"] == pytest.approx(4 / 3)

    assert "games" not in result
    assert "board_planes" not in result
    raw_profile = (output_dir / "profile.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(raw_profile) == 2

    ready = json.loads((output_dir / "ready.json").read_text(encoding="utf-8"))
    assert ready["pid"] == os.getpid()
    assert isinstance(ready["ready_epoch"], float)


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda lines: lines[0].update(backend_calls=9), "backend_calls"),
        (lambda lines: lines[1].update(used_boards=99), "used_boards"),
        (lambda lines: lines.pop(), "expected 2"),
    ],
)
def test_rejects_inconsistent_profile(worker, tmp_path, mutate, needle):
    lines = _profile_lines()
    mutate(lines)
    output_dir, _, request_path, binding = _make_request(
        tmp_path, start=True, profile_lines=lines
    )
    worker.import_binding = lambda: binding

    exit_code = worker.run_request(request_path)

    assert exit_code != 0
    assert binding.called == 1
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert needle in result["error"]
    assert result["traceback"]
    assert (output_dir / "profile.jsonl").is_file()


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (
            lambda stats, games: stats.update(server_threads=3),
            "server_threads mismatch",
        ),
        (
            lambda stats, games: stats.update(max_batch_size=8),
            "max_batch_size mismatch",
        ),
        (lambda stats, games: games.pop(), "game_count mismatch"),
    ],
)
def test_rejects_return_mismatch(worker, tmp_path, mutate, needle):
    stats = _stats()
    games = _games()
    mutate(stats, games)
    output_dir, _, request_path, binding = _make_request(
        tmp_path, start=True, stats=stats, games=games
    )
    worker.import_binding = lambda: binding

    exit_code = worker.run_request(request_path)

    assert exit_code != 0
    assert binding.called == 1
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert needle in result["error"]
    assert result["traceback"]


def test_ready_barrier_timeout(worker, tmp_path):
    output_dir, _, request_path, binding = _make_request(
        tmp_path, timeout=0.2, start=False
    )
    worker.import_binding = lambda: binding

    started = time.monotonic()
    exit_code = worker.run_request(request_path)
    elapsed = time.monotonic() - started

    assert exit_code != 0
    assert elapsed >= 0.15
    assert binding.called == 0
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert "timed out" in result["error"]
    assert result["traceback"]
    assert (output_dir / "ready.json").is_file()
    assert not (output_dir / "profile.jsonl").exists()


@pytest.mark.parametrize(
    "timeout", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0]
)
def test_rejects_invalid_barrier_timeout(worker, tmp_path, timeout):
    output_dir, _, request_path, _ = _make_request(
        tmp_path, timeout=timeout, start=True
    )

    def forbidden_import():
        raise AssertionError("binding must not be imported for invalid request")

    worker.import_binding = forbidden_import

    exit_code = worker.run_request(request_path)

    assert exit_code != 0
    assert sorted(path.name for path in output_dir.iterdir()) == ["request.json"]


@pytest.mark.parametrize("name", ["result.json", "profile.jsonl", "ready.json"])
def test_refuses_to_overwrite(worker, tmp_path, name):
    output_dir, _, request_path, _ = _make_request(tmp_path, start=True)
    marker = output_dir / name
    marker.write_text("keep", encoding="utf-8")

    def forbidden_import():
        raise AssertionError("binding must not be imported when refusing")

    worker.import_binding = forbidden_import

    exit_code = worker.run_request(request_path)

    assert exit_code != 0
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        ["request.json", name]
    )
