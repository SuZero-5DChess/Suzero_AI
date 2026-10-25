"""CPU-only tests for the multiprocess self-play benchmark driver.

No GPU, no torch, no real worker process: the module is loaded straight from
its file and only pure helpers are exercised.
"""

from __future__ import annotations

import hashlib
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "multiprocess_selfplay"
    / "benchmark.py"
)

FORMAL_PLAN = {
    "processes": [1, 2],
    "total_server_threads": 2,
    "total_workers": 70,
    "games": 300,
    "repeats": 2,
    "seed": 20261023,
}


@pytest.fixture(scope="module")
def bench():
    assert BENCH_PATH.is_file(), BENCH_PATH
    spec = importlib.util.spec_from_file_location(
        "multiprocess_selfplay_benchmark_under_test", BENCH_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(
    *,
    status: str = "ok",
    start: float = 100.0,
    end: float = 110.0,
    samples: int = 100,
    rows: int = 1000,
    inference: int = 900,
    forced: int = 100,
    ort_runs: int = 9,
    setup: float = 1.0,
    total: float = 10.0,
    histogram: dict | None = None,
) -> dict:
    return {
        "status": status,
        "stats": {
            "server_threads": 1,
            "max_batch_size": 128,
            "requests": rows,
            "batches": ort_runs,
            "rows": rows,
            "max_observed_batch": 128,
            "total_seconds": total,
            "setup_seconds": setup,
        },
        "samples": samples,
        "game_count": 1,
        "game_signature": "sig",
        "call_start_epoch": start,
        "call_end_epoch": end,
        "api_wall_seconds": end - start,
        "run_cpu_seconds": 2.0,
        "native_profile_summary": {
            "backend_calls": ort_runs,
            "backend_rows": rows,
            "inference_rows": inference,
            "forced_rows": forced,
            "ort_runs": ort_runs,
            "used_boards": inference,
            "board_slots": inference * 2,
            "used_actions": inference * 10,
            "action_slots": inference * 20,
            "prepare_ns": 1,
            "encode_ns": 2,
            "allocate_ns": 3,
            "pack_ns": 4,
            "lock_wait_ns": 5,
            "run_ns": 6,
            "decode_ns": 7,
            "batch_histogram": (
                {"128": ort_runs} if histogram is None else histogram
            ),
        },
    }


def _record(index: int, *, exit_code: int = 0, **result_kwargs) -> dict:
    return {
        "index": index,
        "exit_code": exit_code,
        "result": _result(**result_kwargs),
        "launch_mono": 1000.0 + index,
        "exit_mono": 1010.0 + index,
    }


# ------------------------------------------------------------------ static


def test_parent_module_has_no_gpu_or_binding_imports(bench):
    text = BENCH_PATH.read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "import suzero_5dchess" not in text
    assert "from suzero_5dchess" not in text


def test_default_total_workers_matches_readme_main_example(bench):
    args = bench.build_parser().parse_args(
        ["--model", "model.onnx", "--output-dir", "out"]
    )
    assert args.total_workers == 70
    assert args.provider == "cuda"


# ------------------------------------------------------------- entry-wide lock


def test_entry_lock_is_entry_wide_and_never_deletes_existing(bench, tmp_path):
    # The lock is a module-level entry lock, not derived from --output-dir.
    assert bench.ENTRY_LOCK_PATH.parent == bench.SCRIPT_DIR
    assert bench.ENTRY_LOCK_PATH.name == ".multiprocess_selfplay.lock"

    lock = tmp_path / "entry.lock"
    held = bench.acquire_lock(lock)
    assert held == lock and lock.is_file()

    # A second invocation (even with a different output directory) would use
    # this same entry lock and must be refused.
    with pytest.raises(SystemExit):
        bench.acquire_lock(lock)
    assert lock.is_file()

    bench.release_lock(held)
    assert not lock.exists()

    # A pre-existing lock belongs to someone else: refuse, never delete.
    lock.write_text("someone-else\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        bench.acquire_lock(lock)
    assert lock.read_text(encoding="utf-8") == "someone-else\n"


# ------------------------------------------------------------------ gpu gate


@pytest.mark.parametrize(
    "gpu_before,provider,allowed,status",
    [
        (
            {"available": False, "processes": [], "error": "boom"},
            "cuda",
            False,
            "gpu_check_failed",
        ),
        (
            {"available": True, "processes": ["123 python"]},
            "cuda",
            False,
            "gpu_busy",
        ),
        ({"available": True, "processes": []}, "cuda", True, None),
        ({"available": False, "processes": []}, "cpu", True, None),
    ],
)
def test_gpu_gate_fails_closed(bench, gpu_before, provider, allowed, status):
    got_allowed, got_status, message = bench.gpu_gate(gpu_before, provider)
    assert got_allowed is allowed
    assert got_status == status
    if provider == "cpu":
        assert message and "GPU data not collected" in message


# ------------------------------------------------------------------ timeouts


@pytest.mark.parametrize(
    "timeout,barrier_timeout",
    [
        (float("nan"), 120.0),
        (float("inf"), 120.0),
        (900.0, float("nan")),
        (900.0, float("inf")),
        (0.0, 120.0),
        (900.0, -1.0),
    ],
)
def test_validate_runtime_limits_rejects_non_finite_or_non_positive(
    bench, timeout, barrier_timeout
):
    with pytest.raises(ValueError):
        bench.validate_runtime_limits(timeout, barrier_timeout)


def test_validate_runtime_limits_accepts_positive_values(bench):
    bench.validate_runtime_limits(900.0, 120.0)


# ------------------------------------------------------- exit-code propagation


class _FakeProc:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if self.returncode is None:
            self.returncode = 1

    def kill(self):
        self.terminated = True
        if self.returncode is None:
            self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_wait_for_all_aborts_remaining_children_on_first_nonzero(bench):
    failed = _FakeProc(returncode=3)
    running = _FakeProc(returncode=None)
    codes, _, _, timed_out, aborted = bench.wait_for_all(
        [failed, running], time.monotonic() + 30.0
    )
    assert codes == [3, 1]
    assert running.terminated
    assert aborted is True
    assert timed_out is False


def test_wait_for_all_leaves_clean_zero_children_alone(bench):
    first = _FakeProc(returncode=0)
    second = _FakeProc(returncode=0)
    codes, _, _, timed_out, aborted = bench.wait_for_all(
        [first, second], time.monotonic() + 30.0
    )
    assert codes == [0, 0]
    assert not first.terminated and not second.terminated
    assert aborted is False
    assert timed_out is False


# --------------------------------------------------------------- allocation


def test_split_total_conserves_and_balances(bench):
    for total, parts in ((50, 2), (70, 2), (7, 3), (4, 4), (300, 2)):
        chunks = bench.split_total(total, parts)
        assert sum(chunks) == total
        assert all(chunk >= 1 for chunk in chunks)
        assert max(chunks) - min(chunks) <= 1


def test_game_allocation_conserves_and_covers_workers(bench):
    workers = bench.split_total(70, 2)
    games = bench.allocate_by_weight(300, workers)
    assert workers == [35, 35]
    assert games == [150, 150]
    assert sum(games) == 300
    assert all(game >= worker for game, worker in zip(games, workers))

    odd = bench.allocate_by_weight(301, workers)
    assert sum(odd) == 301
    assert all(game >= worker for game, worker in zip(odd, workers))


def test_process_plan_preserves_totals_and_uses_uint64_seeds(bench):
    plan = bench.build_process_plan(
        FORMAL_PLAN["processes"],
        FORMAL_PLAN["total_server_threads"],
        FORMAL_PLAN["total_workers"],
        FORMAL_PLAN["games"],
        FORMAL_PLAN["seed"],
    )
    for count in (1, 2):
        entries = [entry for entry in plan if entry["processes"] == count]
        assert len(entries) == count
        assert sum(entry["workers"] for entry in entries) == 70
        assert sum(entry["games"] for entry in entries) == 300
        assert all(
            entry["server_threads"] == 2 // count for entry in entries
        )
        assert all(entry["games"] >= entry["workers"] for entry in entries)
        seeds = [entry["seed"] for entry in entries]
        assert len(set(seeds)) == count
        assert seeds[0] == FORMAL_PLAN["seed"]
        assert all(0 <= seed <= 2**64 - 1 for seed in seeds)


@pytest.mark.parametrize(
    "overrides",
    [
        {"processes": []},
        {"processes": [0]},
        {"processes": [-1]},
        {"processes": [1, 1]},
        {"total_server_threads": 3},
        {"total_server_threads": 0},
        {"total_workers": 1},
        {"games": 10},
        {"repeats": 0},
        {"seed": -1},
        {"seed": 2**64},
    ],
)
def test_validate_plan_rejects_invalid_inputs(bench, overrides):
    with pytest.raises(ValueError):
        bench.validate_plan(**{**FORMAL_PLAN, **overrides})


def test_validate_plan_accepts_formal_plan(bench):
    bench.validate_plan(**FORMAL_PLAN)


# ------------------------------------------------------------------- rounds


def test_round_order_is_interleaved_one_two_two_one(bench):
    rounds = bench.build_rounds([2, 1], 2)
    assert [
        (round_spec["processes"], round_spec["repeat"]) for round_spec in rounds
    ] == [(1, 1), (2, 1), (2, 2), (1, 2)]

    single = bench.build_rounds([2, 1], 1)
    assert [
        (round_spec["processes"], round_spec["repeat"]) for round_spec in single
    ] == [(1, 1), (2, 1)]


# -------------------------------------------------------------- aggregation


def test_aggregate_uses_one_shared_span_not_summed_rates(bench):
    outcome = bench.aggregate_worker_records(
        [
            _record(0, start=100.0, end=110.0, samples=100),
            _record(1, start=105.0, end=115.0, samples=100),
        ],
        2,
    )
    assert outcome["ok"], outcome["failure_reasons"]
    metrics = outcome["metrics"]
    assert metrics["throughput_denominator_seconds"] == pytest.approx(15.0)
    assert metrics["samples_per_span_second"] == pytest.approx(200 / 15)
    # Summed per-process rates would have been 10 + 10 = 20.
    assert metrics["samples_per_span_second"] != pytest.approx(20.0)
    assert metrics["rows_per_span_second"] == pytest.approx(2000 / 15)
    assert metrics["samples_total"] == 200
    assert metrics["rows_total"] == 2000
    assert metrics["inference_rows_total"] == 1800
    assert metrics["ort_runs_total"] == 18
    assert metrics["avg_batch_inference_per_ort"] == pytest.approx(100.0)
    assert metrics["overlap_seconds"] == pytest.approx(5.0)
    assert metrics["overlap_fraction"] == pytest.approx(5 / 15)
    assert metrics["setup_seconds_raw"] == [1.0, 1.0]
    assert metrics["setup_seconds_sum"] == pytest.approx(2.0)
    assert metrics["cpu_seconds_sum"] == pytest.approx(4.0)
    assert metrics["group_wall_seconds"] == pytest.approx(11.0)


def test_nonzero_real_exit_is_rejected_even_if_result_status_ok(bench):
    outcome = bench.aggregate_worker_records([_record(0, exit_code=1)], 1)
    assert not outcome["ok"]
    assert any(
        "exit code 1" in reason for reason in outcome["failure_reasons"]
    )


def test_result_status_must_be_ok(bench):
    outcome = bench.aggregate_worker_records([_record(0, status="error")], 1)
    assert not outcome["ok"]
    assert any(
        "result.status" in reason for reason in outcome["failure_reasons"]
    )


def test_missing_result_is_a_failure(bench):
    record = _record(0)
    record["result"] = None
    record["result_error"] = "missing result.json"
    outcome = bench.aggregate_worker_records([record], 1)
    assert not outcome["ok"]
    assert any(
        "missing result.json" in reason
        for reason in outcome["failure_reasons"]
    )


def test_p_gt_1_requires_interval_intersection(bench):
    outcome = bench.aggregate_worker_records(
        [
            _record(0, start=100.0, end=110.0),
            _record(1, start=120.0, end=130.0),
        ],
        2,
    )
    assert not outcome["ok"]
    assert any(
        "do not intersect" in reason
        for reason in outcome["failure_reasons"]
    )


def test_histogram_merge_preserves_rows_and_counts(bench):
    assert bench.merge_histograms(
        [{"128": 2}, {"128": 3, "1": 1}]
    ) == {"1": 1, "128": 5}


# -------------------------------------------------------------------- freeze


def test_freeze_model_preserves_source_and_refuses_overwrite(bench, tmp_path):
    source = tmp_path / "model.onnx"
    payload = b"onnx payload bytes"
    source.write_bytes(payload)
    sidecar = tmp_path / "model.onnx.json"
    sidecar_payload = b'{"side": true}'
    sidecar.write_bytes(sidecar_payload)
    source_before = source.read_bytes()
    sidecar_before = sidecar.read_bytes()

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    manifest = bench.freeze_model(source, output_dir)

    # The source is read-only from the experiment's point of view.
    assert source.read_bytes() == source_before
    assert sidecar.read_bytes() == sidecar_before

    frozen = output_dir / "model" / "model.onnx"
    frozen_sidecar = output_dir / "model" / "model.onnx.json"
    assert frozen.read_bytes() == payload
    assert frozen_sidecar.read_bytes() == sidecar_payload
    assert manifest["files"]["model.onnx"]["sha256"] == (
        hashlib.sha256(payload).hexdigest()
    )
    assert manifest["files"]["model.onnx.json"]["sha256"] == (
        hashlib.sha256(sidecar_payload).hexdigest()
    )
    assert (output_dir / "model" / "freeze_manifest.json").is_file()

    # A second freeze must not overwrite the frozen copy or the source.
    with pytest.raises(FileExistsError):
        bench.freeze_model(source, output_dir)
    assert source.read_bytes() == source_before


# ------------------------------------------------------------- binding kwargs


def test_binding_kwargs_carry_cpu_diagnostic_and_very_small_params(bench):
    args = SimpleNamespace(
        provider="cpu",
        cuda_device=0,
        sims=200,
        fast_sims=50,
        playout_cap_randomization=True,
        full_search_prob=0.25,
        c_puct=2.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        max_depth=512,
        temperature=1.0,
        temperature_final=0.1,
        temperature_threshold=30,
        board_min=15,
        board_max=25,
        material_scale=2.0,
        max_batch_size=128,
    )
    frozen = {
        "frozen_model": "model.onnx",
        "frozen_sidecar": "model.onnx.json",
    }
    kwargs = bench.binding_kwargs(
        args, frozen, games=4, workers=2, server_threads=1, seed=7
    )
    assert kwargs["provider"] == "cpu"
    assert kwargs["model_path"] == "model.onnx"
    assert kwargs["metadata_path"] == "model.onnx.json"
    assert kwargs["board_side"] == 4
    assert kwargs["line_shift"] == 5
    assert "Very Small" in kwargs["variant_pgn"]
    assert kwargs["num_games"] == 4
    assert kwargs["num_threads"] == 2
    assert kwargs["num_server_threads"] == 1
    assert kwargs["seed"] == 7
    assert kwargs["max_batch_size"] == 128
    assert kwargs["min_board_limit"] == 15
    assert kwargs["max_board_limit"] == 25
    assert "max_game_length" not in kwargs
