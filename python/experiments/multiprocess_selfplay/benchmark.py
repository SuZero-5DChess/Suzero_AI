"""Multi-process ONNX self-play benchmark (experiment; does not touch training).

One ``worker.py`` child per "process" is launched concurrently per round.  Each
child creates one ONNX Runtime session **per server thread** (the requested
total is split across processes) and calls the real
``_5dchess.run_selfplay`` binding once, so the experiment measures several
independent sessions sharing one GPU.

The parent uses **only the standard library** and never imports
``suzero_5dchess``/torch/onnxruntime.  Children are started as
``python worker.py --request request.json``.

Configuration model
-------------------
``--total-workers`` and ``--total-server-threads`` are *totals*, not per
process.  For a round with P processes:

* server threads are split evenly (``total // P``; validation requires an
  exact division);
* game threads are split as evenly as possible with the remainder going to the
  first processes, and the sum is preserved;
* games are split proportionally to the game-thread counts with the remainder
  preserved, so every worker still gets at least one game
  (``--games >= --total-workers`` is required).

Rounds are interleaved ``1, 2, 2, 1`` for ``--processes 1 2 --repeats 2``.

Throughput denominator
----------------------
The main throughput denominator is one span shared by every process in the
round: ``max(call_end_epoch) - min(call_start_epoch)``.  Aggregated samples /
rows / inference rows are divided by that single span.  Per-process rates are
never summed, and user-space setup/teardown time is not silently appended.
Process-group wall time, CPU seconds and raw setup values are reported
separately.

This is *not* multi-process training; it only emits statistics and never feeds
a replay buffer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
PYTHON_ROOT = ROOT / "python"
WORKER = SCRIPT_DIR / "worker.py"
# One lock for the whole experiment entry, independent of --output-dir.
ENTRY_LOCK_PATH = SCRIPT_DIR / ".multiprocess_selfplay.lock"

MODEL_SUBDIR = "model"
PROFILE_ENV = "SUZERO_ONNX_PROFILE_PATH"
VARIANT_PGN = '[Board "Very Small - Open"]\n[Mode "5D"]\n'
BARRIER_TIMEOUT_DEFAULT = 120.0
GPU_TS_FORMAT = "%Y/%m/%d %H:%M:%S.%f"
SEED_STRIDE = 0x9E3779B97F4A7C15
UINT64_MASK = (1 << 64) - 1

SOURCE_FILES = (
    ROOT / "src" / "5dchess" / "python_binding.cpp",
    ROOT / "src" / "5dchess" / "selfplay_bridge.cpp",
    ROOT / "src" / "5dchess" / "onnx_model.cpp",
    ROOT / "src" / "5dchess" / "onnx_batch_profile.h",
)

NOTES = {
    "scope": (
        "VerySmall 4x4 self-play experiment (board_side=4, line_shift=5, "
        "board limits 15-25). Not a training run."
    ),
    "throughput_denominator": (
        "samples_per_span_second / rows_per_span_second / "
        "inference_rows_per_span_second use ONE round span = "
        "max(call_end_epoch) - min(call_start_epoch). They are NOT the sum of "
        "per-process rates and NOT divided by the sum of per-process times. "
        "group_wall_seconds, cpu_seconds_sum and setup_seconds_raw are listed "
        "separately."
    ),
    "gpu_sampling": (
        "nvidia-smi -lms 500 samples the device; samples are filtered to the "
        "API union interval [min(call_start), max(call_end)]. The reported "
        "utilization is the sampled GPU utilization average/max, not SM "
        "occupancy. memory.used is device-level memory, not per-process or "
        "per-session memory."
    ),
    "seed_semantics": (
        "Each process gets a deterministic uint64 offset from --seed. Runs are "
        "therefore NOT the same trajectory as a single-process run; compare "
        "throughput only as approximate, load-equivalent work."
    ),
    "replay": "Statistics only; no replay buffer is written or consumed.",
    "cpu_diagnostic": (
        "--provider cpu is a diagnostic smoke only: the GPU busy check and "
        "nvidia-smi sampling are skipped, GPU data is marked as not collected, "
        "and no GPU performance conclusion may be drawn from it."
    ),
}

CSV_FIELDS = [
    "label",
    "round_index",
    "processes",
    "repeat",
    "worker_count",
    "ok",
    "exit_codes",
    "timed_out",
    "failure_reasons",
    "span_seconds",
    "group_wall_seconds",
    "cpu_seconds_sum",
    "api_wall_seconds_sum",
    "setup_seconds_sum",
    "setup_seconds_max",
    "samples_total",
    "rows_total",
    "inference_rows_total",
    "forced_rows_total",
    "backend_calls_total",
    "ort_runs_total",
    "used_boards_total",
    "board_slots_total",
    "used_actions_total",
    "action_slots_total",
    "samples_per_span_second",
    "rows_per_span_second",
    "inference_rows_per_span_second",
    "avg_batch_inference_per_ort",
    "max_observed_batch",
    "overlap_seconds",
    "overlap_fraction",
    "game_count_total",
    "gpu_samples",
    "gpu_util_avg",
    "gpu_util_max",
    "gpu_mem_used_max_mib",
]


# --------------------------------------------------------------------------- io


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_model(source: Path, output_dir: Path) -> dict:
    """Copy the ONNX model and its sidecar; never modify or overwrite them."""

    source = source.resolve()
    sidecar = Path(str(source) + ".json")
    for path in (source, sidecar):
        if not path.is_file():
            raise FileNotFoundError(f"missing model file: {path}")

    dest_dir = output_dir / MODEL_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for path in (source, sidecar):
        destination = dest_dir / path.name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite frozen file: {destination}")
        before = sha256_file(path)
        shutil.copy2(path, destination)
        after = sha256_file(path)
        copied = sha256_file(destination)
        if not (before == after == copied):
            raise RuntimeError(
                f"model freeze hash mismatch for {path}: "
                f"source_before={before} source_after={after} copy={copied}"
            )
        files[path.name] = {
            "source_path": str(path),
            "source_sha256": before,
            "frozen_path": str(destination.resolve()),
            "sha256": copied,
            "size": destination.stat().st_size,
            "source_mtime_ns": path.stat().st_mtime_ns,
        }

    manifest = {
        "source_model": str(source),
        "source_sidecar": str(sidecar),
        "frozen_dir": str(dest_dir.resolve()),
        "files": files,
        "note": "copy2 + SHA256 verification; the source model is never modified",
    }
    write_json(dest_dir / "freeze_manifest.json", manifest)
    return manifest


def preflight() -> dict:
    """Extension hash + related source mtimes, with a stale-build refusal."""

    report: dict[str, object] = {
        "extension": None,
        "sources": {},
        "stale_sources": [],
        "ok": False,
    }
    extensions = sorted((PYTHON_ROOT / "suzero_5dchess").glob("_5dchess*.pyd"))
    if not extensions:
        report["error"] = (
            "no compiled _5dchess*.pyd under "
            f"{PYTHON_ROOT / 'suzero_5dchess'}"
        )
        return report

    extension = extensions[0]
    extension_mtime = extension.stat().st_mtime_ns
    sources: dict[str, int | None] = {}
    for path in SOURCE_FILES:
        sources[str(path)] = path.stat().st_mtime_ns if path.is_file() else None
    stale = [
        path
        for path, mtime in sources.items()
        if mtime is not None and mtime > extension_mtime
    ]
    report.update(
        {
            "extension": {
                "path": str(extension),
                "sha256": sha256_file(extension),
                "mtime_ns": extension_mtime,
            },
            "sources": sources,
            "stale_sources": stale,
            "ok": not stale,
        }
    )
    if stale:
        report["stale_reason"] = (
            "installed _5dchess.pyd is older than: " + ", ".join(stale)
        )
    return report


# ---------------------------------------------------------------------- planning


def split_total(total: int, parts: int) -> list[int]:
    if parts < 1:
        raise ValueError("parts must be >= 1")
    if total < parts:
        raise ValueError("total must be >= parts so every part gets one")
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def allocate_by_weight(total: int, weights: list[int]) -> list[int]:
    """Largest-remainder integer allocation that preserves ``total``."""

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    allocation = [total * weight // total_weight for weight in weights]
    remainder = total - sum(allocation)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(total * weights[index] % total_weight), index),
    )
    for index in order[:remainder]:
        allocation[index] += 1
    return allocation


def process_seed(base_seed: int, process_index: int) -> int:
    """Deterministic uint64 offset so processes do not replay identical games.

    Process 0 keeps the requested seed; each additional process is shifted by
    a fixed 64-bit stride.
    """

    return (base_seed + process_index * SEED_STRIDE) & UINT64_MASK


def validate_plan(
    processes: list[int],
    total_server_threads: int,
    total_workers: int,
    games: int,
    repeats: int,
    seed: int,
) -> None:
    if not processes:
        raise ValueError("--processes must not be empty")
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if any(value < 1 for value in processes):
        raise ValueError("--processes values must be positive integers")
    if len(set(processes)) != len(processes):
        raise ValueError("--processes values must be unique")
    if total_server_threads < 1:
        raise ValueError("--total-server-threads must be >= 1")
    if total_workers < 1:
        raise ValueError("--total-workers must be >= 1")
    if games < 1:
        raise ValueError("--games must be >= 1")
    if not (0 <= seed <= UINT64_MASK):
        raise ValueError("--seed must fit in uint64")
    for count in processes:
        if total_server_threads % count != 0:
            raise ValueError(
                f"--total-server-threads {total_server_threads} is not "
                f"divisible by --processes {count}"
            )
        if total_workers < count:
            raise ValueError(
                f"--total-workers {total_workers} < --processes {count}; "
                "every process needs at least one game thread"
            )
    if games < total_workers:
        raise ValueError(
            f"--games {games} < --total-workers {total_workers}; the actual "
            "number of active game threads would drop below the requested "
            "total and comparisons would not be load-equivalent"
        )


def validate_runtime_limits(timeout: float, barrier_timeout: float) -> None:
    for name, value in (
        ("--timeout", timeout),
        ("--barrier-timeout", barrier_timeout),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")


def build_process_plan(
    processes: list[int],
    total_server_threads: int,
    total_workers: int,
    games: int,
    seed: int,
) -> list[dict]:
    entries: list[dict] = []
    for count in sorted(processes):
        workers = split_total(total_workers, count)
        games_per_process = allocate_by_weight(games, workers)
        server_threads = total_server_threads // count
        for index in range(count):
            entries.append(
                {
                    "processes": count,
                    "process_index": index,
                    "server_threads": server_threads,
                    "workers": workers[index],
                    "games": games_per_process[index],
                    "seed": process_seed(seed, index),
                }
            )
    return entries


def build_rounds(processes: list[int], repeats: int) -> list[dict]:
    ordered = sorted(processes)
    rounds: list[dict] = []
    for repeat in range(1, repeats + 1):
        sequence = ordered if repeat % 2 == 1 else list(reversed(ordered))
        for count in sequence:
            rounds.append({"processes": count, "repeat": repeat})
    return rounds


def binding_kwargs(
    args: argparse.Namespace,
    frozen: dict,
    *,
    games: int,
    workers: int,
    server_threads: int,
    seed: int,
) -> dict:
    return {
        "model_path": frozen["frozen_model"],
        "metadata_path": frozen["frozen_sidecar"],
        "provider": args.provider,
        "cuda_device_id": args.cuda_device,
        "variant_pgn": VARIANT_PGN,
        "board_side": 4,
        "line_shift": 5,
        "num_games": games,
        "num_threads": workers,
        "num_simulations": args.sims,
        "fast_simulations": args.fast_sims,
        "playout_cap_randomization": args.playout_cap_randomization,
        "full_search_probability": args.full_search_prob,
        "c_puct": args.c_puct,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_epsilon": args.dirichlet_epsilon,
        "max_depth": args.max_depth,
        "temperature": args.temperature,
        "temperature_final": args.temperature_final,
        "temperature_threshold": args.temperature_threshold,
        "min_board_limit": args.board_min,
        "max_board_limit": args.board_max,
        "material_scale": args.material_scale,
        "max_batch_size": args.max_batch_size,
        "seed": seed,
        "num_server_threads": server_threads,
    }


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    # The worker configures its own profile path; never inherit one.
    env.pop(PROFILE_ENV, None)
    python_path = str(PYTHON_ROOT)
    env["PYTHONPATH"] = (
        python_path + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH")
        else python_path
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def acquire_lock(lock_path: Path) -> Path:
    """Exclusive experiment-wide lock; never deletes an existing lock.

    The path is shared by every invocation of this entry point regardless of
    ``--output-dir`` (and optionally by GPU id), so different output
    directories still exclude each other.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(
            "another multiprocess benchmark may be running (shared lock "
            f"exists: {lock_path}); remove it manually only if it is stale"
        ) from exc
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
    return lock_path


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


# ------------------------------------------------------------------ aggregation


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def merge_histograms(histograms: list[object]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for histogram in histograms:
        if not isinstance(histogram, dict):
            continue
        for key, value in histogram.items():
            text = str(key)
            merged[text] = merged.get(text, 0) + _as_int(value)

    def order(item: tuple[str, int]) -> tuple[int, object]:
        key = item[0]
        return (0, int(key)) if key.isdigit() else (1, key)

    return dict(sorted(merged.items(), key=order))


def aggregate_worker_records(
    records: list[dict], expected_processes: int
) -> dict:
    """Aggregate one round with a single shared call-interval denominator."""

    reasons: list[str] = []
    if len(records) != expected_processes:
        reasons.append(
            f"expected {expected_processes} worker records, got {len(records)}"
        )

    intervals: list[tuple[float, float]] = []
    samples_total = 0
    game_count_total = 0
    rows_total = 0
    inference_total = 0
    forced_total = 0
    backend_calls_total = 0
    ort_runs_total = 0
    used_boards_total = 0
    board_slots_total = 0
    used_actions_total = 0
    action_slots_total = 0
    max_observed_batch = 0
    cpu_seconds_sum = 0.0
    api_wall_seconds_sum = 0.0
    setup_raw: list[float | None] = []
    total_seconds_raw: list[float | None] = []
    histograms: list[object] = []
    launch_monos: list[float] = []
    exit_monos: list[float] = []

    for record in records:
        index = record.get("index")
        exit_code = record.get("exit_code")
        result = record.get("result")
        status = result.get("status") if isinstance(result, dict) else None
        if exit_code != 0:
            reasons.append(
                f"worker {index}: real exit code {exit_code} "
                f"(result.status={status!r})"
            )
        if not isinstance(result, dict):
            reasons.append(
                f"worker {index}: {record.get('result_error') or 'invalid result'}"
            )
            continue
        if result.get("status") != "ok":
            reasons.append(f"worker {index}: result.status={status!r}")

        stats = result.get("stats")
        if not isinstance(stats, dict):
            reasons.append(f"worker {index}: result.stats is missing")
            stats = {}
        native = result.get("native_profile_summary")
        if not isinstance(native, dict):
            native = {}

        samples_total += _as_int(result.get("samples"))
        game_count_total += _as_int(result.get("game_count"))
        rows_total += _as_int(stats.get("rows"))
        inference_total += _as_int(native.get("inference_rows"))
        forced_total += _as_int(native.get("forced_rows"))
        backend_calls_total += _as_int(native.get("backend_calls"))
        ort_runs_total += _as_int(native.get("ort_runs"))
        used_boards_total += _as_int(native.get("used_boards"))
        board_slots_total += _as_int(native.get("board_slots"))
        used_actions_total += _as_int(native.get("used_actions"))
        action_slots_total += _as_int(native.get("action_slots"))
        max_observed_batch = max(
            max_observed_batch, _as_int(stats.get("max_observed_batch"))
        )
        histograms.append(native.get("batch_histogram"))
        cpu_seconds_sum += _as_float(result.get("run_cpu_seconds"))
        api_wall_seconds_sum += _as_float(result.get("api_wall_seconds"))
        setup_raw.append(
            _as_float(stats["setup_seconds"])
            if "setup_seconds" in stats
            else None
        )
        total_seconds_raw.append(
            _as_float(stats["total_seconds"])
            if "total_seconds" in stats
            else None
        )

        try:
            start = float(result["call_start_epoch"])
            end = float(result["call_end_epoch"])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"worker {index}: missing call start/end epoch")
        else:
            intervals.append((start, end))

        launch_mono = record.get("launch_mono")
        exit_mono = record.get("exit_mono")
        if launch_mono is not None and exit_mono is not None:
            launch_monos.append(_as_float(launch_mono))
            exit_monos.append(_as_float(exit_mono))

    span = None
    overlap = None
    overlap_fraction = None
    start_min = end_max = start_max = end_min = None
    if intervals:
        start_min = min(start for start, _ in intervals)
        end_max = max(end for _, end in intervals)
        start_max = max(start for start, _ in intervals)
        end_min = min(end for _, end in intervals)
        span = end_max - start_min
        overlap = max(0.0, end_min - start_max)
        overlap_fraction = overlap / span if span > 0 else None
        if expected_processes > 1 and overlap <= 0:
            reasons.append(
                "P>1 but the real call intervals do not intersect: "
                f"max(call_start)={start_max} >= min(call_end)={end_min}"
            )
        if span <= 0:
            reasons.append(f"non-positive shared span: {span}")
    else:
        reasons.append("no valid call interval in the round")

    valid_span = span is not None and span > 0
    metrics = {
        "worker_count": len(records),
        "call_start_min": start_min,
        "call_end_max": end_max,
        "call_start_max": start_max,
        "call_end_min": end_min,
        "throughput_denominator_seconds": span,
        "overlap_seconds": overlap,
        "overlap_fraction": overlap_fraction,
        "samples_total": samples_total,
        "rows_total": rows_total,
        "inference_rows_total": inference_total,
        "forced_rows_total": forced_total,
        "backend_calls_total": backend_calls_total,
        "ort_runs_total": ort_runs_total,
        "used_boards_total": used_boards_total,
        "board_slots_total": board_slots_total,
        "used_actions_total": used_actions_total,
        "action_slots_total": action_slots_total,
        "samples_per_span_second": (
            samples_total / span if valid_span else None
        ),
        "rows_per_span_second": rows_total / span if valid_span else None,
        "inference_rows_per_span_second": (
            inference_total / span if valid_span else None
        ),
        "avg_batch_inference_per_ort": (
            inference_total / ort_runs_total if ort_runs_total > 0 else None
        ),
        "board_fill_fraction": (
            used_boards_total / board_slots_total
            if board_slots_total > 0
            else None
        ),
        "action_fill_fraction": (
            used_actions_total / action_slots_total
            if action_slots_total > 0
            else None
        ),
        "max_observed_batch": max_observed_batch,
        "game_count_total": game_count_total,
        "cpu_seconds_sum": cpu_seconds_sum,
        "api_wall_seconds_sum": api_wall_seconds_sum,
        "setup_seconds_raw": setup_raw,
        "setup_seconds_sum": (
            sum(value for value in setup_raw if value is not None)
            if setup_raw
            else 0.0
        ),
        "setup_seconds_max": (
            max((value for value in setup_raw if value is not None), default=None)
        ),
        "stats_total_seconds_raw": total_seconds_raw,
        "group_wall_seconds": (
            max(exit_monos) - min(launch_monos)
            if launch_monos and exit_monos
            else None
        ),
        "batch_histogram": merge_histograms(histograms),
    }
    return {"ok": not reasons, "failure_reasons": reasons, "metrics": metrics}


# --------------------------------------------------------------------- GPU info


def gpu_compute_processes() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {
            "available": False,
            "processes": [],
            "error": "nvidia-smi not found",
        }
    try:
        proc = subprocess.run(
            [
                exe,
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never fatal here
        return {"available": False, "processes": [], "error": repr(exc)}
    if proc.returncode != 0:
        return {
            "available": False,
            "processes": [],
            "error": (proc.stderr or proc.stdout).strip(),
        }
    processes = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return {"available": True, "processes": processes}


def gpu_device_info() -> dict:
    exe = shutil.which("nvidia-smi")
    info: dict = {
        "available": False,
        "nvidia_smi": exe,
        "name": None,
        "memory_total_mib": None,
    }
    if not exe:
        return info
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        info["error"] = repr(exc)
        return info
    if proc.returncode != 0:
        info["error"] = (proc.stderr or proc.stdout).strip()
        return info
    first = next((line for line in proc.stdout.splitlines() if line.strip()), "")
    parts = [part.strip() for part in first.split(",")]
    if len(parts) >= 2:
        info.update(
            {
                "available": True,
                "name": parts[0],
                "memory_total_mib": _as_int(parts[1].split()[0], 0),
            }
        )
    return info


def gpu_gate(gpu_before: dict, provider: str) -> tuple[bool, str | None, str | None]:
    """Decide whether a run may start.  CUDA fails closed on unknown occupancy."""

    if provider == "cpu":
        return (
            True,
            None,
            "provider=cpu diagnostic mode: GPU busy check and sampling "
            "skipped; GPU data not collected",
        )
    if not gpu_before.get("available"):
        return (
            False,
            "gpu_check_failed",
            "GPU occupancy check unavailable; refusing to start workers "
            f"(fail closed): {gpu_before.get('error')}",
        )
    if gpu_before.get("processes"):
        return (
            False,
            "gpu_busy",
            "GPU already has compute processes; refusing to benchmark "
            "(existing processes are never terminated)",
        )
    return (True, None, None)


def parse_gpu_timestamp(text: str) -> float | None:
    text = text.strip()
    try:
        return datetime.strptime(text, GPU_TS_FORMAT).timestamp()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _leading_number(text: str) -> int | None:
    head = text.strip().split()[0] if text.strip() else ""
    return int(head) if head.isdigit() else None


def parse_gpu_line(line: str) -> tuple[float, int | None, int | None, str] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return None
    timestamp = parse_gpu_timestamp(parts[0])
    if timestamp is None:
        return None
    return timestamp, _leading_number(parts[1]), _leading_number(parts[2]), parts[0]


def summarize_gpu_lines(
    lines: list[str],
    start: float | None,
    end: float | None,
    filtered_path: Path | None = None,
) -> dict:
    rows: list[dict] = []
    if start is not None and end is not None:
        for line in lines:
            parsed = parse_gpu_line(line)
            if parsed is None:
                continue
            timestamp, utilization, memory, timestamp_text = parsed
            if start <= timestamp <= end:
                rows.append(
                    {
                        "timestamp": timestamp_text,
                        "utilization_gpu": utilization,
                        "memory_used_mib": memory,
                    }
                )

    if filtered_path is not None:
        with filtered_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "utilization_gpu",
                    "memory_used_mib",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    utils = [row["utilization_gpu"] for row in rows if row["utilization_gpu"] is not None]
    memories = [
        row["memory_used_mib"] for row in rows if row["memory_used_mib"] is not None
    ]
    return {
        "samples": len(rows),
        "collected": True,
        "util_avg": (sum(utils) / len(utils)) if utils else None,
        "util_max": max(utils) if utils else None,
        "mem_used_max_mib": max(memories) if memories else None,
        "interval_start_epoch": start,
        "interval_end_epoch": end,
        "filter_basis": "API union interval [min(call_start), max(call_end)]",
    }


def empty_gpu_summary(reason: str) -> dict:
    return {
        "samples": 0,
        "collected": False,
        "util_avg": None,
        "util_max": None,
        "mem_used_max_mib": None,
        "interval_start_epoch": None,
        "interval_end_epoch": None,
        "filter_basis": None,
        "note": reason,
    }


def overall_gpu(rounds: list[dict]) -> dict:
    weighted_sum = 0.0
    total = 0
    util_max = None
    mem_max = None
    for round_summary in rounds:
        gpu = round_summary.get("gpu") or {}
        count = _as_int(gpu.get("samples"))
        average = gpu.get("util_avg")
        total += count
        if average is not None:
            weighted_sum += _as_float(average) * count
        if gpu.get("util_max") is not None:
            util_max = max(util_max or 0, _as_int(gpu["util_max"]))
        if gpu.get("mem_used_max_mib") is not None:
            mem_max = max(mem_max or 0, _as_int(gpu["mem_used_max_mib"]))
    return {
        "samples": total,
        "collected": any(
            (round_summary.get("gpu") or {}).get("collected")
            and _as_int((round_summary.get("gpu") or {}).get("samples")) > 0
            for round_summary in rounds
        ),
        "util_avg": (weighted_sum / total) if total else None,
        "util_max": util_max,
        "mem_used_max_mib": mem_max,
        "note": NOTES["gpu_sampling"],
    }


class GpuSampler:
    """One ``nvidia-smi -lms 500`` child per round; stdout goes straight to a file."""

    def __init__(self, raw_path: Path, err_path: Path) -> None:
        self.raw_path = raw_path
        self.err_path = err_path
        self.proc: subprocess.Popen | None = None
        self._raw = None
        self._err = None

    def start(self) -> None:
        exe = shutil.which("nvidia-smi")
        if not exe:
            self.raw_path.write_text("", encoding="utf-8")
            self.err_path.write_text(
                "nvidia-smi not found; no GPU samples\n", encoding="utf-8"
            )
            return
        self._raw = self.raw_path.open("wb")
        self._err = self.err_path.open("wb")
        try:
            self.proc = subprocess.Popen(
                [
                    exe,
                    "--query-gpu=timestamp,utilization.gpu,memory.used",
                    "--format=csv,noheader",
                    "-lms",
                    "500",
                ],
                stdout=self._raw,
                stderr=self._err,
            )
        except Exception as exc:  # noqa: BLE001
            self._err.write(f"sampler start failed: {exc!r}\n".encode())
            self._close()
            if not self.raw_path.exists():
                self.raw_path.write_text("", encoding="utf-8")

    def stop(self) -> None:
        if self.proc is not None:
            terminate_process(self.proc)
            self.proc = None
        self._close()
        if not self.raw_path.exists():
            self.raw_path.write_text("", encoding="utf-8")

    def _close(self) -> None:
        for handle in (self._raw, self._err):
            if handle is not None:
                handle.close()
        self._raw = None
        self._err = None


# --------------------------------------------------------------- process control


def terminate_process(proc: subprocess.Popen, grace: float = 10.0) -> None:
    """Terminate only our own child; never user processes."""

    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass


def wait_for_ready(
    procs: list[subprocess.Popen], worker_dirs: list[Path], deadline: float
) -> str | None:
    while True:
        for index, proc in enumerate(procs):
            if proc.poll() is not None:
                return (
                    f"worker {index} exited before the barrier "
                    f"(real exit code {proc.returncode})"
                )
        ready_all = True
        for index, worker_dir in enumerate(worker_dirs):
            ready = read_json(worker_dir / "ready.json")
            if not isinstance(ready, dict) or "pid" not in ready:
                ready_all = False
                break
        if ready_all:
            for index, worker_dir in enumerate(worker_dirs):
                ready = read_json(worker_dir / "ready.json") or {}
                if _as_int(ready.get("pid"), -1) != procs[index].pid:
                    return (
                        f"worker {index} ready.json pid "
                        f"{ready.get('pid')!r} != spawned pid {procs[index].pid}"
                    )
            return None
        if time.monotonic() >= deadline:
            return "barrier timeout while waiting for ready.json"
        time.sleep(0.05)


def wait_for_all(
    procs: list[subprocess.Popen], deadline: float
) -> tuple[
    list[int | None],
    list[float | None],
    list[float | None],
    bool,
    bool,
]:
    count = len(procs)
    exit_codes: list[int | None] = [None] * count
    exit_monos: list[float | None] = [None] * count
    exit_epochs: list[float | None] = [None] * count
    timed_out = False
    aborted_after_failure = False

    def fill_finished() -> None:
        for index, proc in enumerate(procs):
            if exit_codes[index] is not None:
                continue
            code = proc.poll()
            if code is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                code = proc.returncode
            exit_codes[index] = code
            exit_monos[index] = time.perf_counter()
            exit_epochs[index] = time.time()

    while True:
        for index, proc in enumerate(procs):
            if exit_codes[index] is None:
                code = proc.poll()
                if code is not None:
                    exit_codes[index] = code
                    exit_monos[index] = time.perf_counter()
                    exit_epochs[index] = time.time()

        # First real non-zero exit: stop our remaining active children now,
        # then preserve every real exit code.  Do not wait for --timeout.
        if any(code is not None and code != 0 for code in exit_codes):
            active = [proc for proc in procs if proc.poll() is None]
            aborted_after_failure = bool(active)
            for proc in procs:
                terminate_process(proc)
            fill_finished()
            break

        if all(code is not None for code in exit_codes):
            break

        if time.monotonic() >= deadline:
            timed_out = True
            for proc in procs:
                terminate_process(proc)
            fill_finished()
            break
        time.sleep(0.05)

    return exit_codes, exit_monos, exit_epochs, timed_out, aborted_after_failure


# ----------------------------------------------------------------- round runner


def concat_logs(dest: Path, records: list[dict], key: str) -> None:
    with dest.open("wb") as output:
        for record in records:
            header = (
                f"===== worker {record['index']} pid={record['pid']} "
                f"exit={record['exit_code']} =====\n"
            ).encode()
            output.write(header)
            path = Path(record[key])
            if path.is_file():
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output, 1 << 20)


def write_merged_profile(dest: Path, records: list[dict]) -> None:
    with dest.open("w", encoding="utf-8") as output:
        for record in records:
            path = Path(record["profile_path"])
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = {"raw_line": line}
                if not isinstance(payload, dict):
                    payload = {"payload": payload}
                payload["worker_index"] = record["index"]
                output.write(json.dumps(payload, default=str) + "\n")


def run_round(
    args: argparse.Namespace,
    output_dir: Path,
    round_spec: dict,
    plan_entries: list[dict],
    frozen: dict,
    round_index: int,
) -> dict:
    process_count = round_spec["processes"]
    repeat = round_spec["repeat"]
    label = f"p{process_count:02d}_rep{repeat:02d}"
    round_dir = output_dir / "runs" / label
    round_dir.mkdir(parents=True)
    workers_root = round_dir / "workers"
    workers_root.mkdir()

    entries = [
        entry for entry in plan_entries if entry["processes"] == process_count
    ]
    if len(entries) != process_count:
        raise RuntimeError(f"process plan mismatch for {label}")

    worker_specs: list[dict] = []
    for index, entry in enumerate(entries):
        worker_dir = workers_root / f"w{index:02d}"
        worker_dir.mkdir()
        start_file = worker_dir / "go"
        kwargs = binding_kwargs(
            args,
            frozen,
            games=entry["games"],
            workers=entry["workers"],
            server_threads=entry["server_threads"],
            seed=entry["seed"],
        )
        request = {
            "output_dir": str(worker_dir.resolve()),
            "start_file": str(start_file.resolve()),
            "barrier_timeout_seconds": float(args.barrier_timeout),
            "binding_kwargs": kwargs,
        }
        request_path = worker_dir / "request.json"
        write_json(request_path, request)
        worker_specs.append(
            {
                "index": index,
                "dir": worker_dir,
                "request_path": request_path,
                "start_file": start_file,
                "kwargs": kwargs,
            }
        )

    round_request = {
        "label": label,
        "processes": process_count,
        "repeat": repeat,
        "total_workers": args.total_workers,
        "total_server_threads": args.total_server_threads,
        "games": args.games,
        "worker_plan": [
            {
                "index": index,
                "output_dir": str(spec["dir"].resolve()),
                "request_path": str(spec["request_path"].resolve()),
                "start_file": str(spec["start_file"].resolve()),
                "binding_kwargs": spec["kwargs"],
            }
            for index, spec in enumerate(worker_specs)
        ],
    }
    write_json(round_dir / "request.json", round_request)

    sampler = GpuSampler(round_dir / "gpu_raw.csv", round_dir / "gpu_monitor.err")
    gpu_enabled = args.provider != "cpu" and args.gpu_monitor
    if gpu_enabled:
        sampler.start()
    else:
        sampler.raw_path.write_text("", encoding="utf-8")
        sampler.err_path.write_text(
            "GPU sampling skipped (provider=cpu diagnostic mode or "
            "--no-gpu-monitor); no GPU data collected\n",
            encoding="utf-8",
        )

    procs: list[subprocess.Popen] = []
    handles: list[tuple] = []
    launches: list[tuple[float, float]] = []
    barrier_error: str | None = None
    timed_out = False
    aborted_after_failure = False
    go_epoch: float | None = None
    exit_codes: list[int | None] = []
    exit_monos: list[float | None] = []
    exit_epochs: list[float | None] = []

    try:
        for spec in worker_specs:
            stdout_handle = (spec["dir"] / "stdout.log").open("wb")
            stderr_handle = (spec["dir"] / "stderr.log").open("wb")
            handles.append((stdout_handle, stderr_handle))
            launch_mono = time.perf_counter()
            launch_epoch = time.time()
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(WORKER),
                    "--request",
                    str(spec["request_path"].resolve()),
                ],
                cwd=str(ROOT),
                env=child_env(),
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            procs.append(proc)
            launches.append((launch_mono, launch_epoch))

        barrier_error = wait_for_ready(
            procs,
            [spec["dir"] for spec in worker_specs],
            time.monotonic() + float(args.barrier_timeout),
        )
        if barrier_error is None:
            go_epoch = time.time()
            # All children have imported the binding; release them together.
            for spec in worker_specs:
                spec["start_file"].write_text(
                    f"go {go_epoch:.6f}\n", encoding="utf-8"
                )
            exit_codes, exit_monos, exit_epochs, timed_out, aborted_after_failure = (
                wait_for_all(procs, time.monotonic() + float(args.timeout))
            )
        else:
            for proc in procs:
                terminate_process(proc)
            (
                exit_codes,
                exit_monos,
                exit_epochs,
                _barrier_timeout,
                aborted_after_failure,
            ) = wait_for_all(procs, time.monotonic() + 15.0)
    except BaseException:
        for proc in procs:
            terminate_process(proc)
        raise
    finally:
        for stdout_handle, stderr_handle in handles:
            stdout_handle.close()
            stderr_handle.close()
        sampler.stop()

    records: list[dict] = []
    for index, spec in enumerate(worker_specs):
        result_path = spec["dir"] / "result.json"
        result = read_json(result_path)
        if result is None:
            result_error = (
                "missing result.json"
                if not result_path.is_file()
                else "unreadable result.json"
            )
        else:
            result_error = None
        records.append(
            {
                "index": index,
                "pid": procs[index].pid,
                "exit_code": exit_codes[index],
                "result": result,
                "result_error": result_error,
                "launch_mono": launches[index][0],
                "exit_mono": exit_monos[index],
                "launch_epoch": launches[index][1],
                "exit_epoch": exit_epochs[index],
                "output_dir": str(spec["dir"]),
                "request_path": str(spec["request_path"]),
                "result_path": str(result_path),
                "stdout_path": str(spec["dir"] / "stdout.log"),
                "stderr_path": str(spec["dir"] / "stderr.log"),
                "profile_path": str(spec["dir"] / "profile.jsonl"),
                "binding_kwargs": spec["kwargs"],
            }
        )

    aggregate = aggregate_worker_records(records, process_count)
    reasons = list(aggregate["failure_reasons"])
    if barrier_error is not None:
        reasons.insert(0, barrier_error)
    if timed_out:
        reasons.insert(0, f"round exceeded --timeout {args.timeout}s")
    metrics = aggregate["metrics"]

    raw_lines = (
        (round_dir / "gpu_raw.csv").read_text(encoding="utf-8", errors="replace").splitlines()
        if gpu_enabled and (round_dir / "gpu_raw.csv").is_file()
        else []
    )
    if gpu_enabled:
        gpu_summary = summarize_gpu_lines(
            raw_lines,
            metrics.get("call_start_min"),
            metrics.get("call_end_max"),
            filtered_path=round_dir / "gpu_filtered.csv",
        )
    else:
        gpu_summary = empty_gpu_summary(
            "provider=cpu diagnostic mode or --no-gpu-monitor: "
            "GPU busy check and sampling skipped; GPU data not collected"
        )
        with (round_dir / "gpu_filtered.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            csv.DictWriter(
                handle,
                fieldnames=["timestamp", "utilization_gpu", "memory_used_mib"],
            ).writeheader()

    gpu_warning = None
    if gpu_enabled and gpu_summary["samples"] == 0:
        gpu_warning = "no GPU samples fell inside the API union interval"

    ok = not reasons
    public_workers = [
        {
            "index": record["index"],
            "pid": record["pid"],
            "exit_code": record["exit_code"],
            "status": (
                record["result"].get("status")
                if isinstance(record["result"], dict)
                else None
            ),
            "result_error": record["result_error"],
            "output_dir": record["output_dir"],
            "request_path": record["request_path"],
            "result_path": record["result_path"],
            "stdout_path": record["stdout_path"],
            "stderr_path": record["stderr_path"],
            "profile_path": record["profile_path"],
            "binding_kwargs": record["binding_kwargs"],
            "stats": (
                record["result"].get("stats")
                if isinstance(record["result"], dict)
                else None
            ),
            "native_profile_summary": (
                record["result"].get("native_profile_summary")
                if isinstance(record["result"], dict)
                else None
            ),
            "samples": (
                record["result"].get("samples")
                if isinstance(record["result"], dict)
                else None
            ),
            "game_count": (
                record["result"].get("game_count")
                if isinstance(record["result"], dict)
                else None
            ),
            "game_signature": (
                record["result"].get("game_signature")
                if isinstance(record["result"], dict)
                else None
            ),
            "call_start_epoch": (
                record["result"].get("call_start_epoch")
                if isinstance(record["result"], dict)
                else None
            ),
            "call_end_epoch": (
                record["result"].get("call_end_epoch")
                if isinstance(record["result"], dict)
                else None
            ),
            "api_wall_seconds": (
                record["result"].get("api_wall_seconds")
                if isinstance(record["result"], dict)
                else None
            ),
            "run_cpu_seconds": (
                record["result"].get("run_cpu_seconds")
                if isinstance(record["result"], dict)
                else None
            ),
            "launch_epoch": record["launch_epoch"],
            "exit_epoch": record["exit_epoch"],
        }
        for record in records
    ]

    concat_logs(round_dir / "stdout.log", records, "stdout_path")
    concat_logs(round_dir / "stderr.log", records, "stderr_path")
    write_merged_profile(round_dir / "profile.jsonl", records)

    round_result = {
        "label": label,
        "processes": process_count,
        "repeat": repeat,
        "ok": ok,
        "failure_reasons": reasons,
        "barrier_error": barrier_error,
        "timed_out": timed_out,
        "aborted_after_failure": aborted_after_failure,
        "go_epoch": go_epoch,
        "exit_codes": exit_codes,
        "metrics": metrics,
        "gpu": gpu_summary,
        "gpu_warning": gpu_warning,
        "workers": [
            {
                "index": record["index"],
                "pid": record["pid"],
                "exit_code": record["exit_code"],
                "result_error": record["result_error"],
                "request_path": record["request_path"],
                "result": record["result"],
            }
            for record in records
        ],
    }
    write_json(round_dir / "result.json", round_result)

    round_summary = {
        "label": label,
        "round_index": round_index,
        "processes": process_count,
        "repeat": repeat,
        "ok": ok,
        "timed_out": timed_out,
        "aborted_after_failure": aborted_after_failure,
        "barrier_error": barrier_error,
        "failure_reasons": reasons,
        "exit_codes": exit_codes,
        "go_epoch": go_epoch,
        "metrics": metrics,
        "gpu": gpu_summary,
        "gpu_warning": gpu_warning,
        "workers": public_workers,
        "paths": {
            "round_dir": str(round_dir),
            "request": str(round_dir / "request.json"),
            "result": str(round_dir / "result.json"),
            "stdout": str(round_dir / "stdout.log"),
            "stderr": str(round_dir / "stderr.log"),
            "profile": str(round_dir / "profile.jsonl"),
            "gpu_raw": str(round_dir / "gpu_raw.csv"),
            "gpu_filtered": str(round_dir / "gpu_filtered.csv"),
        },
    }
    print(
        json.dumps(
            {
                "round": label,
                "ok": ok,
                "failure_reasons": reasons,
                "exit_codes": exit_codes,
                "samples_total": metrics.get("samples_total"),
                "rows_total": metrics.get("rows_total"),
                "span_seconds": metrics.get("throughput_denominator_seconds"),
                "samples_per_span_second": metrics.get("samples_per_span_second"),
                "group_wall_seconds": metrics.get("group_wall_seconds"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return round_summary


# --------------------------------------------------------------------- summary


def round_csv_row(round_summary: dict) -> dict:
    metrics = round_summary.get("metrics") or {}
    gpu = round_summary.get("gpu") or {}

    def rounded(value: object, digits: int = 6) -> float | None:
        return round(float(value), digits) if value is not None else None

    return {
        "label": round_summary.get("label"),
        "round_index": round_summary.get("round_index"),
        "processes": round_summary.get("processes"),
        "repeat": round_summary.get("repeat"),
        "worker_count": metrics.get("worker_count"),
        "ok": round_summary.get("ok"),
        "exit_codes": ",".join(
            str(code) for code in round_summary.get("exit_codes", [])
        ),
        "timed_out": round_summary.get("timed_out"),
        "failure_reasons": " | ".join(round_summary.get("failure_reasons", [])),
        "span_seconds": rounded(metrics.get("throughput_denominator_seconds")),
        "group_wall_seconds": rounded(metrics.get("group_wall_seconds")),
        "cpu_seconds_sum": rounded(metrics.get("cpu_seconds_sum")),
        "api_wall_seconds_sum": rounded(metrics.get("api_wall_seconds_sum")),
        "setup_seconds_sum": rounded(metrics.get("setup_seconds_sum")),
        "setup_seconds_max": rounded(metrics.get("setup_seconds_max")),
        "samples_total": metrics.get("samples_total"),
        "rows_total": metrics.get("rows_total"),
        "inference_rows_total": metrics.get("inference_rows_total"),
        "forced_rows_total": metrics.get("forced_rows_total"),
        "backend_calls_total": metrics.get("backend_calls_total"),
        "ort_runs_total": metrics.get("ort_runs_total"),
        "used_boards_total": metrics.get("used_boards_total"),
        "board_slots_total": metrics.get("board_slots_total"),
        "used_actions_total": metrics.get("used_actions_total"),
        "action_slots_total": metrics.get("action_slots_total"),
        "samples_per_span_second": rounded(metrics.get("samples_per_span_second")),
        "rows_per_span_second": rounded(metrics.get("rows_per_span_second")),
        "inference_rows_per_span_second": rounded(
            metrics.get("inference_rows_per_span_second")
        ),
        "avg_batch_inference_per_ort": rounded(
            metrics.get("avg_batch_inference_per_ort")
        ),
        "max_observed_batch": metrics.get("max_observed_batch"),
        "overlap_seconds": rounded(metrics.get("overlap_seconds")),
        "overlap_fraction": rounded(metrics.get("overlap_fraction")),
        "game_count_total": metrics.get("game_count_total"),
        "gpu_samples": gpu.get("samples"),
        "gpu_util_avg": rounded(gpu.get("util_avg"), 3),
        "gpu_util_max": gpu.get("util_max"),
        "gpu_mem_used_max_mib": gpu.get("mem_used_max_mib"),
    }


def write_summary(output_dir: Path, summary: dict) -> None:
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            round_csv_row(round_summary)
            for round_summary in summary.get("rounds", [])
        )


# ------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run several independent ONNX self-play sessions on one GPU and "
            "aggregate throughput on a single shared call-interval span."
        )
    )
    parser.add_argument("--model", type=Path, required=True, help="source ONNX model")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new output directory"
    )
    parser.add_argument(
        "--processes", type=int, nargs="+", default=[1, 2],
        help="process counts, e.g. --processes 1 2",
    )
    parser.add_argument("--total-workers", type=int, default=70)
    parser.add_argument("--total-server-threads", type=int, default=2)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--fast-sims", type=int, default=50)
    parser.add_argument("--full-search-prob", type=float, default=0.25)
    parser.add_argument(
        "--playout-cap-randomization", action="store_true", default=True
    )
    parser.add_argument(
        "--no-playout-cap-randomization",
        dest="playout_cap_randomization",
        action="store_false",
    )
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--board-min", type=int, default=15)
    parser.add_argument("--board-max", type=int, default=25)
    parser.add_argument("--material-scale", type=float, default=2.0)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=512)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-final", type=float, default=0.1)
    parser.add_argument("--temperature-threshold", type=int, default=30)
    parser.add_argument(
        "--provider", choices=["cuda", "cpu"], default="cuda",
        help="cuda for the hardware experiment; cpu only for a diagnostic smoke",
    )
    parser.add_argument(
        "--barrier-timeout", type=float, default=BARRIER_TIMEOUT_DEFAULT
    )
    parser.add_argument(
        "--gpu-monitor", dest="gpu_monitor", action="store_true", default=True
    )
    parser.add_argument(
        "--no-gpu-monitor", dest="gpu_monitor", action="store_false"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_plan(
            args.processes,
            args.total_server_threads,
            args.total_workers,
            args.games,
            args.repeats,
            args.seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        validate_runtime_limits(args.timeout, args.barrier_timeout)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = args.output_dir.resolve()
    model = args.model.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    if not WORKER.is_file():
        raise SystemExit(f"worker not found: {WORKER}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "status": "error",
        "created_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output_dir),
        "provider": args.provider,
        "gpu_data_collected": args.provider != "cpu" and args.gpu_monitor,
        "args": jsonable(vars(args)),
        "process_plan": build_process_plan(
            args.processes,
            args.total_server_threads,
            args.total_workers,
            args.games,
            args.seed,
        ),
        "rounds_plan": build_rounds(args.processes, args.repeats),
        "notes": NOTES,
        "warnings": [],
        "rounds": [],
    }
    manifest: dict | None = None
    exit_code = 2
    lock_path = acquire_lock(ENTRY_LOCK_PATH)
    try:
        output_dir.mkdir(parents=True)
        frozen = freeze_model(model, output_dir)
        frozen_paths = {
            "frozen_model": frozen["files"][model.name]["frozen_path"],
            "frozen_sidecar": frozen["files"][Path(str(model) + ".json").name][
                "frozen_path"
            ],
            "manifest": frozen,
        }
        preflight_report = preflight()
        cpu_mode = args.provider == "cpu"
        if cpu_mode:
            gpu_before = {
                "available": False,
                "processes": [],
                "skipped": True,
                "reason": "provider=cpu diagnostic mode",
            }
            gpu_device = {
                "available": False,
                "skipped": True,
                "reason": "provider=cpu diagnostic mode",
            }
        else:
            gpu_before = gpu_compute_processes()
            gpu_device = gpu_device_info()

        manifest = {
            "created_wall_time": summary["created_wall_time"],
            "benchmark_path": str(Path(__file__).resolve()),
            "worker_path": str(WORKER),
            "repository_root": str(ROOT),
            "output_dir": str(output_dir),
            "frozen_model": frozen,
            "preflight": preflight_report,
            "gpu_before": gpu_before,
            "gpu_device": gpu_device,
            "process_plan": summary["process_plan"],
            "rounds_plan": summary["rounds_plan"],
            "args": jsonable(vars(args)),
            "notes": NOTES,
        }
        write_json(output_dir / "manifest.json", manifest)
        summary["frozen_model"] = frozen
        summary["preflight"] = preflight_report
        summary["gpu_before"] = gpu_before
        summary["gpu_device"] = gpu_device

        if not preflight_report.get("ok"):
            summary["status"] = "preflight_failed"
            summary["warnings"].append(
                str(
                    preflight_report.get(
                        "stale_reason",
                        preflight_report.get("error", "preflight did not pass"),
                    )
                )
            )
            print(json.dumps(preflight_report, indent=2), file=sys.stderr)
            return 2

        allowed, gate_status, gate_message = gpu_gate(
            gpu_before, args.provider
        )
        if gate_message:
            summary["warnings"].append(gate_message)
        if not allowed:
            summary["status"] = gate_status or "gpu_check_failed"
            return 2

        all_ok = True
        for round_index, round_spec in enumerate(summary["rounds_plan"], start=1):
            round_summary = run_round(
                args,
                output_dir,
                round_spec,
                summary["process_plan"],
                frozen_paths,
                round_index,
            )
            summary["rounds"].append(round_summary)
            if not round_summary["ok"]:
                all_ok = False
                summary["status"] = "round_failed"
                break

        summary["sampled_gpu"] = overall_gpu(summary["rounds"])
        if all_ok:
            summary["status"] = "ok"
            exit_code = 0
    except BaseException as exc:  # noqa: BLE001 - failure still writes summary
        summary["status"] = "error"
        summary["exception"] = traceback.format_exc()
        summary["warnings"].append(repr(exc))
        exit_code = 2
    finally:
        summary["finished_wall_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if "sampled_gpu" not in summary:
            summary["sampled_gpu"] = overall_gpu(summary.get("rounds", []))
        summary["gpu_data_collected"] = bool(
            summary["sampled_gpu"].get("collected")
        )
        if output_dir.is_dir():
            write_summary(output_dir, summary)
            if manifest is not None:
                manifest["finished_wall_time"] = summary["finished_wall_time"]
                manifest["status"] = summary["status"]
                write_json(output_dir / "manifest.json", manifest)
        release_lock(lock_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
