"""One process, one public ``_5dchess.run_selfplay`` call.

This worker is the child side of the multiprocess single-GPU experiment.  The
parent creates an exclusive worker directory with ``request.json`` and later
creates ``start_file`` once every child reports ready.  The worker never
imports benchmark scripts from ``build/`` and never spawns further processes.

Protocol
--------
``python worker.py --request ABS_JSON``

request::

    {
      "output_dir": "/path/to/worker_00",
      "start_file": "/path/to/start.flag",
      "barrier_timeout_seconds": 120,
      "binding_kwargs": { ... original run_selfplay keyword arguments ... }
    }

Lifecycle:
1. refuse to touch a directory that already has result/profile/ready output;
2. set ``SUZERO_ONNX_PROFILE_PATH`` before importing the binding;
3. import the public binding exactly like the old benchmark worker (torch
   first so the Windows DLL directory is prepared);
4. verify that ``run_selfplay`` exposes ``num_server_threads``;
5. write ``ready.json`` = {pid, ready_epoch};
6. wait for ``start_file`` (barrier timeout is fatal);
7. call ``run_selfplay`` exactly once;
8. read and validate every per-session native profile line;
9. write ``result.json`` and let the interpreter exit normally.

The native ``profile.jsonl`` is retained verbatim: one line per OnnxModel
(one per server thread).  No per-scalar event log and no NumPy training
payloads are stored; ``used_board_counts.shape[0]`` counts samples in either transport layout.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

PROFILE_ENV = "SUZERO_ONNX_PROFILE_PATH"
READY_NAME = "ready.json"
RESULT_NAME = "result.json"
PROFILE_NAME = "profile.jsonl"
BARRIER_POLL_SECONDS = 0.05
PROFILE_WAIT_SECONDS = 10.0

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

PROFILE_REQUIRED = (
    "backend_calls",
    "backend_rows",
    "inference_rows",
    "forced_rows",
    "ort_runs",
    "used_boards",
    "board_slots",
    "used_actions",
    "action_slots",
    "prepare_ns",
    "encode_ns",
    "allocate_ns",
    "pack_ns",
    "lock_wait_ns",
    "run_ns",
    "decode_ns",
    "batch_histogram",
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


@dataclass(frozen=True)
class Request:
    output_dir: Path
    start_file: Path
    barrier_timeout_seconds: float
    binding_kwargs: dict


def import_binding() -> object:
    """Import the public pybind11 extension without importing build scripts.

    ``torch`` must be imported first: on Windows it installs the DLL search
    path needed by ``_5dchess.pyd`` and the ONNX Runtime provider DLLs.  This
    mirrors the initialization used by the old benchmark worker.
    """

    python_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(python_root))
    import torch  # noqa: F401

    from suzero_5dchess import env as env_mod

    if env_mod._5dchess is None:
        raise RuntimeError(
            "the _5dchess extension failed to import: "
            f"{env_mod._IMPORT_ERROR}"
        )
    return env_mod._5dchess


def binding_parameters(binding: object) -> list[str]:
    try:
        return list(inspect.signature(binding.run_selfplay).parameters)
    except (TypeError, ValueError):
        text = getattr(binding.run_selfplay, "__doc__", None) or ""
        head = text.split("->", 1)[0]
        return re.findall(r"([A-Za-z_]\w*)\s*[:=]", head)


def binding_supports_num_server_threads(binding: object) -> bool:
    return "num_server_threads" in binding_parameters(binding)


def load_request(path: Path) -> Request:
    if not path.is_absolute():
        raise ValueError(f"request path must be absolute: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("request must be a JSON object")

    output_dir = Path(data["output_dir"])
    start_file = Path(data["start_file"])
    if not output_dir.is_absolute():
        raise ValueError("output_dir must be absolute")
    if not start_file.is_absolute():
        raise ValueError("start_file must be absolute")
    if not output_dir.is_dir():
        raise ValueError(f"output_dir does not exist: {output_dir}")

    timeout = float(data.get("barrier_timeout_seconds", 120.0))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "barrier_timeout_seconds must be finite and positive"
        )

    kwargs = data.get("binding_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("binding_kwargs must be a JSON object")

    return Request(
        output_dir=output_dir,
        start_file=start_file,
        barrier_timeout_seconds=timeout,
        binding_kwargs=dict(kwargs),
    )


def _write_json_exclusive(path: Path, payload: dict) -> None:
    """Create ``path`` exclusively (``open('x')``); never overwrite."""

    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _refuse_existing(output_dir: Path) -> list[Path]:
    existing = [
        output_dir / RESULT_NAME,
        output_dir / PROFILE_NAME,
        output_dir / READY_NAME,
    ]
    return [path for path in existing if path.exists()]


def wait_for_start(start_file: Path, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if start_file.is_file():
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "start barrier timed out after "
                f"{timeout_seconds:g}s waiting for {start_file}"
            )
        time.sleep(min(BARRIER_POLL_SECONDS, remaining))


def _as_int(mapping: dict, key: str, where: str) -> int:
    try:
        return int(mapping[key])
    except KeyError as exc:
        raise RuntimeError(f"{where} missing key: {key}") from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{where} key {key} is not an integer: {mapping[key]!r}") from exc


def read_profile(path: Path, timeout_seconds: float = PROFILE_WAIT_SECONDS) -> list[dict]:
    """Wait for the real native profile and parse its per-session lines."""

    deadline = time.monotonic() + timeout_seconds
    last_error: object | None = None
    while True:
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
                lines = [line for line in text.splitlines() if line.strip()]
                records = [json.loads(line) for line in lines]
                if all(isinstance(record, dict) for record in records):
                    return records
                last_error = ValueError("profile line is not a JSON object")
            except Exception as exc:  # partial append while C++ writes
                last_error = exc
        if time.monotonic() >= deadline:
            if not path.is_file():
                raise RuntimeError(f"native profile not found: {path}")
            raise RuntimeError(
                f"could not parse native profile {path}: {last_error!r}"
            )
        time.sleep(BARRIER_POLL_SECONDS)


def validate_profile(records: list[dict], stats: dict) -> dict:
    """Validate native profile invariants and build the merged summary."""

    expected_lines = int(stats["server_threads"])
    if len(records) != expected_lines:
        raise RuntimeError(
            f"native profile has {len(records)} line(s), expected "
            f"{expected_lines} (server_threads)"
        )

    sum_backend_calls = 0
    sum_backend_rows = 0
    sum_inference_rows = 0
    sum_forced_rows = 0
    sum_ort_runs = 0
    sum_used_boards = 0
    sum_board_slots = 0
    sum_used_actions = 0
    sum_action_slots = 0
    timing_totals = {field: 0 for field in TIMING_FIELDS}
    merged_histogram: dict[int, int] = {}

    for index, record in enumerate(records):
        where = f"native profile line {index}"
        missing = [key for key in PROFILE_REQUIRED if key not in record]
        if missing:
            raise RuntimeError(f"{where} missing keys: {missing}")

        backend_calls = _as_int(record, "backend_calls", where)
        backend_rows = _as_int(record, "backend_rows", where)
        inference_rows = _as_int(record, "inference_rows", where)
        forced_rows = _as_int(record, "forced_rows", where)
        ort_runs = _as_int(record, "ort_runs", where)
        used_boards = _as_int(record, "used_boards", where)
        board_slots = _as_int(record, "board_slots", where)
        used_actions = _as_int(record, "used_actions", where)
        action_slots = _as_int(record, "action_slots", where)

        if backend_rows != inference_rows + forced_rows:
            raise RuntimeError(
                f"{where}: backend_rows {backend_rows} != "
                f"inference_rows {inference_rows} + forced_rows {forced_rows}"
            )

        histogram = record["batch_histogram"]
        if not isinstance(histogram, dict):
            raise RuntimeError(f"{where}: batch_histogram is not an object")
        line_hist_count = 0
        line_hist_weighted = 0
        for rows_key, count_value in histogram.items():
            try:
                rows = int(rows_key)
                count = int(count_value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{where}: invalid batch_histogram entry "
                    f"{rows_key!r}: {count_value!r}"
                ) from exc
            if rows < 0 or count < 0:
                raise RuntimeError(f"{where}: negative batch_histogram entry")
            line_hist_count += count
            line_hist_weighted += rows * count
            merged_histogram[rows] = merged_histogram.get(rows, 0) + count

        if line_hist_weighted != inference_rows:
            raise RuntimeError(
                f"{where}: histogram weighted rows {line_hist_weighted} != "
                f"inference_rows {inference_rows}"
            )
        if line_hist_count != ort_runs:
            raise RuntimeError(
                f"{where}: histogram entry count {line_hist_count} != "
                f"ort_runs {ort_runs}"
            )
        if used_boards > board_slots:
            raise RuntimeError(
                f"{where}: used_boards {used_boards} > board_slots {board_slots}"
            )
        if used_actions > action_slots:
            raise RuntimeError(
                f"{where}: used_actions {used_actions} > action_slots "
                f"{action_slots}"
            )

        sum_backend_calls += backend_calls
        sum_backend_rows += backend_rows
        sum_inference_rows += inference_rows
        sum_forced_rows += forced_rows
        sum_ort_runs += ort_runs
        sum_used_boards += used_boards
        sum_board_slots += board_slots
        sum_used_actions += used_actions
        sum_action_slots += action_slots
        for field in TIMING_FIELDS:
            timing_totals[field] += _as_int(record, field, where)

    stats_batches = int(stats["batches"])
    stats_rows = int(stats["rows"])
    stats_requests = int(stats["requests"])

    if sum_backend_calls != stats_batches:
        raise RuntimeError(
            f"native backend_calls {sum_backend_calls} != stats.batches "
            f"{stats_batches}"
        )
    if not (sum_backend_rows == stats_rows == stats_requests):
        raise RuntimeError(
            "native backend_rows / stats mismatch: "
            f"backend_rows={sum_backend_rows} stats.rows={stats_rows} "
            f"stats.requests={stats_requests}"
        )
    if sum_inference_rows + sum_forced_rows != stats_rows:
        raise RuntimeError(
            "native inference_rows + forced_rows != stats.rows: "
            f"{sum_inference_rows} + {sum_forced_rows} != {stats_rows}"
        )
    hist_count = sum(merged_histogram.values())
    hist_weighted = sum(rows * count for rows, count in merged_histogram.items())
    if hist_count != sum_ort_runs:
        raise RuntimeError(
            f"native histogram entry count {hist_count} != ort_runs "
            f"{sum_ort_runs}"
        )
    if hist_weighted != sum_inference_rows:
        raise RuntimeError(
            f"native histogram weighted rows {hist_weighted} != "
            f"inference_rows {sum_inference_rows}"
        )
    if sum_used_boards > sum_board_slots:
        raise RuntimeError(
            f"native used_boards {sum_used_boards} > board_slots "
            f"{sum_board_slots}"
        )
    if sum_used_actions > sum_action_slots:
        raise RuntimeError(
            f"native used_actions {sum_used_actions} > action_slots "
            f"{sum_action_slots}"
        )

    summary = {
        "line_count": len(records),
        "backend_calls": sum_backend_calls,
        "backend_rows": sum_backend_rows,
        "inference_rows": sum_inference_rows,
        "forced_rows": sum_forced_rows,
        "ort_runs": sum_ort_runs,
        "used_boards": sum_used_boards,
        "board_slots": sum_board_slots,
        "used_actions": sum_used_actions,
        "action_slots": sum_action_slots,
        "batch_histogram": {
            str(rows): merged_histogram[rows] for rows in sorted(merged_histogram)
        },
        "ort_avg_batch": (
            sum_inference_rows / sum_ort_runs if sum_ort_runs > 0 else None
        ),
    }
    summary.update(timing_totals)
    return summary


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def game_signature(games: list[dict]) -> str:
    # Sort the games themselves, then canonicalize: concurrent children must
    # agree even if the binding returns the same games in a different order.
    ordered = sorted(games, key=_canonical_json)
    canonical = _canonical_json(ordered).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _extract_stats(raw: dict) -> dict:
    stats = raw.get("stats")
    if not isinstance(stats, dict):
        raise RuntimeError("run_selfplay result has no raw 'stats' dict")
    missing = [key for key in REQUIRED_STATS if key not in stats]
    if missing:
        raise RuntimeError(f"raw stats missing keys: {missing}")
    return {key: stats[key] for key in REQUIRED_STATS}


def _extract_games(raw: dict) -> list[dict]:
    games = raw.get("games")
    if not isinstance(games, (list, tuple)):
        raise RuntimeError("run_selfplay result has no 'games' sequence")
    return [dict(entry) for entry in games]


def _extract_samples(raw: dict) -> int:
    counts = raw.get("used_board_counts")
    if counts is not None:
        shape = getattr(counts, "shape", None)
        if shape is None or len(shape) != 1:
            raise RuntimeError("run_selfplay result has invalid used_board_counts")
        return int(shape[0])
    # Compatibility with older diagnostic fixtures/extensions.
    board_planes = raw.get("board_planes")
    shape = getattr(board_planes, "shape", None)
    if shape is None or len(shape) < 1:
        raise RuntimeError("run_selfplay result has no board_planes array")
    return int(shape[0])


REQUESTED_INT_KEYS = (
    "num_server_threads",
    "max_batch_size",
    "num_games",
)


def _requested_ints(kwargs: dict) -> dict[str, int]:
    """Read the request-side values used for returned-result checks."""

    return {
        key: _as_int(kwargs, key, "binding_kwargs")
        for key in REQUESTED_INT_KEYS
    }


def _validate_against_request(
    stats: dict, games: list[dict], requested: dict[str, int]
) -> None:
    """Reject results that disagree with the request that produced them."""

    returned_threads = int(stats["server_threads"])
    if returned_threads != requested["num_server_threads"]:
        raise RuntimeError(
            "server_threads mismatch: returned "
            f"{returned_threads}, requested {requested['num_server_threads']}"
        )

    returned_batch = int(stats["max_batch_size"])
    if returned_batch != requested["max_batch_size"]:
        raise RuntimeError(
            "max_batch_size mismatch: returned "
            f"{returned_batch}, requested {requested['max_batch_size']}"
        )

    if len(games) != requested["num_games"]:
        raise RuntimeError(
            f"game_count mismatch: returned {len(games)}, "
            f"requested {requested['num_games']}"
        )


def _write_error_result(output_dir: Path, result: dict) -> int:
    try:
        _write_json_exclusive(output_dir / RESULT_NAME, result)
    except Exception as exc:
        print(f"worker: could not write {output_dir / RESULT_NAME}: {exc}", file=sys.stderr)
        return 2
    return 1


def run_request(request_path: Path) -> int:
    try:
        request = load_request(Path(request_path))
    except Exception as exc:
        print(f"worker: invalid request {request_path}: {exc}", file=sys.stderr)
        return 2

    output_dir = request.output_dir
    existing = _refuse_existing(output_dir)
    if existing:
        print(
            "worker: refusing to overwrite existing output: "
            + ", ".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 2

    result: dict = {
        "status": "error",
        "pid": os.getpid(),
        "binding_kwargs": dict(request.binding_kwargs),
    }
    call_metrics: dict = {}
    profile_path = output_dir / PROFILE_NAME

    try:
        # Must be configured before importing/constructing any OnnxModel.
        os.environ[PROFILE_ENV] = str(profile_path)

        binding = import_binding()
        if not binding_supports_num_server_threads(binding):
            raise RuntimeError(
                "installed _5dchess.run_selfplay has no num_server_threads "
                "argument; rebuild the extension"
            )
        requested = _requested_ints(request.binding_kwargs)

        _write_json_exclusive(
            output_dir / READY_NAME,
            {"pid": os.getpid(), "ready_epoch": time.time()},
        )
        wait_for_start(
            request.start_file, request.barrier_timeout_seconds
        )

        cpu_before = time.process_time()
        wall_started = time.perf_counter()
        epoch_started = time.time()
        try:
            raw = binding.run_selfplay(**request.binding_kwargs)
        finally:
            call_metrics = {
                "call_start_epoch": epoch_started,
                "call_end_epoch": time.time(),
                "api_wall_seconds": time.perf_counter() - wall_started,
                "run_cpu_seconds": time.process_time() - cpu_before,
            }
        result.update(call_metrics)

        if not isinstance(raw, dict):
            raise RuntimeError(
                f"run_selfplay returned {type(raw)!r}, expected dict"
            )

        stats = _extract_stats(raw)
        games = _extract_games(raw)
        samples = _extract_samples(raw)
        _validate_against_request(stats, games, requested)

        records = read_profile(profile_path)
        native_profile_summary = validate_profile(records, stats)

        result.update(
            {
                "status": "ok",
                "stats": stats,
                "samples": samples,
                "game_count": len(games),
                "game_signature": game_signature(games),
                "native_profile_summary": native_profile_summary,
            }
        )
        _write_json_exclusive(output_dir / RESULT_NAME, result)
        return 0
    except Exception as exc:
        result.update(call_metrics)
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return _write_error_result(output_dir, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.request.is_absolute():
        parser.error("--request must be an absolute path")
    return run_request(args.request)


if __name__ == "__main__":
    raise SystemExit(main())
