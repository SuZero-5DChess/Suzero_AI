"""Multi-process C++ ONNX self-play backend.

This module is both sides of the optional ``SelfPlayConfig.processes > 1``
path:

* the parent schedules one independent interpreter per process, splits the
  total worker/server-thread/game budgets, collects small progress files and
  merges the returned per-process arrays into :class:`SelfPlayResult` objects;
* ``python -m suzero_5dchess.cpp_selfplay_processes --worker request.json`` is
  the child entry point.  A child only imports the installed ``_5dchess``
  binding, calls ``run_selfplay`` once, and writes numeric ``.npz`` arrays plus
  a small JSON manifest into its exclusive directory.  It never receives the
  live network/optimizer, never exports a model, and never trains.

The parent exports the ONNX model once (in :mod:`cpp_self_play`) and passes the
same read-only paths to every child, so each child gets its own native queues
and ONNX session without duplicating the export.

Transport rules:

* arrays stay in ``.npz`` files (read with ``allow_pickle=False``) so padding
  shapes from different processes are never concatenated;
* stdout/stderr go to per-child files, never to a pipe that could fill up;
* every child is explicitly ``wait()``-ed and its exit code checked, so a
  failure during interpreter teardown is not mistaken for success;
* a first child failure terminates the remaining children immediately and
  raises ``RuntimeError`` with that child's stderr.

There is deliberately no wall-clock runtime limit: production games may take
as long as they need.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - result types are only needed by the parent
    from .self_play import SelfPlayResult

logger = logging.getLogger(__name__)

PROFILE_ENV = "SUZERO_ONNX_PROFILE_PATH"

SEED_STRIDE = 0x9E3779B97F4A7C15
UINT64_MASK = (1 << 64) - 1
POLL_SECONDS = 0.1
TERMINATE_GRACE_SECONDS = 10.0

REQUEST_NAME = "request.json"
PROGRESS_NAME = "progress.json"
ARRAYS_NAME = "samples.npz"
RESULT_NAME = "result.json"
STDOUT_NAME = "stdout.log"
STDERR_NAME = "stderr.log"

STDERR_TAIL_CHARS = 4000

# Exactly the arrays consumed by ``cpp_self_play._samples_by_game``.  Keeping
# the list explicit avoids accidentally serialising metadata or object arrays.
TRANSPORT_ARRAYS = (
    "board_codes",
    "l_coords",
    "t_coords",
    "used_board_counts",
    "urgency",
    "side_to_move",
    "actions",
    "action_counts",
    "policy",
    "has_submit",
    "value",
    "player",
    "move_count",
    "game_index",
)

ProgressCallback = Callable[[int], None]


# --------------------------------------------------------------------- planning


@dataclass(frozen=True)
class ProcessPlan:
    index: int
    workers: int
    games: int
    server_threads: int
    seed: int


class _WorkerExit(RuntimeError):
    """Internal signal: a child exited non-zero while others may still run."""

    def __init__(self, index: int, code: int) -> None:
        super().__init__(f"worker {index} exited with code {code}")
        self.index = index
        self.code = code


def split_total(total: int, parts: int) -> list[int]:
    """Split ``total`` evenly, giving the remainder to the first processes."""

    total = int(total)
    parts = int(parts)
    if parts < 1:
        raise ValueError("parts must be >= 1")
    if total < parts:
        raise ValueError(
            f"total {total} is smaller than the {parts} processes; every "
            "process needs at least one"
        )
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def process_seed(base_seed: int, game_offset: int) -> int:
    """Offset the native seed stream by the number of preceding games."""

    return (int(base_seed) + int(game_offset) * SEED_STRIDE) & UINT64_MASK


def build_plan(
    processes: int,
    total_workers: int,
    total_server_threads: int,
    games: int,
    seed: int,
) -> list[ProcessPlan]:
    """Allocate the three total budgets across ``processes`` children.

    All three values are totals: they are split evenly with the remainder going
    to the lowest indices.  They must each be at least ``processes``, but need
    not divide evenly.  ``games`` may be smaller than ``total_workers``; the
    native bridge then only activates ``min(workers, games)`` threads for that
    child.
    """

    processes = int(processes)
    total_workers = int(total_workers)
    total_server_threads = int(total_server_threads)
    games = int(games)
    if processes < 1:
        raise ValueError("processes must be >= 1")
    if total_workers < processes:
        raise ValueError(
            f"total workers {total_workers} < processes {processes}"
        )
    if total_server_threads < processes:
        raise ValueError(
            "total server threads "
            f"{total_server_threads} < processes {processes}"
        )
    if games < processes:
        raise ValueError(f"total games {games} < processes {processes}")

    worker_split = split_total(total_workers, processes)
    server_split = split_total(total_server_threads, processes)
    game_split = split_total(games, processes)
    plan = []
    game_offset = 0
    for index in range(processes):
        plan.append(
            ProcessPlan(
                index=index,
                workers=worker_split[index],
                games=game_split[index],
                server_threads=server_split[index],
                seed=process_seed(seed, game_offset),
            )
        )
        game_offset += game_split[index]
    return plan


def child_binding_kwargs(
    binding_kwargs: Mapping[str, Any], plan: ProcessPlan
) -> dict[str, Any]:
    """Copy the parent's binding kwargs with this child's share."""

    kwargs = dict(binding_kwargs)
    kwargs["num_games"] = plan.games
    kwargs["num_threads"] = plan.workers
    kwargs["num_server_threads"] = plan.server_threads
    kwargs["seed"] = plan.seed
    return kwargs


# ------------------------------------------------------------------------- io


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value)!r} is not JSON serializable")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a small JSON file atomically (``os.replace``)."""

    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                ensure_ascii=False,
                default=_json_default,
            )
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write numeric arrays without pickling, deflated for the child handoff."""

    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    """Best-effort progress update.

    On Windows a concurrent reader (or an antivirus scan) can briefly hold the
    destination and make ``os.replace`` raise ``PermissionError``.  Progress is
    auxiliary: retry a couple of times, then silently leave the previous value.
    The parent fills the final value from the games that actually completed.
    """

    for attempt in range(3):
        try:
            _atomic_json(path, payload)
            return
        except OSError:
            if attempt == 2:
                return
            time.sleep(0.01)


def _read_text_tail(path: Path, limit: int = STDERR_TAIL_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _child_env(package_root: Path) -> dict[str, str]:
    """Environment for a child interpreter.

    An inherited ``SUZERO_ONNX_PROFILE_PATH`` is removed so several processes
    can never append to the same native profile file.  The package root is
    prepended to ``PYTHONPATH`` because children are started with ``-m``.
    """

    env = os.environ.copy()
    env.pop(PROFILE_ENV, None)
    existing = env.get("PYTHONPATH")
    package_path = str(package_root)
    env["PYTHONPATH"] = (
        package_path + os.pathsep + existing if existing else package_path
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    return env


def worker_command(request_path: Path) -> list[str]:
    """Command line for one child.  Kept separate so tests can substitute it."""

    return [
        sys.executable,
        "-u",
        "-m",
        "suzero_5dchess.cpp_selfplay_processes",
        "--worker",
        str(request_path),
    ]


# ----------------------------------------------------------------- child side


def _load_request(request_path: Path) -> tuple[Path, dict[str, Any]]:
    if not request_path.is_absolute():
        raise ValueError(f"request path must be absolute: {request_path}")
    data = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("request must be a JSON object")

    output_dir = Path(data["output_dir"])
    if not output_dir.is_absolute():
        raise ValueError("output_dir must be absolute")
    if not output_dir.is_dir():
        raise ValueError(f"output_dir does not exist: {output_dir}")

    kwargs = data.get("binding_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("binding_kwargs must be a JSON object")
    if "progress" in kwargs:
        raise ValueError("binding_kwargs must not contain a progress callback")
    try:
        games = int(kwargs["num_games"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("binding_kwargs.num_games must be an integer") from exc
    if games < 1:
        raise ValueError("binding_kwargs.num_games must be >= 1")
    return output_dir, dict(kwargs)


def _load_binding() -> Any:
    """Import the installed extension only when a child actually runs."""

    from . import env as env_module

    if env_module._5dchess is None:
        raise RuntimeError(
            "the _5dchess extension failed to import: "
            f"{env_module._IMPORT_ERROR}"
        )
    return env_module._5dchess


def _extract_arrays(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in TRANSPORT_ARRAYS:
        if key == "board_codes":
            # Prefer codes when a caller already provides them; otherwise
            # collapse the native one-hot planes before writing the npz, which
            # keeps the child -> parent transport ~48x smaller.
            source = raw.get("board_codes")
            if source is None:
                source = raw.get("board_planes")
            if source is None:
                raise RuntimeError(
                    "run_selfplay result is missing array 'board_planes'"
                )
            value = np.asarray(source)
            if value.dtype == object:
                raise RuntimeError(
                    "run_selfplay result array 'board_planes' is not numeric"
                )
            if value.dtype == np.uint8 and value.ndim in (2, 3):
                pass  # already codes (flat [boards, S] or [samples, boards, S])
            elif value.ndim == 3:
                from .sample_codec import planes_to_codes

                if value.shape[0] == 0:
                    # A child that produced no samples still carries an empty
                    # payload; keep its square count without validating boards.
                    value = np.zeros((0, value.shape[2]), dtype=np.uint8)
                else:
                    value = planes_to_codes(value)
            else:
                raise RuntimeError(
                    "run_selfplay result array 'board_codes' must be uint8 [N, S]"
                )
            arrays[key] = value
            continue
        if key not in raw:
            raise RuntimeError(f"run_selfplay result is missing array {key!r}")
        value = np.asarray(raw[key])
        if value.dtype == object:
            raise RuntimeError(f"run_selfplay result array {key!r} is not numeric")
        arrays[key] = value
    for key in ("board_offsets", "action_offsets"):
        if key in raw:
            value = np.asarray(raw[key])
            if value.dtype != np.int64:
                raise RuntimeError(f"run_selfplay result array {key!r} must be int64")
            arrays[key] = value
    return arrays


def _extract_games(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    games = raw.get("games")
    if not isinstance(games, (list, tuple)):
        raise RuntimeError("run_selfplay result has no 'games' sequence")
    result: list[dict[str, Any]] = []
    for entry in games:
        if not isinstance(entry, Mapping):
            raise RuntimeError("run_selfplay game entry is not a mapping")
        result.append(dict(entry))
    return result


def _extract_stats(raw: Mapping[str, Any]) -> dict[str, Any]:
    stats = raw.get("stats", {})
    if not isinstance(stats, Mapping):
        raise RuntimeError("run_selfplay result 'stats' is not a mapping")
    return dict(stats)


def _write_error_result(
    result_path: Path, exc: BaseException
) -> None:
    try:
        _atomic_json(
            result_path,
            {
                "status": "error",
                "pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
    except Exception:
        # The parent still sees the non-zero exit code and stderr.
        pass


def run_worker(request_path: str | os.PathLike[str]) -> int:
    """Child entry: run exactly one native ``run_selfplay`` call."""

    # A parent must never share a native profile file between interpreters.
    os.environ.pop(PROFILE_ENV, None)

    path = Path(request_path)
    try:
        output_dir, kwargs = _load_request(path)
    except Exception as exc:
        print(f"self-play worker: invalid request {path}: {exc}", file=sys.stderr)
        return 2

    games = int(kwargs["num_games"])
    progress_path = output_dir / PROGRESS_NAME
    arrays_path = output_dir / ARRAYS_NAME
    result_path = output_dir / RESULT_NAME
    existing = [item for item in (arrays_path, result_path) if item.exists()]
    if existing:
        print(
            "self-play worker: refusing to overwrite "
            + ", ".join(str(item) for item in existing),
            file=sys.stderr,
        )
        return 2

    try:
        _write_progress(progress_path, {"done": 0, "games": games})
        binding = _load_binding()

        # The native bridge invokes this from several game threads, and the
        # Python binding only takes the GIL (file I/O releases it).  Hold the
        # lock across the whole atomic write: every callback in this process
        # shares the same ``.tmp`` path, and ``done`` must never move back.
        lock = threading.Lock()
        last = [0]

        def on_game(done: int, _index: int) -> None:
            try:
                value = int(done)
            except (TypeError, ValueError):
                return
            value = max(0, min(value, games))
            with lock:
                if value < last[0]:
                    return
                last[0] = value
                # Never let an auxiliary progress write abort the batch.
                _write_progress(
                    progress_path, {"done": value, "games": games}
                )

        raw = binding.run_selfplay(**kwargs, progress=on_game)
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                f"run_selfplay returned {type(raw)!r}, expected a mapping"
            )

        arrays = _extract_arrays(raw)
        game_list = _extract_games(raw)
        stats = _extract_stats(raw)
        if len(game_list) != games:
            raise RuntimeError(
                f"run_selfplay returned {len(game_list)} games, requested "
                f"{games}"
            )

        _atomic_npz(arrays_path, arrays)
        _atomic_json(
            result_path,
            {
                "status": "ok",
                "pid": os.getpid(),
                "games": game_list,
                "stats": stats,
                "samples": int(arrays["used_board_counts"].shape[0]),
            },
        )
        _write_progress(progress_path, {"done": games, "games": games})
        return 0
    except Exception as exc:
        _write_error_result(result_path, exc)
        print(
            f"self-play worker failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return 1


# ----------------------------------------------------------------- parent side


def _read_progress(child_dir: Path, games: int, previous: int) -> int:
    path = child_dir / PROGRESS_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        done = int(payload.get("done", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return previous
    return max(previous, max(0, min(done, games)))


def _wait_for_children(
    procs: Sequence[subprocess.Popen],
    child_dirs: Sequence[Path],
    plan: Sequence[ProcessPlan],
    progress: ProgressCallback | None,
) -> int:
    """Poll progress and exit codes until every child has finished.

    Returns the highest total progress reported while waiting.  There is no
    deadline.  The first non-zero exit raises ``_WorkerExit`` so the caller can
    stop the remaining children immediately.
    """

    scene = [0] * len(plan)
    reported = 0

    while True:
        total = 0
        for index, item in enumerate(plan):
            scene[index] = _read_progress(
                child_dirs[index], item.games, scene[index]
            )
            total += scene[index]
        if callable(progress) and total > reported:
            reported = total
            progress(reported)

        active = 0
        failure: tuple[int, int] | None = None
        for index, proc in enumerate(procs):
            code = proc.poll()
            if code is None:
                active += 1
            elif code != 0 and failure is None:
                failure = (index, int(code))
        if failure is not None:
            raise _WorkerExit(failure[0], failure[1])
        if active == 0:
            break
        time.sleep(POLL_SECONDS)

    # ``poll`` already reaped them, but wait explicitly so every real exit
    # code (including a failure during interpreter teardown) is surfaced.
    for proc in procs:
        code = proc.wait()
        if code != 0:
            raise _WorkerExit(procs.index(proc), int(code))

    return reported


def _terminate_all(
    procs: Sequence[subprocess.Popen], grace: float = TERMINATE_GRACE_SECONDS
) -> None:
    """Terminate only our own children, then reap them."""

    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=grace)
            except Exception:
                pass
        except Exception:
            pass


def _load_child_results(child_dir: Path, expected_games: int) -> list[SelfPlayResult]:
    """Read one child's transport files and convert them in isolation."""

    result_path = child_dir / RESULT_NAME
    arrays_path = child_dir / ARRAYS_NAME
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"self-play worker result is missing or unreadable: {result_path}: "
            f"{exc}"
        ) from exc

    if not isinstance(payload, dict) or payload.get("status") != "ok":
        detail = (
            payload.get("error") if isinstance(payload, dict) else None
        )
        raise RuntimeError(
            f"self-play worker did not report success: {result_path}: {detail!r}"
        )

    game_list = payload.get("games")
    if not isinstance(game_list, list):
        raise RuntimeError(f"self-play worker result has no game list: {result_path}")
    if len(game_list) != expected_games:
        raise RuntimeError(
            f"self-play worker returned {len(game_list)} games, expected "
            f"{expected_games}"
        )

    indices = sorted(int(entry["index"]) for entry in game_list)
    if indices != list(range(expected_games)):
        raise RuntimeError(
            "self-play worker game indices are not the expected local "
            f"0..{expected_games - 1} range: {indices}"
        )

    try:
        with np.load(arrays_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
            game_index = arrays.get("game_index")
            if game_index is not None and game_index.size:
                low = int(game_index.min())
                high = int(game_index.max())
                if low < 0 or high >= expected_games:
                    raise RuntimeError(
                        "self-play worker sample game_index is outside its "
                        f"local 0..{expected_games - 1} range: [{low}, {high}]"
                    )
            raw: dict[str, Any] = dict(arrays)
            raw["games"] = game_list
            # Conversion is parent-only; workers serialise the native arrays.
            from .cpp_self_play import _selfplay_results

            results = _selfplay_results(raw)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"self-play worker arrays are missing or unreadable: {arrays_path}: "
            f"{exc}"
        ) from exc

    if len(results) != expected_games:
        raise RuntimeError(
            f"self-play worker produced {len(results)} converted games, "
            f"expected {expected_games}"
        )
    return results


def _failure_message(
    index: int,
    code: int,
    procs: Sequence[subprocess.Popen],
    stderr_paths: Sequence[Path],
) -> str:
    pid = getattr(procs[index], "pid", "?")
    stderr = _read_text_tail(stderr_paths[index]).strip() or "<empty>"
    return (
        f"self-play worker {index} (pid {pid}) exited with code {code}; "
        "its remaining siblings were stopped. stderr:\n"
        f"{stderr}"
    )


def _read_child_stats(child_dir: Path) -> dict[str, Any]:
    """Read the small stats object a child stored next to its arrays."""

    try:
        payload = json.loads(
            (child_dir / RESULT_NAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return {}
    stats = payload.get("stats") if isinstance(payload, dict) else None
    return dict(stats) if isinstance(stats, Mapping) else {}


def _log_summary(
    plan: Sequence[ProcessPlan],
    total_workers: int,
    total_server_threads: int,
    games: int,
    results: Sequence[SelfPlayResult],
    child_stats: Sequence[Mapping[str, Any]],
    wall_seconds: float,
) -> None:
    samples = sum(len(result.samples) for result in results)
    backend_rows = 0
    setup_values: list[float] = []
    for stats in child_stats:
        try:
            backend_rows += int(stats.get("rows", 0))
        except (TypeError, ValueError):
            pass
        if "setup_seconds" in stats:
            try:
                setup_values.append(float(stats["setup_seconds"]))
            except (TypeError, ValueError):
                pass

    active_threads = sum(min(item.workers, item.games) for item in plan)
    samples_rate = samples / wall_seconds if wall_seconds > 0 else 0.0
    rows_rate = backend_rows / wall_seconds if wall_seconds > 0 else 0.0
    logger.info(
        "cpp self-play multiprocess: processes=%d workers=%d "
        "server_threads=%d active_game_threads=%d games=%d samples=%d "
        "backend_rows=%d",
        len(plan),
        total_workers,
        total_server_threads,
        active_threads,
        games,
        samples,
        backend_rows,
    )
    # Only the parent's real end-to-end span is a denominator.  Child
    # total/setup seconds are raw diagnostic values, never summed into it.
    logger.info(
        "cpp self-play multiprocess wall: %.2fs parent end-to-end "
        "(Popen startup + self-play + IPC/result load): %.1f samples/s, "
        "%.1f backend_rows/s on that wall (per-child rates are not summed)",
        wall_seconds,
        samples_rate,
        rows_rate,
    )
    if setup_values:
        logger.info(
            "cpp self-play multiprocess setup: per_process_seconds=%s "
            "max_seconds=%.2f (raw child values, not a denominator)",
            [round(value, 3) for value in setup_values],
            max(setup_values),
        )


def run_self_play_processes(
    binding_kwargs: Mapping[str, Any],
    *,
    processes: int,
    total_server_threads: int,
    progress: ProgressCallback | None = None,
) -> list[SelfPlayResult]:
    """Run one iteration's self-play across ``processes`` child interpreters.

    ``binding_kwargs`` is the parent's original ``run_selfplay`` keyword set
    with the ``progress`` callback removed.  ``num_games`` and ``num_threads``
    are treated as totals and split across children.
    """

    kwargs = dict(binding_kwargs)
    if "progress" in kwargs:
        raise ValueError("binding_kwargs must not contain progress")
    processes = int(processes)
    if processes < 2:
        raise ValueError(
            "run_self_play_processes requires processes >= 2; the single-process "
            "path stays in cpp_self_play.generate_games"
        )
    try:
        total_workers = int(kwargs["num_threads"])
        games = int(kwargs["num_games"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "binding_kwargs must contain integer num_threads and num_games"
        ) from exc
    base_seed = int(kwargs.get("seed", 0)) & UINT64_MASK

    plan = build_plan(
        processes, total_workers, total_server_threads, games, base_seed
    )
    package_root = Path(__file__).resolve().parents[1]
    env = _child_env(package_root)
    work_root = Path(tempfile.mkdtemp(prefix="suzero_cpp_selfplay_"))

    procs: list[subprocess.Popen] = []
    handles: list[tuple[Any, Any]] = []
    child_dirs: list[Path] = []
    stderr_paths: list[Path] = []
    started = time.perf_counter()
    keep_diagnostics = False

    try:
        for item in plan:
            child_dir = work_root / f"child_{item.index:02d}"
            child_dir.mkdir()
            child_dirs.append(child_dir)

            request_path = child_dir / REQUEST_NAME
            _atomic_json(
                request_path,
                {
                    "output_dir": str(child_dir.resolve()),
                    "binding_kwargs": child_binding_kwargs(kwargs, item),
                },
            )

            stderr_path = child_dir / STDERR_NAME
            stdout_handle = (child_dir / STDOUT_NAME).open("wb")
            stderr_handle = stderr_path.open("wb")
            handles.append((stdout_handle, stderr_handle))
            stderr_paths.append(stderr_path)

            popen_kwargs: dict[str, Any] = {}
            if os.name == "nt":
                # Never let the first console Ctrl+C land in a child group.
                popen_kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
            else:
                popen_kwargs["start_new_session"] = True

            proc = subprocess.Popen(
                worker_command(request_path.resolve()),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                **popen_kwargs,
            )
            procs.append(proc)

        try:
            reported = _wait_for_children(procs, child_dirs, plan, progress)
        except _WorkerExit as exc:
            keep_diagnostics = True
            _terminate_all(procs)
            raise RuntimeError(
                _failure_message(exc.index, exc.code, procs, stderr_paths)
                + f"\ndiagnostics directory: {work_root.resolve()}"
            ) from None
        except BaseException:
            _terminate_all(procs)
            raise

        results: list[SelfPlayResult] = []
        child_stats: list[Mapping[str, Any]] = []
        for item, child_dir in zip(plan, child_dirs):
            results.extend(_load_child_results(child_dir, item.games))
            child_stats.append(_read_child_stats(child_dir))

        if len(results) != games:
            raise RuntimeError(
                f"multiprocess self-play returned {len(results)} games, "
                f"expected {games}"
            )

        # Fill the final progress value from what actually succeeded, even if
        # a child's auxiliary progress files were unwritable on Windows.
        if callable(progress) and len(results) > reported:
            progress(len(results))

        wall = time.perf_counter() - started
        _log_summary(
            plan,
            total_workers,
            total_server_threads,
            games,
            results,
            child_stats,
            wall,
        )
        return results
    except BaseException:
        _terminate_all(procs)
        raise
    finally:
        _terminate_all(procs)
        for stdout_handle, stderr_handle in handles:
            for handle in (stdout_handle, stderr_handle):
                try:
                    handle.close()
                except OSError:
                    pass
        if not keep_diagnostics:
            shutil.rmtree(work_root, ignore_errors=True)


# --------------------------------------------------------------------- child CLI


def _ignore_sigint() -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        # Not the main thread (or an unsupported platform): the parent still
        # detaches the child group, and a second Ctrl+C is handled there.
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker",
        required=True,
        type=Path,
        help="absolute path to this child's request JSON",
    )
    args = parser.parse_args(argv)
    _ignore_sigint()
    return run_worker(args.worker)


if __name__ == "__main__":
    raise SystemExit(main())
