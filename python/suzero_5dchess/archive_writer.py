"""Asynchronous GN5D archive writing in one persistent subprocess."""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
import types
from collections import deque
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import IO, Mapping, Sequence


# Running this file directly is deliberate: ``python -m suzero_5dchess...``
# would execute the package's heavyweight __init__ before reaching this module.
# Install a package shell instead, so relative imports resolve without importing
# the training stack (and therefore torch/onnxruntime) in the archive worker.
if __package__ in (None, ""):
    _PACKAGE_NAME = "suzero_5dchess"
    _PACKAGE_DIR = Path(__file__).resolve().parent
    if _PACKAGE_NAME not in sys.modules:
        _package = types.ModuleType(_PACKAGE_NAME)
        _package.__file__ = str(_PACKAGE_DIR / "__init__.py")
        _package.__package__ = _PACKAGE_NAME
        _package.__path__ = [str(_PACKAGE_DIR)]  # type: ignore[attr-defined]
        _package.__spec__ = ModuleSpec(
            _PACKAGE_NAME, loader=None, is_package=True
        )
        _package.__spec__.submodule_search_locations = [str(_PACKAGE_DIR)]
        sys.modules[_PACKAGE_NAME] = _package
    __package__ = _PACKAGE_NAME

from .protocol import VariantSpec

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_SIZE = 2
_QUEUE_POLL_SECONDS = 0.1
_JOIN_TIMEOUT_SECONDS = 10.0
_STDERR_TAIL_LINES = 100


@dataclass(frozen=True)
class ArchiveMove:
    """Primitive-only transport form of one canonical recorded action."""

    action_type: str
    player: int
    source: tuple[int, int, int, int] | None = None
    target: tuple[int, int, int, int] | None = None
    promotion: str | None = None

    @classmethod
    def from_mapping(cls, raw: object) -> ArchiveMove:
        if not isinstance(raw, Mapping):
            raise ValueError("recorded action must be a mapping")
        action_type = str(raw.get("type", ""))
        player = int(raw.get("player"))  # type: ignore[arg-type]
        if action_type == "submit":
            return cls(action_type=action_type, player=player)
        if action_type != "move":
            raise ValueError("recorded action must be a canonical move or submit")
        source = _coordinate(raw.get("from"), "from")
        target = _coordinate(raw.get("to"), "to")
        promotion = str(raw.get("promotion", ""))
        return cls(action_type, player, source, target, promotion)

    def as_mapping(self) -> dict[str, object]:
        if self.action_type == "submit":
            return {"type": "submit", "player": self.player}
        assert self.source is not None and self.target is not None
        return {
            "type": "move",
            "player": self.player,
            "from": list(self.source),
            "to": list(self.target),
            "promotion": self.promotion,
        }


def _coordinate(value: object, label: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"recorded action {label} coordinate must contain four integers")
    converted = tuple(int(item) for item in value)
    return converted  # type: ignore[return-value]


@dataclass(frozen=True)
class ArchiveGame:
    """Self-play result fields needed by :func:`save_game_record` only."""

    moves: tuple[ArchiveMove, ...]
    white_outcome: float
    terminal_reason: str
    semimoves: int
    turns: int
    board_limit: int
    board_count: int
    final_player: int
    final_fen: str

    @classmethod
    def from_result(cls, result: object) -> ArchiveGame:
        raw_moves = getattr(result, "moves")
        if not isinstance(raw_moves, list):
            raise ValueError("result.moves must be a list")
        return cls(
            moves=tuple(ArchiveMove.from_mapping(move) for move in raw_moves),
            white_outcome=float(getattr(result, "white_outcome")),
            terminal_reason=str(getattr(result, "terminal_reason")),
            semimoves=int(getattr(result, "semimoves")),
            turns=int(getattr(result, "turns")),
            board_limit=int(getattr(result, "board_limit")),
            board_count=int(getattr(result, "board_count")),
            final_player=int(getattr(result, "final_player")),
            final_fen=str(getattr(result, "final_fen")),
        )

    def as_record_result(self) -> _RecordResult:
        return _RecordResult(
            moves=[move.as_mapping() for move in self.moves],
            white_outcome=self.white_outcome,
            terminal_reason=self.terminal_reason,
            semimoves=self.semimoves,
            turns=self.turns,
            board_limit=self.board_limit,
            board_count=self.board_count,
            final_player=self.final_player,
            final_fen=self.final_fen,
        )


@dataclass(frozen=True)
class ArchiveVariant:
    """Explicit, primitive-only transport form of ``VariantSpec``."""

    name: str
    pgn: str
    board_side: int
    line_shift: int
    default_board_limit: int
    min_board_limit: int
    max_board_limit: int

    @classmethod
    def from_spec(cls, spec: VariantSpec) -> ArchiveVariant:
        return cls(
            name=str(spec.name),
            pgn=str(spec.pgn),
            board_side=int(spec.board_side),
            line_shift=int(spec.line_shift),
            default_board_limit=int(spec.default_board_limit),
            min_board_limit=int(spec.min_board_limit),
            max_board_limit=int(spec.max_board_limit),
        )

    def as_spec(self) -> VariantSpec:
        return VariantSpec(
            name=self.name,
            pgn=self.pgn,
            board_side=self.board_side,
            line_shift=self.line_shift,
            default_board_limit=self.default_board_limit,
            min_board_limit=self.min_board_limit,
            max_board_limit=self.max_board_limit,
        )


@dataclass(frozen=True)
class ArchiveBatch:
    """Serialization-safe payload for one complete self-play round."""

    batch_id: int
    games: tuple[ArchiveGame, ...]
    spec: ArchiveVariant
    log_dir: str
    iteration: int
    first_game_id: int
    backend: str
    material_scale: float

    @classmethod
    def from_results(
        cls,
        results: Sequence[object],
        spec: VariantSpec,
        log_dir: str | Path,
        *,
        iteration: int,
        first_game_id: int,
        backend: str,
        material_scale: float = 2.0,
        batch_id: int = 0,
    ) -> ArchiveBatch:
        return cls(
            batch_id=int(batch_id),
            games=tuple(ArchiveGame.from_result(result) for result in results),
            spec=ArchiveVariant.from_spec(spec),
            log_dir=str(log_dir),
            iteration=int(iteration),
            first_game_id=int(first_game_id),
            backend=str(backend),
            material_scale=float(material_scale),
        )


@dataclass(frozen=True)
class ArchiveBatchCompletion:
    """Batch-level completion status returned by ``check`` and ``flush``."""

    batch_id: int
    game_count: int


@dataclass(frozen=True)
class _RecordResult:
    moves: list[dict[str, object]]
    white_outcome: float
    terminal_reason: str
    semimoves: int
    turns: int
    board_limit: int
    board_count: int
    final_player: int
    final_fen: str


@dataclass(frozen=True)
class _WorkerStatus:
    kind: str
    batch_id: int | None
    game_count: int
    message: str = ""
    traceback_text: str = ""


def _batch_to_wire(batch: ArchiveBatch) -> bytes:
    document = {
        "batch_id": batch.batch_id,
        "games": [
            {
                "moves": [move.as_mapping() for move in game.moves],
                "white_outcome": game.white_outcome,
                "terminal_reason": game.terminal_reason,
                "semimoves": game.semimoves,
                "turns": game.turns,
                "board_limit": game.board_limit,
                "board_count": game.board_count,
                "final_player": game.final_player,
                "final_fen": game.final_fen,
            }
            for game in batch.games
        ],
        "spec": {
            "name": batch.spec.name,
            "pgn": batch.spec.pgn,
            "board_side": batch.spec.board_side,
            "line_shift": batch.spec.line_shift,
            "default_board_limit": batch.spec.default_board_limit,
            "min_board_limit": batch.spec.min_board_limit,
            "max_board_limit": batch.spec.max_board_limit,
        },
        "log_dir": batch.log_dir,
        "iteration": batch.iteration,
        "first_game_id": batch.first_game_id,
        "backend": batch.backend,
        "material_scale": batch.material_scale,
    }
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _batch_from_wire(payload: bytes) -> ArchiveBatch:
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise TypeError("archive payload must be an object")
    raw_spec = raw["spec"]
    raw_games = raw["games"]
    if not isinstance(raw_spec, dict) or not isinstance(raw_games, list):
        raise TypeError("archive payload has invalid spec or games")
    spec = ArchiveVariant(
        name=str(raw_spec["name"]),
        pgn=str(raw_spec["pgn"]),
        board_side=int(raw_spec["board_side"]),
        line_shift=int(raw_spec["line_shift"]),
        default_board_limit=int(raw_spec["default_board_limit"]),
        min_board_limit=int(raw_spec["min_board_limit"]),
        max_board_limit=int(raw_spec["max_board_limit"]),
    )
    games: list[ArchiveGame] = []
    for raw_game in raw_games:
        if not isinstance(raw_game, dict) or not isinstance(raw_game.get("moves"), list):
            raise TypeError("archive payload contains an invalid game")
        games.append(
            ArchiveGame(
                moves=tuple(
                    ArchiveMove.from_mapping(move) for move in raw_game["moves"]
                ),
                white_outcome=float(raw_game["white_outcome"]),
                terminal_reason=str(raw_game["terminal_reason"]),
                semimoves=int(raw_game["semimoves"]),
                turns=int(raw_game["turns"]),
                board_limit=int(raw_game["board_limit"]),
                board_count=int(raw_game["board_count"]),
                final_player=int(raw_game["final_player"]),
                final_fen=str(raw_game["final_fen"]),
            )
        )
    return ArchiveBatch(
        batch_id=int(raw["batch_id"]),
        games=tuple(games),
        spec=spec,
        log_dir=str(raw["log_dir"]),
        iteration=int(raw["iteration"]),
        first_game_id=int(raw["first_game_id"]),
        backend=str(raw["backend"]),
        material_scale=float(raw["material_scale"]),
    )


def _status_to_wire(status: _WorkerStatus) -> str:
    return json.dumps(
        {
            "kind": status.kind,
            "batch_id": status.batch_id,
            "game_count": status.game_count,
            "message": status.message,
            "traceback_text": status.traceback_text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _status_from_wire(line: str) -> _WorkerStatus:
    raw = json.loads(line)
    if not isinstance(raw, dict):
        raise TypeError("worker status must be an object")
    batch_id = raw.get("batch_id")
    return _WorkerStatus(
        kind=str(raw["kind"]),
        batch_id=None if batch_id is None else int(batch_id),
        game_count=int(raw["game_count"]),
        message=str(raw.get("message", "")),
        traceback_text=str(raw.get("traceback_text", "")),
    )


def _emit_worker_status(status: _WorkerStatus) -> None:
    sys.stdout.write(_status_to_wire(status) + "\n")
    sys.stdout.flush()


def _worker_runtime_diagnostics() -> dict[str, bool]:
    """Report heavyweight runtimes accidentally imported by the CLI worker."""

    diagnostics = {
        "torch": "torch" in sys.modules,
        "onnxruntime": "onnxruntime" in sys.modules,
        "libiomp5md": False,
    }
    if os.name == "nt":
        import ctypes

        diagnostics["libiomp5md"] = bool(
            ctypes.windll.kernel32.GetModuleHandleW("libiomp5md.dll")
        )
    else:
        maps = Path("/proc/self/maps")
        if maps.is_file():
            diagnostics["libiomp5md"] = "libiomp5" in maps.read_text(
                encoding="utf-8", errors="ignore"
            ).lower()
    return diagnostics


def _archive_worker() -> int:
    """Read line-framed batches from stdin and report statuses on stdout."""

    # The parent owns cooperative interruption and decides whether to drain or
    # abort. CREATE_NEW_PROCESS_GROUP also prevents ordinary parent Ctrl+C from
    # being treated as worker-owned shutdown on Windows.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from .game_record import prune_game_records, save_game_record

    diagnostics = _worker_runtime_diagnostics()
    forbidden = [
        name for name in ("torch", "onnxruntime") if diagnostics[name]
    ]
    if forbidden:
        _emit_worker_status(
            _WorkerStatus(
                "error",
                None,
                0,
                "archive worker loaded forbidden runtime(s): " + ", ".join(forbidden),
            )
        )
        return 1
    _emit_worker_status(
        _WorkerStatus("ready", None, 0, json.dumps(diagnostics, sort_keys=True))
    )

    for line in sys.stdin.buffer:
        if line == b"stop\n":
            return 0
        batch_id: int | None = None
        try:
            if not line.startswith(b"batch\t"):
                raise ValueError("archive worker received an invalid command")
            batch = _batch_from_wire(line[len(b"batch\t") :].rstrip(b"\r\n"))
            batch_id = batch.batch_id
            spec = batch.spec.as_spec()
            for offset, game in enumerate(batch.games):
                save_game_record(
                    game.as_record_result(),
                    spec,
                    batch.log_dir,
                    iteration=batch.iteration,
                    game_id=batch.first_game_id + offset,
                    backend=batch.backend,
                    material_scale=batch.material_scale,
                    prune=False,
                )
            prune_game_records(batch.log_dir)
            _emit_worker_status(
                _WorkerStatus("complete", batch.batch_id, len(batch.games))
            )
        except BaseException as exc:
            _emit_worker_status(
                _WorkerStatus(
                    "error",
                    batch_id,
                    0,
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                )
            )
            return 1
    return 0


class ArchiveWriter:
    """Write complete self-play rounds through a persistent CLI subprocess.

    At most ``queue_size`` batches may wait behind the batch presumed active in
    the child, preserving bounded backpressure without multiprocessing queues.
    ``enqueue`` transfers primitive archive data and never samples or tensors.
    All public synchronization methods surface child failures.
    """

    def __init__(self, queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be at least one")
        self._queue_size = int(queue_size)
        self._process: subprocess.Popen[bytes] | None = None
        self._status_queue: queue.Queue[_WorkerStatus] = queue.Queue()
        self._stdout_closed = threading.Event()
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._reader_threads: tuple[threading.Thread, threading.Thread] | None = None
        self._ready = False
        self._worker_diagnostics: dict[str, bool] | None = None
        self._pending: set[int] = set()
        self._next_batch_id = 1
        self._failure: str | None = None
        self._closed = False
        self._stop_sent = False

    def start(self) -> None:
        """Start the child process, or do nothing when already started."""

        if self._closed:
            raise RuntimeError("archive writer is closed")
        if self._process is not None:
            self.check()
            return

        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"]
        popen_kwargs: dict[str, object] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(command, **popen_kwargs)  # type: ignore[arg-type]
            self._process = process
            assert process.stdout is not None and process.stderr is not None
            stdout_thread = threading.Thread(
                target=self._read_statuses,
                args=(process.stdout,),
                name="gn5d-archive-status-reader",
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self._read_stderr,
                args=(process.stderr,),
                name="gn5d-archive-stderr-reader",
                daemon=True,
            )
            self._reader_threads = (stdout_thread, stderr_thread)
            stdout_thread.start()
            stderr_thread.start()
            self._wait_until_ready()
        except BaseException:
            self._abort()
            self._closed = True
            raise

    def enqueue(
        self,
        results: Sequence[object],
        spec: VariantSpec,
        log_dir: str | Path,
        *,
        iteration: int,
        first_game_id: int,
        backend: str,
        material_scale: float = 2.0,
    ) -> int:
        """Enqueue one whole round and return its monotonically increasing id."""

        self._ensure_open()
        self.start()
        self.check()
        batch_id = self._next_batch_id
        batch = ArchiveBatch.from_results(
            results,
            spec,
            log_dir,
            iteration=iteration,
            first_game_id=first_game_id,
            backend=backend,
            material_scale=material_scale,
            batch_id=batch_id,
        )
        # Encode synchronously so malformed payloads fail before enqueue returns.
        payload = _batch_to_wire(batch)
        # Register before writing: a tiny batch can complete during the post-write
        # health check, and that completion must clear an already-known id.
        self._pending.add(batch_id)
        try:
            self._put_with_backpressure(payload)
        except BaseException:
            self._pending.discard(batch_id)
            raise
        self._next_batch_id += 1
        self.check()
        return batch_id

    def check(self) -> tuple[ArchiveBatchCompletion, ...]:
        """Collect available statuses and raise immediately on child failure."""

        completions = self._collect_available()
        self._raise_failure()
        self._raise_if_worker_exited()
        return tuple(completions)

    def flush(self) -> tuple[ArchiveBatchCompletion, ...]:
        """Wait until every batch enqueued so far has completed."""

        if self._process is None:
            self._ensure_open_or_closed()
            return ()
        completions = list(self.check())
        while self._pending:
            status = self._get_status(_QUEUE_POLL_SECONDS)
            if status is not None:
                completion = self._handle_status(status)
                if completion is not None:
                    completions.append(completion)
            self._raise_failure()
            self._raise_if_worker_exited()
        return tuple(completions)

    def close(self) -> None:
        """Drain queued work, stop and join the child; safe to call twice."""

        if self._closed:
            return
        try:
            if self._process is not None:
                self.check()
                self.flush()
                self._put_with_backpressure(None, stopping=True)
                self._close_stdin()
                try:
                    exit_code = self._process.wait(_JOIN_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        "archive worker did not stop after its input drained"
                    ) from exc
                self._join_reader_threads()
                self._collect_available()
                self._raise_failure()
                if exit_code != 0:
                    raise RuntimeError(self._exit_message(exit_code))
        except BaseException:
            self._abort()
            self._closed = True
            raise
        self._closed = True
        self._close_streams()

    def abort(self) -> None:
        """Terminate immediately without draining queued archive batches."""

        if self._closed:
            return
        self._abort()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("archive writer is closed")

    def _ensure_open_or_closed(self) -> None:
        self._raise_failure()

    def _put_with_backpressure(
        self, payload: bytes | None, *, stopping: bool = False
    ) -> None:
        process = self._process
        assert process is not None and process.stdin is not None

        if payload is not None:
            # One batch may be active while queue_size batches wait behind it.
            while len(self._pending) > self._queue_size + 1:
                status = self._get_status(_QUEUE_POLL_SECONDS)
                if status is not None:
                    self._handle_status(status)
                self._raise_failure()
                self._raise_if_worker_exited()
            message = b"batch\t" + payload + b"\n"
        else:
            message = b"stop\n"

        self.check()
        try:
            process.stdin.write(message)
            process.stdin.flush()
            if stopping:
                # Mark before checking because the worker may consume stop and
                # exit immediately on a fast machine.
                self._stop_sent = True
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._collect_available()
            self._raise_failure()
            exit_code = process.poll()
            raise RuntimeError(
                self._exit_message(exit_code, "archive worker input closed unexpectedly")
            ) from exc
        self.check()

    def _read_statuses(self, stream: IO[bytes]) -> None:
        try:
            for raw_line in stream:
                try:
                    line = raw_line.decode("utf-8")
                    status = _status_from_wire(line)
                except BaseException as exc:
                    status = _WorkerStatus(
                        "error",
                        None,
                        0,
                        f"invalid archive worker status: {type(exc).__name__}: {exc}",
                        traceback.format_exc(),
                    )
                self._status_queue.put(status)
        finally:
            self._stdout_closed.set()

    def _read_stderr(self, stream: IO[bytes]) -> None:
        for raw_line in stream:
            self._stderr_tail.append(raw_line.decode("utf-8", errors="replace").rstrip())

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + _JOIN_TIMEOUT_SECONDS
        while not self._ready:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("archive worker did not report ready")
            status = self._get_status(min(_QUEUE_POLL_SECONDS, remaining))
            if status is not None:
                self._handle_status(status)
            self._raise_failure()
            self._raise_if_worker_exited()

    def _get_status(self, timeout: float | None = None) -> _WorkerStatus | None:
        try:
            if timeout is None:
                return self._status_queue.get_nowait()
            return self._status_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _collect_available(self) -> list[ArchiveBatchCompletion]:
        completions: list[ArchiveBatchCompletion] = []
        while True:
            status = self._get_status()
            if status is None:
                break
            completion = self._handle_status(status)
            if completion is not None:
                completions.append(completion)
        return completions

    def _handle_status(self, status: _WorkerStatus) -> ArchiveBatchCompletion | None:
        if status.kind == "ready":
            if self._ready or status.batch_id is not None or status.game_count != 0:
                self._failure = f"archive worker returned invalid status {status!r}"
                return None
            try:
                diagnostics = json.loads(status.message)
            except (TypeError, ValueError) as exc:
                self._failure = f"archive worker returned invalid diagnostics: {exc}"
                return None
            if not isinstance(diagnostics, dict) or not all(
                isinstance(name, str) and isinstance(loaded, bool)
                for name, loaded in diagnostics.items()
            ):
                self._failure = (
                    f"archive worker returned invalid diagnostics {diagnostics!r}"
                )
                return None
            self._worker_diagnostics = diagnostics
            self._ready = True
            return None
        if status.kind == "error":
            location = (
                f"batch {status.batch_id}"
                if status.batch_id is not None
                else "an unreadable batch"
            )
            self._failure = (
                f"archive worker failed in {location}: {status.message}\n"
                f"{status.traceback_text}"
            ).rstrip()
            return None
        if status.kind != "complete" or status.batch_id is None:
            self._failure = f"archive worker returned invalid status {status!r}"
            return None
        if status.batch_id not in self._pending:
            self._failure = (
                f"archive worker completed unknown batch {status.batch_id}"
            )
            return None
        self._pending.remove(status.batch_id)
        completion = ArchiveBatchCompletion(status.batch_id, status.game_count)
        logger.info(
            "archive batch %d completed: %d games",
            completion.batch_id,
            completion.game_count,
        )
        return completion

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError(self._failure)

    def _raise_if_worker_exited(self) -> None:
        process = self._process
        if process is None or process.poll() is None or self._stop_sent:
            return
        # Once the process has exited, stdout is finite. Wait for the reader to
        # publish every final status so a worker traceback wins over bare death.
        self._stdout_closed.wait(_JOIN_TIMEOUT_SECONDS)
        self._collect_available()
        self._raise_failure()
        raise RuntimeError(
            self._exit_message(process.returncode, "archive worker exited unexpectedly")
        )

    def _exit_message(self, exit_code: int | None, prefix: str = "archive worker exited") -> str:
        details = "\n".join(self._stderr_tail).strip()
        message = f"{prefix} with code {exit_code}"
        return f"{message}\n{details}" if details else message

    def _close_stdin(self) -> None:
        process = self._process
        if process is not None and process.stdin is not None and not process.stdin.closed:
            process.stdin.close()

    def _join_reader_threads(self) -> None:
        if self._reader_threads is None:
            return
        for thread in self._reader_threads:
            thread.join(_JOIN_TIMEOUT_SECONDS)

    def _abort(self) -> None:
        process = self._process
        self._close_stdin()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(_JOIN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(_JOIN_TIMEOUT_SECONDS)
        self._join_reader_threads()
        self._close_streams()

    def _close_streams(self) -> None:
        process = self._process
        if process is None:
            return
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def _main() -> int:
    if sys.argv[1:] != ["--worker"]:
        sys.stderr.write("usage: archive_writer.py --worker\n")
        return 2
    return _archive_worker()


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ArchiveBatch",
    "ArchiveBatchCompletion",
    "ArchiveGame",
    "ArchiveMove",
    "ArchiveVariant",
    "ArchiveWriter",
]
