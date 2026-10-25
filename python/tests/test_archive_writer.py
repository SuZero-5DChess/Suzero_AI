from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from suzero_5dchess.archive_writer import ArchiveBatch, ArchiveWriter
from suzero_5dchess.protocol import VERY_SMALL
from suzero_5dchess.self_play import SelfPlayResult


def _empty_result(*, samples: list[object] | None = None) -> SelfPlayResult:
    try:
        from suzero_5dchess.env import GameEnv

        env = GameEnv(VERY_SMALL, board_limit=VERY_SMALL.default_board_limit)
        snapshot = env.record_snapshot()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"native 5D chess record API is unavailable: {exc}")
    return SelfPlayResult(
        samples=[] if samples is None else samples,  # type: ignore[arg-type]
        white_outcome=0.0,
        terminal_reason="aborted",
        semimoves=0,
        turns=0,
        board_limit=env.board_limit,
        board_count=env.board_count,
        final_player=env.current_player,
        moves=[],
        final_fen=str(snapshot["fen"]),
    )


def test_archive_batch_omits_samples_and_is_pickle_safe(tmp_path: Path) -> None:
    # A lambda is deliberately not pickleable. DTO serialization still succeeds
    # because samples (including this tensor) never enter the transport object.
    result = _empty_result(samples=[torch.tensor([1.0]), lambda: None])
    batch = ArchiveBatch.from_results(
        [result],
        VERY_SMALL,
        tmp_path,
        iteration=3,
        first_game_id=7,
        backend="test",
        material_scale=1.5,
    )

    restored = pickle.loads(pickle.dumps(batch))
    assert len(restored.games) == 1
    assert not hasattr(restored.games[0], "samples")
    assert restored.games[0].moves == ()
    assert restored.spec.name == VERY_SMALL.name
    assert restored.iteration == 3
    assert restored.first_game_id == 7


def test_enqueue_registers_batch_before_fast_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from suzero_5dchess import archive_writer as module

    writer = ArchiveWriter()
    monkeypatch.setattr(writer, "start", lambda: None)

    def complete_during_put(_payload: object, **_kwargs: object) -> None:
        writer._handle_status(module._WorkerStatus("complete", 1, 1))

    monkeypatch.setattr(writer, "_put_with_backpressure", complete_during_put)
    writer.enqueue(
        [_empty_result()],
        VERY_SMALL,
        tmp_path,
        iteration=1,
        first_game_id=1,
        backend="test",
    )

    assert writer._pending == set()
    writer.abort()


def test_spawn_writer_saves_two_game_batch_and_loads_records(tmp_path: Path) -> None:
    from suzero_5dchess.game_record import load_record

    results = [_empty_result(), _empty_result()]
    writer = ArchiveWriter(queue_size=1)
    try:
        batch_id = writer.enqueue(
            results,
            VERY_SMALL,
            tmp_path,
            iteration=4,
            first_game_id=11,
            backend="test_async",
            material_scale=2.0,
        )
        assert batch_id == 1
        writer.flush()
        writer.close()
        writer.close()
    finally:
        writer.abort()

    archives = sorted((tmp_path / "games").rglob("*.gn5d"))
    assert len(archives) == 2
    records = [load_record(path) for path in archives]
    assert {record["meta"]["game_id"] for record in records} == {11, 12}
    assert all(record["meta"]["iteration"] == 4 for record in records)
    assert all(record["meta"]["backend"] == "test_async" for record in records)


def test_worker_error_is_reported_without_hanging_close(tmp_path: Path) -> None:
    writer = ArchiveWriter(queue_size=1)
    unsupported = replace(VERY_SMALL, name="unsupported_archive_test_variant")
    try:
        writer.enqueue(
            [_empty_result()],
            unsupported,
            tmp_path,
            iteration=1,
            first_game_id=1,
            backend="test",
        )
        with pytest.raises(RuntimeError, match="archive worker failed.*unsupported variant"):
            writer.flush()
    finally:
        writer.abort()


def test_started_writer_closes_normally_and_idempotently() -> None:
    writer = ArchiveWriter()
    writer.start()
    writer.close()
    writer.close()


def test_worker_uses_standalone_cli_and_avoids_heavy_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from suzero_5dchess import archive_writer as module

    real_popen = module.subprocess.Popen
    launch: dict[str, object] = {}

    def recording_popen(*args: object, **kwargs: object) -> object:
        launch["args"] = args
        launch["kwargs"] = kwargs.copy()
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", recording_popen)
    writer = ArchiveWriter()
    try:
        writer.start()
        process = writer._process
        assert process is not None
        command = process.args
        assert isinstance(command, list)
        assert command[-1] == "--worker"
        assert "-m" not in command
        assert command[2] == str(Path(module.__file__).resolve())
        diagnostics = writer._worker_diagnostics
        assert diagnostics is not None
        assert diagnostics["torch"] is False
        assert diagnostics["onnxruntime"] is False
        assert "libiomp5md" in diagnostics
        kwargs = launch["kwargs"]
        assert isinstance(kwargs, dict)
        if module.os.name == "nt":
            assert kwargs["creationflags"] == module.subprocess.CREATE_NEW_PROCESS_GROUP
            assert "start_new_session" not in kwargs
        else:
            assert kwargs["start_new_session"] is True
            assert "creationflags" not in kwargs
        writer.close()
        assert process.returncode == 0
    finally:
        writer.abort()


def test_large_pipe_payloads_flush_without_deadlock(tmp_path: Path) -> None:
    writer = ArchiveWriter(queue_size=1)
    large_backend = "archive-pipe-probe-" + ("x" * 1_000_000)
    try:
        first = writer.enqueue(
            [],
            VERY_SMALL,
            tmp_path,
            iteration=1,
            first_game_id=1,
            backend=large_backend,
        )
        second = writer.enqueue(
            [],
            VERY_SMALL,
            tmp_path,
            iteration=1,
            first_game_id=2,
            backend=large_backend,
        )
        assert (first, second) == (1, 2)
        writer.flush()
        writer.close()
    finally:
        writer.abort()
