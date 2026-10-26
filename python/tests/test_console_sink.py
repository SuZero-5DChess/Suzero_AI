from __future__ import annotations

import sys
import threading
import time

import pytest

from suzero_5dchess import console_sink


class RecordingStream:
    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.condition = threading.Condition()
        self.paused = threading.Event()

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        # A paused console (text selected) holds the writer inside this call
        # until the selection goes away.
        while self.paused.is_set():
            time.sleep(0.005)
        with self.condition:
            self.chunks.append(text)
            self.condition.notify_all()
        return len(text)

    def flush(self) -> None:
        return None

    def output(self) -> str:
        with self.condition:
            return "".join(self.chunks)

    def wait_for(self, text: str, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        with self.condition:
            while text not in "".join(self.chunks):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
        return True


def test_writes_reach_the_stream_from_a_background_thread() -> None:
    stream = RecordingStream()
    sink = console_sink.ConsoleSink(stream)
    try:
        assert sink.write("hello ") == 6
        assert sink.write("world\n") == 6
        assert stream.wait_for("world")
        assert stream.output() == "hello world\n"
    finally:
        sink.close()


def test_write_never_blocks_while_the_console_is_paused() -> None:
    stream = RecordingStream()
    sink = console_sink.ConsoleSink(stream, max_chunks=2)
    try:
        stream.paused.set()
        started = time.monotonic()
        for index in range(50):
            sink.write(f"line {index}\n")
        sink.flush()
        assert time.monotonic() - started < 1.0
        assert sink.dropped > 0
        assert sink.pending <= 2
    finally:
        stream.paused.clear()
        sink.close()


def test_dropped_output_is_announced_once_the_console_resumes() -> None:
    stream = RecordingStream()
    sink = console_sink.ConsoleSink(stream, max_chunks=1)
    try:
        stream.paused.set()
        sink.write("first\n")
        deadline = time.monotonic() + 2.0
        while sink.pending and time.monotonic() < deadline:
            time.sleep(0.005)  # wait until the writer holds "first"
        assert sink.pending == 0
        sink.write("second\n")
        sink.write("third\n")
        assert sink.dropped == 1
        stream.paused.clear()
        assert stream.wait_for("second")
        # The queue is drained again, so this write is kept, not dropped.
        sink.write("after\n")
        assert stream.wait_for("after")
        output = stream.output()
        assert console_sink.DROPPED_NOTICE % 1 in output
        assert "after" in output
        assert sink.dropped == 0
    finally:
        stream.paused.clear()
        sink.close()


def test_close_is_idempotent_and_stops_accepting_output() -> None:
    stream = RecordingStream()
    sink = console_sink.ConsoleSink(stream)
    sink.write("kept\n")
    assert stream.wait_for("kept")
    sink.close()
    sink.close()
    assert sink.write("dropped\n") == len("dropped\n")
    assert sink.dropped == 1
    assert "dropped" not in stream.output()


def test_install_is_skipped_for_redirected_output(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    console_sink.reset()
    try:
        assert console_sink.install(RecordingStream()) is not None
        console_sink.reset()
        assert console_sink.install(_RedirectedStream()) is None
    finally:
        console_sink.reset()


def test_install_is_skipped_outside_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    console_sink.reset()
    try:
        assert console_sink.install(RecordingStream()) is None
        assert console_sink.active() is None
    finally:
        console_sink.reset()


def test_install_shares_one_sink(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    console_sink.reset()
    stream = RecordingStream()
    try:
        first = console_sink.install(stream)
        second = console_sink.install(stream)
        assert first is not None and first is second
        assert console_sink.active() is first
    finally:
        console_sink.reset()
    assert console_sink.active() is None


class _RedirectedStream:
    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


@pytest.mark.parametrize("chunks", [0, -1])
def test_invalid_queue_size_is_rejected(chunks: int) -> None:
    with pytest.raises(ValueError):
        console_sink.ConsoleSink(RecordingStream(), max_chunks=chunks)
