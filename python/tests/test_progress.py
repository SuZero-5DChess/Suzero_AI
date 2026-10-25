"""Self-play progress display: one line on a terminal, silent when redirected."""

from __future__ import annotations

import io
import logging

import pytest

from suzero_5dchess import progress


class TerminalStream(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_bar_updates_on_a_single_line() -> None:
    stream = TerminalStream()
    with progress.self_play_progress("第 2 轮自对弈", 3, stream=stream) as advance:
        for done in range(1, 4):
            advance(done)
    output = stream.getvalue()
    assert "第 2 轮自对弈" in output
    assert "3/3" in output
    assert output.count("\n") == 1
    assert output.count("\r") >= 3


def test_redirected_stream_stays_silent() -> None:
    stream = io.StringIO()
    with progress.self_play_progress("第 1 轮自对弈", 3, stream=stream) as advance:
        for done in range(1, 4):
            advance(done)
    assert stream.getvalue() == ""


def test_redirected_stream_logs_periodically(caplog: pytest.LogCaptureFixture) -> None:
    stream = io.StringIO()
    with caplog.at_level(logging.INFO, logger=progress.__name__):
        with progress.self_play_progress("第 1 轮自对弈", 10, stream=stream) as advance:
            for done in range(1, 11):
                advance(done)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 10
    assert messages[-1] == "第 1 轮自对弈: 10/10 games"


def test_advance_ignores_repeated_or_lower_counts() -> None:
    stream = TerminalStream()
    with progress.self_play_progress("第 1 轮自对弈", 3, stream=stream) as advance:
        advance(2)
        advance(2)
        advance(1)
    assert "3/3" not in stream.getvalue()


def test_missing_tqdm_falls_back_to_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(progress, "tqdm", None)
    stream = TerminalStream()
    with caplog.at_level(logging.INFO, logger=progress.__name__):
        with progress.self_play_progress("第 1 轮自对弈", 2, stream=stream) as advance:
            advance(1)
            advance(2)
    assert stream.getvalue() == ""
    assert [record.getMessage() for record in caplog.records] == [
        "第 1 轮自对弈: 1/2 games",
        "第 1 轮自对弈: 2/2 games",
    ]
