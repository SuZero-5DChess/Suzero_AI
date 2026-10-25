"""Console progress display for the self-play phase.

One log line per game drowns the console over a long run, so self-play renders
the reference trainer's single updating line instead::

    第 3 轮自对弈: 12/32 局 | 已用时 00:07

The bar only appears on an interactive stream.  Redirected output (a file, a
pipe, a CI log) gets periodic log records instead, so no carriage-return
control characters ever end up in captured output.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TextIO

try:  # tqdm is optional; training still runs without it.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback path tested via monkeypatch
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

BAR_FORMAT = "{desc}: {n_fmt}/{total_fmt} 局 | 已用时 {elapsed}"


@contextmanager
def self_play_progress(
    desc: str,
    total: int,
    *,
    stream: TextIO | None = None,
) -> Iterator[Callable[[int], None]]:
    """Yield ``advance(done)`` where ``done`` is the number of games finished."""

    target = sys.stderr if stream is None else stream
    bar = _open_bar(desc, total, target)
    if bar is None:
        yield _logging_progress(desc, total)
        return
    try:
        yield lambda done: _advance(bar, done)
    finally:
        bar.close()


def _open_bar(desc: str, total: int, target: TextIO) -> Any | None:
    if tqdm is None or total <= 0 or not _is_interactive(target):
        return None
    return tqdm(
        total=total,
        desc=desc,
        bar_format=BAR_FORMAT,
        file=target,
        disable=False,
        dynamic_ncols=True,
        mininterval=0,
    )


def _advance(bar: Any, done: int) -> None:
    step = done - bar.n
    if step > 0:
        bar.update(step)


def _is_interactive(stream: TextIO) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def _logging_progress(desc: str, total: int) -> Callable[[int], None]:
    interval = max(1, total // 10)

    def advance(done: int) -> None:
        if done == total or done % interval == 0:
            logger.info("%s: %d/%d games", desc, done, total)

    return advance
