"""Cooperative Ctrl+C handling for long training runs.

Pressing Ctrl+C once during self-play asks the backend to stop handing out new
games: the games already in flight finish, every finished game is written to
the replay store, and the trainer saves its checkpoint and exits before the
training phase.  The next run resumes from those files.  Outside self-play the
current iteration (training, evaluation) finishes and is checkpointed before
the process exits, so a completed iteration is never lost to an abrupt
KeyboardInterrupt mid-write.

A second Ctrl+C gives up waiting and aborts immediately (the in-flight
iteration may then be lost).

While the C++ self-play bridge runs, the Python handler cannot execute: the
main thread is blocked inside the native call with the GIL released, so the
signal is only delivered once the call returns.  The native guard acknowledges
the press itself, sets the scheduler's stop flag, and stops starting new games
(see ``src/5dchess/console_interrupt.h``); the cooperative semantics in this
module still apply after the call returns.
"""

from __future__ import annotations

import logging
import signal
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

_stop = threading.Event()
_previous: object | None = None


def _handle(signum: int, frame: object) -> None:
    del signum, frame
    if _stop.is_set():
        # Second press: stop waiting and let the default handler abort.  The
        # signal is re-raised so the caller sees a normal KeyboardInterrupt.
        _restore()
        logger.warning("second Ctrl+C received; aborting immediately")
        raise KeyboardInterrupt
    _stop.set()
    logger.warning(
        "Ctrl+C received; self-play will finish its in-flight games and save "
        "before exit (press again to abort immediately)"
    )


def _restore() -> None:
    global _previous
    if _previous is not None:
        signal.signal(signal.SIGINT, _previous)  # type: ignore[arg-type]
        _previous = None


def stop_requested() -> bool:
    """True once the user asked to stop after the current iteration."""

    return _stop.is_set()


def install() -> None:
    """Install the cooperative handler (main thread only)."""

    global _previous
    _stop.clear()
    if threading.current_thread() is not threading.main_thread():
        # signal.signal only works in the main thread; a worker-hosted run
        # simply cannot be stopped cooperatively.
        _previous = None
        return
    try:
        _previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handle)
    except (ValueError, OSError):
        _previous = None


def uninstall() -> None:
    """Restore the previous SIGINT handler and clear the flag."""

    _restore()
    _stop.clear()


@contextmanager
def stop_on_sigint() -> Iterator[None]:
    """Install the cooperative handler for the duration of a block."""

    install()
    try:
        yield
    finally:
        uninstall()
