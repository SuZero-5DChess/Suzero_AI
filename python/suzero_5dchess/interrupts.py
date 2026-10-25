"""Cooperative Ctrl+C handling for long training runs.

Pressing Ctrl+C once asks the trainer to stop *after the current iteration*:
the iteration already in progress (self-play, training, evaluation) finishes,
its checkpoint is written, and the process exits normally.  This keeps every
completed iteration on disk instead of losing it to an abrupt KeyboardInterrupt
in the middle of a replay-buffer write.

A second Ctrl+C gives up waiting and aborts immediately (the in-flight
iteration may then be lost).

While the C++ self-play bridge runs, the Python handler cannot execute: the
main thread is blocked inside the native call with the GIL released, so the
signal is only delivered once the batch returns.  The bridge acknowledges the
press itself in the meantime (see ``src/5dchess/console_interrupt.h``), so the
console shows a notice right away while this cooperative semantics keeps
applying after the call returns.
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
        "Ctrl+C received; finishing the current iteration and saving before "
        "exit (press again to abort immediately)"
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
