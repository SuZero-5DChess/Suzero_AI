"""Console output that never blocks the process that produced it.

The legacy Windows console stops accepting output while text is selected
(QuickEdit). A synchronous write therefore blocks whichever thread emits it -
the training loop, the log handler, the progress bar - for as long as the
selection exists. This module writes console output from a daemon thread
through a bounded queue instead: while the console is paused, output is
dropped (the file log keeps every record) and the trainer keeps running.

QuickEdit must stay enabled for the user to select and copy text with the
mouse, so this is the default protection; ``console_mode.disable_quick_edit``
remains available as an explicit opt-in for unattended runs.
"""

from __future__ import annotations

import atexit
import sys
import threading
from collections import deque
from typing import Iterable, TextIO

DEFAULT_MAX_CHUNKS = 4096
DROPPED_NOTICE = "\n[控制台暂停期间有 %d 条输出被丢弃；完整记录见 training.log]\n"

_active: "ConsoleSink | None" = None
_install_lock = threading.Lock()


def _is_interactive(stream: object) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


class ConsoleSink:
    """A ``TextIO``-like sink that hands output to a background thread."""

    def __init__(self, stream: TextIO, *, max_chunks: int = DEFAULT_MAX_CHUNKS) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        self._stream = stream
        self._max_chunks = int(max_chunks)
        self._chunks: deque[str] = deque()
        self._dropped = 0
        self._closed = False
        self._wake = threading.Condition(threading.Lock())
        self._thread = threading.Thread(
            target=self._drain, name="console-writer", daemon=True
        )
        self._thread.start()

    # -- TextIO surface ----------------------------------------------------

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._wake:
            if self._closed or len(self._chunks) >= self._max_chunks:
                self._dropped += 1
                return len(text)
            self._chunks.append(text)
            self._wake.notify()
        return len(text)

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        """Never blocks: a paused console must not stall the caller."""

    def isatty(self) -> bool:
        return _is_interactive(self._stream)

    @property
    def encoding(self) -> str | None:
        return getattr(self._stream, "encoding", None)

    @property
    def dropped(self) -> int:
        with self._wake:
            return self._dropped

    @property
    def pending(self) -> int:
        with self._wake:
            return len(self._chunks)

    def close(self, timeout: float = 0.2) -> None:
        """Stop the writer, waiting only briefly for a paused console."""

        with self._wake:
            if self._closed:
                return
            self._closed = True
            self._wake.notify_all()
        if (
            self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout)

    # -- background writer -------------------------------------------------

    def _drain(self) -> None:
        while True:
            with self._wake:
                while not self._chunks and not self._closed:
                    self._wake.wait()
                if not self._chunks:
                    return
                chunk = self._chunks.popleft()
                dropped, self._dropped = self._dropped, 0
            if dropped:
                chunk = DROPPED_NOTICE % dropped + chunk
            try:
                self._stream.write(chunk)
                self._stream.flush()
            except (OSError, ValueError):
                # The console was closed under us; stop instead of spinning.
                with self._wake:
                    self._closed = True
                return


def install(stream: TextIO | None = None) -> ConsoleSink | None:
    """Start the shared sink for an interactive Windows console.

    Returns ``None`` on other platforms and for redirected output, where a
    synchronous write cannot be paused by a text selection.
    """

    global _active
    target = sys.stderr if stream is None else stream
    if sys.platform != "win32" or not _is_interactive(target):
        return None
    with _install_lock:
        if _active is None:
            _active = ConsoleSink(target)
            atexit.register(_active.close)
        return _active


def active() -> ConsoleSink | None:
    """Return the installed sink, if any."""

    return _active


def reset() -> None:
    """Close and forget the shared sink (tests and process teardown)."""

    global _active
    with _install_lock:
        sink, _active = _active, None
    if sink is not None:
        sink.close()


__all__ = ["ConsoleSink", "DROPPED_NOTICE", "active", "install", "reset"]
