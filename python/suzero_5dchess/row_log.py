"""Single append-only row log behind the file replay backend.

The replay window is one file, ``replay.log``, whose unit is a position row
rather than a whole game.  Every finished self-play game becomes one frame::

    header   <4s I I H I I q i>
             magic, payload bytes, crc32(payload), version,
             rows, skipped, game_id, iteration
    payload  rows x [u32 blob bytes][zlib sample record]

``rows`` counts every row the game ever had and ``skipped`` the leading ones
the window has already trimmed away, so the payload only carries the surviving
rows.  Row identities are ``(game_id, local row)`` with ``local`` counted from
the first row of the game, which stays stable when a game is trimmed from the
front: the surviving rows keep their numbers.

Appending takes an OS-level lock on ``replay.lock``, so several self-play
worker processes can stream into the same file.  A writer re-reads the frames
other writers appended, then writes its frame with one append at the end.  A
frame is verified with its CRC while indexing, so the next writer repairs a
frame left half-written by a crashed process while a concurrent reader simply
stops at the torn tail.

Trimming is row-granular: the oldest rows go, even in the middle of a game.
``drop`` rewrites the surviving frames into a temporary file and atomically
replaces the log.  Only the parent process trims, between iterations, when no
self-play worker has the file open.
"""

from __future__ import annotations

import logging
import os
import struct
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

MAGIC = b"RWL1"
VERSION = 1
LOG_NAME = "replay.log"
LOCK_NAME = "replay.lock"

#: magic, payload bytes, crc32, version, rows, skipped, game id, iteration
HEADER = struct.Struct("<4sIIHIIqi")
_ROW = struct.Struct("<I")

_LOCK_TIMEOUT = 300.0
_LOCK_POLL = 0.01


if os.name == "nt":  # pragma: no cover - one platform at a time
    import msvcrt

    def _try_lock(handle) -> bool:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:  # pragma: no cover - one platform at a time
    import fcntl

    def _try_lock(handle) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class Frame:
    """One game in the log, addressed by its byte offset."""

    offset: int
    length: int
    rows: int
    skipped: int
    game_id: int
    iteration: int

    @property
    def live_rows(self) -> int:
        return self.rows - self.skipped

    @property
    def end(self) -> int:
        return self.offset + self.length


def _encode_payload(blobs: Sequence[bytes]) -> bytes:
    return b"".join(_ROW.pack(len(blob)) + blob for blob in blobs)


def _decode_payload(payload: bytes, rows: int) -> list[bytes]:
    blobs: list[bytes] = []
    position = 0
    for _ in range(rows):
        if position + _ROW.size > len(payload):
            raise ValueError("replay frame row table is truncated")
        (size,) = _ROW.unpack_from(payload, position)
        position += _ROW.size
        if position + size > len(payload):
            raise ValueError("replay frame row blob is truncated")
        blobs.append(payload[position:position + size])
        position += size
    if position != len(payload):
        raise ValueError("replay frame has trailing bytes")
    return blobs


def _header_bytes(
    payload: bytes, *, rows: int, skipped: int, game_id: int, iteration: int
) -> bytes:
    return HEADER.pack(
        MAGIC,
        len(payload),
        zlib.crc32(payload),
        VERSION,
        rows,
        skipped,
        int(game_id),
        int(iteration),
    )


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _scan(path: Path, start: int = 0) -> tuple[list[Frame], int]:
    """Read frames starting at ``start``; return them and the valid end.

    A frame whose header is incomplete, whose payload is short, or whose CRC
    does not match ends the scan: everything from there on is a torn write.
    The caller decides whether to repair or simply ignore the tail.
    """

    frames: list[Frame] = []
    offset = start
    try:
        handle = path.open("rb")
    except FileNotFoundError:
        return frames, 0
    with handle:
        end_of_file = os.fstat(handle.fileno()).st_size
        while True:
            if offset + HEADER.size > end_of_file:
                break
            handle.seek(offset)
            header = handle.read(HEADER.size)
            magic, length, checksum, version, rows, skipped, game_id, iteration = (
                HEADER.unpack(header)
            )
            if magic != MAGIC or version != VERSION:
                break
            if skipped > rows or offset + HEADER.size + length > end_of_file:
                break
            payload = handle.read(length)
            if len(payload) != length or zlib.crc32(payload) != checksum:
                break
            frames.append(
                Frame(
                    offset=offset,
                    length=HEADER.size + length,
                    rows=rows,
                    skipped=skipped,
                    game_id=game_id,
                    iteration=iteration,
                )
            )
            offset += HEADER.size + length
    return frames, offset


def _read_frame(handle, frame: Frame) -> list[bytes]:
    handle.seek(frame.offset)
    raw = handle.read(frame.length)
    if len(raw) != frame.length:
        raise ValueError(
            f"replay frame for game {frame.game_id} is gone; the log shrank"
        )
    magic, length, checksum, version, rows, skipped, game_id, iteration = (
        HEADER.unpack_from(raw, 0)
    )
    payload = raw[HEADER.size:]
    if (
        magic != MAGIC
        or version != VERSION
        or length != len(payload)
        or zlib.crc32(payload) != checksum
        or (rows, skipped, game_id, iteration)
        != (frame.rows, frame.skipped, frame.game_id, frame.iteration)
    ):
        raise ValueError(
            f"replay frame for game {frame.game_id} moved; the index is stale"
        )
    return _decode_payload(payload, frame.live_rows)


class RowLog:
    """Append-only log of game frames, shared by every self-play process."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.path = self.directory / LOG_NAME
        self.lock_path = self.directory / LOCK_NAME
        self.frames: list[Frame] = []
        self.end = 0
        self.loaded = False

    # ------------------------------------------------------------ inspecting

    @property
    def rows(self) -> int:
        return sum(frame.live_rows for frame in self.frames)

    def load(self) -> list[Frame]:
        """Re-index the whole file, tolerating a torn tail."""

        self.frames, self.end = _scan(self.path)
        self.loaded = True
        return self.frames

    def refresh(self) -> int:
        """Index frames appended since the last call; return the new row count."""

        if not self.loaded:
            before = 0
            self.load()
            return self.rows - before
        before = self.rows
        frames, end = _scan(self.path, self.end)
        if end > self.end or frames:
            self.frames.extend(frames)
            self.end = end
        return self.rows - before

    def read(self, frame: Frame) -> list[bytes]:
        """Return the surviving encoded rows of ``frame``."""

        with self.path.open("rb") as handle:
            return _read_frame(handle, frame)

    # -------------------------------------------------------------- writing

    @contextmanager
    def exclusive(self, timeout: float = _LOCK_TIMEOUT) -> Iterator[None]:
        """Hold the cross-process write lock."""

        self.directory.mkdir(parents=True, exist_ok=True)
        handle = open(self.lock_path, "a+b")
        try:
            deadline = time.monotonic() + timeout
            while not _try_lock(handle):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"timed out waiting for the replay log lock: "
                        f"{self.lock_path}"
                    )
                time.sleep(_LOCK_POLL)
            try:
                yield
            finally:
                _unlock(handle)
        finally:
            handle.close()

    def append(
        self,
        *,
        game_id: int,
        iteration: int,
        blobs: Sequence[bytes],
        skipped: int = 0,
        unique: bool = False,
    ) -> Frame:
        """Stream one game to the end of the log."""

        return self.append_many(
            [(game_id, iteration, blobs, skipped)], unique=unique
        )[0]

    def append_many(
        self,
        games: Iterable[tuple[int, int, Sequence[bytes], int]],
        *,
        unique: bool = False,
    ) -> list[Frame]:
        """Append several games with the lock held once.

        Each item is ``(game_id, iteration, blobs, skipped)``.  The writer
        re-reads frames appended by other processes first, so its in-memory
        index and the file agree, and repairs a torn frame left by a process
        that died mid-write before appending behind it.  With ``unique`` the
        append refuses an id the log already holds, which catches a reused
        self-play id across processes.
        """

        encoded: list[bytes] = []
        pending: list[tuple[int, int, int, int]] = []
        for game_id, iteration, blobs, skipped in games:
            blobs = list(blobs)
            if not blobs:
                raise ValueError("a replay frame needs at least one row")
            if skipped < 0:
                raise ValueError("skipped must be non-negative")
            payload = _encode_payload(blobs)
            encoded.append(
                _header_bytes(
                    payload,
                    rows=skipped + len(blobs),
                    skipped=skipped,
                    game_id=game_id,
                    iteration=iteration,
                ) + payload
            )
            pending.append((int(game_id), int(iteration), skipped + len(blobs), skipped))
        if not encoded:
            return []

        with self.exclusive():
            self._sync_locked()
            if unique:
                known = {frame.game_id for frame in self.frames}
                for game_id, *_rest in pending:
                    if game_id in known:
                        raise FileExistsError(
                            f"game {game_id} is already in the replay log"
                        )
            new_frames = []
            offset = self.end
            with self.path.open("ab") as handle:
                for data, (game_id, iteration, rows, skipped) in zip(encoded, pending):
                    handle.write(data)
                    new_frames.append(
                        Frame(
                            offset=offset,
                            length=len(data),
                            rows=rows,
                            skipped=skipped,
                            game_id=game_id,
                            iteration=iteration,
                        )
                    )
                    offset += len(data)
                handle.flush()
            self.frames.extend(new_frames)
            self.end = offset
            self.loaded = True
        return new_frames

    def _sync_locked(self) -> None:
        """Bring the index up to date and cut a torn tail; lock must be held."""

        size = _size(self.path)
        if not self.loaded:
            self.load()
            size = _size(self.path)
        if self.end > size:
            logger.warning("replay log shrank; re-indexing %s", self.path)
            self.load()
            size = _size(self.path)
        else:
            frames, end = _scan(self.path, self.end)
            if frames:
                self.frames.extend(frames)
            self.end = end
        if size > self.end:
            # A writer died between its header and the end of its frame.
            with self.path.open("r+b") as handle:
                handle.truncate(self.end)
            logger.warning(
                "replay log: discarded %d torn bytes at offset %d",
                size - self.end,
                self.end,
            )

    def drop(self, rows: int) -> int:
        """Drop the oldest ``rows`` rows, rewriting the live frames."""

        rows = int(rows)
        if rows < 0:
            raise ValueError("rows to drop must be non-negative")
        if not rows:
            return 0

        with self.exclusive():
            self._sync_locked()
            dropped = min(rows, self.rows)
            if not dropped:
                return 0
            tmp = self.path.with_name(f"{self.path.name}.compact")
            new_frames: list[Frame] = []
            remaining = dropped
            base = 0
            try:
                with self.path.open("rb") as source, tmp.open("wb") as target:
                    for frame in self.frames:
                        if remaining >= frame.live_rows:
                            remaining -= frame.live_rows
                            continue
                        skipped = frame.skipped
                        if remaining:
                            blobs = _read_frame(source, frame)[remaining:]
                            skipped += remaining
                            remaining = 0
                            payload = _encode_payload(blobs)
                            data = _header_bytes(
                                payload,
                                rows=frame.rows,
                                skipped=skipped,
                                game_id=frame.game_id,
                                iteration=frame.iteration,
                            ) + payload
                        else:
                            source.seek(frame.offset)
                            data = source.read(frame.length)
                            if len(data) != frame.length:
                                raise ValueError(
                                    f"replay frame for game {frame.game_id} is gone"
                                )
                        target.write(data)
                        new_frames.append(
                            Frame(
                                offset=base,
                                length=len(data),
                                rows=frame.rows,
                                skipped=skipped,
                                game_id=frame.game_id,
                                iteration=frame.iteration,
                            )
                        )
                        base += len(data)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(tmp, self.path)
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
            self.frames = new_frames
            self.end = base
            self.loaded = True
        logger.info(
            "replay log: dropped %d oldest rows, %d games left, %d bytes",
            dropped, len(new_frames), base,
        )
        return dropped

    def __len__(self) -> int:
        return self.rows


__all__ = [
    "Frame",
    "HEADER",
    "LOCK_NAME",
    "LOG_NAME",
    "MAGIC",
    "RowLog",
    "VERSION",
]
