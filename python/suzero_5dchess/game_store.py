"""Single-file replay window and the training source that reads it.

The default replay backend keeps the window in one append-only log,
``replay.log`` (see :mod:`suzero_5dchess.row_log`), whose unit is a position
row: every finished self-play game streams in as a frame as soon as it
completes, and trimming drops the oldest rows even in the middle of a game.
Nothing holds the window's sample blobs in RAM; the log is the source of truth
and the index only carries frame offsets, row counts and the counters.

Samples use the existing compact zlib codec.  A game's surviving rows keep
their original local row numbers (the frame records how many leading rows were
trimmed), so ``(game_id, local row)`` identities stay valid when a game is cut.
Trimming is a full rewrite of the log, so it happens in batches: the window may
run about a fifth over its target before the oldest rows go in one go.

Training draws shuffle rounds from the window.  The last drawn round is
recorded in ``last_round.json`` beside the log; it carries only row
identities, never sample blobs, and lets the next round (or phase) exclude the
rows the previous round selected.

A directory written by the older per-game layout (``g<id>-<rows>.npz``) is
imported into the log on the first scan and moved to ``legacy/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
import torch

from suzero_train.data_source import TrainingDataSource
from suzero_train.types import TrainingBatch

from .endgame_sampling import (
    EndgameSamplingConfig,
    endgame_weights,
    repeat_by_weight,
    weighted_without_replacement,
)
from .file_replay import ROUND_SEED_STRIDE
from .protocol import VariantSpec
from .replay import REPLAY_FORMAT, ReplayBuffer
from .row_log import Frame, RowLog
from .sample import PositionSample, collate_samples, seal_decoded, validate_sample
from .sample_codec import decode_sample, encode_sample, peek_move_count
from .train_counts import (
    COUNT_DTYPE,
    evictable_prefix,
    new_counts,
    record,
    untrained_first,
)

if TYPE_CHECKING:  # pragma: no cover - only used for type hints
    from .self_play import SelfPlayResult

logger = logging.getLogger(__name__)

GAME_STORE_FORMAT = "suzero_game_store_v2"
GAME_ROUND_FORMAT = "suzero_game_round_v2"
GAME_FILE_FORMAT = "suzero_game_file_v1"
LEGACY_STORE_FORMAT = "suzero_game_store_v1"
LEGACY_ROUND_FORMAT = "suzero_game_round_v1"
TRAIN_COUNTS_FORMAT = "suzero_train_counts_v1"
ROUND_FILE = "last_round.json"
LEGACY_DIR = "legacy"

#: The window is allowed to run over its target by this fraction of it before
#: a batch is dropped: trimming is a full rewrite of the log, so it happens in
#: spurts with room for the next games instead of after every game.
TRIM_MARGIN_DIVISOR = 5

_LEGACY_NAME = re.compile(r"g([0-9]{12})-([0-9]{8})\.npz")


def _round_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_round(data: object) -> None:
    """Validate the round sidecar before trusting its row identities."""

    if not isinstance(data, dict) or data.get("format") != GAME_ROUND_FORMAT:
        raise ValueError("unrecognised game round state")
    for name in ("iteration", "round", "pool_samples", "count"):
        _round_int(data.get(name), name)
    for name in ("phase_seed", "seed"):
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    keep = data.get("keep_samples")
    if keep is not None:
        _round_int(keep, "keep_samples", minimum=1)
    games = data.get("games")
    if not isinstance(games, list):
        raise ValueError("game round games must be a list")
    for item in games:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("invalid game round game entry")
        _round_int(item[0], "game id")
        _round_int(item[1], "game rows", minimum=1)
    selected = data.get("selected")
    if not isinstance(selected, list) or len(selected) != data["count"]:
        raise ValueError("game round selection does not match its count")
    seen: set[tuple[int, int]] = set()
    endgame = data.get("endgame")
    if endgame is not None:
        if not isinstance(endgame, dict):
            raise ValueError("game round endgame settings must be an object")
        try:
            EndgameSamplingConfig(
                enabled=True,
                strength=endgame.get("strength"),
                exponent=endgame.get("exponent"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid game round endgame settings") from exc
    for item in selected:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("invalid game round row")
        row = (
            _round_int(item[0], "game id"),
            _round_int(item[1], "game row"),
        )
        # Endgame sampling deliberately repeats late rows inside a round, so
        # duplicates are only allowed when the round declares that setting.
        if row in seen and endgame is None:
            raise ValueError("duplicate game round row")
        seen.add(row)


@dataclass(frozen=True)
class StoredGame:
    """One game frame in the row log, as the window sees it."""

    frame: Frame

    @property
    def game_id(self) -> int:
        return self.frame.game_id

    @property
    def count(self) -> int:
        """Rows of the game still inside the window."""

        return self.frame.live_rows

    @property
    def iteration(self) -> int:
        return self.frame.iteration

    @property
    def skip(self) -> int:
        """Leading rows already trimmed from this game."""

        return self.frame.skipped

    @property
    def offset(self) -> int:
        return self.frame.offset

    @property
    def size(self) -> int:
        return self.frame.length


def _encode_samples(
    samples: Sequence[PositionSample], spec: VariantSpec
) -> list[bytes]:
    encoded: list[bytes] = []
    for sample in samples:
        if not isinstance(sample, PositionSample):
            raise TypeError("samples must be PositionSample instances")
        validate_sample(sample, spec)
        encoded.append(encode_sample(sample))
    return encoded


def _read_game_file(path: Path, spec: VariantSpec) -> list[bytes]:
    """Load one archive from the per-game layout; used only for migration."""

    if path.is_symlink():
        raise ValueError(f"game archive must not be a symlink: {path}")
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    for key in ("format", "capacity", "count", "total_generated"):
        if key in state:
            state[key] = state[key].item()
    if state.get("format") != GAME_FILE_FORMAT:
        raise ValueError(f"unrecognised game archive format: {path}")
    count = int(state.get("count", 0))
    replay = ReplayBuffer(max(1, count), spec)
    # Reuse the compact replay validator; the payload is the same codec arena.
    replay_state = {
        "format": REPLAY_FORMAT,
        "capacity": replay.capacity,
        "count": count,
        "total_generated": int(state.get("total_generated", count)),
        "offsets": state["offsets"],
        "blobs": state["blobs"],
    }
    # Migration only needs the encoded rows; weighting never samples them.
    replay.load_state_dict(replay_state, derive_progress=False)
    if len(replay) != count:
        raise ValueError(f"game archive row count differs from its header: {path}")
    return replay.encoded_at(range(count))


class GameStore:
    """Row log of finished games, acting as the replay backend."""

    def __init__(
        self,
        directory: Path,
        spec: VariantSpec,
        capacity: int,
        *,
        enforce_capacity: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.directory = Path(directory)
        self.spec = spec
        self.capacity = int(capacity)
        self.enforce_capacity = bool(enforce_capacity)
        self.log = RowLog(self.directory)
        self._games: list[StoredGame] = []
        self._rows = 0
        self._bytes = 0
        #: Training passes per live window row, aligned with the row order of
        #: ``_games``.  Front trims drop the matching counters.
        self.train_counts: list[int] = []
        #: Per-game ``move_count`` arrays for endgame weighting, keyed by game
        #: id and cached with the front skip they were read at.  Rows never
        #: change, so a front trim only slices the cached array.
        self._move_counts: dict[int, tuple[int, np.ndarray]] = {}
        self.total_generated = 0
        self.next_game_id = 0
        self.shuffle_state: dict[str, object] | None = None
        self._lock = threading.RLock()
        self._warned_untrained_trim = False
        self.directory.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- indexing

    @property
    def games(self) -> tuple[StoredGame, ...]:
        with self._lock:
            return tuple(self._games)

    def __len__(self) -> int:
        with self._lock:
            return self._rows

    def scan(self) -> None:
        """(Re)build the index and reconcile the generation counter."""

        with self._lock:
            self._check_directory()
            self._migrate_legacy()
            self.log.load()
            self._apply()
            if self._rows > self.total_generated:
                self.total_generated = self._rows

    def refresh(self) -> int:
        """Pick up frames appended by another process; return new rows."""

        with self._lock:
            self._check_directory()
            before = self._rows
            self._migrate_legacy()
            if self.log.loaded:
                self.log.refresh()
            else:
                self.log.load()
            self._apply()
            added = self._rows - before
            if added > 0:
                self.total_generated += added
            return max(0, added)

    def _check_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise ValueError("game store directory must not be a symlink")

    def _apply(self) -> None:
        self._games = [StoredGame(frame) for frame in self.log.frames]
        self._rows = self.log.rows
        self._bytes = self.log.end
        # New frames extend the window at the end; front trims are sliced by
        # their callers because only they know how many rows went away.
        if len(self.train_counts) < self._rows:
            self.train_counts.extend(new_counts(self._rows - len(self.train_counts)))
        if self._move_counts:
            live = {entry.game_id for entry in self._games}
            self._move_counts = {
                game_id: value
                for game_id, value in self._move_counts.items()
                if game_id in live
            }
        if self._games:
            # Frames are in append order, and parallel self-play workers append
            # interleaved id ranges, so the highest id is not always the last.
            highest = max(game.game_id for game in self._games)
            self.next_game_id = max(self.next_game_id, highest + 1)

    def _ensure_loaded(self) -> None:
        if self.log.loaded:
            return
        self._check_directory()
        self._migrate_legacy()
        if not self.log.loaded:
            self.log.load()
        self._apply()
        if self._rows > self.total_generated:
            self.total_generated = self._rows

    # -------------------------------------------------------------- migration

    def _legacy_archives(self) -> list[tuple[int, Path]]:
        if not self.directory.is_dir():
            return []
        found: list[tuple[int, Path]] = []
        for path in self.directory.iterdir():
            match = _LEGACY_NAME.fullmatch(path.name)
            if match is None:
                continue
            if path.is_symlink():
                raise ValueError(f"game store must not contain symlinks: {path}")
            if not path.is_file():
                continue
            found.append((int(match.group(1)), path))
        found.sort(key=lambda item: item[0])
        return found

    def _migrate_legacy(self) -> int:
        """Import per-game archives into the log, then park them in ``legacy/``.

        Games the log already holds are not imported again: an append that was
        cut short by a crash is repaired by the next writer and re-imported
        from its still-present archive.
        """

        archives = self._legacy_archives()
        if not archives:
            return 0
        if not self.log.loaded:
            self.log.load()
        highest = max(
            (frame.game_id for frame in self.log.frames), default=-1
        )
        frames = []
        batch: list[tuple[int, int, list[bytes], int]] = []
        for game_id, path in archives:
            if game_id <= highest:
                continue
            blobs = _read_game_file(path, self.spec)
            if not blobs:
                continue
            batch.append((game_id, -1, blobs, 0))
            # Import in chunks so a large legacy directory does not have to
            # sit in RAM all at once.
            if len(batch) >= 512:
                frames.extend(self.log.append_many(batch))
                batch = []
        if batch:
            frames.extend(self.log.append_many(batch))
        if frames:
            logger.info(
                "replay store: imported %d game archives (%d rows) into %s",
                len(frames), sum(frame.live_rows for frame in frames),
                self.log.path,
            )
        legacy = self.directory / LEGACY_DIR
        legacy.mkdir(parents=True, exist_ok=True)
        for _game_id, path in archives:
            try:
                os.replace(path, legacy / path.name)
            except FileNotFoundError:
                pass
        self._apply()
        return len(frames)

    # --------------------------------------------------------------- writing

    def append(
        self,
        samples: Sequence[PositionSample],
        *,
        game_id: int | None = None,
        iteration: int = -1,
    ) -> int:
        """Encode and append one finished game; return its row count."""

        encoded = _encode_samples(samples, self.spec)
        if not encoded:
            return 0
        with self._lock:
            self._ensure_loaded()
            if game_id is None:
                game_id = self.next_game_id
            game_id = int(game_id)
            if game_id < 0:
                raise ValueError("game_id must be non-negative")
            if any(game.game_id == game_id for game in self._games):
                raise FileExistsError(
                    f"game {game_id} is already in the replay log"
                )
            # The check above sees this process's index; ``unique`` also
            # catches an id another writer appended in between.
            frame = self.log.append(
                game_id=game_id, iteration=iteration, blobs=encoded, unique=True
            )
            self._apply()
            self.total_generated += frame.live_rows
        if self.enforce_capacity:
            self._trim_above(self.capacity)
        return frame.live_rows

    def append_result(
        self,
        game_id: int,
        result: "SelfPlayResult",
        *,
        iteration: int = -1,
        release: bool = True,
    ) -> int:
        """Stream one finished :class:`SelfPlayResult` into the store.

        ``release`` frees the in-RAM rows once they are on disk; callers that
        still need them (tests, alternative backends) can opt out.
        """

        count = self.append(result.samples, game_id=game_id, iteration=iteration)
        result.persisted = True
        if release:
            result.release_samples()
        return count

    # --------------------------------------------------------------- trimming

    def margin(self, size: int) -> int:
        """Rows the window may hold above ``size`` before a batch is dropped."""

        return max(1, int(size) // TRIM_MARGIN_DIVISOR)

    def _drop_trained(self, requested: int) -> int:
        """Drop up to ``requested`` leading rows that were trained on.

        Rows leave only after the trainer served them at least once.  An
        untrained row at the front therefore keeps itself and every row behind
        it, so the live window can stay above its target until the training
        rounds reach the backlog.
        """

        requested = max(0, min(int(requested), self._rows))
        if not requested:
            return 0
        allowed = evictable_prefix(self.train_counts, requested)
        if allowed < requested:
            message = (
                "replay store: keeping %d untrained front rows; dropped %d of "
                "%d requested rows (buffered=%d)"
            )
            fields = (requested - allowed, allowed, requested, self._rows)
            if self._warned_untrained_trim:
                logger.debug(message, *fields)
            else:
                self._warned_untrained_trim = True
                logger.warning(message, *fields)
        if not allowed:
            return 0
        dropped = self.log.drop(allowed)
        if dropped:
            self._apply()
            del self.train_counts[:dropped]
            logger.info(
                "replay store: dropped %d trained rows, %d buffered",
                dropped,
                self._rows,
            )
        return dropped

    def trim_to(self, size: int) -> int:
        """Drop the oldest trained rows in a batch once the window runs over ``size``.

        The window is allowed to keep up to :meth:`margin` rows above ``size``;
        while it stays inside that buffer nothing is rewritten.  Once it runs
        over, the oldest rows are dropped in one go down to ``size``.  Rows the
        trainer has not served yet are never dropped, so the window may stay
        larger than the target while an untrained backlog remains.
        """

        size = int(size)
        if size < 0:
            raise ValueError("trim size must be non-negative")
        with self._lock:
            self._ensure_loaded()
            if self._rows - size <= self.margin(size):
                return 0
            before = self._rows
            self._drop_trained(self._rows - size)
            return before - self._rows

    def _trim_above(self, size: int) -> int:
        """Enforce capacity from appends, in batches with room to spare."""

        with self._lock:
            self._ensure_loaded()
            margin = self.margin(size)
            if self._rows - size <= margin:
                return 0
            before = self._rows
            # Cut into the lower part of the band so the next batch of games
            # can arrive before the log is rewritten again.
            self._drop_trained(self._rows - max(0, size - margin))
            return before - self._rows

    def estimated_bytes(self) -> int:
        """On-disk size of the current window (no sample blobs live in RAM)."""

        with self._lock:
            return self._bytes

    # ------------------------------------------------------- training counts

    def record_trained(self, rows: Sequence[int]) -> int:
        """Count window rows a data source just served to the trainer."""

        with self._lock:
            return record(self.train_counts, rows)

    def save_train_counts(self, path: Path) -> None:
        """Atomically persist counters beside the replay state.

        The sidecar stores row identities rather than offsets, so trimming
        between the save and the next load only drops counters instead of
        shifting every remaining row.
        """

        path = Path(path)
        if path.is_symlink():
            raise ValueError("training counter state must not be a symlink")
        with self._lock:
            games = np.fromiter(
                (entry.game_id for entry in self._games),
                dtype=np.int64, count=len(self._games),
            )
            skips = np.fromiter(
                (entry.skip for entry in self._games),
                dtype=np.int64, count=len(self._games),
            )
            lengths = np.fromiter(
                (entry.count for entry in self._games),
                dtype=np.int64, count=len(self._games),
            )
            counts = np.asarray(self.train_counts, dtype=COUNT_DTYPE)
            if counts.size != int(lengths.sum()):
                raise ValueError("training counters do not match the window")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("wb") as handle:
                np.savez(
                    handle,
                    format=np.asarray(TRAIN_COUNTS_FORMAT),
                    games=games,
                    skips=skips,
                    lengths=lengths,
                    counts=counts,
                )
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def load_train_counts(self, path: Path) -> int:
        """Merge persisted counters onto the current window; return rows kept."""

        path = Path(path)
        if not path.is_file():
            return 0
        if path.is_symlink():
            raise ValueError("training counter state must not be a symlink")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        if str(np.asarray(data.get("format")).item()) != TRAIN_COUNTS_FORMAT:
            raise ValueError("unrecognised training counter state")
        games = np.asarray(data["games"], dtype=np.int64)
        skips = np.asarray(data["skips"], dtype=np.int64)
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        counts = np.asarray(data["counts"], dtype=np.int64)
        if not (
            games.ndim == skips.ndim == lengths.ndim == 1
            and games.shape == skips.shape == lengths.shape
            and counts.ndim == 1
            and int(lengths.sum()) == counts.size
        ):
            raise ValueError("training counter state is inconsistent")
        if counts.size and int(counts.min()) < 0:
            raise ValueError("training counters must be non-negative")
        if counts.size == 0:
            return 0
        saved: dict[int, tuple[int, np.ndarray]] = {}
        position = 0
        for game_id, skip, length in zip(games, skips, lengths):
            end = position + int(length)
            saved[int(game_id)] = (int(skip), counts[position:end])
            position = end
        with self._lock:
            restored = 0
            position = 0
            for entry in self._games:
                segment = saved.get(entry.game_id)
                if segment is not None:
                    saved_skip, values = segment
                    offset = entry.skip - saved_skip
                    if 0 <= offset <= len(values):
                        length = min(entry.count, len(values) - offset)
                        saved_values = values[offset : offset + length].tolist()
                        for index, value in enumerate(saved_values):
                            row = position + index
                            # Never let a stale sidecar erase a newer count.
                            self.train_counts[row] = max(
                                self.train_counts[row], int(value)
                            )
                        restored += length
                position += entry.count
            return restored

    def read_game(self, game: StoredGame) -> list[bytes]:
        """Return the encoded rows of one game frame."""

        return self.log.read(game.frame)

    def row_move_counts(self, game: StoredGame) -> np.ndarray:
        """Return the live rows' ``move_count`` values for one game.

        Endgame weighting needs one integer per row; inflating only the fixed
        record header keeps this far cheaper than decoding the samples.  The
        result is cached per game id, and a front trim slices the cached array
        because surviving rows keep their identity.
        """

        with self._lock:
            cached = self._move_counts.get(game.game_id)
            if cached is not None:
                skip, values = cached
                start = game.skip - skip
                if start >= 0 and start + game.count <= values.size:
                    return values[start : start + game.count]
            blobs = self.read_game(game)
            values = np.fromiter(
                (peek_move_count(blob) for blob in blobs),
                dtype=np.int64,
                count=len(blobs),
            )
            self._move_counts[game.game_id] = (game.skip, values)
            return values

    def iter_samples(self) -> Iterator[PositionSample]:
        """Yield the stored rows one game at a time.

        Intended for inspection and tests; training uses
        :class:`GameFileDataSource`, which never decodes a whole window at once.
        """

        for entry in self.games:
            for blob in self.read_game(entry):
                yield seal_decoded(decode_sample(blob), self.spec)

    # ------------------------------------------------------------ data source

    def read_last_round(self) -> dict[str, object] | None:
        """Return the most recently drawn shuffle round, if one is recorded.

        The sidecar lives beside the log, so the next training phase (or a
        restarted process) can avoid drawing the previous round's rows again.
        """

        path = self.directory / ROUND_FILE
        if not path.is_file():
            return None
        if path.is_symlink():
            raise ValueError("game round state must not be a symlink")
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("format") == LEGACY_ROUND_FORMAT:
            logger.warning(
                "replay store: ignoring a round record written for the "
                "per-game layout"
            )
            return None
        _validate_round(data)
        return data

    def write_last_round(self, state: dict[str, object]) -> None:
        """Atomically record the round that was just drawn."""

        if self.directory.is_symlink():
            raise ValueError("game store directory must not be a symlink")
        data = {"format": GAME_ROUND_FORMAT, **state}
        _validate_round(data)
        path = self.directory / ROUND_FILE
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with self._lock:
            try:
                with tmp.open("w", encoding="utf-8") as handle:
                    json.dump(data, handle, sort_keys=True)
                    handle.write("\n")
                os.replace(tmp, path)
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def training_source(
        self,
        batch_size: int,
        steps_per_epoch: int,
        spec: VariantSpec,
        *,
        iteration: int = 0,
        seed: int,
        keep_samples: int | None = None,
        no_repeat_rows: bool = True,
        endgame: EndgameSamplingConfig | None = None,
    ) -> "GameFileDataSource":
        return GameFileDataSource(
            self,
            batch_size,
            steps_per_epoch,
            spec,
            iteration=iteration,
            seed=seed,
            keep_samples=keep_samples,
            no_repeat_rows=no_repeat_rows,
            endgame=endgame,
        )

    def child_transport(self, iteration: int) -> dict[str, object]:
        """Small JSON-safe descriptor a self-play child uses to append directly."""

        return {
            "directory": str(self.directory.resolve()),
            "variant": self.spec.name,
            "iteration": int(iteration),
        }

    # --------------------------------------------------------- checkpointing

    def state_dict(self) -> dict[str, object]:
        with self._lock:
            state: dict[str, object] = {
                "format": GAME_STORE_FORMAT,
                "capacity": self.capacity,
                "count": self._rows,
                "frames": len(self._games),
                "bytes": self._bytes,
                "total_generated": self.total_generated,
                "next_game_id": self.next_game_id,
            }
            if self.shuffle_state is not None:
                state["shuffle_state"] = json.loads(
                    json.dumps(self.shuffle_state)
                )
            return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        if not isinstance(state, dict):
            raise ValueError("replay state must be a dictionary")
        if state.get("format") not in (GAME_STORE_FORMAT, LEGACY_STORE_FORMAT):
            raise ValueError("replay state is not a file game store")
        total_generated = int(state.get("total_generated", 0))
        if total_generated < 0:
            raise ValueError("replay total_generated must be non-negative")
        next_game_id = int(state.get("next_game_id", 0))
        if next_game_id < 0:
            raise ValueError("replay next_game_id must be non-negative")
        shuffle_state = state.get("shuffle_state")
        if shuffle_state is not None and not isinstance(shuffle_state, dict):
            raise ValueError("replay shuffle_state must be a dictionary")
        with self._lock:
            self.total_generated = max(self.total_generated, total_generated)
            self.next_game_id = max(self.next_game_id, next_game_id)
            self.shuffle_state = (
                json.loads(json.dumps(shuffle_state))
                if shuffle_state is not None
                else None
            )


class GameFileDataSource(TrainingDataSource):
    """Serve adjacent shuffle rounds over the games in the row log.

    Each round draws rows from the current window except those selected by the
    immediately preceding round, so successive shuffles never repeat a row.
    With ``no_repeat_rows=False`` every round is drawn from the whole window
    instead, so adjacent rounds may overlap.  A round keeps its rows grouped by
    game, so reading stays sequential and only the game being read is in RAM.
    One epoch may need several rounds; when the current one is spent the next
    is drawn at once.  The last drawn round is recorded beside the log so the
    next training phase (and a restarted process) excludes it as well.
    """

    def __init__(
        self,
        store: GameStore,
        batch_size: int,
        steps_per_epoch: int,
        spec: VariantSpec,
        *,
        iteration: int = 0,
        seed: int,
        keep_samples: int | None = None,
        no_repeat_rows: bool = True,
        endgame: EndgameSamplingConfig | None = None,
    ) -> None:
        if batch_size <= 0 or steps_per_epoch <= 0:
            raise ValueError("batch_size and steps_per_epoch must be positive")
        if keep_samples is not None and keep_samples <= 0:
            raise ValueError("keep_samples must be positive")
        if not isinstance(no_repeat_rows, bool):
            raise ValueError("no_repeat_rows must be a boolean")
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < 0
        ):
            raise ValueError("iteration must be a non-negative integer")
        self.store = store
        self.spec = spec
        self.batch_size = int(batch_size)
        self.requested_steps = int(steps_per_epoch)
        self.iteration = int(iteration)
        self.seed = int(seed)
        self.keep_samples = keep_samples
        self.no_repeat_rows = no_repeat_rows
        self.endgame = endgame or EndgameSamplingConfig()
        self.entries = store.games
        self._starts = np.zeros(len(self.entries), dtype=np.int64)
        counts = np.zeros(len(self.entries), dtype=np.int64)
        for index, entry in enumerate(self.entries):
            if index:
                self._starts[index] = self._starts[index - 1] + counts[index - 1]
            counts[index] = entry.count
        self._ends = self._starts + counts
        self.pool_samples = int(self._ends[-1]) if len(self._ends) else 0
        self._window = [[entry.game_id, entry.count] for entry in self.entries]

        self._published = 0
        self._rounds = 0
        self._order = np.empty(0, dtype=np.int64)
        self._selected: list[list[int]] = []
        self._round_samples = 0
        self._position = 0
        self._served = 0
        self._steps = 0
        self._waiting = False
        self._loaded_index = -1
        self._loaded_blobs: list[bytes] | None = None
        self.generation = self._generation(0, self.seed)
        self._start_phase()

    # ----------------------------------------------------------- round drawing

    def _generation(self, index: int, seed: int) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "iteration": self.iteration, "phase_seed": self.seed,
                    "round": index, "seed": seed, "games": self._window,
                    "endgame": self.endgame.identity(),
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        return "generation-" + digest.hexdigest()

    def _same_phase(self, previous: dict[str, object]) -> bool:
        """True when the sidecar is this phase's as-yet unused first round."""

        return (
            int(previous["round"]) == 0
            and int(previous["iteration"]) == self.iteration
            and int(previous["phase_seed"]) == self.seed
            and previous["keep_samples"] == self.keep_samples
            and previous["games"] == self._window
            and previous.get("endgame") == self.endgame.identity()
        )

    def _start_phase(self) -> None:
        previous = self.store.read_last_round()
        if previous is not None and self._same_phase(previous):
            self._install(
                previous["selected"], int(previous["seed"]), int(previous["round"])
            )
            self._published = 1
            return
        exclude = (
            previous["selected"]
            if previous is not None and self.no_repeat_rows
            else []
        )
        self._waiting = not self._publish(exclude)

    def _install(self, selected, seed: int, index: int) -> None:
        positions = {
            entry.game_id: position for position, entry in enumerate(self.entries)
        }
        rows = []
        for game_id, local in selected:
            position = positions.get(int(game_id))
            if position is None:
                raise ValueError("game round refers to a missing game")
            entry = self.entries[position]
            local = int(local)
            if not entry.skip <= local < entry.skip + entry.count:
                raise ValueError("game round row is outside its game")
            rows.append(int(self._starts[position]) + local - entry.skip)
        self._order = np.asarray(rows, dtype=np.int64)
        self._selected = [[int(game_id), int(local)] for game_id, local in selected]
        self._round_samples = int(self._order.size)
        self._position = 0
        self._waiting = False
        self.generation = self._generation(index, seed)

    def _excluded_rows(self, selected) -> np.ndarray:
        positions = {
            entry.game_id: position for position, entry in enumerate(self.entries)
        }
        counts = self.store.train_counts
        rows: set[int] = set()
        for game_id, local in selected:
            position = positions.get(int(game_id))
            if position is None:
                # The whole game fell out of the window; its rows are gone too.
                continue
            entry = self.entries[position]
            local = int(local)
            if local < entry.skip:
                # The row was trimmed from the front of this game since the
                # round was drawn; there is nothing left to exclude.
                continue
            if local >= entry.skip + entry.count:
                raise ValueError("game round row is outside its game")
            row = int(self._starts[position]) + local - entry.skip
            # The previous round was drawn whole but may not have been served
            # whole.  Leaving its untrained rows out of the exclusion lets the
            # next round train them instead of skipping them once again.
            if row < len(counts) and counts[row] > 0:
                rows.add(row)
        return np.fromiter(sorted(rows), dtype=np.int64, count=len(rows))

    def _row_weights(self) -> np.ndarray | None:
        """Sampling weight of every pool row, or ``None`` when disabled."""

        if not self.endgame.enabled:
            return None
        weights = np.ones(self.pool_samples, dtype=np.float64)
        for index, entry in enumerate(self.entries):
            counts = self.store.row_move_counts(entry)
            if counts.size != entry.count:
                raise ValueError(
                    f"game {entry.game_id} exposes {counts.size} move counts, "
                    f"expected {entry.count}"
                )
            weights[self._starts[index] : self._ends[index]] = endgame_weights(
                counts, self.endgame
            )
        return weights

    def _publish(self, selected) -> bool:
        if self.pool_samples == 0:
            return False
        index = self._published
        seed = self.seed + index * ROUND_SEED_STRIDE
        if self.no_repeat_rows:
            excluded = self._excluded_rows(selected)
            eligible = np.setdiff1d(
                np.arange(self.pool_samples, dtype=np.int64),
                excluded,
                assume_unique=True,
            )
        else:
            excluded = np.empty(0, dtype=np.int64)
            eligible = np.arange(self.pool_samples, dtype=np.int64)
        if eligible.size == 0:
            logger.info(
                "replay store: all %d buffered rows were selected by the "
                "previous round; no new round", self.pool_samples,
            )
            return False
        rng = np.random.default_rng(seed)
        weights = self._row_weights()
        if self.keep_samples is not None and self.keep_samples < eligible.size:
            if weights is not None:
                pool = weighted_without_replacement(
                    eligible, weights[eligible], self.keep_samples, rng
                )
            else:
                pool = rng.choice(eligible, size=self.keep_samples, replace=False)
        else:
            pool = eligible
        base_rows = int(pool.size)
        if weights is not None:
            # Keep every eligible row once and repeat the late-game ones.
            pool = repeat_by_weight(pool, weights[pool], rng)
        order = rng.permutation(pool)
        # Spend the phase budget on never-trained rows first; the partition is
        # stable, so the shuffle is unchanged when every row is untrained.
        counts = self.store.train_counts
        if len(counts) == self.pool_samples:
            order = untrained_first(order, counts)
        # Group rows by game for sequential IO; in-game order stays shuffled.
        # Games keep the order in which the shuffle first reaches them: sorting
        # by game index alone would always serve the oldest games first, so a
        # phase that consumes only part of its round would train the same rows
        # every time and leave the rest untrained until they are trimmed.
        game_of_row = np.searchsorted(self._ends, order, side="right")
        seen, first_index = np.unique(game_of_row, return_index=True)
        rank = np.zeros(len(self._ends), dtype=np.int64)
        rank[seen[np.argsort(first_index, kind="stable")]] = np.arange(seen.size)
        grouped = np.argsort(rank[game_of_row], kind="stable")
        order = order[grouped].astype(np.int64, copy=False)
        game_of_row = game_of_row[grouped]
        rows = [
            [
                int(self.entries[int(position)].game_id),
                int(row) - int(self._starts[int(position)])
                + self.entries[int(position)].skip,
            ]
            for position, row in zip(game_of_row, order)
        ]
        self._order = order
        self._selected = rows
        self._round_samples = len(rows)
        self._position = 0
        self._waiting = False
        self.generation = self._generation(index, seed)
        self.store.write_last_round({
            "iteration": self.iteration,
            "phase_seed": self.seed,
            "round": index,
            "seed": seed,
            "keep_samples": self.keep_samples,
            "pool_samples": self.pool_samples,
            "games": self._window,
            "count": len(rows),
            "selected": rows,
            "endgame": self.endgame.identity(),
        })
        self._published += 1
        logger.info(
            "replay store round: iteration=%d round=%d rows=%d base=%d "
            "extra=%d excluded=%d no_repeat_rows=%s endgame=%s",
            self.iteration, index, len(rows), base_rows,
            len(rows) - base_rows, int(excluded.size),
            str(self.no_repeat_rows).lower(),
            json.dumps(self.endgame.identity(), sort_keys=True),
        )
        return True

    def _ensure_round(self) -> bool:
        if self._position < len(self._order):
            self._waiting = False
            return True
        if self._waiting:
            return False
        if not self._publish(self._selected):
            self._waiting = True
            return False
        self._rounds += 1
        return True

    # --------------------------------------------------------------- serving

    @property
    def remaining_samples(self) -> int:
        return int(len(self._order) - self._position)

    @property
    def waiting_for_shuffle(self) -> bool:
        return self._waiting or self._position >= len(self._order)

    def begin_epoch(self) -> None:
        self._served = 0
        self._ensure_round()
        self._steps = 0 if self.waiting_for_shuffle else self.requested_steps

    def is_exhausted(self) -> bool:
        return self._served >= self._steps or self._waiting

    def _game_blobs(self, index: int) -> list[bytes]:
        if self._loaded_index == index and self._loaded_blobs is not None:
            return self._loaded_blobs
        entry = self.entries[index]
        blobs = self.store.read_game(entry)
        if len(blobs) != entry.count:
            raise ValueError(
                f"game {entry.game_id} holds {len(blobs)} rows, "
                f"expected {entry.count}"
            )
        self._loaded_index = index
        self._loaded_blobs = blobs
        return blobs

    def _decode_rows(self, rows: np.ndarray) -> list[PositionSample]:
        game_indices = np.searchsorted(self._ends, rows, side="right")
        samples: list[PositionSample] = []
        index = 0
        while index < len(rows):
            game_index = int(game_indices[index])
            end = index + 1
            while end < len(rows) and int(game_indices[end]) == game_index:
                end += 1
            blobs = self._game_blobs(game_index)
            start = int(self._starts[game_index])
            for position in range(index, end):
                local = int(rows[position]) - start
                samples.append(
                    seal_decoded(decode_sample(blobs[local]), self.spec)
                )
            index = end
        return samples

    def next_batch(self) -> TrainingBatch | None:
        if self.is_exhausted():
            return None
        samples: list[PositionSample] = []
        while len(samples) < self.batch_size:
            if not self._ensure_round():
                break
            stop = min(
                len(self._order),
                self._position + self.batch_size - len(samples),
            )
            rows = self._order[self._position:stop]
            samples.extend(self._decode_rows(rows))
            self.store.record_trained(rows)
            self._position = stop
        if not samples:
            return None
        raw = collate_samples(samples, self.spec)
        data = {
            key: torch.from_numpy(np.ascontiguousarray(value))
            for key, value in raw.items()
        }
        self._served += 1
        return TrainingBatch(data=data, sample_count=len(samples))

    def state_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "rounds": self._rounds,
            "pool_samples": self.pool_samples,
            "round_samples": self._round_samples,
            "remaining_samples": self.remaining_samples,
        }

    def summary(self) -> dict[str, object]:
        return {
            **self.state_dict(),
            "games": len(self.entries),
            "no_repeat_rows": self.no_repeat_rows,
            "waiting_for_shuffle": self.waiting_for_shuffle,
            "backend": "file",
            "keep_samples": self.keep_samples,
            "endgame": self.endgame.identity(),
        }

    def validation_batches(self) -> Iterator[TrainingBatch]:
        return iter(())

    def has_validation(self) -> bool:
        return False

    @property
    def steps_per_epoch(self) -> int:
        return self._steps


__all__ = [
    "GAME_FILE_FORMAT",
    "GAME_ROUND_FORMAT",
    "GAME_STORE_FORMAT",
    "GameFileDataSource",
    "GameStore",
    "StoredGame",
]
