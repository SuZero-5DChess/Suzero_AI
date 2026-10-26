"""Published shuffle rounds and a row-level no-repeat training source.

Only the current lossy window is shuffled. Each publication is one shuffle
round: a subset of the window written out in random order. A new round is
drawn only from rows that the immediately preceding round did not select, so
successive shuffles never repeat a row. One training epoch may consume several
rounds when ``keep_samples`` is smaller than the epoch's step budget.

Rounds, not output files, are the unit of repetition. Writing a new directory
is merely how a round is handed to the trainer; it does not reset the row
history.
"""

from __future__ import annotations

import hashlib
import json
import logging
import operator
import os
from pathlib import Path
import re
import uuid
from typing import Callable, Iterator, Sequence

import numpy as np
import torch

from suzero_train.data_source import TrainingDataSource
from suzero_train.types import TrainingBatch

from .endgame_sampling import (
    EndgameSamplingConfig,
    repeat_by_weight,
    sampling_weights,
    weighted_without_replacement,
)
from .protocol import VariantSpec
from .replay import ReplayBuffer
from .sample import collate_samples
from .train_counts import record, untrained_first

logger = logging.getLogger(__name__)
SHUFFLE_FORMAT = "suzero_replay_shuffle_v2"
ROWS_PER_FILE = 4096
ROUND_SEED_STRIDE = 7919
_GENERATION = re.compile(r"generation-[0-9a-f]{32}")
_STAGING = re.compile(r"generation-[0-9a-f]{32}\.tmp")
_SHARD = re.compile(r"[0-9]{6}\.npz")

RoundPublisher = Callable[[Sequence[int]], "dict[str, object] | None"]


def _read_manifest(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError("shuffle manifest must not be a symlink")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != SHUFFLE_FORMAT:
        raise ValueError("unrecognised shuffle manifest")
    if not _GENERATION.fullmatch(str(data.get("generation", ""))):
        raise ValueError("invalid shuffle generation name")
    files = data.get("files")
    if not isinstance(files, list):
        raise ValueError("shuffle files must be a list")
    names = []
    count = 0
    for item in files:
        if not isinstance(item, dict) or not _SHARD.fullmatch(str(item.get("name", ""))):
            raise ValueError("invalid shuffle shard name")
        rows = item.get("count")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError("invalid shuffle shard count")
        names.append(item["name"])
        count += rows
    if len(set(names)) != len(names) or count != data.get("count"):
        raise ValueError("shuffle manifest counts or names are inconsistent")
    pool_samples = data.get("pool_samples")
    if (
        isinstance(pool_samples, bool)
        or not isinstance(pool_samples, int)
        or pool_samples < 0
    ):
        raise ValueError("invalid shuffle pool size")
    selected = data.get("rows")
    if not isinstance(selected, list) or len(selected) != count:
        raise ValueError("shuffle rows do not match the manifest count")
    producer = data.get("producer")
    endgame = producer.get("endgame") if isinstance(producer, dict) else None
    if endgame is not None:
        if not isinstance(endgame, dict):
            raise ValueError("shuffle endgame settings must be an object")
        try:
            EndgameSamplingConfig(
                enabled=True,
                strength=endgame.get("strength"),
                exponent=endgame.get("exponent"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid shuffle endgame settings") from exc
    seen: set[int] = set()
    for row in selected:
        if isinstance(row, bool) or not isinstance(row, int):
            raise ValueError("invalid shuffle row index")
        if not 0 <= row < pool_samples:
            raise ValueError("shuffle row index is outside the pool")
        # Endgame sampling deliberately repeats late rows inside one round.
        if row in seen and endgame is None:
            raise ValueError("duplicate shuffle row index")
        seen.add(row)
    return data


def _write_json(path: Path, data: dict[str, object]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _remove_generation(path: Path) -> None:
    """Delete only a recognised, owned generation, never unrelated contents."""

    if path.is_symlink():
        raise ValueError("shuffle generation must not be a symlink")
    manifest = _read_manifest(path / "manifest.json")
    if path.name != manifest["generation"]:
        raise ValueError("shuffle generation ownership mismatch")
    expected = {item["name"] for item in manifest["files"]} | {"manifest.json", "owner.json"}
    entries = list(path.iterdir())
    if any(p.name not in expected or p.is_symlink() or not p.is_file() for p in entries):
        raise ValueError(f"refusing to delete unexpected shuffle contents in {path}")
    for entry in entries:
        entry.unlink()
    path.rmdir()


def _remove_staging(path: Path) -> None:
    """Clean crash leftovers only after checking our explicit ownership marker."""

    if path.is_symlink():
        raise ValueError("shuffle staging directory must not be a symlink")
    entries = list(path.iterdir())
    if entries:
        marker = path / "owner.json"
        if marker.is_symlink():
            raise ValueError("shuffle ownership marker must not be a symlink")
        owner = json.loads(marker.read_text(encoding="utf-8"))
        if owner != {"format": SHUFFLE_FORMAT, "generation": path.name[:-4]}:
            raise ValueError("shuffle staging ownership mismatch")
        allowed = {"owner.json", "manifest.json", "manifest.json.tmp"}
        if any(
            (p.name not in allowed and not _SHARD.fullmatch(p.name))
            or p.is_symlink() or not p.is_file()
            for p in entries
        ):
            raise ValueError(f"refusing to delete unexpected shuffle contents in {path}")
    for entry in entries:
        entry.unlink()
    path.rmdir()


def _clean_obsolete(directory: Path, current: str | None = None) -> None:
    for path in directory.iterdir():
        if _GENERATION.fullmatch(path.name) and path.name != current:
            _remove_generation(path)
        elif _STAGING.fullmatch(path.name):
            _remove_staging(path)


def clear_shuffle(directory: Path) -> None:
    """Discard owned training views; never search them for lost replay rows."""

    directory = Path(directory)
    if not directory.exists():
        return
    if directory.is_symlink():
        raise ValueError("shuffle directory must not be a symlink")
    pointer = directory / "current.json"
    if pointer.exists():
        _read_manifest(pointer)
    # Invalidate the pointer before deleting its data, including after a crash.
    if pointer.exists():
        pointer.unlink()
    _clean_obsolete(directory)


def _excluded_rows(exclude: Sequence[int], size: int) -> np.ndarray:
    values: set[int] = set()
    for item in exclude:
        try:
            index = operator.index(item)
        except TypeError as error:
            raise ValueError("excluded rows must be integers") from error
        if not 0 <= index < size:
            raise ValueError("excluded row index is outside the pool")
        values.add(index)
    return np.fromiter(sorted(values), dtype=np.int64, count=len(values))


def _buffer_digest(buffer: ReplayBuffer) -> str:
    digest = hashlib.sha256()
    for blob in buffer.encoded_at(range(len(buffer))):
        digest.update(len(blob).to_bytes(8, "little"))
        digest.update(blob)
    return digest.hexdigest()


def _producer(
    buffer: ReplayBuffer,
    *,
    iteration: int,
    seed: int,
    keep_samples: int | None,
    rows_per_file: int,
    endgame: dict[str, float] | None,
) -> dict[str, object]:
    """Identity used to reuse a round when identical input is retried.

    Deliberately excludes the excluded-row set: the exclusion follows from the
    previous round, so a retry of the same phase must reuse the already
    published round instead of republishing a different one.
    """

    return {
        "iteration": int(iteration), "total_generated": buffer.total_generated,
        "buffered": len(buffer), "digest": _buffer_digest(buffer),
        "seed": int(seed), "keep_samples": keep_samples,
        "rows_per_file": int(rows_per_file), "endgame": endgame,
    }


def _same_round(stored: object, expected: dict[str, object]) -> bool:
    if not isinstance(stored, dict):
        return False
    return {key: value for key, value in stored.items() if key != "exclude_digest"} == expected


def publish_shuffle(
    buffer: ReplayBuffer,
    directory: Path,
    *,
    iteration: int,
    seed: int,
    keep_samples: int | None = None,
    rows_per_file: int = ROWS_PER_FILE,
    exclude: Sequence[int] = (),
    no_repeat_rows: bool = True,
    endgame: EndgameSamplingConfig | None = None,
) -> dict[str, object] | None:
    """Publish one shuffle round drawn from rows outside ``exclude``.

    ``exclude`` holds pool row indices selected by the previous round; those
    rows are never selected again by this round. Pass an empty ``exclude`` and
    ``no_repeat_rows=False`` to draw every round from the whole pool. Returns
    ``None`` when every row is excluded, because there is no new round to train
    on. Retrying the same input and seed reuses the already published round.

    With ``endgame`` enabled, rows near the end of their game also receive
    extra copies inside the round, so they are trained on more often without
    dropping any other row.
    """

    if rows_per_file <= 0 or (keep_samples is not None and keep_samples <= 0):
        raise ValueError("rows_per_file and keep_samples must be positive")
    if not isinstance(no_repeat_rows, bool):
        raise ValueError("no_repeat_rows must be a boolean")
    sampling = endgame or EndgameSamplingConfig()
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("shuffle directory must not be a symlink")
    excluded = _excluded_rows(exclude, len(buffer))
    producer = _producer(
        buffer, iteration=iteration, seed=seed,
        keep_samples=keep_samples, rows_per_file=rows_per_file,
        endgame=sampling.identity(),
    )
    pointer = directory / "current.json"
    if pointer.exists():
        current = _read_manifest(pointer)
        if _same_round(current.get("producer"), producer):
            generation_dir = directory / str(current["generation"])
            if generation_dir.is_symlink():
                raise ValueError("shuffle generation must not be a symlink")
            # Missing/corrupt published data is an error, not permission to
            # regenerate consumed files under a new identity.
            stored = _read_manifest(generation_dir / "manifest.json")
            if stored != current:
                raise ValueError("shuffle publication manifests differ")
            for item in current["files"]:
                path = generation_dir / item["name"]
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"missing or unsafe shuffle shard {path}")
            _clean_obsolete(directory, str(current["generation"]))
            return current

    available = len(buffer) - int(excluded.size)
    if available <= 0 and len(buffer):
        logger.info(
            "replay shuffle: all %d buffered rows were selected by the previous "
            "round; no new round", len(buffer),
        )
        return None

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if len(buffer) == 0:
        order = np.empty(0, dtype=np.int64)
    else:
        eligible = np.setdiff1d(
            np.arange(len(buffer), dtype=np.int64), excluded, assume_unique=True
        )
        weights = None
        if sampling.enabled and len(buffer.progress) == len(buffer):
            weights = sampling_weights(np.asarray(buffer.progress), sampling)
        if keep_samples is not None and keep_samples < eligible.size:
            if weights is not None:
                pool = weighted_without_replacement(
                    eligible, weights[eligible], keep_samples, rng
                )
            else:
                pool = rng.choice(eligible, size=keep_samples, replace=False)
        else:
            pool = eligible
        if weights is not None:
            pool = repeat_by_weight(pool, weights[pool], rng)
        order = rng.permutation(pool)
        if len(buffer.train_counts) == len(buffer):
            # Serve never-trained rows before rows that already had a pass.
            order = untrained_first(order, buffer.train_counts)
    generation = "generation-" + uuid.uuid4().hex
    staging = directory / (generation + ".tmp")
    staging.mkdir()
    exclude_digest = hashlib.sha256(
        np.asarray(excluded, dtype="<i8").tobytes()
    ).hexdigest()
    manifest = {
        "format": SHUFFLE_FORMAT, "generation": generation,
        "producer": {**producer, "exclude_digest": exclude_digest},
        "count": len(order), "pool_samples": len(buffer),
        "rows": [int(row) for row in order], "files": [],
    }
    try:
        # Written before any blobs, so a later process can identify a partial
        # generation without treating arbitrary .tmp directories as ours.
        (staging / "owner.json").write_text(
            json.dumps({"format": SHUFFLE_FORMAT, "generation": generation}),
            encoding="utf-8",
        )
        for start in range(0, len(order), rows_per_file):
            indices = order[start : start + rows_per_file]
            name = f"{len(manifest['files']):06d}.npz"
            # The sample codec is unchanged: NPZ just carries compact offsets
            # and the existing zlib byte records, without object pickling.
            np.savez(staging / name, **buffer.state_dict(indices))
            manifest["files"].append({"name": name, "count": len(indices)})
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, directory / generation)
        _write_json(pointer, manifest)
    except BaseException:
        # Staging was created by this call in an empty, unique directory.
        if staging.exists():
            _remove_staging(staging)
        raise
    _clean_obsolete(directory, generation)
    logger.info(
        "published replay shuffle: generation=%s rows=%d excluded=%d files=%d "
        "no_repeat_rows=%s",
        generation, len(order), excluded.size, len(manifest["files"]),
        str(no_repeat_rows).lower(),
    )
    return manifest


def previous_round_exclusion(
    directory: Path, *, pool_samples: int, total_generated: int
) -> list[int]:
    """Map the last published round's rows onto the current replay window.

    Rows are only removed from the front of the window, so an old index simply
    shifts by the number of rows evicted since publication. Rows that fell out
    of the window are dropped; there is nothing left to exclude for them.
    """

    for value, name in ((pool_samples, "pool_samples"), (total_generated, "total_generated")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    path = Path(directory) / "current.json"
    if not path.is_file():
        return []
    manifest = _read_manifest(path)
    producer = manifest.get("producer")
    generated_then = (
        int(producer.get("total_generated", total_generated))
        if isinstance(producer, dict) else total_generated
    )
    added = max(0, int(total_generated) - generated_then)
    shift = max(0, int(manifest["pool_samples"]) + added - pool_samples)
    shifted = []
    for row in manifest["rows"]:
        value = int(row) - shift
        if 0 <= value < pool_samples:
            shifted.append(value)
    return shifted


class FileReplayDataSource(TrainingDataSource):
    """Consume published shuffle rounds without repeating adjacent rounds.

    The file/row cursor continues across epochs, so a round may span epoch
    boundaries. When a round is exhausted and the epoch still has steps left,
    the configured publisher draws the next round from rows the exhausted
    round did not select. A new epoch does not reset that rule; the publisher
    is only asked when another round is actually needed. With
    ``no_repeat_rows=False`` the publisher is asked for a round from the whole
    pool, so adjacent rounds may share rows.
    """

    def __init__(
        self,
        directory: Path,
        batch_size: int,
        steps_per_epoch: int,
        spec: VariantSpec,
        *,
        state: dict[str, object] | None = None,
        publisher: RoundPublisher | None = None,
        no_repeat_rows: bool = True,
        train_counts: list[int] | None = None,
    ) -> None:
        if batch_size <= 0 or steps_per_epoch <= 0:
            raise ValueError("batch_size and steps_per_epoch must be positive")
        if not isinstance(no_repeat_rows, bool):
            raise ValueError("no_repeat_rows must be a boolean")
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("shuffle directory must not be a symlink")
        self.batch_size = int(batch_size)
        self.requested_steps = int(steps_per_epoch)
        self.spec = spec
        self.publisher = publisher
        self.no_repeat_rows = no_repeat_rows
        #: Live buffer counters, indexed by the pool rows recorded in the
        #: manifest.  Shard consumption itself never changes those rows.
        self.train_counts = train_counts
        self._uncounted = 0
        self._warned_uncounted = False
        self._rounds = 0
        self._waiting = False
        self._install(_read_manifest(self.directory / "current.json"))
        self._served = 0
        self._steps = 0
        if state is not None and state.get("generation") == self.generation:
            self._file_index = int(state["file_index"])
            self._row_offset = int(state["row_offset"])
            if not 0 <= self._file_index <= len(self.files):
                raise ValueError("invalid shuffle file cursor")
            if self._file_index == len(self.files):
                if self._row_offset != 0:
                    raise ValueError("exhausted shuffle has a row offset")
            elif not 0 <= self._row_offset < self.files[self._file_index]["count"]:
                raise ValueError("invalid shuffle row cursor")

    def _install(self, manifest: dict[str, object]) -> None:
        self.manifest = manifest
        self.generation = str(manifest["generation"])
        self._generation_dir = self.directory / self.generation
        if self._generation_dir.is_symlink():
            raise ValueError("shuffle generation must not be a symlink")
        if _read_manifest(self._generation_dir / "manifest.json") != self.manifest:
            raise ValueError("shuffle publication manifests differ")
        self.files = self.manifest["files"]
        self.rows = [int(row) for row in self.manifest["rows"]]
        self.pool_samples = int(self.manifest["pool_samples"])
        self._file_starts = np.zeros(len(self.files) + 1, dtype=np.int64)
        if self.files:
            self._file_starts[1:] = np.cumsum(
                np.fromiter(
                    (item["count"] for item in self.files),
                    dtype=np.int64,
                    count=len(self.files),
                )
            )
        self._file_index = 0
        self._row_offset = 0
        self._shard: ReplayBuffer | None = None

    @property
    def remaining_samples(self) -> int:
        return sum(item["count"] for item in self.files[self._file_index:]) - self._row_offset

    @property
    def waiting_for_shuffle(self) -> bool:
        return self._waiting or self._file_index == len(self.files)

    def _ensure_round(self) -> bool:
        """Return True when the current round has unread rows.

        Publishes the next round when the current one is spent. The previous
        round's entire selection is excluded, including rows that were never
        read, so the next round cannot repeat anything it drew.
        """

        if self._file_index < len(self.files):
            self._waiting = False
            return True
        if self.publisher is None or not self.files:
            self._waiting = True
            return False
        manifest = self.publisher(
            list(self.rows) if self.no_repeat_rows else []
        )
        if manifest is None:
            self._waiting = True
            return False
        self._install(manifest)
        self._rounds += 1
        self._waiting = False
        return bool(self.files)

    def begin_epoch(self) -> None:
        self._served = 0
        self._ensure_round()
        self._steps = 0 if self.waiting_for_shuffle else self.requested_steps

    def is_exhausted(self) -> bool:
        return self._served >= self._steps or self._waiting

    def _load_shard(self) -> ReplayBuffer:
        item = self.files[self._file_index]
        path = self._generation_dir / item["name"]
        if path.is_symlink():
            raise ValueError("shuffle shard must not be a symlink")
        with np.load(path, allow_pickle=False) as archive:
            state = {key: archive[key] for key in archive.files}
        for key in ("format", "capacity", "count", "total_generated"):
            state[key] = state[key].item()
        if state["count"] != item["count"]:
            raise ValueError("shuffle shard count differs from manifest")
        replay = ReplayBuffer(item["count"], self.spec)
        # Shards are read sequentially and never sampled from, so their
        # per-row progress is not needed.
        replay.load_state_dict(state, derive_progress=False)
        return replay

    def _record_round_rows(self, file_index: int, start: int, stop: int) -> None:
        """Count served shard rows against their pool rows."""

        if self.train_counts is None or stop <= start:
            return
        base = int(self._file_starts[file_index])
        rows = [self.rows[base + index] for index in range(start, stop)]
        missed = len(rows) - record(self.train_counts, rows)
        if missed:
            self._uncounted += missed
            if not self._warned_uncounted:
                self._warned_uncounted = True
                logger.warning(
                    "replay shuffle: served rows are outside the live buffer; "
                    "their training counts are not recorded (further misses "
                    "are silent)"
                )

    def next_batch(self) -> TrainingBatch | None:
        if self.is_exhausted():
            return None
        samples = []
        while len(samples) < self.batch_size:
            if not self._ensure_round():
                break
            if self._shard is None:
                self._shard = self._load_shard()
            stop = min(len(self._shard), self._row_offset + self.batch_size - len(samples))
            samples.extend(self._shard.sample_at(range(self._row_offset, stop)))
            self._record_round_rows(self._file_index, self._row_offset, stop)
            self._row_offset = stop
            if stop == len(self._shard):
                self._file_index += 1
                self._row_offset = 0
                self._shard = None
        if not samples:
            return None
        raw = collate_samples(samples, self.spec)
        data = {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in raw.items()}
        self._served += 1
        return TrainingBatch(data=data, sample_count=len(samples))

    def state_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation, "file_index": self._file_index,
            "row_offset": self._row_offset,
            "files_used": [item["name"] for item in self.files[:self._file_index]],
        }

    def summary(self) -> dict[str, object]:
        return {
            **self.state_dict(), "files": len(self.files),
            "pool_samples": self.pool_samples,
            "round_samples": int(self.manifest["count"]),
            "remaining_samples": self.remaining_samples,
            "rounds": self._rounds,
            "no_repeat_rows": self.no_repeat_rows,
            "waiting_for_shuffle": self.waiting_for_shuffle,
            "uncounted_samples": self._uncounted,
        }

    def validation_batches(self) -> Iterator[TrainingBatch]:
        return iter(())

    def has_validation(self) -> bool:
        return False

    @property
    def steps_per_epoch(self) -> int:
        return self._steps


__all__ = [
    "FileReplayDataSource",
    "RoundPublisher",
    "ROUND_SEED_STRIDE",
    "ROWS_PER_FILE",
    "SHUFFLE_FORMAT",
    "clear_shuffle",
    "previous_round_exclusion",
    "publish_shuffle",
]
