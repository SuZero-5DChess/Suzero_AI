"""Uniform replay buffer and its ``TrainingDataSource`` adapter.

Samples are stored as compact zlib records (see :mod:`sample_codec`), not as
float32 one-hot tensors.  `~50 KB` of planes per sample collapses to roughly
half a kilobyte, and decoding happens only for the batches actually trained on.
"""

from __future__ import annotations

import copy
import logging
import operator
import sys
from dataclasses import fields
from math import isqrt
from typing import Iterator, Sequence

import numpy as np
import torch

from suzero_train.data_source import TrainingDataSource
from suzero_train.types import TrainingBatch

from .endgame_sampling import (
    EndgameSamplingConfig,
    game_progress,
    sampling_weights,
    weighted_without_replacement,
)
from .protocol import VariantSpec
from .sample import (
    PositionSample,
    collate_samples,
    seal_decoded,
    validate_sample,
)
from .sample_codec import (
    decode_sample,
    encode_sample,
    peek_move_count,
    planes_to_codes,
)
from .train_counts import (
    evictable_prefix,
    new_counts,
    record,
    untrained_first,
)

logger = logging.getLogger(__name__)

REPLAY_FORMAT = "gn5d_replay_zlib_v1"


def _numpy(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _record_from_legacy(record: object) -> dict[str, object]:
    """Normalise a legacy ``PositionSample`` / dict / tensor dict to fields."""

    if isinstance(record, PositionSample):
        data = dict(getattr(record, "__dict__", {}))
        if not data:
            data = {item.name: getattr(record, item.name) for item in fields(PositionSample)}
    elif isinstance(record, dict):
        data = {key: _numpy(value) for key, value in record.items()}
    else:
        raise ValueError(
            "replay state samples must be PositionSample instances or dicts"
        )

    data.pop("_sealed", None)
    if "board_planes" in data and "board_codes" not in data:
        data["board_codes"] = planes_to_codes(data.pop("board_planes"))
    if "metadata" in data:
        data["metadata"] = {
            str(key): float(value) for key, value in dict(data["metadata"]).items()
        }
    allowed = {item.name for item in fields(PositionSample)}
    return {key: value for key, value in data.items() if key in allowed}


def _legacy_sample(record: object) -> PositionSample:
    return PositionSample(**_record_from_legacy(record))


def _derive_spec(codes: np.ndarray) -> VariantSpec:
    squares = codes.shape[1] if codes.ndim == 2 else 0
    return VariantSpec(name="replay", pgn="", board_side=isqrt(squares))


class ReplayBuffer:
    """FIFO buffer of compressed self-play samples, sampled uniformly."""

    def __init__(
        self, capacity: int = 100_000, spec: VariantSpec | None = None
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.spec = spec
        self._blobs: list[bytes] = []
        #: Times each live blob was served to the trainer, aligned with
        #: ``_blobs``.  Trimmed rows drop their counter with the blob.
        self.train_counts: list[int] = []
        #: In-game progress (0..1) of each live blob, aligned with ``_blobs``.
        #: One ``push`` call is one game, so the batch maximum is that game's
        #: end.  Endgame sampling turns this into per-row weights.
        self.progress: list[float] = []
        self.total_generated = 0
        self.shuffle_state: dict[str, object] | None = None

    def __len__(self) -> int:
        return len(self._blobs)

    def push(self, samples: Sequence[PositionSample]) -> None:
        if not samples:
            return
        encoded: list[bytes] = []
        for sample in samples:
            if not isinstance(sample, PositionSample):
                raise TypeError("samples must be PositionSample instances")
            spec = self.spec or _derive_spec(np.asarray(sample.board_codes))
            validate_sample(sample, spec)
            encoded.append(encode_sample(sample))
        self._blobs.extend(encoded)
        self.train_counts.extend(new_counts(len(encoded)))
        move_counts = np.fromiter(
            (sample.move_count for sample in samples),
            dtype=np.int64,
            count=len(samples),
        )
        self.progress.extend(game_progress(move_counts).tolist())
        self.total_generated += len(encoded)
        overflow = len(self._blobs) - self.capacity
        if overflow > 0:
            self._evict(overflow)

    def _evict(self, requested: int) -> int:
        """Drop up to ``requested`` leading records that were trained on.

        A record is only discarded after the trainer has served it at least
        once; while an untrained record sits at the front, the buffer keeps it
        and every record behind it, so the live size may exceed ``capacity``
        until training catches up.
        """

        requested = max(0, min(int(requested), len(self._blobs)))
        if not requested:
            return 0
        allowed = evictable_prefix(self.train_counts, requested)
        if allowed < requested:
            logger.info(
                "replay buffer: keeping %d untrained front rows; evicted %d "
                "of %d requested rows (%d buffered, capacity %d)",
                requested - allowed,
                allowed,
                requested,
                len(self._blobs),
                self.capacity,
            )
        if not allowed:
            return 0
        del self._blobs[:allowed]
        del self.train_counts[:allowed]
        del self.progress[:allowed]
        return allowed

    def trim_to(self, size: int) -> int:
        """Discard the oldest records beyond ``size`` once they were trained."""

        size = operator.index(size)
        if size < 0:
            raise ValueError("trim size must be non-negative")
        return self._evict(max(0, len(self._blobs) - size))

    def encoded_at(self, indices: Sequence[int]) -> list[bytes]:
        """Return immutable codec records without decoding or recompressing."""

        result = []
        for value in indices:
            index = operator.index(value)
            if not 0 <= index < len(self._blobs):
                raise IndexError("replay sample index out of range")
            result.append(self._blobs[index])
        return result

    def sample_at(self, indices: Sequence[int]) -> list[PositionSample]:
        """Decode records in the supplied index order, using the usual checks."""

        return [
            seal_decoded(decode_sample(blob), self.spec)
            for blob in self.encoded_at(indices)
        ]

    def estimated_bytes(self) -> int:
        """Estimate resident compressed records, including Python object overhead."""

        return sys.getsizeof(self._blobs) + sum(
            len(blob) + sys.getsizeof(b"") for blob in self._blobs
        )

    def sample_indices(
        self, count: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draw distinct row indices for a uniform batch."""

        if count <= 0:
            raise ValueError("count must be positive")
        if count > len(self._blobs):
            raise ValueError(
                f"cannot sample {count} samples from a buffer of "
                f"{len(self._blobs)}"
            )
        generator = rng or np.random.default_rng()
        return generator.choice(len(self._blobs), size=count, replace=False)

    def record_trained(self, indices: Sequence[int]) -> int:
        """Count rows that a data source just served to the trainer."""

        return record(self.train_counts, indices)

    def sample(
        self, count: int, rng: np.random.Generator | None = None
    ) -> list[PositionSample]:
        return self.sample_at(self.sample_indices(count, rng))

    def state_dict(
        self, indices: Sequence[int] | None = None
    ) -> dict[str, object]:
        records = self._blobs if indices is None else self.encoded_at(indices)
        count = len(records)
        offsets = np.zeros(count + 1, dtype=np.int64)
        if count:
            offsets[1:] = np.cumsum(
                np.fromiter((len(blob) for blob in records), dtype=np.int64)
            )
            arena = np.frombuffer(b"".join(records), dtype=np.uint8)
        else:
            arena = np.zeros(0, dtype=np.uint8)
        state: dict[str, object] = {
            "format": REPLAY_FORMAT,
            "capacity": self.capacity,
            "count": count,
            "total_generated": self.total_generated,
            "offsets": offsets,
            "blobs": arena,
        }
        if indices is None:
            # Published shuffle shards are disposable views; only the full
            # buffer state carries the training counters and row progress.
            state["train_counts"] = np.asarray(self.train_counts, dtype=np.int64)
            state["progress"] = np.asarray(self.progress, dtype=np.float32)
            if self.shuffle_state is not None:
                state["shuffle_state"] = copy.deepcopy(self.shuffle_state)
        return state

    def load_state_dict(
        self, state: dict[str, object], *, derive_progress: bool = True
    ) -> None:
        """Restore a buffer snapshot.

        ``derive_progress`` controls how a snapshot without per-row progress
        (written before endgame sampling) is handled: peeking every blob keeps
        the ordering usable after a resume, while disposable shuffle shards and
        legacy migration can skip that pass because nothing samples from them.
        """

        if not isinstance(state, dict):
            raise ValueError("replay state must be a dictionary")
        total_generated = int(state.get("total_generated", 0))
        if total_generated < 0:
            raise ValueError("replay total_generated must be non-negative")
        shuffle_state = state.get("shuffle_state")
        if shuffle_state is not None and not isinstance(shuffle_state, dict):
            raise ValueError("replay shuffle_state must be a dictionary")
        if state.get("format") == REPLAY_FORMAT:
            self._load_compact(state, derive_progress=derive_progress)
        elif "samples" in state:
            self._load_legacy(state)
        else:
            raise ValueError("replay state has no recognised format")
        # Old snapshots have no counter; the pipeline reconciles checkpoint
        # extra.total_samples before pushing any new records.
        self.total_generated = total_generated
        self.shuffle_state = copy.deepcopy(shuffle_state)

    def _load_compact(
        self, state: dict[str, object], *, derive_progress: bool = True
    ) -> None:
        offsets = np.asarray(_numpy(state.get("offsets")))
        arena = np.asarray(_numpy(state.get("blobs")))
        if offsets.dtype != np.int64 or offsets.ndim != 1:
            raise ValueError("replay offsets must be a flat int64 array")
        if arena.dtype != np.uint8 or arena.ndim != 1:
            raise ValueError("replay blobs must be a flat uint8 array")
        count = int(state.get("count", max(0, offsets.size - 1)))
        if offsets.shape != (count + 1,):
            raise ValueError("replay offsets do not match the sample count")
        if count and (
            offsets[0] != 0
            or (np.diff(offsets) < 0).any()
            or int(offsets[-1]) != arena.size
        ):
            raise ValueError("replay offsets are inconsistent")
        values = state.get("train_counts")
        saved_counts = None
        if values is not None:
            saved_counts = np.asarray(_numpy(values))
            if saved_counts.ndim != 1 or saved_counts.size < count:
                raise ValueError("replay train_counts do not match the sample count")
            if saved_counts.size and int(saved_counts.min()) < 0:
                raise ValueError("replay train_counts must be non-negative")
        # A saved buffer larger than the current capacity may only lose rows
        # the trainer already served.  Without counters (older snapshots)
        # nothing is proven trained, so keep every row rather than silently
        # discarding history the resume cannot recover.
        start = (
            0
            if saved_counts is None
            else evictable_prefix(saved_counts[:count], count - self.capacity)
        )
        if saved_counts is None and count > self.capacity:
            logger.info(
                "replay buffer: saved state has no training counters; keeping "
                "%d rows above the current capacity of %d",
                count,
                self.capacity,
            )
        self._blobs = [
            arena[int(offsets[index]) : int(offsets[index + 1])].tobytes()
            for index in range(start, count)
        ]
        self.train_counts = new_counts(len(self._blobs))
        if saved_counts is not None:
            tail = np.asarray(saved_counts[start:count], dtype=np.int64)
            self.train_counts = [int(value) for value in tail]
        self.progress = self._loaded_progress(
            state, start, count, derive=derive_progress
        )

    def _loaded_progress(
        self, state: dict[str, object], start: int, count: int, *, derive: bool
    ) -> list[float]:
        values = state.get("progress")
        if values is None and not derive:
            return [0.0] * count
        if values is not None:
            array = np.asarray(_numpy(values), dtype=np.float32)
            if array.ndim != 1 or array.size < count:
                raise ValueError("replay progress does not match the sample count")
            tail = array[start:count].astype(np.float64)
            if not np.isfinite(tail).all() or (tail < 0).any() or (tail > 1).any():
                raise ValueError("replay progress must be in [0, 1]")
            return [float(value) for value in tail]
        # Snapshots written before endgame sampling stored no progress.  The
        # per-game boundary is gone with the game frames, so fall back to the
        # largest stored move_count as the scale: rows keep a usable ordering
        # instead of silently reverting to uniform sampling.
        counts = np.fromiter(
            (peek_move_count(blob) for blob in self._blobs),
            dtype=np.int64,
            count=len(self._blobs),
        )
        return game_progress(counts).tolist()

    def _load_legacy(self, state: dict[str, object]) -> None:
        samples = state.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError("replay state 'samples' must be a list")

        encoded: list[bytes] = []
        move_counts: list[int] = []
        # Legacy records carry no training counters, so nothing proves them
        # trained: keep them all and let the trained-only trim dismiss them
        # after a later pass instead of discarding them on load.
        for record in samples:
            sample = _legacy_sample(record)
            spec = self.spec or _derive_spec(np.asarray(sample.board_codes))
            validate_sample(sample, spec)
            encoded.append(encode_sample(sample))
            move_counts.append(int(sample.move_count))
        self._blobs = encoded
        self.train_counts = new_counts(len(encoded))
        self.progress = game_progress(
            np.asarray(move_counts, dtype=np.int64)
        ).tolist()


class ReplayDataSource(TrainingDataSource):
    """Serve a fixed budget of uniform or epoch-shuffled replay batches."""

    def __init__(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        steps_per_epoch: int,
        spec: VariantSpec,
        seed: int = 0,
        sampling: str = "uniform",
        keep_samples: int | None = None,
        endgame: EndgameSamplingConfig | None = None,
    ) -> None:
        if batch_size <= 0 or steps_per_epoch <= 0:
            raise ValueError("batch_size and steps_per_epoch must be positive")
        if sampling not in ("uniform", "shuffle"):
            raise ValueError("sampling must be 'uniform' or 'shuffle'")
        if keep_samples is not None and keep_samples <= 0:
            raise ValueError("keep_samples must be positive")
        self.buffer = buffer
        self.batch_size = batch_size
        self.requested_steps = steps_per_epoch
        self.spec = spec
        self.seed = int(seed)
        self.endgame = endgame or EndgameSamplingConfig()
        self._epoch = 0
        self._served = 0
        self._steps = 0
        self._rng = np.random.default_rng(self.seed)
        self.sampling = sampling
        self._pool = np.empty(0, dtype=np.int64)
        self._order = self._pool
        if sampling == "shuffle":
            available = len(buffer)
            if keep_samples is not None and keep_samples < available:
                self._pool = self._select_pool(available, keep_samples)
            else:
                self._pool = np.arange(available, dtype=np.int64)

    def _weights(self) -> np.ndarray:
        return sampling_weights(np.asarray(self.buffer.progress), self.endgame)

    def _select_pool(self, available: int, keep_samples: int) -> np.ndarray:
        """Choose the epoch's shuffled subset, weighted when enabled."""

        rows = np.arange(available, dtype=np.int64)
        if self.endgame.enabled:
            return weighted_without_replacement(
                rows, self._weights(), keep_samples, self._rng
            )
        return self._rng.choice(available, size=keep_samples, replace=False)

    def begin_epoch(self) -> None:
        self._epoch += 1
        self._served = 0
        available = len(self._pool) if self.sampling == "shuffle" else len(self.buffer)
        if available == 0:
            self._steps = 0
            return
        # Match the reference's effective-steps cap so one epoch does not
        # sample more batches than the buffer can support.
        self._steps = min(self.requested_steps, max(1, available // self.batch_size))
        self._rng = np.random.default_rng(self.seed + self._epoch)
        if self.sampling == "shuffle":
            # Serve rows that were never trained before spending the epoch on
            # rows that already had a pass.
            self._order = untrained_first(
                self._rng.permutation(self._pool), self.buffer.train_counts
            )

    def is_exhausted(self) -> bool:
        return self._served >= self._steps

    def next_batch(self) -> TrainingBatch | None:
        if self.is_exhausted():
            return None
        if self.sampling == "shuffle":
            start = self._served * self.batch_size
            indices = self._order[start : start + self.batch_size]
            samples = self.buffer.sample_at(indices)
            size = len(indices)
        else:
            size = min(self.batch_size, len(self.buffer))
            if self.endgame.enabled:
                indices = weighted_without_replacement(
                    np.arange(len(self.buffer), dtype=np.int64),
                    self._weights(),
                    size,
                    self._rng,
                )
            else:
                indices = self.buffer.sample_indices(size, self._rng)
            samples = self.buffer.sample_at(indices)
        self.buffer.record_trained(indices)
        raw = collate_samples(samples, self.spec)
        data = {
            key: torch.from_numpy(np.ascontiguousarray(value))
            for key, value in raw.items()
        }
        self._served += 1
        return TrainingBatch(data=data, sample_count=size)

    def validation_batches(self) -> Iterator[TrainingBatch]:
        return iter(())

    def has_validation(self) -> bool:
        return False

    @property
    def steps_per_epoch(self) -> int:
        return self._steps
