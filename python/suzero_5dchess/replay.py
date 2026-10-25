"""Uniform replay buffer and its ``TrainingDataSource`` adapter.

Samples are stored as compact zlib records (see :mod:`sample_codec`), not as
float32 one-hot tensors.  `~50 KB` of planes per sample collapses to roughly
half a kilobyte, and decoding happens only for the batches actually trained on.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from math import isqrt
from typing import Iterator, Sequence

import numpy as np
import torch

from suzero_train.data_source import TrainingDataSource
from suzero_train.types import TrainingBatch

from .protocol import VariantSpec
from .sample import (
    PositionSample,
    collate_samples,
    seal_decoded,
    validate_sample,
)
from .sample_codec import decode_sample, encode_sample, planes_to_codes

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
        self, capacity: int = 50_000, spec: VariantSpec | None = None
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.spec = spec
        self._blobs: list[bytes] = []

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
        overflow = len(self._blobs) - self.capacity
        if overflow > 0:
            del self._blobs[:overflow]

    def sample(
        self, count: int, rng: np.random.Generator | None = None
    ) -> list[PositionSample]:
        if count <= 0:
            raise ValueError("count must be positive")
        if count > len(self._blobs):
            raise ValueError(
                f"cannot sample {count} samples from a buffer of "
                f"{len(self._blobs)}"
            )
        generator = rng or np.random.default_rng()
        indices = generator.choice(len(self._blobs), size=count, replace=False)
        return [
            seal_decoded(decode_sample(self._blobs[int(index)]), self.spec)
            for index in indices
        ]

    def state_dict(self) -> dict[str, object]:
        count = len(self._blobs)
        offsets = np.zeros(count + 1, dtype=np.int64)
        if count:
            offsets[1:] = np.cumsum(
                np.fromiter((len(blob) for blob in self._blobs), dtype=np.int64)
            )
            arena = np.frombuffer(b"".join(self._blobs), dtype=np.uint8)
        else:
            arena = np.zeros(0, dtype=np.uint8)
        return {
            "format": REPLAY_FORMAT,
            "capacity": self.capacity,
            "count": count,
            "offsets": offsets,
            "blobs": arena,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if not isinstance(state, dict):
            raise ValueError("replay state must be a dictionary")
        if state.get("format") == REPLAY_FORMAT:
            self._load_compact(state)
            return
        if "samples" in state:
            self._load_legacy(state)
            return
        raise ValueError("replay state has no recognised format")

    def _load_compact(self, state: dict[str, object]) -> None:
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
        start = max(0, count - self.capacity)
        self._blobs = [
            arena[int(offsets[index]) : int(offsets[index + 1])].tobytes()
            for index in range(start, count)
        ]

    def _load_legacy(self, state: dict[str, object]) -> None:
        samples = state.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError("replay state 'samples' must be a list")

        encoded: list[bytes] = []
        for record in samples[-self.capacity :]:
            sample = _legacy_sample(record)
            spec = self.spec or _derive_spec(np.asarray(sample.board_codes))
            validate_sample(sample, spec)
            encoded.append(encode_sample(sample))
        self._blobs = encoded


class ReplayDataSource(TrainingDataSource):
    """Serve ``steps_per_epoch`` uniformly sampled batches from a buffer."""

    def __init__(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        steps_per_epoch: int,
        spec: VariantSpec,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0 or steps_per_epoch <= 0:
            raise ValueError("batch_size and steps_per_epoch must be positive")
        self.buffer = buffer
        self.batch_size = batch_size
        self.requested_steps = steps_per_epoch
        self.spec = spec
        self.seed = int(seed)
        self._epoch = 0
        self._served = 0
        self._steps = 0
        self._rng = np.random.default_rng(self.seed)

    def begin_epoch(self) -> None:
        self._epoch += 1
        self._served = 0
        available = len(self.buffer)
        if available == 0:
            self._steps = 0
            return
        # Match the reference's effective-steps cap so one epoch does not
        # sample more batches than the buffer can support.
        self._steps = min(self.requested_steps, max(1, available // self.batch_size))
        self._rng = np.random.default_rng(self.seed + self._epoch)

    def is_exhausted(self) -> bool:
        return self._served >= self._steps

    def next_batch(self) -> TrainingBatch | None:
        if self.is_exhausted():
            return None
        size = min(self.batch_size, len(self.buffer))
        samples = self.buffer.sample(size, self._rng)
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
