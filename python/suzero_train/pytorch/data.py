"""Game-agnostic data sources for the PyTorch trainer.

Both sources produce :class:`~suzero_train.types.TrainingBatch` objects whose
``data`` mapping is consumed by the trainer's ``objective``.  They deliberately
know nothing about board games, tensor shapes or storage formats.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch

from ..data_source import TrainingDataSource
from ..types import TrainingBatch

CollateFn = Callable[[Sequence[Any]], Mapping[str, Any]]
BatchFactory = Callable[[], Iterable[TrainingBatch]]


def default_collate(samples: Sequence[Any]) -> Mapping[str, Any]:
    """Stack tensor samples into ``{"inputs": tensor}``."""
    return {"inputs": torch.stack(list(samples))}


class IterableDataSource(TrainingDataSource):
    """Adapt two batch-iterable factories to the data-source contract.

    Each call to :meth:`begin_epoch` obtains a fresh iterator from
    ``epoch_factory``, so a factory may stream from disk, generate self-play
    data, or simply replay a list.
    """

    def __init__(
        self,
        epoch_factory: BatchFactory,
        validation_factory: BatchFactory | None = None,
    ) -> None:
        self._epoch_factory = epoch_factory
        self._validation_factory = validation_factory
        self._batches: Iterator[TrainingBatch] | None = None
        self._exhausted = False

    def begin_epoch(self) -> None:
        self._batches = iter(self._epoch_factory())
        self._exhausted = False

    def next_batch(self) -> TrainingBatch | None:
        if self._batches is None:
            self.begin_epoch()
        assert self._batches is not None
        batch = next(self._batches, None)
        if batch is None:
            self._exhausted = True
        return batch

    def is_exhausted(self) -> bool:
        # A generic iterator cannot be inspected without consuming it.  Once
        # next_batch() has returned None this is exact; before that the loop
        # falls back to a one-shot probe at a step-limit boundary.
        return self._exhausted

    def validation_batches(self) -> Iterator[TrainingBatch]:
        if self._validation_factory is None:
            return iter(())
        return iter(self._validation_factory())

    def has_validation(self) -> bool:
        return self._validation_factory is not None


class InMemoryDataSource(TrainingDataSource):
    """Serve batches from an in-memory sequence.

    ``collate_fn`` turns a list of samples into the mapping consumed by the
    trainer's objective.  Training batches are shuffled each epoch when
    ``shuffle`` is set; with a ``seed`` the shuffle is reproducible per epoch.
    Validation data, when supplied, is never shuffled and keeps a final
    partial batch (``drop_last`` applies to training only), so validation
    metrics may weight the last batch by fewer samples than training does.

    ``validation_collate_fn`` overrides ``collate_fn`` for validation, e.g.
    when validation samples must not be collated into a training-shaped batch.
    It defaults to ``collate_fn``.

    ``start_epoch`` sets the epoch counter used to seed the shuffle.  Set it
    when resuming from a checkpoint so the resumed run draws the same epoch
    order it would have had without interruption.
    """

    def __init__(
        self,
        samples: Sequence[Any],
        batch_size: int,
        collate_fn: CollateFn = default_collate,
        *,
        validation_samples: Sequence[Any] | None = None,
        validation_batch_size: int | None = None,
        validation_collate_fn: CollateFn | None = None,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int | None = None,
        start_epoch: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if validation_batch_size is not None and validation_batch_size <= 0:
            raise ValueError("validation_batch_size must be positive")
        if start_epoch < 0:
            raise ValueError("start_epoch must be >= 0")
        if len(samples) == 0:
            raise ValueError("samples must not be empty")
        if drop_last and len(samples) < batch_size:
            raise ValueError(
                "samples must contain at least one full batch when drop_last is "
                f"True (got {len(samples)} samples for batch_size {batch_size})"
            )
        self._samples = samples
        self._batch_size = batch_size
        self._collate_fn = collate_fn
        self._validation_collate_fn = (
            collate_fn if validation_collate_fn is None else validation_collate_fn
        )
        self._validation_samples = validation_samples
        self._validation_batch_size = validation_batch_size or batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._seed = seed
        self._order: list[int] = []
        self._cursor = 0
        self._epoch = start_epoch
        self._prepare_order()

    def _prepare_order(self) -> None:
        order = list(range(len(self._samples)))
        if self._shuffle:
            rng = random.Random(
                None if self._seed is None else self._seed + self._epoch
            )
            rng.shuffle(order)
        self._order = order
        self._cursor = 0

    def begin_epoch(self) -> None:
        self._prepare_order()
        self._epoch += 1

    def next_batch(self) -> TrainingBatch | None:
        if self._cursor >= len(self._order):
            return None
        end = min(self._cursor + self._batch_size, len(self._order))
        if end - self._cursor < self._batch_size and self._drop_last:
            self._cursor = len(self._order)
            return None
        indices = self._order[self._cursor : end]
        self._cursor = end
        return self._make_batch(
            [self._samples[i] for i in indices], self._collate_fn
        )

    def is_exhausted(self) -> bool:
        return self._cursor >= len(self._order)

    def validation_batches(self) -> Iterator[TrainingBatch]:
        if self._validation_samples is None:
            return iter(())
        return self._iter_validation()

    def _iter_validation(self) -> Iterator[TrainingBatch]:
        assert self._validation_samples is not None
        for start in range(0, len(self._validation_samples), self._validation_batch_size):
            chunk = self._validation_samples[start : start + self._validation_batch_size]
            yield self._make_batch(list(chunk), self._validation_collate_fn)

    def _make_batch(
        self, samples: Sequence[Any], collate_fn: CollateFn
    ) -> TrainingBatch:
        data = collate_fn(samples)
        if not isinstance(data, Mapping):
            raise TypeError("collate_fn must return a mapping of tensors")
        return TrainingBatch(data=data, sample_count=len(samples))

    def has_validation(self) -> bool:
        return (
            self._validation_samples is not None
            and len(self._validation_samples) > 0
        )
