"""Abstract source of training and validation minibatches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from .types import TrainingBatch


class TrainingDataSource(ABC):
    """Supplies minibatches for one training epoch at a time.

    Contract
    --------
    * The loop calls :meth:`begin_epoch` once before each epoch.
    * :meth:`next_batch` returns batches until the epoch is exhausted, then
      returns ``None``.  The next epoch starts with another ``begin_epoch``.
    * :meth:`validation_batches` is a one-shot iterator independent of the
      training stream; it may yield nothing when no validation data exists.
    """

    @abstractmethod
    def begin_epoch(self) -> None:
        """Prepare a fresh pass over the training data."""

    @abstractmethod
    def next_batch(self) -> TrainingBatch | None:
        """Return the next training batch, or ``None`` when the epoch is done."""

    @abstractmethod
    def validation_batches(self) -> Iterator[TrainingBatch]:
        """Yield validation batches from the beginning of the validation set."""

    def is_exhausted(self) -> bool:
        """Whether the current epoch has no batches left.

        The loop uses this to recognize a step limit that lands exactly on an
        epoch boundary.  Sources that cannot answer cheaply may keep the
        default; the loop then probes once with :meth:`next_batch`.
        """
        return False

    def has_validation(self) -> bool:
        """Whether this source can serve validation data at all."""
        return True
