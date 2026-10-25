"""Abstract training backend."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterable, Mapping

from .types import TrainStepResult, TrainingBatch


class Trainer(ABC):
    """Owns a model and optimizer and performs one step at a time.

    The training loop is responsible only for sequencing batches, logging and
    checkpoint cadence.  Anything model-, loss- or optimizer-specific belongs to
    an implementation of this interface.
    """

    @abstractmethod
    def train_step(self, batch: TrainingBatch) -> TrainStepResult:
        """Forward, backward and optimizer step for a single batch."""

    @abstractmethod
    def validate(self, batches: Iterable[TrainingBatch]) -> Mapping[str, float]:
        """Evaluate without updating weights and return averaged metrics."""

    @abstractmethod
    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Persist everything needed to resume this trainer."""

    @abstractmethod
    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Restore state from a checkpoint written by :meth:`save_checkpoint`."""

    @property
    @abstractmethod
    def samples_seen(self) -> int:
        """Number of samples consumed by optimizer steps so far."""

    @property
    @abstractmethod
    def global_step(self) -> int:
        """Number of optimizer steps performed so far."""

    def on_epoch_start(self) -> None:
        """Called by the loop before the first batch of each epoch.

        Backends may use this to refresh schedules or reset per-epoch windows.
        The default is a no-op.
        """

    def on_log(self) -> None:
        """Called after the metrics sink has logged a training step.

        Backends may use this for bookkeeping tied to logging.  The
        :class:`~suzero_train.pytorch.TorchTrainer` does not rely on it:
        KataGo rotates its ``_batch`` metric window on a fixed step cadence
        (``batch_metric_window_steps``), not on when logging happens.  The
        default is a no-op.
        """

    def close(self) -> None:
        """Release backend resources.  The default is a no-op."""
