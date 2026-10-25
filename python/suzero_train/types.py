"""Backend-neutral value types exchanged by the training loop.

The abstract layer deliberately knows nothing about tensors, optimizers or
game rules.  A backend (for example :mod:`suzero_train.pytorch`) defines what
lives inside ``TrainingBatch.data`` and how a step is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingBatch:
    """One minibatch of training samples.

    ``data`` is opaque to the loop; only the backend interprets it.  The loop
    reads ``sample_count`` to keep sample-based budgets and reports independent
    of the batch representation.
    """

    data: Mapping[str, Any]
    sample_count: int

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("TrainingBatch.sample_count must be positive")


@dataclass(frozen=True)
class TrainStepResult:
    """Outcome of a single optimizer step.

    ``metrics`` holds this step's values.  ``running_metrics``, when the
    backend provides it, holds the decayed running averages that upstream
    KataGo prints; both are optional for simple backends.
    """

    metrics: Mapping[str, float]
    sample_count: int
    learning_rate: float = 0.0
    running_metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass
class TrainingReport:
    """Summary returned by :meth:`TrainingLoop.run`."""

    epochs_completed: int = 0
    steps: int = 0
    samples: int = 0
    elapsed_seconds: float = 0.0
    last_train_metrics: Mapping[str, float] = field(default_factory=dict)
    last_validation_metrics: Mapping[str, float] = field(default_factory=dict)
