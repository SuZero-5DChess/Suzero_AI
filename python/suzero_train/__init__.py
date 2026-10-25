"""Game-agnostic training core extracted from KataGo's training recipe.

The abstract layer (this package's top level) has no PyTorch or game
dependency.  :mod:`suzero_train.pytorch` supplies a concrete PyTorch trainer
and data sources; importing this package alone does not import PyTorch.
"""

from .data_source import TrainingDataSource
from .loop import LoopConfig, TrainingLoop
from .metrics import (
    CompositeMetricsSink,
    LoggingMetricsSink,
    MetricsSink,
    NullMetricsSink,
)
from .trainer import Trainer
from .types import TrainStepResult, TrainingBatch, TrainingReport

__all__ = [
    "CompositeMetricsSink",
    "LoggingMetricsSink",
    "LoopConfig",
    "MetricsSink",
    "NullMetricsSink",
    "Trainer",
    "TrainingBatch",
    "TrainingDataSource",
    "TrainingLoop",
    "TrainingReport",
    "TrainStepResult",
]
