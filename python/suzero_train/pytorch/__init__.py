"""PyTorch trainer and data sources, independent of any specific game.

The abstract contracts live in :mod:`suzero_train`; this subpackage imports
PyTorch and implements them::

    from suzero_train.pytorch import (
        InMemoryDataSource,
        TorchTrainer,
        TorchTrainerConfig,
    )

Importing :mod:`suzero_train` alone does not import PyTorch.
"""

from .config import ParamGroupSpec, TorchTrainerConfig
from .data import (
    InMemoryDataSource,
    IterableDataSource,
    default_collate,
)
from .device import resolve_device
from .trainer import TorchTrainer

__all__ = [
    "InMemoryDataSource",
    "IterableDataSource",
    "ParamGroupSpec",
    "TorchTrainer",
    "TorchTrainerConfig",
    "default_collate",
    "resolve_device",
]
