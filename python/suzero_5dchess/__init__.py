"""5D chess AlphaZero training built on the suzero_train core."""

# The compiled extension depends on onnxruntime.dll. Register its directory
# before any submodule imports the extension.
from ._ort_runtime import configure_onnxruntime_dlls

configure_onnxruntime_dlls()

from .config import MctsConfig, NetworkConfig, SelfPlayConfig, TrainConfig
from .env import GameEnv
from .network import SuzeroNetwork, build_network
from .protocol import (
    ACTION_ENCODING,
    ACTION_INPUT_ENCODING,
    IO_PROTOCOL,
    VERY_SMALL,
    VariantSpec,
)

__all__ = [
    "ACTION_ENCODING",
    "ACTION_INPUT_ENCODING",
    "GameEnv",
    "IO_PROTOCOL",
    "MctsConfig",
    "NetworkConfig",
    "SelfPlayConfig",
    "TrainConfig",
    "VariantSpec",
    "VERY_SMALL",
    "SuzeroNetwork",
    "build_network",
]
