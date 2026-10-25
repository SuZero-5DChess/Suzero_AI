"""Device selection shared by the PyTorch trainer and data sources."""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve ``"auto"`` to CUDA, then MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
