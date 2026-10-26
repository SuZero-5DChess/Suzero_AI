"""Configuration and size schedule for the optional dynamic replay window."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_real(value: object, name: str) -> None:
    try:
        finite = isinstance(value, Real) and math.isfinite(value)
    except OverflowError:
        finite = False
    if isinstance(value, bool) or not finite:
        raise ValueError(f"{name} must be finite")


@dataclass
class WindowConfig:
    """Opt-in growth, storage and sampling settings for replay history."""

    enabled: bool = False
    expand_per_row: float = 0.4
    taper_exponent: float = 0.65
    taper_scale: int | None = None
    max_samples: int = 500_000
    keep_samples: int | None = None
    sampling: str = "shuffle"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("window enabled must be a bool")
        _finite_real(self.expand_per_row, "expand_per_row")
        if self.expand_per_row < 0:
            raise ValueError("expand_per_row must be non-negative")
        _finite_real(self.taper_exponent, "taper_exponent")
        if not 0 < self.taper_exponent <= 1:
            raise ValueError("taper_exponent must be in (0, 1]")
        if self.taper_scale is not None:
            _positive_integer(self.taper_scale, "taper_scale")
        _positive_integer(self.max_samples, "max_samples")
        if self.keep_samples is not None:
            _positive_integer(self.keep_samples, "keep_samples")
        if self.sampling not in ("shuffle", "uniform"):
            raise ValueError("sampling must be shuffle or uniform")


def compute_window_size(
    total_generated: int, min_rows: int, window: WindowConfig
) -> int:
    """Return the floored, clamped tapered window size at generated row ``N``.

    ``min_rows`` is the configured replay-buffer size, not the training gate.
    The caller decides whether window mode is enabled. With ``scale`` defaulting
    to ``min_rows``, the schedule is exactly::

        min_rows + expand * (N**e - scale**e) / (e * scale**(e - 1))
    """

    if (
        isinstance(total_generated, bool)
        or not isinstance(total_generated, Integral)
        or total_generated < 0
    ):
        raise ValueError("total_generated must be a non-negative integer")
    _positive_integer(min_rows, "min_rows")
    window.__post_init__()
    if window.max_samples < min_rows:
        raise ValueError("max_samples must be >= min_rows")

    scale = window.taper_scale if window.taper_scale is not None else min_rows
    # Below the taper scale the growth term is non-positive. Handle the zero
    # rate explicitly so an otherwise unused power cannot overflow.
    if total_generated <= scale or window.expand_per_row == 0:
        return int(min_rows)
    exponent = window.taper_exponent
    growth = (total_generated**exponent - scale**exponent) / (
        exponent * scale ** (exponent - 1)
    )
    size = min_rows + window.expand_per_row * growth
    return math.floor(max(min_rows, min(window.max_samples, size)))


__all__ = ["WindowConfig", "compute_window_size"]
