"""Oversample positions from the late part of each self-play game.

Every sample records the game's ``move_count`` at capture time.  Within one
game the largest stored ``move_count`` is the end of that game, so a row's
progress is ``move_count / game_end``.  Its sampling weight is::

    1 + strength * progress ** exponent

Rounds still contain every eligible row once; rows with a weight above one get
extra copies (Efraimidis style: the whole part always, the fraction with that
probability).  The training step budget and row coverage stay unchanged, while
positions nearer the end of their game are trained on more often.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # numpy is imported lazily to keep config stdlib-only
    import numpy as np


def _np():
    """Import numpy on first use; ``config`` must not depend on it."""

    import numpy

    return numpy


@dataclass(frozen=True)
class EndgameSamplingConfig:
    """Weighting for rows that are close to the end of their game.

    ``strength`` is the number of extra copies the final position of a game
    expects; ``exponent`` shapes the ramp (2 concentrates the boost in the
    last part of the game, 1 is linear).  Disabled by default so existing
    checkpoints keep their replay distribution.
    """

    enabled: bool = False
    strength: float = 1.0
    exponent: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("endgame sampling enabled must be a bool")
        for name, value in (("strength", self.strength), ("exponent", self.exponent)):
            try:
                finite = isinstance(value, Real) and math.isfinite(value)
            except OverflowError:
                finite = False
            if isinstance(value, bool) or not finite:
                raise ValueError(f"endgame sampling {name} must be finite")
        if self.strength < 0:
            raise ValueError("endgame sampling strength must be non-negative")
        if self.exponent <= 0:
            raise ValueError("endgame sampling exponent must be positive")

    def identity(self) -> dict[str, float] | None:
        """Stable sidecar/generation description, or ``None`` when disabled."""

        if not self.enabled:
            return None
        return {"strength": float(self.strength), "exponent": float(self.exponent)}


def game_progress(move_counts: np.ndarray) -> np.ndarray:
    """Map one game's ``move_count`` values to ``[0, 1]`` progress.

    The maximum is that game's end.  A game whose rows all share ``move_count``
    zero (a synthetic or legacy row set) has no measurable progress and maps to
    zero, so it is never boosted.
    """

    np = _np()
    values = np.asarray(move_counts, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("move counts must be a flat array")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("move counts must be finite and non-negative")
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    end = float(values.max())
    if end <= 0.0:
        return np.zeros(values.size, dtype=np.float64)
    return np.clip(values / end, 0.0, 1.0)


def sampling_weights(
    progress: np.ndarray, config: EndgameSamplingConfig
) -> np.ndarray:
    """Turn per-row progress into sampling weights (one when disabled)."""

    np = _np()
    values = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    if not config.enabled or values.size == 0:
        return np.ones(values.shape, dtype=np.float64)
    return 1.0 + config.strength * np.power(values, config.exponent)


def endgame_weights(
    move_counts: np.ndarray, config: EndgameSamplingConfig
) -> np.ndarray:
    """Convenience: progress and weight for one game's rows."""

    return sampling_weights(game_progress(move_counts), config)


def extra_copies(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stochastic extra copies; the expected count is ``weight - 1``."""

    np = _np()
    extra = np.maximum(np.asarray(weights, dtype=np.float64) - 1.0, 0.0)
    whole = np.floor(extra).astype(np.int64)
    fraction = extra - whole
    return whole + (rng.random(extra.size) < fraction).astype(np.int64)


def repeat_by_weight(
    rows: np.ndarray, weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Repeat each row once per drawn copy, preserving the input order."""

    np = _np()
    rows = np.asarray(rows, dtype=np.int64)
    if rows.shape != np.asarray(weights).shape:
        raise ValueError("rows and weights must have the same shape")
    if rows.size == 0:
        return rows.copy()
    return np.repeat(rows, 1 + extra_copies(weights, rng))


def weighted_without_replacement(
    rows: np.ndarray,
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``count`` distinct rows with probability proportional to weight.

    Uses the Gumbel top-k trick: adding Gumbel noise to ``log(weight)`` and
    keeping the largest keys is an exact weighted sample without replacement.
    """

    np = _np()
    rows = np.asarray(rows, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if rows.ndim != 1 or weights.ndim != 1 or rows.shape != weights.shape:
        raise ValueError("rows and weights must be flat and equally sized")
    if count <= 0 or count > rows.size:
        raise ValueError("count must be in [1, len(rows)]")
    if (weights <= 0).any() or not np.isfinite(weights).all():
        raise ValueError("weights must be finite and positive")
    keys = np.log(weights) + rng.gumbel(size=rows.size)
    chosen = np.argpartition(keys, rows.size - count)[rows.size - count :]
    return rows[chosen]


__all__ = [
    "EndgameSamplingConfig",
    "endgame_weights",
    "extra_copies",
    "game_progress",
    "repeat_by_weight",
    "sampling_weights",
    "weighted_without_replacement",
]
