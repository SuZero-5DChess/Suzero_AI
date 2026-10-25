"""KataGo's sample-based schedule pieces, independent of any model."""

from __future__ import annotations

_WARMUP_CURVE: tuple[tuple[float, float], ...] = (
    (0.125, 1.0 / 20.0),
    (0.250, 1.0 / 14.0),
    (0.375, 1.0 / 10.0),
    (0.500, 1.0 / 7.0),
    (0.625, 1.0 / 5.0),
    (0.750, 1.0 / 3.0),
    (0.875, 1.0 / 2.0),
    (1.000, 1.0 / 1.4),
)


def warmup_scale(samples_seen: int, warmup_samples: int) -> float:
    """Piecewise warmup from KataGo, scaled to an arbitrary sample budget.

    KataGo ramps over its first two million samples in eight equal segments
    (1/20, 1/14, ..., 1/1.4).  Dividing by ``warmup_samples`` reproduces the
    same curve for any budget; ``warmup_samples <= 0`` disables it.
    """
    if warmup_samples <= 0:
        return 1.0
    progress = samples_seen / warmup_samples
    for boundary, scale in _WARMUP_CURVE:
        if progress < boundary:
            return scale
    return 1.0


def batch_scale(
    batch_size: int,
    reference: int,
    exponent: float,
    enabled: bool = True,
) -> float:
    """KataGo's batch-size scaling for the learning rate and weight decay."""
    if not enabled:
        return 1.0
    if reference <= 0:
        raise ValueError("reference batch size must be positive")
    return (batch_size / reference) ** exponent
