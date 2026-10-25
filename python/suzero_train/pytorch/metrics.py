"""Metric accumulation and averaging, following KataGo's conventions.

A metric name ending in ``_sum`` is a batch sum and is reported per sample; a
name ending in ``_batch`` is already batch-normalized and is reported as a
batch average.  Any other name is accumulated as a running sum, matching
KataGo's ``accumulate_metrics``/``log_metrics`` pair.

The convention lets an objective report e.g. ``{"loss_sum": loss * n,
"gnorm_batch": gnorm}`` and have the trainer report ``loss`` per sample and
``gnorm_batch`` per batch.
"""

from __future__ import annotations

import logging
import math
from typing import Mapping

import torch

logger = logging.getLogger(__name__)


def detensorify_metrics(
    metrics: Mapping[str, float | torch.Tensor],
) -> dict[str, float]:
    """Convert tensor metrics to Python floats with a single device sync."""
    result: dict[str, float] = {}
    tensor_items: list[tuple[str, torch.Tensor]] = []
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            tensor_items.append((key, value))
        else:
            result[key] = float(value)
    if tensor_items:
        stacked = torch.stack(
            [value.detach().reshape([]).float() for _, value in tensor_items]
        ).cpu().tolist()
        for (key, _), value in zip(tensor_items, stacked):
            result[key] = value
    # Non-finite gradient norms come from skipped FP16 steps; drop them as
    # upstream does rather than poisoning the running average.  ``exgnorm_sum``
    # is derived from the norm, so it must disappear with it; otherwise a
    # single inf would decay forever and keep reporting inf.
    if "gnorm_batch" in result and not (
        math.isfinite(result["gnorm_batch"]) and result["gnorm_batch"] < 1e30
    ):
        del result["gnorm_batch"]
        result.pop("exgnorm_sum", None)
    elif "exgnorm_sum" in result and not (
        math.isfinite(result["exgnorm_sum"]) and result["exgnorm_sum"] < 1e30
    ):
        del result["exgnorm_sum"]
    return result


class MetricsAccumulator:
    """Running sums/weights with optional exponential decay on ``_sum`` keys."""

    def __init__(self, decay: float = 1.0) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("metrics decay must be in (0, 1]")
        self.decay = decay
        self.sums: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def update(
        self,
        metrics: Mapping[str, float],
        batch_size: int,
        new_weight: float = 1.0,
    ) -> None:
        if self.decay != 1.0:
            for key in self.sums:
                if key.endswith("_sum"):
                    self.sums[key] *= self.decay
                    self.weights[key] *= self.decay
        for key, value in metrics.items():
            if key.endswith("_sum"):
                self.sums[key] = self.sums.get(key, 0.0) + value * new_weight
                self.weights[key] = (
                    self.weights.get(key, 0.0) + batch_size * new_weight
                )
            elif key.endswith("_batch"):
                self.sums[key] = self.sums.get(key, 0.0) + value * new_weight
                self.weights[key] = self.weights.get(key, 0.0) + new_weight
            else:
                self.sums[key] = self.sums.get(key, 0.0) + value
                self.weights[key] = self.weights.get(key, 0.0) + batch_size

    def averaged(self) -> dict[str, float]:
        result: dict[str, float] = {}
        # Normalized keys win over plain keys that collapse to the same name
        # (``loss_sum`` -> ``loss``).  A plain ``loss`` metric is a running sum
        # and must not overwrite the per-sample average.
        for key, value in self.sums.items():
            if key.endswith("_sum") or key.endswith("_batch"):
                weight = self.weights.get(key, 0.0)
                result[key[:-4] if key.endswith("_sum") else key] = (
                    value / weight if weight != 0.0 else math.nan
                )
        for key, value in self.sums.items():
            if not (key.endswith("_sum") or key.endswith("_batch")):
                result.setdefault(key, value)
        return result

    def shrink_batch_window(self, factor: float = 0.001) -> None:
        """Shrink ``_batch`` sums/weights after a log, as upstream does.

        ``_batch`` metrics are averages over the window since the last log.
        Calling this after logging makes the next window start small without
        discarding the just-reported value.
        """
        if not 0.0 < factor <= 1.0:
            raise ValueError("batch window factor must be in (0, 1]")
        for key in self.sums:
            if key.endswith("_batch"):
                self.sums[key] *= factor
                self.weights[key] *= factor

    def clear_nonfinite(self) -> None:
        """Reset keys whose running sum became non-finite (upstream behaviour)."""
        for key in list(self.sums):
            if not math.isfinite(self.sums[key]):
                logger.warning(
                    "non-finite value of metric %s; clearing it back to empty", key
                )
                self.sums[key] = 0.0
                self.weights[key] = 0.0

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"sums": dict(self.sums), "weights": dict(self.weights)}

    def load_state_dict(self, state: Mapping[str, Mapping[str, float]]) -> None:
        self.sums = dict(state.get("sums", {}))
        self.weights = dict(state.get("weights", {}))
