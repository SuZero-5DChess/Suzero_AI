"""Metrics sinks: the loop's only output boundary."""

from __future__ import annotations

import logging
from abc import ABC
from typing import Iterable, Mapping

from .types import TrainStepResult

logger = logging.getLogger(__name__)


class MetricsSink(ABC):
    """Observer for training events.

    All hooks default to no-ops so a sink only overrides what it needs.
    """

    def on_train_step(self, step: int, result: TrainStepResult) -> None:
        pass

    def on_epoch_end(self, epoch: int, samples_seen: int) -> None:
        pass

    def on_validation(self, epoch: int, metrics: Mapping[str, float]) -> None:
        pass

    def on_checkpoint(self, step: int, path: str) -> None:
        pass


class NullMetricsSink(MetricsSink):
    """Discards every event."""


class LoggingMetricsSink(MetricsSink):
    """Writes a compact, human-readable training trace to ``logging``."""

    def __init__(self, logger_name: str = "suzero_train") -> None:
        self._logger = logging.getLogger(logger_name)

    def on_train_step(self, step: int, result: TrainStepResult) -> None:
        metrics = result.running_metrics or result.metrics
        loss = metrics.get("loss", metrics.get("loss_sum"))
        if loss is None:
            loss = result.metrics.get("loss", result.metrics.get("loss_sum"))
        if loss is None:
            return
        self._logger.info(
            "step %d | samples %d | loss %.6f | lr %.3g",
            step,
            result.sample_count,
            loss,
            result.learning_rate,
        )

    def on_epoch_end(self, epoch: int, samples_seen: int) -> None:
        self._logger.info("epoch %d finished | samples seen %d", epoch, samples_seen)

    def on_validation(self, epoch: int, metrics: Mapping[str, float]) -> None:
        loss = metrics.get("loss", metrics.get("loss_sum"))
        if loss is not None:
            self._logger.info("epoch %d validation | loss %.6f", epoch, loss)

    def on_checkpoint(self, step: int, path: str) -> None:
        self._logger.info("checkpoint at step %d -> %s", step, path)


class CompositeMetricsSink(MetricsSink):
    """Fan out every event to several sinks in order."""

    def __init__(self, sinks: Iterable[MetricsSink]) -> None:
        self._sinks = list(sinks)

    def on_train_step(self, step: int, result: TrainStepResult) -> None:
        for sink in self._sinks:
            sink.on_train_step(step, result)

    def on_epoch_end(self, epoch: int, samples_seen: int) -> None:
        for sink in self._sinks:
            sink.on_epoch_end(epoch, samples_seen)

    def on_validation(self, epoch: int, metrics: Mapping[str, float]) -> None:
        for sink in self._sinks:
            sink.on_validation(epoch, metrics)

    def on_checkpoint(self, step: int, path: str) -> None:
        for sink in self._sinks:
            sink.on_checkpoint(step, path)
