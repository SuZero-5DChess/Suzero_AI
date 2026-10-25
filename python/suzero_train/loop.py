"""Game-agnostic training loop.

The loop owns sequencing only: epochs, step/sample counters, logging cadence,
validation cadence and checkpoint cadence.  Model math lives in a
:class:`~suzero_train.trainer.Trainer`.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Mapping

from .data_source import TrainingDataSource
from .metrics import MetricsSink, NullMetricsSink
from .trainer import Trainer
from .types import TrainingReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopConfig:
    """Scheduling knobs for :class:`TrainingLoop`.

    ``epochs <= 0`` means "run until ``max_steps`` is reached"; in that case
    ``max_steps`` must be set, otherwise :meth:`TrainingLoop.run` raises.

    ``max_steps`` counts optimizer steps performed by this ``run()`` call, not
    steps recorded in a resumed checkpoint.
    """

    epochs: int = 1
    max_steps: int | None = None
    log_every_steps: int = 0
    checkpoint_every_steps: int = 0
    checkpoint_dir: str | None = None
    checkpoint_prefix: str = "checkpoint"
    validate_every_epochs: int = 1
    save_final_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        if self.checkpoint_every_steps < 0:
            raise ValueError("checkpoint_every_steps must be >= 0")
        if self.validate_every_epochs < 0:
            raise ValueError("validate_every_epochs must be >= 0")
        needs_checkpoint_dir = (
            self.checkpoint_every_steps > 0 or self.save_final_checkpoint
        )
        if needs_checkpoint_dir and self.checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir is required when checkpointing is enabled"
            )


class TrainingLoop:
    """Drive a :class:`Trainer` from a :class:`TrainingDataSource`."""

    def __init__(
        self,
        trainer: Trainer,
        data_source: TrainingDataSource,
        config: LoopConfig | None = None,
        metrics_sink: MetricsSink | None = None,
    ) -> None:
        self._trainer = trainer
        self._data_source = data_source
        self._config = config or LoopConfig()
        self._sink = metrics_sink or NullMetricsSink()

    def run(self) -> TrainingReport:
        cfg = self._config
        report = TrainingReport()
        started = time.monotonic()
        steps_this_run = 0
        epoch_index = 0
        stop = False

        while not stop:
            if cfg.epochs > 0 and epoch_index >= cfg.epochs:
                break
            if cfg.epochs <= 0 and cfg.max_steps is None:
                raise ValueError(
                    "epochs <= 0 requires max_steps so the loop can terminate"
                )
            if cfg.max_steps is not None and steps_this_run >= cfg.max_steps:
                break

            self._data_source.begin_epoch()
            self._trainer.on_epoch_start()
            batches_this_epoch = 0
            stopped_at_step_limit = False

            while True:
                if cfg.max_steps is not None and steps_this_run >= cfg.max_steps:
                    # A limit that lands exactly on the end of an epoch should
                    # still count and validate that epoch.  is_exhausted() is
                    # cheap for in-memory sources; other sources may not know,
                    # so probe once.  The probe can consume one batch, but the
                    # run is stopping, and the next run starts a fresh epoch.
                    if batches_this_epoch > 0 and (
                        self._data_source.is_exhausted()
                        or self._data_source.next_batch() is None
                    ):
                        break
                    stopped_at_step_limit = True
                    break

                batch = self._data_source.next_batch()
                if batch is None:
                    break

                result = self._trainer.train_step(batch)
                steps_this_run += 1
                batches_this_epoch += 1
                report.steps = steps_this_run
                report.samples += result.sample_count
                report.last_train_metrics = dict(
                    result.running_metrics or result.metrics
                )

                if (
                    cfg.log_every_steps > 0
                    and steps_this_run % cfg.log_every_steps == 0
                ):
                    self._sink.on_train_step(self._trainer.global_step, result)
                    self._trainer.on_log()

                if (
                    cfg.checkpoint_every_steps > 0
                    and steps_this_run % cfg.checkpoint_every_steps == 0
                ):
                    self._save_checkpoint(
                        self._step_checkpoint_path(self._trainer.global_step)
                    )

            if stopped_at_step_limit:
                break

            if batches_this_epoch == 0:
                logger.warning(
                    "Training data source produced no batches in epoch %d; stopping",
                    epoch_index + 1,
                )
                break

            epoch_index += 1
            report.epochs_completed = epoch_index
            self._sink.on_epoch_end(epoch_index, self._trainer.samples_seen)

            if cfg.validate_every_epochs > 0 and self._data_source.has_validation():
                if epoch_index % cfg.validate_every_epochs == 0:
                    metrics = dict(
                        self._trainer.validate(self._data_source.validation_batches())
                    )
                    report.last_validation_metrics = metrics
                    self._sink.on_validation(epoch_index, metrics)

        if (
            cfg.save_final_checkpoint
            and cfg.checkpoint_dir is not None
            and steps_this_run > 0
        ):
            self._save_checkpoint(self._final_checkpoint_path())

        report.elapsed_seconds = time.monotonic() - started
        return report

    def _save_checkpoint(self, path: str) -> None:
        self._trainer.save_checkpoint(path)
        self._sink.on_checkpoint(self._trainer.global_step, path)

    def _step_checkpoint_path(self, step: int) -> str:
        cfg = self._config
        assert cfg.checkpoint_dir is not None
        name = f"{cfg.checkpoint_prefix}_{step:09d}.ckpt"
        return os.path.join(cfg.checkpoint_dir, name)

    def _final_checkpoint_path(self) -> str:
        cfg = self._config
        assert cfg.checkpoint_dir is not None
        return os.path.join(cfg.checkpoint_dir, f"{cfg.checkpoint_prefix}.ckpt")


def summarize(metrics: Mapping[str, float]) -> str:
    """Format a metrics mapping deterministically, useful for logs and tests."""
    return ", ".join(f"{key}={metrics[key]:.6g}" for key in sorted(metrics))
