"""Tests for the game-agnostic abstract layer.

These use fakes only: no PyTorch code is imported.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping

import pytest

from suzero_train import (
    LoopConfig,
    TrainStepResult,
    Trainer,
    TrainingBatch,
    TrainingDataSource,
    TrainingLoop,
)


class FakeSource(TrainingDataSource):
    def __init__(self, batches_per_epoch: int = 2, validation: bool = True) -> None:
        self.batches_per_epoch = batches_per_epoch
        self.validation = validation
        self.epochs_started = 0
        self._served = 0

    def begin_epoch(self) -> None:
        self.epochs_started += 1
        self._served = 0

    def next_batch(self) -> TrainingBatch | None:
        if self._served >= self.batches_per_epoch:
            return None
        self._served += 1
        return TrainingBatch(data={"value": self._served}, sample_count=4)

    def is_exhausted(self) -> bool:
        return self._served >= self.batches_per_epoch

    def validation_batches(self) -> Iterable[TrainingBatch]:
        if self.validation:
            yield TrainingBatch(data={"value": 0}, sample_count=4)

    def has_validation(self) -> bool:
        return self.validation


class FakeTrainer(Trainer):
    def __init__(self) -> None:
        self._steps = 0
        self._samples = 0
        self.validation_calls = 0
        self.epoch_starts = 0
        self.log_calls = 0
        self.saved: list[str] = []

    def train_step(self, batch: TrainingBatch) -> TrainStepResult:
        self._steps += 1
        self._samples += batch.sample_count
        return TrainStepResult(
            metrics={"loss_sum": 1.0 / self._steps},
            sample_count=batch.sample_count,
            learning_rate=0.01,
        )

    def validate(self, batches: Iterable[TrainingBatch]) -> Mapping[str, float]:
        self.validation_calls += 1
        assert list(batches)
        return {"loss": 0.5}

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        text = os.fspath(path)
        with open(text, "w", encoding="utf-8") as handle:
            handle.write(str(self._steps))
        self.saved.append(text)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        self._steps = int(open(path, encoding="utf-8").read().strip())

    def on_epoch_start(self) -> None:
        self.epoch_starts += 1

    def on_log(self) -> None:
        self.log_calls += 1

    @property
    def samples_seen(self) -> int:
        return self._samples

    @property
    def global_step(self) -> int:
        return self._steps


def test_two_epochs_validate_once(tmp_path):
    source = FakeSource(batches_per_epoch=2)
    trainer = FakeTrainer()
    loop = TrainingLoop(
        trainer,
        source,
        LoopConfig(
            epochs=2,
            validate_every_epochs=2,
            checkpoint_dir=str(tmp_path),
            save_final_checkpoint=True,
        ),
    )

    report = loop.run()

    assert report.epochs_completed == 2
    assert report.steps == 4
    assert report.samples == 16
    assert report.last_train_metrics["loss_sum"] == pytest.approx(0.25)
    assert trainer.validation_calls == 1
    assert source.epochs_started == 2
    assert os.path.exists(tmp_path / "checkpoint.ckpt")


def test_max_steps_stops_mid_epoch(tmp_path):
    loop = TrainingLoop(
        FakeTrainer(),
        FakeSource(batches_per_epoch=2),
        LoopConfig(
            epochs=10,
            max_steps=3,
            validate_every_epochs=1,
            checkpoint_every_steps=2,
            checkpoint_dir=str(tmp_path),
            save_final_checkpoint=True,
        ),
    )

    report = loop.run()

    assert report.steps == 3
    assert report.samples == 12
    # The first epoch completes (2 batches), the second stops after one step.
    assert report.epochs_completed == 1
    assert os.path.exists(tmp_path / "checkpoint_000000002.ckpt")
    assert os.path.exists(tmp_path / "checkpoint.ckpt")


def test_max_steps_at_epoch_boundary_still_validates(tmp_path):
    trainer = FakeTrainer()
    loop = TrainingLoop(
        trainer,
        FakeSource(batches_per_epoch=2),
        LoopConfig(
            epochs=10,
            max_steps=2,
            validate_every_epochs=1,
            checkpoint_dir=str(tmp_path),
        ),
    )

    report = loop.run()

    assert report.steps == 2
    assert report.epochs_completed == 1
    assert trainer.validation_calls == 1


def test_no_final_checkpoint_when_source_is_empty(tmp_path):
    loop = TrainingLoop(
        FakeTrainer(),
        FakeSource(batches_per_epoch=0),
        LoopConfig(
            epochs=1,
            checkpoint_dir=str(tmp_path),
            save_final_checkpoint=True,
        ),
    )

    report = loop.run()

    assert report.steps == 0
    assert not os.path.exists(tmp_path / "checkpoint.ckpt")


def test_epoch_start_and_log_hooks():
    trainer = FakeTrainer()
    loop = TrainingLoop(
        trainer,
        FakeSource(batches_per_epoch=2),
        LoopConfig(epochs=2, log_every_steps=1, validate_every_epochs=0),
    )

    report = loop.run()

    assert report.steps == 4
    assert trainer.epoch_starts == 2
    assert trainer.log_calls == report.steps


def test_validation_can_be_disabled():
    source = FakeSource(validation=False)
    trainer = FakeTrainer()
    loop = TrainingLoop(trainer, source, LoopConfig(epochs=1, validate_every_epochs=0))

    loop.run()

    assert trainer.validation_calls == 0


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        LoopConfig(checkpoint_every_steps=5, checkpoint_dir=None)
    with pytest.raises(ValueError):
        LoopConfig(max_steps=-1)


def test_unbounded_epochs_requires_max_steps():
    loop = TrainingLoop(FakeTrainer(), FakeSource(), LoopConfig(epochs=0))
    with pytest.raises(ValueError):
        loop.run()
