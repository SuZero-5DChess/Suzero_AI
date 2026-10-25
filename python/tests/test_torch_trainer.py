"""End-to-end tests for the game-agnostic PyTorch trainer.

A tiny regression model stands in for a game network: the trainer never sees
anything game-specific.
"""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F

from suzero_train import LoopConfig, TrainingBatch, TrainingLoop
from suzero_train.pytorch import (
    InMemoryDataSource,
    IterableDataSource,
    ParamGroupSpec,
    TorchTrainer,
    TorchTrainerConfig,
)
from suzero_train.pytorch.metrics import MetricsAccumulator, detensorify_metrics
from suzero_train.pytorch.trainer import _grad_norm


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(2, 4)
        self.head = torch.nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.hidden(inputs)))


def objective(model, data):
    predictions = model(data["inputs"]).squeeze(-1)
    targets = data["targets"].squeeze(-1)
    loss = F.mse_loss(predictions, targets)
    count = predictions.shape[0]
    return loss, {
        "loss_sum": loss * count,
        "mae_sum": (predictions - targets).abs().sum(),
    }


def collate(samples):
    return {
        "inputs": torch.stack([sample[0] for sample in samples]),
        "targets": torch.stack([sample[1] for sample in samples]),
    }


def make_samples(count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(7)
    return [
        (
            torch.randn(2, generator=generator),
            torch.randn(1, generator=generator),
        )
        for _ in range(count)
    ]


def make_source() -> InMemoryDataSource:
    return InMemoryDataSource(
        make_samples(8),
        batch_size=4,
        collate_fn=collate,
        validation_samples=make_samples(3),
        validation_batch_size=2,
        seed=1,
    )


def next_training_batch():
    source = make_source()
    source.begin_epoch()
    batch = source.next_batch()
    assert batch is not None
    return batch


def make_config(**overrides) -> TorchTrainerConfig:
    values = {
        "device": "cpu",
        "optimizer": "adamw",
        "base_learning_rate": 1e-2,
        "batch_lr_scaling": False,
        # LR and WD scaling are decoupled; tests that want batch-scaled weight
        # decay opt in explicitly.
        "weight_decay_batch_scaling": False,
        "warmup_samples": 0,
    }
    values.update(overrides)
    return TorchTrainerConfig(**values)


class CountingTrainer(TorchTrainer):
    """Counts schedule refreshes so the update cadence can be asserted."""

    def __init__(self, *args, **kwargs) -> None:
        self.schedule_updates = 0
        super().__init__(*args, **kwargs)

    def _update_lr_and_wd(self, batch_size: int):
        self.schedule_updates += 1
        return super()._update_lr_and_wd(batch_size)


def test_loop_trains_validates_and_checkpoints(tmp_path):
    trainer = TorchTrainer(TinyModel(), objective, config=make_config())
    loop = TrainingLoop(
        trainer,
        make_source(),
        LoopConfig(
            epochs=2,
            validate_every_epochs=1,
            checkpoint_dir=str(tmp_path),
            save_final_checkpoint=True,
        ),
    )

    report = loop.run()

    assert report.epochs_completed == 2
    assert report.steps == 4
    assert report.samples == 16
    assert report.last_train_metrics["loss"] >= 0.0
    assert "loss" in report.last_validation_metrics
    assert trainer.global_step == 4
    assert trainer.samples_seen == 16
    assert os.path.exists(tmp_path / "checkpoint.ckpt")

    restored = TorchTrainer(TinyModel(), objective, config=make_config())
    restored.load_checkpoint(tmp_path / "checkpoint.ckpt")
    assert restored.global_step == 4
    assert restored.samples_seen == 16
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])


def test_learning_rate_matches_katago_recipe():
    trainer = TorchTrainer(
        TinyModel(),
        objective,
        config=make_config(
            base_learning_rate=3e-5,
            batch_lr_scaling=True,
            warmup_samples=0,
        ),
    )
    batch = next_training_batch()

    result = trainer.train_step(batch)

    expected = 3e-5 * 1.33 * (4 / 256) ** 0.5
    assert result.learning_rate == pytest.approx(expected)


def test_loss_reduction_sum_reports_batch_sum_per_sample():
    def sum_objective(model, data):
        predictions = model(data["inputs"]).squeeze(-1)
        targets = data["targets"].squeeze(-1)
        return F.mse_loss(predictions, targets, reduction="sum"), {}

    trainer = TorchTrainer(
        TinyModel(), sum_objective, config=make_config(loss_reduction="sum")
    )
    batch = next_training_batch()
    with torch.no_grad():
        expected_total = float(
            F.mse_loss(
                trainer.model(batch.data["inputs"]).squeeze(-1),
                batch.data["targets"].squeeze(-1),
                reduction="sum",
            )
        )

    result = trainer.train_step(batch)

    # ``loss_sum`` was reconstructed as the batch total, so the reported loss
    # is the total divided by the sample count, not the total itself.
    assert result.metrics["loss"] == pytest.approx(
        expected_total / batch.sample_count, rel=1e-5
    )
    assert result.running_metrics["loss"] == pytest.approx(
        expected_total / batch.sample_count, rel=1e-5
    )


def test_loss_reduction_mean_converts_to_per_sample_sum():
    def mean_objective(model, data):
        predictions = model(data["inputs"]).squeeze(-1)
        targets = data["targets"].squeeze(-1)
        return F.mse_loss(predictions, targets), {}

    trainer = TorchTrainer(
        TinyModel(), mean_objective, config=make_config(loss_reduction="mean")
    )
    batch = next_training_batch()

    result = trainer.train_step(batch)

    # A batch mean times the sample count is the reconstructed sum, so the
    # reported per-sample loss equals the original mean.
    assert result.metrics["loss"] == pytest.approx(
        trainer.running_metrics.sums["loss_sum"] / batch.sample_count, rel=1e-6
    )
    assert result.metrics["loss"] == pytest.approx(
        trainer.running_metrics.averaged()["loss"], rel=1e-6
    )


def test_split_forward_runs_before_objective():
    calls = {"forward": 0, "objective": 0}

    def forward(model, data):
        calls["forward"] += 1
        return model(data["inputs"]).squeeze(-1)

    def split_objective(model, data, outputs):
        calls["objective"] += 1
        targets = data["targets"].squeeze(-1)
        loss = F.mse_loss(outputs, targets, reduction="sum")
        return loss, {"loss_sum": loss}

    trainer = TorchTrainer(
        TinyModel(), split_objective, forward=forward, config=make_config()
    )

    result = trainer.train_step(next_training_batch())

    assert calls == {"forward": 1, "objective": 1}
    assert result.metrics["loss"] >= 0.0


def test_schedule_batch_size_fixes_batch_scaling():
    # The actual batch has 4 samples, but the schedule is told upstream's
    # configured batch size; the LR factor must use 256, not 4.
    trainer = TorchTrainer(
        TinyModel(),
        objective,
        config=make_config(
            base_learning_rate=3e-5,
            batch_lr_scaling=True,
            schedule_batch_size=256,
        ),
    )

    result = trainer.train_step(next_training_batch())

    assert result.learning_rate == pytest.approx(3e-5 * 1.33)


def test_unclipped_gradient_norm_does_not_scale_infinite_gradients():
    parameter = torch.nn.Parameter(torch.zeros(1))
    parameter.grad = torch.tensor([float("inf")])

    norm = _grad_norm([parameter])

    assert torch.isinf(norm)
    # A max_norm=inf clip would compute inf/inf and destroy the gradient.
    assert torch.isinf(parameter.grad).all()


def test_warmup_scales_learning_rate():
    trainer = TorchTrainer(
        TinyModel(),
        objective,
        config=make_config(
            base_learning_rate=1e-3,
            warmup_samples=1_000_000,
        ),
    )
    batch = next_training_batch()

    result = trainer.train_step(batch)

    # At zero samples seen, KataGo's curve starts at 1/20.
    assert result.learning_rate == pytest.approx(1e-3 * 1.33 / 20.0)


def test_param_groups_control_lr_and_weight_decay():
    model = TinyModel()
    groups = [
        ParamGroupSpec("hidden", tuple(model.hidden.parameters())),
        ParamGroupSpec(
            "head",
            tuple(model.head.parameters()),
            lr_scale=0.5,
            weight_decay_scale=0.0,
        ),
    ]
    trainer = TorchTrainer(
        model,
        objective,
        config=make_config(weight_decay=0.1),
        param_groups=groups,
    )
    batch = next_training_batch()

    trainer.train_step(batch)

    hidden, head = trainer.optimizer.param_groups
    assert hidden["weight_decay"] == pytest.approx(0.1)
    assert head["weight_decay"] == pytest.approx(0.0)
    assert head["lr"] == pytest.approx(hidden["lr"] * 0.5)


def test_unassigned_parameters_are_rejected():
    model = TinyModel()
    with pytest.raises(ValueError):
        TorchTrainer(
            model,
            objective,
            config=make_config(),
            param_groups=[ParamGroupSpec("hidden", tuple(model.hidden.parameters()))],
        )


def test_duplicate_parameters_in_a_group_are_rejected():
    model = TinyModel()
    weight = model.hidden.weight
    with pytest.raises(ValueError):
        TorchTrainer(
            model,
            objective,
            config=make_config(),
            param_groups=[
                ParamGroupSpec("all", (weight,) + tuple(model.parameters()))
            ],
        )


def test_reference_group_drives_reported_learning_rate():
    model = TinyModel()
    trainer = TorchTrainer(
        model,
        objective,
        config=make_config(),
        param_groups=[
            ParamGroupSpec("hidden", tuple(model.hidden.parameters())),
            ParamGroupSpec("head", tuple(model.head.parameters()), lr_scale=0.5),
        ],
        reference_group="head",
    )

    result = trainer.train_step(next_training_batch())

    hidden, head = trainer.optimizer.param_groups
    assert result.learning_rate == pytest.approx(head["lr"])
    assert result.learning_rate == pytest.approx(hidden["lr"] * 0.5)
    assert result.metrics["lr_batch"] == pytest.approx(head["lr"])


def test_unknown_reference_group_is_rejected():
    model = TinyModel()
    with pytest.raises(ValueError):
        TorchTrainer(
            model,
            objective,
            config=make_config(),
            param_groups=[
                ParamGroupSpec("hidden", tuple(model.hidden.parameters())),
                ParamGroupSpec("head", tuple(model.head.parameters())),
            ],
            reference_group="missing",
        )


def test_iterable_data_source_rebuilds_each_epoch():
    epoch_calls = []

    def epoch_factory():
        epoch_calls.append(len(epoch_calls))
        yield TrainingBatch(data={"x": torch.zeros(1)}, sample_count=1)

    source = IterableDataSource(epoch_factory)
    assert not source.has_validation()
    assert list(source.validation_batches()) == []

    source.begin_epoch()
    assert not source.is_exhausted()
    assert source.next_batch() is not None
    assert not source.is_exhausted()
    assert source.next_batch() is None
    assert source.is_exhausted()
    source.begin_epoch()
    assert not source.is_exhausted()
    assert source.next_batch() is not None

    assert epoch_calls == [0, 1]


def test_in_memory_source_requires_a_full_batch_when_dropping_last():
    with pytest.raises(ValueError):
        InMemoryDataSource(make_samples(3), batch_size=4, collate_fn=collate)

    # Keeping a partial batch is allowed and yields one smaller batch.
    source = InMemoryDataSource(
        make_samples(3), batch_size=4, collate_fn=collate, drop_last=False
    )
    source.begin_epoch()
    batch = source.next_batch()
    assert batch is not None and batch.sample_count == 3
    assert source.is_exhausted()
    assert source.next_batch() is None


def test_in_memory_source_uses_a_separate_validation_collate_fn():
    def validation_collate(samples):
        return {"inputs": torch.stack([sample[0] for sample in samples])}

    source = InMemoryDataSource(
        make_samples(4),
        batch_size=2,
        collate_fn=collate,
        validation_samples=make_samples(2),
        validation_batch_size=2,
        validation_collate_fn=validation_collate,
    )
    source.begin_epoch()

    training_batch = source.next_batch()
    validation_batch = next(source.validation_batches())

    assert training_batch is not None
    assert "targets" in training_batch.data
    assert "targets" not in validation_batch.data


def test_start_epoch_reproduces_the_epoch_shuffle_order():
    def first_batch_inputs(start_epoch):
        source = InMemoryDataSource(
            make_samples(8),
            batch_size=8,
            collate_fn=collate,
            seed=1,
            start_epoch=start_epoch,
        )
        source.begin_epoch()
        batch = source.next_batch()
        assert batch is not None
        return batch.data["inputs"]

    uninterrupted = InMemoryDataSource(
        make_samples(8), batch_size=8, collate_fn=collate, seed=1
    )
    uninterrupted.begin_epoch()  # epoch 0
    uninterrupted.begin_epoch()  # epoch 1
    resumed_batch = uninterrupted.next_batch()
    assert resumed_batch is not None

    assert not torch.equal(first_batch_inputs(0), first_batch_inputs(1))
    assert torch.equal(resumed_batch.data["inputs"], first_batch_inputs(1))


def test_averaged_metrics_report_nan_for_an_empty_window():
    accumulator = MetricsAccumulator(decay=1.0)
    accumulator.sums["gnorm_batch"] = 0.0
    accumulator.weights["gnorm_batch"] = 0.0

    assert math.isnan(accumulator.averaged()["gnorm_batch"])


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        TorchTrainerConfig(optimizer="lamb")
    with pytest.raises(ValueError):
        TorchTrainerConfig(amp="fp8")
    with pytest.raises(ValueError):
        TorchTrainerConfig(metrics_decay=0.0)
    with pytest.raises(ValueError):
        TorchTrainerConfig(loss_reduction="median")
    with pytest.raises(ValueError):
        TorchTrainerConfig(grad_norm_clip_scale=0.0)
    with pytest.raises(ValueError):
        TorchTrainerConfig(base_learning_rate=float("nan"))
    with pytest.raises(ValueError):
        TorchTrainerConfig(batch_metric_window_steps=-1)
    with pytest.raises(ValueError):
        TorchTrainerConfig(schedule_batch_size=0)


def test_validation_averages_loss_per_sample():
    trainer = TorchTrainer(TinyModel(), objective, config=make_config())
    batches = list(make_source().validation_batches())
    assert [batch.sample_count for batch in batches] == [2, 1]

    # ``objective`` reports ``loss_sum = batch_mean * batch_size``, so the
    # per-sample average is the total squared error divided by all samples.
    with torch.no_grad():
        squared_error = 0.0
        absolute_error = 0.0
        samples = 0
        for batch in batches:
            predictions = trainer.model(batch.data["inputs"]).squeeze(-1)
            targets = batch.data["targets"].squeeze(-1)
            squared_error += float(
                F.mse_loss(predictions, targets, reduction="sum")
            )
            absolute_error += float((predictions - targets).abs().sum())
            samples += batch.sample_count

    metrics = trainer.validate(batches)

    assert samples == 3
    assert metrics["loss"] == pytest.approx(squared_error / samples, rel=1e-5)
    assert metrics["mae"] == pytest.approx(absolute_error / samples, rel=1e-5)


def test_validation_derives_missing_loss_sum_from_batch_mean():
    def mean_objective(model, data):
        predictions = model(data["inputs"]).squeeze(-1)
        targets = data["targets"].squeeze(-1)
        # No loss_sum: ``loss_reduction='mean'`` says the scalar is a batch
        # mean, so the trainer must weight it by each batch's sample count.
        return F.mse_loss(predictions, targets), {}

    trainer = TorchTrainer(
        TinyModel(),
        mean_objective,
        config=make_config(loss_reduction="mean"),
    )
    batches = list(make_source().validation_batches())
    assert [batch.sample_count for batch in batches] == [2, 1]

    expected = 0.0
    with torch.no_grad():
        for batch in batches:
            predictions = trainer.model(batch.data["inputs"]).squeeze(-1)
            targets = batch.data["targets"].squeeze(-1)
            expected += float(F.mse_loss(predictions, targets)) * batch.sample_count

    metrics = trainer.validate(batches)
    assert metrics["loss"] == pytest.approx(expected / 3, rel=1e-5)


def test_validation_accepts_mixed_loss_sum_reporting():
    calls = {"count": 0}

    def mixed_objective(model, data):
        calls["count"] += 1
        predictions = model(data["inputs"]).squeeze(-1)
        targets = data["targets"].squeeze(-1)
        loss = F.mse_loss(predictions, targets, reduction="sum")
        metrics = {}
        if calls["count"] % 2 == 0:
            metrics["loss_sum"] = loss
        return loss, metrics

    trainer = TorchTrainer(TinyModel(), mixed_objective, config=make_config())
    batches = list(make_source().validation_batches())

    # Under ``loss_reduction='sum'`` a missing loss_sum is reconstructed from
    # the scalar, so mixed reporting no longer makes the average ambiguous.
    metrics = trainer.validate(batches)
    expected = 0.0
    with torch.no_grad():
        for batch in batches:
            predictions = trainer.model(batch.data["inputs"]).squeeze(-1)
            targets = batch.data["targets"].squeeze(-1)
            expected += float(F.mse_loss(predictions, targets, reduction="sum"))
    assert metrics["loss"] == pytest.approx(expected / 3, rel=1e-5)


def test_sgd_learning_rate_is_not_batch_scaled():
    trainer = TorchTrainer(
        TinyModel(),
        objective,
        config=make_config(
            optimizer="sgd",
            base_learning_rate=3e-5,
            batch_lr_scaling=True,
        ),
    )
    batch = next_training_batch()
    assert batch.sample_count == 4

    result = trainer.train_step(batch)

    # SGD's default batch exponent is 0, so the LR stays per-sample.
    assert result.learning_rate == pytest.approx(3e-5)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)


def test_learning_rate_and_weight_decay_scale_independently():
    source = InMemoryDataSource(
        make_samples(8),
        batch_size=8,
        collate_fn=collate,
        seed=1,
    )
    source.begin_epoch()
    batch = source.next_batch()
    assert batch is not None and batch.sample_count == 8

    trainer = TorchTrainer(
        TinyModel(),
        objective,
        config=make_config(
            optimizer="sgd",
            base_learning_rate=3e-5,
            batch_lr_scaling=False,
            weight_decay=0.1,
            weight_decay_batch_scaling=True,
        ),
    )

    result = trainer.train_step(batch)

    expected_wd = 0.1 * (8 / 256) ** 1.0
    assert result.learning_rate == pytest.approx(3e-5)
    assert result.metrics["wd_batch"] == pytest.approx(expected_wd)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        expected_wd
    )


def test_adamw_lr_and_wd_batch_scaling_are_independent():
    trainer = TorchTrainer(
        TinyModel(),
        objective,
        config=make_config(
            base_learning_rate=3e-5,
            batch_lr_scaling=False,
            weight_decay=0.1,
            weight_decay_batch_scaling=True,
        ),
    )
    batch = next_training_batch()

    result = trainer.train_step(batch)

    # LR scaling is off, but weight decay still uses its own sqrt scaling.
    assert result.learning_rate == pytest.approx(3e-5 * 1.33)
    assert result.metrics["wd_batch"] == pytest.approx(
        0.1 * (4 / 256) ** 0.5
    )


def test_weight_decay_follows_learning_rate_warmup_power():
    trainer = TorchTrainer(
        TinyModel(),
        objective,
        config=make_config(
            weight_decay=0.1,
            weight_decay_lr_power=0.75,
            warmup_samples=1_000_000,
        ),
    )
    batch = next_training_batch()

    result = trainer.train_step(batch)

    # At zero samples seen the warmup curve starts in its first segment at
    # 1/20; weight decay is multiplied by (lr_scale * warmup) ** power.
    expected = 0.1 * (1.0 / 20.0) ** 0.75
    assert result.metrics["wd_batch"] == pytest.approx(expected)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        expected
    )


def test_running_metrics_are_per_sample_and_windowed():
    trainer = TorchTrainer(
        TinyModel(),
        objective,
        config=make_config(batch_metric_window_steps=1),
    )
    batch = next_training_batch()

    with torch.no_grad():
        predictions = trainer.model(batch.data["inputs"]).squeeze(-1)
        targets = batch.data["targets"].squeeze(-1)
        expected_loss = float(F.mse_loss(predictions, targets))

    result = trainer.train_step(batch)

    # The first update: running averages equal this step's per-sample loss.
    assert "loss" in result.running_metrics
    assert result.running_metrics["loss"] == pytest.approx(
        expected_loss, rel=1e-5
    )
    assert result.metrics["loss"] == pytest.approx(expected_loss, rel=1e-5)

    # The window shrinks on its own step cadence, not on logging: the _batch
    # sum and weight are scaled, while _sum totals (and their per-sample
    # averages) are untouched.
    assert trainer.running_metrics.weights["gnorm_batch"] == pytest.approx(0.001)
    assert trainer.running_metrics.sums["loss_sum"] == pytest.approx(
        expected_loss * batch.sample_count, rel=1e-5
    )
    assert trainer.running_metrics.averaged()["loss"] == pytest.approx(
        expected_loss, rel=1e-5
    )


def test_shrink_batch_window_scales_batch_sums_and_weights():
    accumulator = MetricsAccumulator(decay=1.0)
    accumulator.update({"gnorm_batch": 4.0, "loss_sum": 8.0}, batch_size=2)

    accumulator.shrink_batch_window(0.001)

    assert accumulator.sums["gnorm_batch"] == pytest.approx(4.0 * 0.001)
    assert accumulator.weights["gnorm_batch"] == pytest.approx(0.001)
    assert accumulator.sums["loss_sum"] == pytest.approx(8.0)
    assert accumulator.weights["loss_sum"] == pytest.approx(2.0)
    # Scaling both the sum and the weight leaves the reported average intact.
    assert accumulator.averaged()["gnorm_batch"] == pytest.approx(4.0)


def test_clear_nonfinite_resets_poisoned_keys():
    accumulator = MetricsAccumulator(decay=1.0)
    accumulator.update(
        {"gnorm_batch": float("inf"), "loss_sum": 1.0}, batch_size=2
    )

    accumulator.clear_nonfinite()

    assert accumulator.sums["gnorm_batch"] == 0.0
    assert accumulator.weights["gnorm_batch"] == 0.0
    assert accumulator.sums["loss_sum"] == pytest.approx(1.0)


def test_nonfinite_gnorm_also_drops_exgnorm_sum():
    dropped = detensorify_metrics(
        {
            "gnorm_batch": float("inf"),
            "exgnorm_sum": 3.0,
            "lr_batch": 0.1,
        }
    )

    assert "gnorm_batch" not in dropped
    assert "exgnorm_sum" not in dropped
    assert dropped["lr_batch"] == pytest.approx(0.1)

    nan_gnorm = detensorify_metrics(
        {"gnorm_batch": torch.tensor(float("nan")), "exgnorm_sum": 2.0}
    )
    assert "gnorm_batch" not in nan_gnorm
    assert "exgnorm_sum" not in nan_gnorm

    finite = detensorify_metrics({"gnorm_batch": 2.0, "exgnorm_sum": 1.0})
    assert finite["gnorm_batch"] == pytest.approx(2.0)
    assert finite["exgnorm_sum"] == pytest.approx(1.0)


def test_checkpoint_reapplies_current_param_group_scales(tmp_path):
    def groups_for(model, head_lr_scale, head_wd_scale):
        return [
            ParamGroupSpec("hidden", tuple(model.hidden.parameters())),
            ParamGroupSpec(
                "head",
                tuple(model.head.parameters()),
                lr_scale=head_lr_scale,
                weight_decay_scale=head_wd_scale,
            ),
        ]

    path = tmp_path / "checkpoint.ckpt"
    original_model = TinyModel()
    original = TorchTrainer(
        original_model,
        objective,
        config=make_config(weight_decay=0.2),
        param_groups=groups_for(original_model, 0.5, 0.0),
    )
    original.save_checkpoint(path)

    restored_model = TinyModel()
    restored = TorchTrainer(
        restored_model,
        objective,
        config=make_config(weight_decay=0.2),
        param_groups=groups_for(restored_model, 0.25, 0.5),
    )

    restored.load_checkpoint(path)

    hidden, head = restored.optimizer.param_groups
    assert hidden["group_name"] == "hidden"
    assert head["group_name"] == "head"
    # The checkpoint's scales must not win over the current specs.
    assert hidden["lr_scale"] == pytest.approx(1.0)
    assert head["lr_scale"] == pytest.approx(0.25)
    assert hidden["weight_decay_scale"] == pytest.approx(1.0)
    assert head["weight_decay_scale"] == pytest.approx(0.5)

    restored.on_epoch_start()
    restored.train_step(next_training_batch())

    hidden, head = restored.optimizer.param_groups
    assert head["lr"] == pytest.approx(hidden["lr"] * 0.25)
    assert head["weight_decay"] == pytest.approx(0.2 * 0.5)


def test_sample_counter_accumulates_across_steps():
    trainer = TorchTrainer(TinyModel(), objective, config=make_config())
    batch = next_training_batch()

    trainer.train_step(batch)
    trainer.train_step(batch)

    assert trainer.running_metrics.sums["nsamp"] == 8
    assert trainer.running_metrics.averaged()["nsamp"] == 8


def test_resume_seeds_sample_counter_when_checkpoint_lacks_it(tmp_path):
    path = tmp_path / "checkpoint.ckpt"
    trainer = TorchTrainer(TinyModel(), objective, config=make_config())
    trainer.train_step(next_training_batch())
    trainer.save_checkpoint(path)

    # Simulate a checkpoint written before the nsamp metric existed.
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["running_metrics"]["sums"].pop("nsamp", None)
    state["running_metrics"]["weights"].pop("nsamp", None)
    torch.save(state, path)

    restored = TorchTrainer(
        TinyModel(), objective, config=make_config(), resume_from=path
    )

    assert restored.running_metrics.sums["nsamp"] == 4
    assert restored.running_metrics.weights["nsamp"] == 4


def test_checkpoint_rejects_mismatched_param_group_order(tmp_path):
    path = tmp_path / "checkpoint.ckpt"
    original_model = TinyModel()
    original = TorchTrainer(
        original_model,
        objective,
        config=make_config(),
        param_groups=[
            ParamGroupSpec("hidden", tuple(original_model.hidden.parameters())),
            ParamGroupSpec("head", tuple(original_model.head.parameters())),
        ],
    )
    original.save_checkpoint(path)

    restored_model = TinyModel()
    restored = TorchTrainer(
        restored_model,
        objective,
        config=make_config(),
        param_groups=[
            ParamGroupSpec("head", tuple(restored_model.head.parameters())),
            ParamGroupSpec("hidden", tuple(restored_model.hidden.parameters())),
        ],
    )

    with pytest.raises(ValueError):
        restored.load_checkpoint(path)


def test_schedule_updates_every_n_steps():
    trainer = CountingTrainer(
        TinyModel(),
        objective,
        config=make_config(
            schedule_update_interval_early=5,
            schedule_update_interval_late=50,
        ),
    )
    batch = next_training_batch()

    for _ in range(6):
        trainer.train_step(batch)

    # First step is forced, then the schedule refreshes again on step 6.
    assert trainer.schedule_updates == 2

    trainer.on_epoch_start()
    trainer.train_step(batch)

    # A new epoch forces a refresh on the next step.
    assert trainer.schedule_updates == 3

