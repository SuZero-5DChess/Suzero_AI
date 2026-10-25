"""Configuration for the game-agnostic PyTorch trainer.

The learning-rate, weight-decay and gradient-clipping *formulas* mirror the
training recipe that KataGo's ``train.py`` applies per sample.  Nothing here
knows about Go, board encodings or KataGo's model classes: a caller supplies
any ``torch.nn.Module`` and objective.

Several defaults deliberately disable upstream features, so a default-constructed
config is not equivalent to a default KataGo run: ``warmup_samples=0`` disables
warmup, ``weight_decay=0`` disables weight decay and ``grad_norm_cap=0`` only
measures the gradient norm instead of clipping.  Set those fields explicitly
to reproduce upstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

_OPTIMIZERS = ("adamw", "sgd")
_AMP_CHOICES = ("none", "fp16", "bf16")
_LOSS_REDUCTIONS = ("sum", "mean")


@dataclass(frozen=True)
class ParamGroupSpec:
    """Parameters that share learning-rate and weight-decay factors.

    ``lr_scale`` multiplies the scheduled learning rate; ``weight_decay_scale``
    multiplies :attr:`TorchTrainerConfig.weight_decay`.  Splitting a model into
    groups is optional: when the trainer receives no groups it uses a single
    group containing every trainable parameter.
    """

    name: str
    params: Sequence[torch.nn.Parameter]
    lr_scale: float = 1.0
    weight_decay_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParamGroupSpec.name must not be empty")
        if self.lr_scale < 0:
            raise ValueError("ParamGroupSpec.lr_scale must be >= 0")
        if self.weight_decay_scale < 0:
            raise ValueError("ParamGroupSpec.weight_decay_scale must be >= 0")


@dataclass(frozen=True)
class TorchTrainerConfig:
    """Optimizer, schedule and checkpoint settings for :class:`TorchTrainer`.

    The learning-rate recipe follows KataGo:

    ``lr = base_learning_rate * learning_rate_scale
           * (adamw_lr_factor if AdamW else 1)
           * warmup_scale(samples_seen)
           * (batch_size / batch_reference) ** batch_lr_exponent
           * group.lr_scale``

    With the defaults and ``batch_size == 256`` this is the upstream
    per-sample rate ``3e-5 * 1.33`` for AdamW and ``3e-5`` for SGD.  Upstream
    does not batch-scale the SGD learning rate, so the SGD default for
    :attr:`batch_lr_exponent` is ``0``.  The recipe assumes the objective's
    backward loss behaves like a batch sum, as KataGo's does.  With a
    batch-mean loss the SGD effective step is smaller by a factor of
    ``batch_size`` and the gradient norm is smaller than the clipping cap was
    calibrated for; prefer returning a batch sum and use
    :attr:`loss_reduction` only to describe scalar metrics that omit
    ``loss_sum``.

    Weight decay is scaled independently:

    ``wd = weight_decay * group.weight_decay_scale
          * (batch_size / batch_reference) ** batch_wd_exponent
          * (learning_rate_scale * warmup_scale) ** weight_decay_lr_power``

    The batch exponent defaults to ``0.5`` for AdamW and ``1.0`` for SGD,
    matching upstream.  ``weight_decay_lr_power`` is ``0`` by default; set it
    to ``0.75`` to reproduce KataGo's bnorm-family factor.  KataGo's
    model-specific weight-decay table and adaptive-norm factor are not
    reproduced; set :attr:`weight_decay` and per-group
    ``weight_decay_scale`` for the model at hand.

    ``warmup_samples`` is the number of samples over which the piecewise
    KataGo warmup (1/20 ... 1/1.4 in eight equal segments) completes.  Zero
    disables warmup.  ``grad_norm_cap <= 0`` measures the gradient norm but
    does not clip.

    KataGo refreshes learning rate and weight decay every 5 batches for the
    first 200M samples and every 50 batches afterwards, counted from the start
    of each epoch; the ``schedule_update_*`` fields mirror that cadence.  A new
    epoch always refreshes the schedule.

    ``loss_reduction`` declares how to interpret the scalar loss returned by
    the objective when its metrics do not contain ``loss_sum``.  ``"sum"`` (the
    default and KataGo's convention) treats the backward loss as a batch sum and
    reports ``loss = loss / sample_count``; ``"mean"`` treats it as a batch mean.
    When ``loss_sum`` is present it always wins.  The objective's *backward*
    loss must follow the same convention; otherwise the learning rate and
    gradient-norm clipping use a different gradient scale than upstream.

    ``schedule_batch_size`` is the fixed batch size used by the LR, weight-decay
    and gradient-cap batch scaling.  Upstream uses its configured
    ``batch_size``; ``None`` means "use the current batch's ``sample_count``",
    which only differs when a data source emits partial or variable-size
    batches.

    ``batch_metric_window_steps`` mirrors KataGo's ``_batch`` metric window:
    the accumulated sum and weight are multiplied by 0.001 every this many
    training steps, counted from the start of each epoch.  Zero disables the
    window shrink.  This is independent of the loop's ``log_every_steps``.
    """

    device: str = "auto"
    optimizer: str = "adamw"

    base_learning_rate: float = 3e-5
    learning_rate_scale: float = 1.0
    adamw_lr_factor: float = 1.33
    batch_reference: int = 256
    batch_lr_scaling: bool = True
    batch_lr_exponent: float | None = None  # None: 0.5 for AdamW, 0.0 for SGD
    batch_wd_exponent: float | None = None  # None: 0.5 for AdamW, 1.0 for SGD
    warmup_samples: int = 0
    schedule_update_interval_early: int = 5
    schedule_update_interval_late: int = 50
    schedule_update_early_until_samples: int = 200_000_000
    schedule_batch_size: int | None = None

    loss_reduction: str = "sum"
    batch_metric_window_steps: int = 100

    weight_decay: float = 0.0
    weight_decay_batch_scaling: bool = True
    weight_decay_lr_power: float = 0.0

    grad_norm_cap: float = 0.0
    grad_norm_clip_scale: float = 1.0
    metrics_decay: float = 0.995

    amp: str = "none"
    sgd_momentum: float = 0.9
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.optimizer not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {_OPTIMIZERS}")
        if self.amp not in _AMP_CHOICES:
            raise ValueError(f"amp must be one of {_AMP_CHOICES}")
        if self.loss_reduction not in _LOSS_REDUCTIONS:
            raise ValueError(f"loss_reduction must be one of {_LOSS_REDUCTIONS}")

        for name in (
            "base_learning_rate",
            "learning_rate_scale",
            "adamw_lr_factor",
            "weight_decay",
            "grad_norm_cap",
            "grad_norm_clip_scale",
            "sgd_momentum",
            "adamw_eps",
        ):
            _check_finite(name, getattr(self, name))
        for name in ("batch_lr_exponent", "batch_wd_exponent"):
            value = getattr(self, name)
            if value is not None:
                _check_finite(name, value)
        for beta in self.adamw_betas:
            _check_finite("adamw_betas entry", beta)

        if self.base_learning_rate < 0:
            raise ValueError("base_learning_rate must be >= 0")
        if self.learning_rate_scale < 0:
            raise ValueError("learning_rate_scale must be >= 0")
        if self.adamw_lr_factor < 0:
            raise ValueError("adamw_lr_factor must be >= 0")
        if self.batch_reference <= 0:
            raise ValueError("batch_reference must be positive")
        if self.warmup_samples < 0:
            raise ValueError("warmup_samples must be >= 0")
        if self.schedule_update_interval_early < 1:
            raise ValueError("schedule_update_interval_early must be >= 1")
        if self.schedule_update_interval_late < 1:
            raise ValueError("schedule_update_interval_late must be >= 1")
        if self.schedule_update_early_until_samples < 0:
            raise ValueError("schedule_update_early_until_samples must be >= 0")
        if self.schedule_batch_size is not None and self.schedule_batch_size <= 0:
            raise ValueError("schedule_batch_size must be positive when set")
        if self.batch_metric_window_steps < 0:
            raise ValueError("batch_metric_window_steps must be >= 0")
        if self.batch_lr_exponent is not None and self.batch_lr_exponent < 0:
            raise ValueError("batch_lr_exponent must be >= 0")
        if self.batch_wd_exponent is not None and self.batch_wd_exponent < 0:
            raise ValueError("batch_wd_exponent must be >= 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.weight_decay_lr_power < 0:
            raise ValueError("weight_decay_lr_power must be >= 0")
        if self.grad_norm_cap < 0:
            raise ValueError("grad_norm_cap must be >= 0")
        if self.grad_norm_clip_scale <= 0:
            raise ValueError("grad_norm_clip_scale must be positive")
        if not 0.0 < self.metrics_decay <= 1.0:
            raise ValueError("metrics_decay must be in (0, 1]")
        if self.sgd_momentum < 0:
            raise ValueError("sgd_momentum must be >= 0")
        if len(self.adamw_betas) != 2:
            raise ValueError("adamw_betas must contain exactly 2 values")
        for beta in self.adamw_betas:
            if not 0.0 <= beta < 1.0:
                raise ValueError("adamw_betas entries must be in [0, 1)")
        if self.adamw_eps <= 0:
            raise ValueError("adamw_eps must be positive")


def _check_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
