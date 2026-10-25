"""A game-agnostic PyTorch trainer built from KataGo's training recipe.

What is extracted from KataGo's ``train.py``:

* per-sample base learning rate, AdamW factor and batch-size scaling;
* the piecewise sample-based warmup curve;
* per-parameter-group learning-rate/weight-decay factors;
* global gradient-norm clipping with KataGo's batch/LR-dependent cap;
* AMP with a ``GradScaler`` for fp16;
* exponentially decayed running metrics and ``_sum``/``_batch`` averaging;
* atomic checkpointing of model, optimizer, scaler, metrics and counters.

What is deliberately *not* here: Go board encodings, KataGo's model classes,
its npz selfplay format, DDP, Muon optimizers, SWA/lookahead and
``torch.compile``.  A caller supplies the model, the objective and the data.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from ..trainer import Trainer
from ..types import TrainStepResult, TrainingBatch
from .config import ParamGroupSpec, TorchTrainerConfig
from .device import resolve_device
from .metrics import MetricsAccumulator, detensorify_metrics
from .schedules import batch_scale, warmup_scale

logger = logging.getLogger(__name__)

Forward = Callable[[torch.nn.Module, Mapping[str, Any]], Any]
Objective = Callable[
    ...,
    tuple[torch.Tensor, Mapping[str, float | torch.Tensor]],
]

_CHECKPOINT_FORMAT = "suzero_train.pytorch.v1"


class TorchTrainer(Trainer):
    """Single-process trainer for any ``torch.nn.Module``.

    ``objective(model, data)`` runs the forward pass and returns
    ``(loss, metrics)``; the trainer owns autocast, backward, gradient
    clipping, optimizer scheduling, metric accumulation and checkpoints.  The
    loss must be a scalar tensor.  Metric keys ending in ``_sum`` are treated
    as batch sums and ``_batch`` as batch-normalized values.

    Passing ``forward`` splits the forward pass from the loss, matching
    KataGo's autocast scope: ``forward(model, data)`` runs under autocast and
    ``objective(model, data, outputs)`` runs outside it, so the loss and its
    metrics are computed in fp32.  Without ``forward``, the whole objective
    runs under autocast (the historical behaviour); in that case the objective
    must keep the loss numerically safe for the chosen ``amp`` mode.

    ``reference_group`` selects which parameter group feeds the reported
    ``lr_batch``/``wd_batch`` and ``TrainStepResult.learning_rate``.  It
    defaults to the first group; set it to the group that plays the role of
    KataGo's ``normal`` group.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        objective: Objective,
        *,
        forward: Forward | None = None,
        config: TorchTrainerConfig | None = None,
        param_groups: Sequence[ParamGroupSpec] | None = None,
        reference_group: str | None = None,
        resume_from: str | os.PathLike[str] | None = None,
    ) -> None:
        self.model = model
        self.objective = objective
        self.forward = forward
        self.config = config or TorchTrainerConfig()
        self.device = resolve_device(self.config.device)
        if self.config.amp == "fp16" and self.device.type != "cuda":
            raise ValueError("amp='fp16' requires a CUDA device")

        self.model.to(self.device)
        self.model.train()

        self.train_state: dict[str, int] = {
            "global_step_samples": 0,
            "trainer_global_step": 0,
        }
        self._global_step = 0
        self._last_val_metrics: dict[str, float] = {}
        # Game-specific trainers may attach arbitrary JSON/pickle-safe state
        # here (iteration counters, replay buffers, ...).  It is persisted in
        # the checkpoint but otherwise ignored by the generic trainer.
        self.extra_state: dict[str, object] = {}
        self.running_metrics = MetricsAccumulator(decay=self.config.metrics_decay)
        self._batch_count_this_epoch = 0
        self._force_schedule_update = True
        self._last_pslr = 0.0

        self.optimizer = self._build_optimizer(param_groups)
        self._reference_group_index = self._resolve_reference_group(reference_group)
        self._scaler = self._build_scaler()

        if resume_from is not None:
            self.load_checkpoint(resume_from)

    # ------------------------------------------------------------------ setup

    def _build_optimizer(
        self, param_groups: Sequence[ParamGroupSpec] | None
    ) -> torch.optim.Optimizer:
        groups = self._resolve_param_groups(param_groups)
        cfg = self.config
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                groups,
                lr=cfg.base_learning_rate,
                betas=cfg.adamw_betas,
                eps=cfg.adamw_eps,
                weight_decay=cfg.weight_decay,
            )
        return torch.optim.SGD(
            groups,
            lr=cfg.base_learning_rate,
            momentum=cfg.sgd_momentum,
            weight_decay=cfg.weight_decay,
        )

    def _resolve_param_groups(
        self, specs: Sequence[ParamGroupSpec] | None
    ) -> list[dict[str, Any]]:
        model_params = {
            id(param): param
            for param in self.model.parameters()
            if param.requires_grad
        }
        if specs is None:
            specs = [ParamGroupSpec("all", tuple(model_params.values()))]

        groups: list[dict[str, Any]] = []
        seen: set[int] = set()
        for spec in specs:
            params = [param for param in spec.params if param.requires_grad]
            if not params:
                continue
            ids = {id(param) for param in params}
            if len(ids) != len(params):
                raise ValueError(
                    f"parameter group {spec.name!r} lists the same parameter "
                    "more than once"
                )
            if ids & seen:
                raise ValueError(
                    f"parameter group {spec.name!r} reuses parameters from "
                    "another group"
                )
            unknown = ids - set(model_params)
            if unknown:
                raise ValueError(
                    f"parameter group {spec.name!r} contains parameters that "
                    "do not belong to the model"
                )
            seen |= ids
            groups.append(
                {
                    "params": params,
                    "group_name": spec.name,
                    "lr_scale": spec.lr_scale,
                    "weight_decay_scale": spec.weight_decay_scale,
                }
            )

        missing = set(model_params) - seen
        if missing:
            raise ValueError(
                f"{len(missing)} trainable parameters are not assigned to a "
                "parameter group"
            )
        if not groups:
            raise ValueError("the model has no trainable parameters")
        return groups

    def _resolve_reference_group(self, name: str | None) -> int:
        names = [group.get("group_name") for group in self.optimizer.param_groups]
        if name is None:
            return 0
        try:
            return names.index(name)
        except ValueError as exc:
            raise ValueError(
                f"reference_group {name!r} is not one of the parameter groups "
                f"{names}"
            ) from exc

    def _build_scaler(self) -> torch.amp.GradScaler | None:
        if self.config.amp != "fp16":
            return None
        return torch.amp.GradScaler(device=self.device.type, enabled=True)

    # -------------------------------------------------------------- schedules

    def _amp_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.amp == "bf16" else torch.float16

    def _amp_enabled(self) -> bool:
        return self.config.amp != "none"

    def _lr_batch_factor(self, batch_size: int) -> float:
        exponent = self.config.batch_lr_exponent
        if exponent is None:
            exponent = 0.5 if self.config.optimizer == "adamw" else 0.0
        return batch_scale(
            batch_size,
            self.config.batch_reference,
            exponent,
            enabled=self.config.batch_lr_scaling,
        )

    def _wd_batch_factor(self, batch_size: int) -> float:
        exponent = self.config.batch_wd_exponent
        if exponent is None:
            exponent = 0.5 if self.config.optimizer == "adamw" else 1.0
        return batch_scale(
            batch_size,
            self.config.batch_reference,
            exponent,
            enabled=self.config.weight_decay_batch_scaling,
        )

    def _update_lr_and_wd(self, batch_size: int) -> tuple[float, float, float]:
        """Set per-group LR/weight decay; return (base LR, reference WD, LR)."""
        cfg = self.config
        warmup = warmup_scale(
            self.train_state["global_step_samples"], cfg.warmup_samples
        )
        base_lr = cfg.base_learning_rate * cfg.learning_rate_scale
        if cfg.optimizer == "adamw":
            base_lr *= cfg.adamw_lr_factor
        base_lr *= warmup

        schedule_batch = self._schedule_batch_size(batch_size)
        lr_factor = self._lr_batch_factor(schedule_batch)
        wd_factor = self._wd_batch_factor(schedule_batch)
        wd_lr_factor = 1.0
        if cfg.weight_decay_lr_power > 0:
            wd_lr_factor = (
                cfg.learning_rate_scale * warmup
            ) ** cfg.weight_decay_lr_power

        reference_lr = 0.0
        reference_wd = 0.0
        for index, group in enumerate(self.optimizer.param_groups):
            group["lr"] = base_lr * group["lr_scale"] * lr_factor
            group["weight_decay"] = (
                cfg.weight_decay
                * group["weight_decay_scale"]
                * wd_factor
                * wd_lr_factor
            )
            if index == self._reference_group_index:
                reference_lr = group["lr"]
                reference_wd = group["weight_decay"]
        self._last_pslr = base_lr
        return base_lr, reference_wd, reference_lr

    def _schedule_batch_size(self, batch_size: int) -> int:
        return (
            self.config.schedule_batch_size
            if self.config.schedule_batch_size is not None
            else batch_size
        )

    def _should_update_schedule(self) -> bool:
        """KataGo's refresh condition, counted from the start of the epoch."""
        if self._force_schedule_update:
            return True
        cfg = self.config
        count = self._batch_count_this_epoch
        if (
            self.train_state["global_step_samples"]
            <= cfg.schedule_update_early_until_samples
        ):
            return count % cfg.schedule_update_interval_early == 0
        return count % cfg.schedule_update_interval_late == 0

    def _grad_norm_cap(self, batch_size: int) -> float:
        cap = self.config.grad_norm_cap
        if cap <= 0:
            return 0.0
        cap *= self.config.grad_norm_clip_scale
        cap *= math.sqrt(
            self._schedule_batch_size(batch_size) / self.config.batch_reference
        )
        cap /= math.sqrt(max(1e-7, self.config.learning_rate_scale))
        return cap

    # ------------------------------------------------------------------ hooks

    def on_epoch_start(self) -> None:
        # Upstream refreshes the schedule at the start of every epoch, counts
        # its refresh/log cadence from the epoch start, and clears any
        # non-finite metric leftovers before accumulating anew.
        self._force_schedule_update = True
        self._batch_count_this_epoch = 0
        self.running_metrics.clear_nonfinite()

    # -------------------------------------------------------------- training

    def train_step(self, batch: TrainingBatch) -> TrainStepResult:
        batch_size = batch.sample_count
        self.optimizer.zero_grad(set_to_none=True)
        if self._should_update_schedule():
            base_lr, weight_decay, actual_lr = self._update_lr_and_wd(batch_size)
            self._force_schedule_update = False
        else:
            reference = self.optimizer.param_groups[self._reference_group_index]
            base_lr = self._last_pslr
            weight_decay = float(reference["weight_decay"])
            actual_lr = float(reference["lr"])

        loss, raw_metrics = self._run_objective(batch.data)
        metrics = self._loss_metrics(raw_metrics, loss, batch_size)
        # Plain keys accumulate as running sums; ``nsamp`` is KataGo's sample
        # counter and is seeded from ``global_step_samples`` on resume.
        metrics["nsamp"] = batch_size

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        cap = self._grad_norm_cap(batch_size)
        params = [
            param
            for group in self.optimizer.param_groups
            for param in group["params"]
        ]
        if cap > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(params, cap)
        else:
            # Do not call clip_grad_norm_ with max_norm=inf: if the norm is
            # already infinite, inf/inf yields a NaN coefficient and every
            # gradient becomes NaN.  Measuring the norm leaves gradients
            # untouched, as "cap == 0 disables clipping" promises.
            gnorm = _grad_norm(params)
        metrics["gnorm_batch"] = gnorm
        if cap > 0:
            metrics["exgnorm_sum"] = torch.clamp(gnorm - cap, min=0.0) * batch_size
        metrics["pslr_batch"] = base_lr
        metrics["lr_batch"] = actual_lr
        metrics["wd_batch"] = weight_decay
        metrics["gnorm_cap_batch"] = cap

        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

        self._global_step += 1
        self._batch_count_this_epoch += 1
        self.train_state["global_step_samples"] += batch_size
        self.train_state["trainer_global_step"] = self._global_step

        detached = detensorify_metrics(metrics)
        self.running_metrics.update(detached, batch_size)
        window = self.config.batch_metric_window_steps
        if window > 0 and self._batch_count_this_epoch % window == 0:
            self.running_metrics.shrink_batch_window(0.001)

        step_metrics = MetricsAccumulator(decay=1.0)
        step_metrics.update(detached, batch_size)
        reported = step_metrics.averaged()
        return TrainStepResult(
            metrics=reported,
            sample_count=batch_size,
            learning_rate=actual_lr,
            running_metrics=self.running_metrics.averaged(),
        )

    def _run_objective(
        self, data: Mapping[str, Any]
    ) -> tuple[torch.Tensor, Mapping[str, float | torch.Tensor]]:
        """Run the forward pass and objective with KataGo's autocast scope."""
        autocast = torch.amp.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype(),
            enabled=self._amp_enabled(),
        )
        if self.forward is not None:
            with autocast:
                outputs = self.forward(self.model, data)
            loss, metrics = self.objective(self.model, data, outputs)
        else:
            with autocast:
                loss, metrics = self.objective(self.model, data)
        return _require_scalar_loss(loss), metrics

    def _loss_metrics(
        self,
        raw_metrics: Mapping[str, float | torch.Tensor],
        loss: torch.Tensor,
        batch_size: int,
    ) -> dict[str, float | torch.Tensor]:
        """Ensure loss is reported per sample, following ``loss_reduction``."""
        metrics = dict(raw_metrics)
        if "loss_sum" not in metrics:
            if self.config.loss_reduction == "sum":
                metrics["loss_sum"] = loss.detach()
            else:
                metrics["loss_sum"] = loss.detach() * batch_size
        return metrics

    def validate(self, batches: Iterable[TrainingBatch]) -> Mapping[str, float]:
        was_training = self.model.training
        self.model.eval()
        accumulated = MetricsAccumulator(decay=1.0)
        samples = 0
        try:
            with torch.no_grad():
                for batch in batches:
                    loss, raw_metrics = self._run_objective(batch.data)
                    # Missing ``loss_sum`` is derived with the same
                    # ``loss_reduction`` rule as training, so a partial or
                    # mixed final batch still averages per sample.
                    metrics = self._loss_metrics(
                        raw_metrics, loss, batch.sample_count
                    )
                    accumulated.update(
                        detensorify_metrics(metrics), batch.sample_count
                    )
                    samples += batch.sample_count
        finally:
            self.model.train(was_training)

        if samples == 0:
            return {}
        metrics = accumulated.averaged()
        self._last_val_metrics = dict(metrics)
        return metrics

    # ------------------------------------------------------------- checkpoint

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        state = {
            "format": _CHECKPOINT_FORMAT,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self._scaler.state_dict() if self._scaler is not None else None,
            "running_metrics": self.running_metrics.state_dict(),
            "train_state": dict(self.train_state),
            "last_val_metrics": dict(self._last_val_metrics),
            "trainer_config": asdict(self.config),
            "extra_state": self.extra_state,
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        logger.info("saved checkpoint: %s", path)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        state = torch.load(
            os.fspath(path), map_location=self.device, weights_only=False
        )
        if "model" not in state:
            raise ValueError(f"{path!r} is not a trainer checkpoint")
        saved_config = state.get("trainer_config")
        if saved_config is not None:
            saved_optimizer = saved_config.get("optimizer")
            if saved_optimizer not in (None, self.config.optimizer):
                raise ValueError(
                    f"checkpoint was written with optimizer={saved_optimizer!r}, "
                    f"but the trainer uses {self.config.optimizer!r}"
                )

        current_group_meta = [
            {
                "group_name": group.get("group_name"),
                "lr_scale": group.get("lr_scale"),
                "weight_decay_scale": group.get("weight_decay_scale"),
            }
            for group in self.optimizer.param_groups
        ]
        if "optimizer" in state:
            saved_groups = state["optimizer"].get("param_groups", [])
            saved_names = [group.get("group_name") for group in saved_groups]
            current_names = [meta["group_name"] for meta in current_group_meta]
            if all(name is not None for name in saved_names) and (
                saved_names != current_names
            ):
                raise ValueError(
                    "checkpoint optimizer parameter groups do not match the "
                    f"current groups: checkpoint={saved_names}, "
                    f"current={current_names}"
                )

        self.model.load_state_dict(state["model"])
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except ValueError as exc:
                raise ValueError(
                    "checkpoint optimizer state is incompatible with the "
                    f"current parameter groups: {exc}"
                ) from exc
            # torch restores the saved group dict (including lr_scale and
            # weight_decay_scale).  Re-apply the caller's current metadata so
            # changed ParamGroupSpec values are not silently ignored.
            for group, meta in zip(self.optimizer.param_groups, current_group_meta):
                group["group_name"] = meta["group_name"]
                group["lr_scale"] = meta["lr_scale"]
                group["weight_decay_scale"] = meta["weight_decay_scale"]
        if state.get("scaler") is not None and self._scaler is not None:
            self._scaler.load_state_dict(state["scaler"])
        if "running_metrics" in state:
            self.running_metrics.load_state_dict(state["running_metrics"])
        if "train_state" in state:
            self.train_state.update(state["train_state"])
        self.train_state.setdefault("global_step_samples", 0)
        self.train_state.setdefault("trainer_global_step", 0)
        # Upstream seeds the sample counter when loading a checkpoint that
        # predates the ``nsamp`` metric, so resumed runs report lifetime
        # samples rather than only the samples seen since the resume.
        if "nsamp" not in self.running_metrics.sums:
            samples = float(self.train_state["global_step_samples"])
            if samples > 0:
                self.running_metrics.sums["nsamp"] = samples
                self.running_metrics.weights["nsamp"] = samples
        self._last_val_metrics = dict(state.get("last_val_metrics", {}))
        self.extra_state = dict(state.get("extra_state", {}))
        self._global_step = int(self.train_state["trainer_global_step"])
        self._batch_count_this_epoch = 0
        self._force_schedule_update = True
        self._last_pslr = 0.0
        logger.info(
            "loaded checkpoint %s at %d samples",
            os.fspath(path),
            self.train_state["global_step_samples"],
        )

    # --------------------------------------------------------------- counters

    @property
    def samples_seen(self) -> int:
        return int(self.train_state["global_step_samples"])

    @property
    def global_step(self) -> int:
        return self._global_step


def _require_scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise TypeError("objective must return a scalar torch.Tensor loss")
    return loss.reshape(())


def _grad_norm(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """Global gradient L2 norm without scaling the gradients.

    ``clip_grad_norm_(..., inf)`` is not a substitute: when the norm is
    already infinite, it multiplies by ``inf / inf`` and turns every gradient
    into NaN.  Here an infinite gradient merely yields an infinite norm, which
    :func:`detensorify_metrics` drops from the metrics.
    """
    norms = [
        torch.linalg.vector_norm(param.grad.detach(), 2)
        for param in params
        if param.grad is not None
    ]
    if not norms:
        device = params[0].device if params else torch.device("cpu")
        return torch.zeros((), device=device)
    return torch.linalg.vector_norm(torch.stack(norms), 2)
