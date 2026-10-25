"""AlphaZero iteration loop: self-play, training, export, evaluation, resume."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from suzero_train.loop import LoopConfig, TrainingLoop
from suzero_train.metrics import MetricsSink
from suzero_train.pytorch.config import TorchTrainerConfig
from suzero_train.pytorch.trainer import TorchTrainer
from suzero_train.types import TrainStepResult

from . import interrupts
from .archive_writer import ArchiveWriter
from .config import NetworkConfig, TrainConfig
from .cpp_self_play import generate_games as generate_games_cpp
from .evaluation import load_network_checkpoint, play_match
from .export_onnx import export_network
from .network import SuzeroNetwork, build_network
from .objective import AlphaZeroObjective
from .periodic_match import PeriodicModelMatches
from .progress import self_play_progress
from .protocol import ACTION_ENCODING, ACTION_INPUT_ENCODING, resolve_variant
from .replay import ReplayBuffer, ReplayDataSource
from .self_play import SelfPlayResult, generate_games

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "suzero_5dchess.train.v1"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    log_dir: Path
    gn5d_log_dir: Path
    tensorboard_dir: Path
    model_path: Path
    latest: Path
    final: Path
    marked: Path
    replay: Path

    @classmethod
    def from_config(cls, config: TrainConfig) -> "RunPaths":
        root = Path(config.checkpoint_dir)
        if config.checkpoint_dir in ("runs", "runs/", "./runs"):
            root = root / config.variant / "cls_action_mlp"
        log_dir = Path(config.log_dir) if config.log_dir else root / "logs"
        gn5d_log_dir = (
            Path(config.gn5d_data_dir) / config.variant / "cls_action_mlp"
            if config.gn5d_data_dir else log_dir
        )
        tensorboard_dir = (
            Path(config.tensorboard_dir)
            if config.tensorboard_dir
            else log_dir / "tensorboard"
        )
        model_path = (
            Path(config.onnx_output)
            if config.onnx_output
            else root / "models" / f"{config.variant}_latest_fp32.onnx"
        )
        return cls(
            root=root,
            log_dir=log_dir,
            gn5d_log_dir=gn5d_log_dir,
            tensorboard_dir=tensorboard_dir,
            model_path=model_path,
            latest=root / "latest.pt",
            final=root / "agent_final.pt",
            marked=root / "marked.pt",
            replay=root / "replay_buffer.pt",
        )

    def iteration_checkpoint(self, iteration: int) -> Path:
        return self.root / f"agent_iter_{iteration:04d}.pt"


class _TensorBoardSink(MetricsSink):
    def __init__(self, writer: Any) -> None:
        self.writer = writer

    def on_train_step(self, step: int, result: TrainStepResult) -> None:
        metrics = result.running_metrics or result.metrics
        for key, tag in (
            ("loss", "Loss/total"),
            ("value_loss", "Loss/value"),
            ("policy_loss", "Loss/policy"),
            ("gnorm_batch", "Gradients/norm_before_clip"),
        ):
            if key in metrics:
                self.writer.add_scalar(tag, float(metrics[key]), step)
        self.writer.add_scalar("LearningRate/train", result.learning_rate, step)


class AlphaZeroTrainer(TorchTrainer):
    """TorchTrainer with a cosine learning-rate multiplier.

    The generic trainer already applies warmup and batch scaling; here both are
    disabled and this multiplier implements the reference's cosine decay.
    """

    def __init__(self, *args: Any, total_steps: int, **kwargs: Any) -> None:
        self.total_steps = max(1, int(total_steps))
        super().__init__(*args, **kwargs)

    def _update_lr_and_wd(self, batch_size: int) -> tuple[float, float, float]:
        base_lr, weight_decay, actual_lr = super()._update_lr_and_wd(batch_size)
        progress = min(1.0, self._global_step / self.total_steps)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] *= factor
        base_lr *= factor
        actual_lr *= factor
        self._last_pslr = base_lr
        return base_lr, weight_decay, actual_lr


def _make_trainer(
    network: SuzeroNetwork,
    config: TrainConfig,
    total_steps: int,
) -> AlphaZeroTrainer:
    trainer_config = TorchTrainerConfig(
        device=config.device,
        optimizer="adamw",
        base_learning_rate=config.learning_rate,
        learning_rate_scale=1.0,
        adamw_lr_factor=1.0,
        batch_reference=config.batch_size,
        batch_lr_scaling=False,
        weight_decay_batch_scaling=False,
        schedule_batch_size=config.batch_size,
        warmup_samples=0,
        weight_decay=config.weight_decay,
        grad_norm_cap=config.max_grad_norm,
        grad_norm_clip_scale=1.0,
        # ``AlphaZeroObjective`` returns a batch-mean loss for backprop.
        loss_reduction="mean",
        metrics_decay=0.995,
    )
    objective = AlphaZeroObjective(
        value_weight=config.value_loss_weight,
        policy_weight=config.policy_loss_weight,
    )
    return AlphaZeroTrainer(
        network,
        objective,
        config=trainer_config,
        total_steps=total_steps,
    )


def _checkpoint_extra(
    config: TrainConfig,
    iteration: int,
    total_games: int,
    total_samples: int,
    evaluation_state: dict[str, object],
) -> dict[str, object]:
    return {
        "format": CHECKPOINT_FORMAT,
        "action_encoding": ACTION_ENCODING,
        "action_input_encoding": ACTION_INPUT_ENCODING,
        "iteration": int(iteration),
        "total_games": int(total_games),
        "total_samples": int(total_samples),
        "variant": config.variant,
        "network_config": asdict(config.network),
        "train_config": config.to_dict(),
        "evaluation": dict(evaluation_state),
        "rng_state": _rng_state(),
    }


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _read_checkpoint(path: str | os.PathLike[str]) -> dict[str, object]:
    state = torch.load(os.fspath(path), map_location="cpu", weights_only=False)
    if "model" not in state:
        raise ValueError(f"{path!r} is not a training checkpoint")
    extra = state.get("extra_state")
    if not isinstance(extra, dict):
        raise ValueError(f"{path!r} has no suzero_5dchess extra_state")
    if extra.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path!r} has checkpoint format {extra.get('format')!r}, expected "
            f"{CHECKPOINT_FORMAT!r}"
        )
    if extra.get("action_encoding") != ACTION_ENCODING:
        raise ValueError(f"{path!r} uses an incompatible action encoding")
    return extra


def _save_checkpoint(
    trainer: AlphaZeroTrainer,
    path: Path,
    extra: dict[str, object],
) -> None:
    trainer.extra_state = extra
    trainer.save_checkpoint(path)


def _save_replay_buffer(buffer: ReplayBuffer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(buffer.state_dict(), tmp)
    os.replace(tmp, path)


def _load_replay_buffer(
    buffer: ReplayBuffer, path: Path, *, allow_missing: bool = False
) -> bool:
    if not path.is_file():
        if allow_missing:
            logger.warning(
                "no replay buffer at %s; resuming with an empty replay", path
            )
            return False
        raise ValueError(f"training directory has no replay buffer at {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"{path} is not a replay buffer")
    buffer.load_state_dict(state)
    return True


def _publish_latest(path: Path, latest: Path) -> None:
    if path == latest:
        return
    tmp = Path(str(latest) + ".tmp")
    latest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, tmp)
    os.replace(tmp, latest)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True))
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (torch.Tensor,)):
        return float(value.detach().cpu())
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    raise TypeError(f"object of type {type(value)!r} is not JSON serializable")


def _write_game_records(
    path: Path, results: list[SelfPlayResult]
) -> None:
    payload = [
        {
            "white_outcome": result.white_outcome,
            "terminal_reason": result.terminal_reason,
            "semimoves": result.semimoves,
            "turns": result.turns,
            "board_limit": result.board_limit,
            "board_count": result.board_count,
            "final_player": result.final_player,
            "moves": result.moves,
            "final_fen": result.final_fen,
        }
        for result in results
    ]
    tmp = Path(str(path) + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _self_play_summary(results: list[SelfPlayResult]) -> dict[str, object]:
    outcomes = {"white_wins": 0, "black_wins": 0, "draws": 0}
    reasons: dict[str, int] = {}
    samples = 0
    semimoves = 0
    for result in results:
        samples += len(result.samples)
        semimoves += result.semimoves
        reasons[result.terminal_reason] = reasons.get(result.terminal_reason, 0) + 1
        if result.white_outcome > 0:
            outcomes["white_wins"] += 1
        elif result.white_outcome < 0:
            outcomes["black_wins"] += 1
        else:
            outcomes["draws"] += 1
    games = len(results)
    return {
        "games": games,
        "samples": samples,
        "avg_semimoves": (semimoves / games) if games else 0.0,
        "outcomes": outcomes,
        "terminal_reasons": reasons,
    }


def _training_phase(
    trainer: AlphaZeroTrainer,
    buffer: ReplayBuffer,
    config: TrainConfig,
    *,
    iteration: int,
    sink: MetricsSink | None,
) -> dict[str, object]:
    source = ReplayDataSource(
        buffer,
        config.batch_size,
        config.steps_per_epoch,
        spec=resolve_variant(config.variant),
        seed=config.seed + iteration * 1009,
    )
    loop = TrainingLoop(
        trainer,
        source,
        LoopConfig(
            epochs=config.epochs_per_iteration,
            log_every_steps=0,
            validate_every_epochs=0,
        ),
        metrics_sink=sink,
    )
    report = loop.run()
    return {
        "steps": report.steps,
        "samples": report.samples,
        "metrics": dict(report.last_train_metrics),
        "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
    }


def _write_marked_snapshot(
    network: SuzeroNetwork,
    config: TrainConfig,
    paths: RunPaths,
    *,
    iteration: int,
    total_games: int,
    total_samples: int,
    evaluation_state: dict[str, object],
) -> None:
    """Atomically publish the current network as the evaluation baseline."""

    snapshot = {
        "format": CHECKPOINT_FORMAT,
        "model": network.state_dict(),
        "extra_state": _checkpoint_extra(
            config,
            iteration,
            total_games,
            total_samples,
            evaluation_state,
        ),
    }
    tmp = Path(str(paths.marked) + ".tmp")
    paths.marked.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, tmp)
    os.replace(tmp, paths.marked)


def _evaluate(
    trainer: AlphaZeroTrainer,
    network: SuzeroNetwork,
    config: TrainConfig,
    paths: RunPaths,
    *,
    iteration: int,
    total_games: int,
    total_samples: int,
    evaluation_state: dict[str, object],
) -> dict[str, object] | None:
    if not config.evaluation:
        return None

    def write_marked() -> None:
        _write_marked_snapshot(
            network,
            config,
            paths,
            iteration=iteration,
            total_games=total_games,
            total_samples=total_samples,
            evaluation_state=evaluation_state,
        )

    reference = config.reference_checkpoint
    reference_path = Path(reference) if reference else paths.marked
    if not reference_path.exists():
        if reference is not None:
            logger.warning(
                "evaluation enabled but the reference checkpoint %s does not "
                "exist; skipping",
                reference_path,
            )
            return None
        # Seed the baseline on the first evaluation opportunity; the next one
        # compares against it.  Without this, `marked.pt` would never exist
        # and evaluation would be skipped forever.
        write_marked()
        logger.info("initialized evaluation baseline at %s", paths.marked)
        return None

    device = next(network.parameters()).device
    reference_network = load_network_checkpoint(reference_path, network.spec, device)
    eval_mcts = replace(
        config.mcts,
        num_simulations=config.evaluation_simulations,
        dirichlet_epsilon=0.0,
    )
    started = time.monotonic()
    match = play_match(
        network,
        reference_network,
        network.spec,
        eval_mcts,
        games=config.evaluation_games,
        device_a=device,
        device_b=device,
        seed=config.seed + 17,
        board_limit=config.self_play.max_board_limit,
    )
    result = match.as_dict()
    result["reference"] = str(reference_path)
    result["seconds"] = time.monotonic() - started

    # Promotion snapshot: the next evaluation compares against this network.
    write_marked()
    return result


def run_training(
    config: TrainConfig,
    *,
    resume: str | os.PathLike[str] | None = None,
    archive_writer: ArchiveWriter | None = None,
) -> dict[str, object]:
    """Run ``config.num_iterations`` AlphaZero iterations.

    Ctrl+C is cooperative: one press lets the current iteration finish, saves
    its checkpoint, drains pending archives, and exits. A second press aborts
    immediately. An injected ``archive_writer`` remains owned by the caller so
    continuous CLI batches can share one persistent process.
    """

    torch.manual_seed(config.seed)
    paths = RunPaths.from_config(config)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    network = build_network(config.network, config.variant)
    spec = network.spec
    buffer = ReplayBuffer(config.replay_buffer_size, spec=spec)

    iteration = 0
    total_games = 0
    total_samples = 0
    evaluation_state: dict[str, object] = {}

    if resume is not None:
        extra = _read_checkpoint(resume)
        if extra.get("variant") != config.variant:
            raise ValueError(
                f"checkpoint variant {extra.get('variant')!r} does not match "
                f"{config.variant!r}"
            )
        saved_network = NetworkConfig(**extra.get("network_config", {}))
        if saved_network != config.network:
            raise ValueError(
                "checkpoint network geometry differs from the current config; "
                "start from the checkpoint's NetworkConfig"
            )
        iteration = int(extra.get("iteration", 0))
        total_games = int(extra.get("total_games", 0))
        total_samples = int(extra.get("total_samples", 0))
        _load_replay_buffer(
            buffer,
            Path(resume).parent / paths.replay.name,
            allow_missing=True,
        )
        evaluation_state = dict(extra.get("evaluation", {}))
        rng_state = extra.get("rng_state")
        if isinstance(rng_state, dict):
            torch_state = rng_state.get("torch")
            if isinstance(torch_state, torch.Tensor):
                torch.set_rng_state(torch_state)
            cuda_state = rng_state.get("cuda")
            if (
                torch.cuda.is_available()
                and isinstance(cuda_state, list)
                and cuda_state
            ):
                torch.cuda.set_rng_state_all(cuda_state)

    model_matches = PeriodicModelMatches(
        config, paths.root, paths.log_dir, paths.gn5d_log_dir
    )
    model_matches.snapshot(network, iteration)

    total_iterations = iteration + config.num_iterations
    total_steps = max(
        1,
        total_iterations * config.epochs_per_iteration * config.steps_per_epoch,
    )
    trainer = _make_trainer(network, config, total_steps)
    if resume is not None:
        trainer.load_checkpoint(resume)
        # load_checkpoint restores extra_state; keep the validated copy.
        trainer.extra_state = _checkpoint_extra(
            config, iteration, total_games, total_samples, evaluation_state
        )

    if (
        config.evaluation
        and config.reference_checkpoint is None
        and not paths.marked.exists()
    ):
        # Seed the baseline before the first iteration so the first match can
        # happen exactly one evaluation interval later.
        evaluation_state["last_iteration"] = iteration
        _write_marked_snapshot(
            network,
            config,
            paths,
            iteration=iteration,
            total_games=total_games,
            total_samples=total_samples,
            evaluation_state=evaluation_state,
        )
        logger.info("initialized evaluation baseline at %s", paths.marked)

    writer = None
    if config.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(paths.tensorboard_dir))
        except ImportError:
            logger.warning("tensorboard is not installed; continuing without it")
    sink = _TensorBoardSink(writer) if writer is not None else None
    jsonl_path = paths.log_dir / "training_log.jsonl"

    summary: dict[str, object] = {
        "run_dir": str(paths.root),
        "start_iteration": iteration,
        "iterations": [],
    }
    logger.info(
        "starting run: variant=%s params=%d device=%s start_iteration=%d",
        config.variant,
        network.parameter_count,
        trainer.device,
        iteration,
    )

    stopped_early = False
    last_completed = iteration
    selfplay_model_dir = paths.root / "selfplay"
    owns_archive_writer = False
    if config.self_play.record_games and archive_writer is None:
        archive_writer = ArchiveWriter()
        owns_archive_writer = True

    interrupts.install()
    try:
        if archive_writer is not None:
            archive_writer.start()
        for offset in range(config.num_iterations):
            if archive_writer is not None:
                archive_writer.check()
            if interrupts.stop_requested():
                stopped_early = True
                logger.warning(
                    "Ctrl+C: iteration %d already finished; saving and exiting",
                    last_completed,
                )
                break
            started = time.monotonic()
            network.eval()
            completed = iteration + offset + 1
            with self_play_progress(
                f"第 {completed} 轮自对弈", config.self_play.num_games
            ) as advance:
                if config.self_play.backend == "cpp_onnx":
                    results = generate_games_cpp(
                        network,
                        spec,
                        config.mcts,
                        config.self_play,
                        device=trainer.device,
                        seed=config.seed + (iteration + offset) * 100_003,
                        model_path=(
                            selfplay_model_dir
                            / f"model_iter_{completed:04d}.onnx"
                        ),
                        progress=advance,
                    )
                else:
                    results = generate_games(
                        network,
                        spec,
                        config.mcts,
                        config.self_play,
                        device=trainer.device,
                        seed=config.seed + (iteration + offset) * 100_003,
                        progress=lambda done, _total, _result: advance(done),
                    )
            self_play_seconds = time.monotonic() - started
            for result in results:
                buffer.push(result.samples)
            self_play = _self_play_summary(results)
            total_games += len(results)
            total_samples += int(self_play["samples"])
            logger.info(
                "iteration %d self-play: %d games, %d new samples in %.1fs, "
                "avg_semimoves=%.1f, replay=%d",
                completed,
                len(results),
                self_play["samples"],
                self_play_seconds,
                self_play["avg_semimoves"],
                len(buffer),
            )
            logger.info(
                "self-play outcomes: %s | terminal reasons: %s",
                self_play["outcomes"],
                self_play["terminal_reasons"],
            )
            if config.self_play.record_games:
                _write_game_records(
                    paths.log_dir / f"games_iter_{completed:04d}.json", results
                )
                assert archive_writer is not None
                archive_writer.enqueue(
                    results,
                    spec,
                    paths.gn5d_log_dir,
                    iteration=completed,
                    first_game_id=total_games - len(results) + 1,
                    backend=config.self_play.backend,
                    material_scale=config.self_play.material_scale,
                )
                logger.info(
                    "queued %d .gn5d game archives for background saving",
                    len(results),
                )

            trained: dict[str, object] = {"skipped": True}
            if len(buffer) >= config.min_replay_size:
                trainer.model.train()
                trained = _training_phase(
                    trainer,
                    buffer,
                    config,
                    iteration=iteration + offset,
                    sink=sink,
                )
                trained["skipped"] = False
            else:
                logger.info(
                    "replay has %d samples (< %d); skipping training",
                    len(buffer),
                    config.min_replay_size,
                )

            export_path: str | None = None
            if config.export_interval > 0 and completed % config.export_interval == 0:
                exported = export_network(network, spec, paths.model_path, verify=True)
                export_path = str(exported)

            last_eval = int(evaluation_state.get("last_iteration", -10**9))
            evaluation = None
            if (
                config.evaluation
                and completed - last_eval >= config.evaluation_interval
            ):
                evaluation = _evaluate(
                    trainer,
                    network,
                    config,
                    paths,
                    iteration=completed,
                    total_games=total_games,
                    total_samples=total_samples,
                    evaluation_state=evaluation_state,
                )
                # Count baseline seeding and skipped-reference attempts as an
                # attempt so the next one waits a full interval.
                evaluation_state["last_iteration"] = completed
            if evaluation is not None:
                evaluation_state["last"] = evaluation

            model_matches.snapshot(network, completed)
            match_results = model_matches.run(network, completed)
            if match_results is not None:
                evaluation_state["model_matches"] = match_results

            elapsed = time.monotonic() - started
            record = {
                "iteration": completed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_games": total_games,
                "total_samples": total_samples,
                "buffer_size": len(buffer),
                "iteration_seconds": elapsed,
                "self_play": self_play,
                "training": trained,
                "evaluation": evaluation,
                "model_matches": match_results,
                "export": export_path,
                "learning_rate": float(
                    trainer.optimizer.param_groups[0]["lr"]
                ),
            }
            summary["iterations"].append(record)  # type: ignore[union-attr]
            if config.jsonl:
                _append_jsonl(jsonl_path, record)
            if writer is not None:
                _log_tensorboard(
                    writer,
                    record,
                    eval_state=evaluation_state,
                    done=completed,
                )

            save_now = (
                completed % config.save_interval == 0
                or offset == config.num_iterations - 1
            )
            if save_now:
                extra = _checkpoint_extra(
                    config,
                    completed,
                    total_games,
                    total_samples,
                    evaluation_state,
                )
                periodic = paths.iteration_checkpoint(completed)
                _save_replay_buffer(buffer, paths.replay)
                _save_checkpoint(trainer, periodic, extra)
                _publish_latest(periodic, paths.latest)
                logger.info(
                    "saved iteration %d (%d samples in replay) to %s",
                    completed,
                    len(buffer),
                    periodic,
                )

            logger.info(
                "iteration %d/%d | games=%d samples=%d buffer=%d "
                "loss=%s time=%.1fs",
                completed,
                total_iterations,
                self_play["games"],
                self_play["samples"],
                len(buffer),
                trained.get("metrics", {}).get("loss")
                if isinstance(trained.get("metrics"), dict)
                else None,
                elapsed,
            )
            last_completed = completed
    finally:
        body_failed = sys.exc_info()[0] is not None
        try:
            if owns_archive_writer and archive_writer is not None:
                if body_failed:
                    archive_writer.abort()
                else:
                    archive_writer.close()
                    logger.info(
                        "all queued game archives saved to %s",
                        paths.gn5d_log_dir / "games",
                    )
        except BaseException:
            if body_failed:
                logger.exception("failed to abort the game archive writer")
            else:
                raise
        finally:
            if writer is not None:
                writer.flush()
                writer.close()
            # The last iteration has no next loop entry to observe a stop request.
            stopped_early = stopped_early or interrupts.stop_requested()
            interrupts.uninstall()

    if config.num_iterations > 0:
        # `iteration` is the iteration this run started from. Advance it by the
        # iterations that actually completed, so a cooperative Ctrl+C does not
        # overstate progress in the final checkpoint.
        iteration = last_completed
        extra = _checkpoint_extra(
            config,
            iteration,
            total_games,
            total_samples,
            evaluation_state,
        )
        _save_replay_buffer(buffer, paths.replay)
        _save_checkpoint(trainer, paths.final, extra)
        _publish_latest(paths.final, paths.latest)
        logger.info("saved final checkpoint to %s", paths.final)

    summary["end_iteration"] = iteration
    summary["total_games"] = total_games
    summary["total_samples"] = total_samples
    summary["buffer_size"] = len(buffer)
    summary["onnx"] = str(paths.model_path)
    summary["stopped_early"] = stopped_early
    return summary


def _add_scalar(writer: Any, tag: str, value: Any, step: int) -> None:
    if value is None:
        return
    try:
        writer.add_scalar(tag, float(value), step)
    except (TypeError, ValueError):
        return


def _log_tensorboard(
    writer: Any,
    record: Mapping[str, Any],
    *,
    eval_state: Mapping[str, object],
    done: int,
) -> None:
    _add_scalar(writer, "Progress/iteration", done, done)
    _add_scalar(writer, "Progress/total_games", record.get("total_games"), done)
    _add_scalar(writer, "Progress/total_samples", record.get("total_samples"), done)
    _add_scalar(writer, "Replay/buffer_size", record.get("buffer_size"), done)
    _add_scalar(writer, "Time/iteration_seconds", record.get("iteration_seconds"), done)
    _add_scalar(writer, "LearningRate/iteration", record.get("learning_rate"), done)

    training = record.get("training", {})
    metrics = training.get("metrics", {}) if isinstance(training, dict) else {}
    for key, tag in (
        ("loss", "IterationLoss/total"),
        ("value_loss", "IterationLoss/value"),
        ("policy_loss", "IterationLoss/policy"),
    ):
        _add_scalar(writer, tag, metrics.get(key), done)

    self_play = record.get("self_play", {})
    if isinstance(self_play, dict):
        outcomes = self_play.get("outcomes", {})
        for key in ("white_wins", "black_wins", "draws"):
            _add_scalar(writer, f"SelfPlay/{key}", outcomes.get(key), done)
        reasons = self_play.get("terminal_reasons", {})
        for reason, count in reasons.items():
            _add_scalar(writer, f"Termination/count/{reason}", count, done)

    for label, result in (record.get("model_matches") or {}).items():
        if result.get("skipped"):
            continue
        for key, value in result["totals"][0].items():
            _add_scalar(writer, f"ModelMatch/{label}/{key}", value, done)
        _add_scalar(writer, f"ModelMatch/{label}/win_rate", result["win_rate"], done)
        color_rates = result.get("win_rate_by_color", {})
        for participant in ("current", "opponent"):
            participant_rates = color_rates.get(participant, {})
            for color in ("white", "black"):
                _add_scalar(
                    writer,
                    f"ModelMatch/{label}/{participant}/{color}_win_rate",
                    participant_rates.get(color),
                    done,
                )
        _add_scalar(writer, f"ModelMatch/{label}/score_rate", result["score_rate"], done)
        _add_scalar(writer, f"ModelMatch/{label}/material_average_score",
                    result.get("material_average_score"), done)
        _add_scalar(writer, f"ModelMatch/{label}/opponent_iteration",
                    result["opponent_iteration"], done)

    evaluation = record.get("evaluation")
    if isinstance(evaluation, dict):
        for key in ("a_wins", "b_wins", "draws", "a_score"):
            _add_scalar(writer, f"Evaluation/{key}", evaluation.get(key), done)
