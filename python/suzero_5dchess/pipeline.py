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

from suzero_train.data_source import TrainingDataSource
from suzero_train.loop import LoopConfig, TrainingLoop
from suzero_train.metrics import MetricsSink
from suzero_train.pytorch.config import TorchTrainerConfig
from suzero_train.pytorch.trainer import TorchTrainer
from suzero_train.types import TrainStepResult

from . import interrupts
from .archive_writer import ArchiveWriter
from .checkpoint import CHECKPOINT_FORMAT
from .config import NetworkConfig, TrainConfig
from .cpp_self_play import generate_games as generate_games_cpp
from .evaluation import load_network_checkpoint, play_match
from .export_onnx import export_network
from .file_replay import (
    ROUND_SEED_STRIDE,
    FileReplayDataSource,
    clear_shuffle,
    previous_round_exclusion,
    publish_shuffle,
)
from .gating import ModelGate
from .game_store import GAME_STORE_FORMAT, GameStore
from .network import SuzeroNetwork, build_network
from .objective import AlphaZeroObjective
from .periodic_match import PeriodicModelMatches
from .progress import self_play_progress
from .protocol import ACTION_ENCODING, ACTION_INPUT_ENCODING, resolve_variant
from .replay import ReplayBuffer, ReplayDataSource
from .replay_window import compute_window_size
from .self_play import SelfPlayResult, generate_games
from .train_counts import summarize

logger = logging.getLogger(__name__)

# Shift applied to the self-play seed when an interrupted iteration is topped
# up, so resumed games do not replay the RNG streams of games already saved.
_RESUME_SEED_STRIDE = 1_000_003

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
    replay_store: Path
    train_counts: Path
    accepted_dir: Path
    accepted_state: Path
    gate_log_dir: Path

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
            replay_store=root / "replay_store",
            train_counts=root / "train_counts.npz",
            accepted_dir=root / "accepted",
            accepted_state=root / "accepted" / "state.json",
            gate_log_dir=log_dir / "gates",
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
    *,
    accepted_state: dict[str, object] | None = None,
    shuffle_state: dict[str, object] | None = None,
    selfplay_progress: dict[str, object] | None = None,
) -> dict[str, object]:
    extra: dict[str, object] = {
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
    if accepted_state is not None:
        extra["accepted"] = dict(accepted_state)
    if shuffle_state is not None:
        extra["replay_shuffle"] = dict(shuffle_state)
    if selfplay_progress is not None:
        extra["selfplay_progress"] = {
            "iteration": int(selfplay_progress.get("iteration", 0)),
            "games": int(selfplay_progress.get("games", 0)),
        }
    return extra


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
    _save_replay_state(buffer.state_dict(), path)


def _save_replay_state(state: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _save_train_counts(replay: ReplayBuffer | GameStore, paths: RunPaths) -> None:
    """Persist the row counters that the replay state itself cannot hold.

    The in-memory backend carries its counters inside ``state_dict``; the row
    log keeps row identity in the log, so its counters live in a sidecar.
    """

    if isinstance(replay, GameStore):
        replay.save_train_counts(paths.train_counts)


def _load_train_counts(replay: ReplayBuffer | GameStore, paths: RunPaths) -> None:
    if not isinstance(replay, GameStore):
        return
    try:
        restored = replay.load_train_counts(paths.train_counts)
    except (OSError, ValueError) as error:
        # The counters are a diagnostic; a damaged sidecar must not block a
        # resume that can still train from the row log.
        logger.warning(
            "could not restore training counters from %s (%s); starting them "
            "from zero for the current window",
            paths.train_counts, error,
        )
        return
    if restored:
        logger.info(
            "restored training counters for %d of %d replay rows from %s",
            restored, len(replay), paths.train_counts,
        )


def _load_replay_buffer(
    buffer: ReplayBuffer,
    path: Path,
    *,
    allow_missing: bool = False,
    warn_capacity: bool = False,
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
    if warn_capacity:
        saved_count = int(state.get("count", len(state.get("samples", []))))
        saved_capacity = int(state.get("capacity", buffer.capacity))
        if saved_capacity > buffer.capacity or saved_count > buffer.capacity:
            logger.warning(
                "replay window capacity reduced: saved_capacity=%d count=%d "
                "new_max_samples=%d; may discard up to %d oldest trained "
                "samples on load (untrained rows are kept)",
                saved_capacity, saved_count, buffer.capacity,
                max(0, saved_count - buffer.capacity),
            )
        if "total_generated" not in state:
            logger.warning(
                "legacy replay has no total_generated; using checkpoint "
                "extra.total_samples and buffered count as fallbacks"
            )
    buffer.load_state_dict(state)
    return True


def _load_game_store(
    store: GameStore,
    path: Path,
    *,
    allow_missing: bool = False,
) -> bool:
    """Restore the counters that the archives alone cannot reconstruct."""

    if not path.is_file():
        if allow_missing:
            logger.warning(
                "no replay state at %s; resuming from the game archives on disk",
                path,
            )
            return False
        raise ValueError(f"training directory has no replay state at {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"{path} is not replay state")
    if state.get("format") != GAME_STORE_FORMAT:
        logger.warning(
            "replay state at %s belongs to the in-memory backend; ignoring it "
            "and rebuilding the window from %s",
            path, store.directory,
        )
        return False
    store.load_state_dict(state)
    return True


def _initialize_replay_window(
    replay: ReplayBuffer | GameStore,
    config: TrainConfig,
    total_samples: int,
    *,
    replay_loaded: bool,
) -> int:
    """Reconcile cumulative counters and trim a resumed window.

    The caller restores the per-row training counters first; the trim only
    removes rows those counters prove the trainer has already served.
    """

    counts = (total_samples, replay.total_generated, len(replay))
    total_generated = max(counts)
    if len(set(counts)) > 1:
        logger.warning(
            "replay counters differ: checkpoint.total_samples=%d "
            "replay.total_generated=%d buffered=%d; using N=%d",
            *counts, total_generated,
        )
    if total_samples == 0 and replay.total_generated == 0 and len(replay):
        logger.warning("no cumulative replay counter; falling back to buffered count")
    if not replay_loaded and total_generated > 0:
        logger.warning(
            "historical replay samples are permanently lost; window uses "
            "historical N=%d but must accumulate new samples (历史样本已丢失、"
            "窗口只能重新积累)", total_generated,
        )
    replay.total_generated = total_generated
    window = config.window
    size = compute_window_size(total_generated, config.replay_buffer_size, window)
    evicted = replay.trim_to(size)
    estimated_bytes = replay.estimated_bytes()
    average = estimated_bytes / len(replay) if len(replay) else None
    logger.info(
        "lossy replay window: min_rows=%d e=%g expand=%g scale=%d max=%d "
        "N=%d W=%d buffered=%d startup_evicted=%d estimated_bytes=%d "
        "estimated_window_bytes=%s estimated_max_bytes=%s",
        config.replay_buffer_size, window.taper_exponent, window.expand_per_row,
        window.taper_scale or config.replay_buffer_size, window.max_samples,
        total_generated, size, len(replay), evicted, estimated_bytes,
        round(average * size) if average is not None else "unknown (empty buffer)",
        round(average * window.max_samples)
        if average is not None else "unknown (empty buffer)",
    )
    logger.info(
        "Measure actual mean compressed blob size with a tiny run before "
        "choosing max_samples; do not assume 60 KB/sample. Estimates exclude "
        "training tensors and checkpoint serialization peaks. Evicted samples "
        "are permanently lost; no archive or old-window reload is provided."
    )
    return total_generated


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
    tree_samples = 0
    semimoves = 0
    board_squares = 0
    for result in results:
        samples += result.sample_count
        tree_samples += result.tree_samples
        semimoves += result.semimoves
        board_squares += result.board_count * result.board_count
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
        "played_samples": samples - tree_samples,
        "tree_samples": tree_samples,
        "avg_semimoves": (semimoves / games) if games else 0.0,
        # Second moment of the per-game board count: sum(board_count^2) / games.
        "avg_board_count_squared": (board_squares / games) if games else 0.0,
        "outcomes": outcomes,
        "terminal_reasons": reasons,
    }


def _resume_selfplay_offset(
    store: GameStore | None,
    pending_progress: dict[str, object] | None,
    completed: int,
    num_games: int,
) -> int:
    """Games an interrupted run already saved for ``completed``.

    The checkpoint record is authoritative because it survives replay
    trimming; counting the store is the fallback for checkpoints written
    before progress tracking existed and also picks up games archived after
    the checkpoint was last written.
    """

    already = 0
    if pending_progress is not None:
        if int(pending_progress.get("iteration", -1)) == completed:
            already = int(pending_progress.get("games", 0))
    if store is not None:
        already = max(
            already, sum(1 for game in store.games if game.iteration == completed)
        )
    return max(0, min(already, num_games))


def _training_phase(
    trainer: AlphaZeroTrainer,
    replay: ReplayBuffer | GameStore,
    config: TrainConfig,
    *,
    iteration: int,
    sink: MetricsSink | None,
) -> dict[str, object]:
    spec = resolve_variant(config.variant)
    file_store = isinstance(replay, GameStore)
    file_shuffle = (
        not file_store
        and config.window.enabled
        and config.window.sampling == "shuffle"
    )
    if file_store:
        # Shuffle rounds over the row log.  Each round excludes the previous
        # round's rows; nothing is copied into RAM beyond the game being read.
        source: TrainingDataSource = replay.training_source(
            config.batch_size,
            config.steps_per_epoch,
            spec,
            iteration=iteration,
            seed=config.seed + iteration * 1009,
            keep_samples=(
                config.window.keep_samples if config.window.enabled else None
            ),
            no_repeat_rows=config.no_repeat_rows,
            endgame=config.endgame,
        )
    elif file_shuffle:
        directory = RunPaths.from_config(config).root / "replay_shuffle"
        rounds = 0

        def publish_round(exclude):
            nonlocal rounds
            index = rounds
            rounds += 1
            return publish_shuffle(
                replay,
                directory,
                iteration=iteration + 1,
                seed=config.seed + iteration * 1009 + index * ROUND_SEED_STRIDE,
                keep_samples=config.window.keep_samples,
                exclude=exclude,
                no_repeat_rows=config.no_repeat_rows,
                endgame=config.endgame,
            )

        # The first round of this phase never repeats rows the previous
        # phase's last round actually trained.  Rows it selected but never
        # served stay eligible, so an interrupted round cannot hide samples
        # from the next phase.  Rows evicted since then are dropped by the
        # index translation.
        exclude: list[int] = []
        if config.no_repeat_rows:
            counts = replay.train_counts
            exclude = [
                row
                for row in previous_round_exclusion(
                    directory,
                    pool_samples=len(replay),
                    total_generated=replay.total_generated,
                )
                if row < len(counts) and counts[row] > 0
            ]
        if publish_round(exclude) is None:
            logger.info(
                "replay shuffle: every buffered row was selected by the previous "
                "round; no new round for iteration %d", iteration + 1,
            )
            return {
                "skipped": True, "steps": 0, "samples": 0, "metrics": {},
                "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                "train_counts": summarize(replay.train_counts),
                "shuffle": {
                    "generation": None, "files": 0, "file_index": 0,
                    "row_offset": 0, "files_used": [], "rounds": 0,
                    "pool_samples": len(replay), "round_samples": 0,
                    "remaining_samples": 0,
                    "no_repeat_rows": config.no_repeat_rows,
                    "waiting_for_shuffle": True,
                },
            }
        source = FileReplayDataSource(
            directory, config.batch_size, config.steps_per_epoch,
            spec=spec, state=replay.shuffle_state,
            publisher=publish_round,
            no_repeat_rows=config.no_repeat_rows,
            train_counts=replay.train_counts,
        )
    else:
        source = ReplayDataSource(
            replay,
            config.batch_size,
            config.steps_per_epoch,
            spec=spec,
            seed=config.seed + iteration * 1009,
            endgame=config.endgame,
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
    trained = {
        "steps": report.steps,
        "samples": report.samples,
        "metrics": dict(report.last_train_metrics),
        "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
        # Distribution over the live window: how many passes each self-play
        # sample received so far, including rows this phase never reached.
        "train_counts": summarize(replay.train_counts),
    }
    if config.endgame.enabled:
        trained["endgame"] = config.endgame.identity()
    # Insertion order, not sort_keys: the summary already lists its fields and
    # the histogram buckets in reading order, and sorting would put "10+"
    # between "1" and "2".
    logger.info("replay train counts: %s", json.dumps(trained["train_counts"]))
    if file_store:
        replay.shuffle_state = source.state_dict()
        trained["shuffle"] = source.summary()
        logger.info("replay_store: %s", json.dumps(trained["shuffle"], sort_keys=True))
        if source.waiting_for_shuffle:
            logger.info(
                "replay-store rounds exhausted for this phase; every buffered "
                "row was selected by the preceding round. The next phase draws "
                "again after self-play updates the window."
            )
    elif file_shuffle:
        replay.shuffle_state = source.state_dict()
        trained["shuffle"] = source.summary()
        logger.info("replay_shuffle: %s", json.dumps(trained["shuffle"], sort_keys=True))
        if source.waiting_for_shuffle:
            logger.info(
                "shuffle rounds exhausted for this phase; every buffered row was "
                "selected by the preceding round. The next phase publishes again "
                "after self-play updates the window."
            )
    return trained


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

    Ctrl+C is cooperative. During self-play the backend stops handing out new
    games, the in-flight games finish, and their samples are persisted before
    training is skipped and progress is saved; the next run resumes from that
    progress. During training/evaluation the current iteration finishes, its
    checkpoint is saved, pending archives are drained, and the process exits.
    A second press aborts immediately. An injected ``archive_writer`` remains
    owned by the caller so continuous CLI batches can share one persistent
    process.
    """

    torch.manual_seed(config.seed)
    paths = RunPaths.from_config(config)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    network = build_network(config.network, config.variant)
    spec = network.spec
    capacity = (
        config.window.max_samples if config.window.enabled else config.replay_buffer_size
    )
    file_replay = config.replay_backend == "file"
    buffer: ReplayBuffer | None = None
    store: GameStore | None = None
    replay: ReplayBuffer | GameStore
    if file_replay:
        store = GameStore(paths.replay_store, spec, capacity)
        store.scan()
        replay = store
    else:
        buffer = ReplayBuffer(capacity, spec=spec)
        replay = buffer

    iteration = 0
    total_games = 0
    total_samples = 0
    evaluation_state: dict[str, object] = {}
    replay_loaded = False
    # Games already saved for an iteration that was interrupted mid-self-play.
    # ``iteration`` stays at the last *completed* round, so the interrupted
    # round is re-entered and topped up rather than restarted from zero.
    pending_progress: dict[str, object] | None = None

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
        replay_path = Path(resume).parent / paths.replay.name
        if store is not None:
            replay_loaded = _load_game_store(
                store, replay_path, allow_missing=True
            )
        else:
            assert buffer is not None
            replay_loaded = _load_replay_buffer(
                buffer,
                replay_path,
                allow_missing=True,
                warn_capacity=config.window.enabled,
            )
        evaluation_state = dict(extra.get("evaluation", {}))
        recorded = extra.get("selfplay_progress")
        if isinstance(recorded, dict):
            pending_progress = {
                "iteration": int(recorded.get("iteration", 0)),
                "games": int(recorded.get("games", 0)),
            }
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

    # Restore the per-row counters before any trim: trimming may only drop
    # rows the trainer has already served, and the sidecar is the only proof
    # of that for the file backend.
    _load_train_counts(replay, paths)
    if config.window.enabled:
        total_samples = _initialize_replay_window(
            replay,
            config,
            total_samples,
            replay_loaded=replay_loaded or (store is not None and len(store) > 0),
        )
    else:
        replay.total_generated = max(total_samples, replay.total_generated)
        total_samples = replay.total_generated
        replay.trim_to(capacity)
    # Published shards are disposable views, never a source for restoring
    # evicted rows. A resumed synchronous run produces its next generation
    # only after self-play has updated and trimmed the current window.
    if not file_replay:
        clear_shuffle(paths.root / "replay_shuffle")
    replay.shuffle_state = None

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
            config, iteration, total_games, total_samples, evaluation_state,
            selfplay_progress=pending_progress,
        )

    gate = None
    if config.gate_interval > 0:
        gate = ModelGate(config, paths, spec, trainer.device)
        gate.bootstrap_or_load(network, iteration)
        trainer.extra_state = _checkpoint_extra(
            config, iteration, total_games, total_samples, evaluation_state,
            accepted_state=gate.summary(),
            selfplay_progress=pending_progress,
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
    # Carries the saved-game count of an interrupted iteration into the next
    # checkpoint; when no self-play runs at all, keep the resumed record.
    selfplay_state: dict[str, object] | None = pending_progress
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
            selfplay_network = gate.resolve(network) if gate is not None else network
            selfplay_iteration = gate.state.iteration if gate is not None else None
            if gate is not None:
                logger.info(
                    "iteration %d selfplay_model_iteration=%d",
                    completed, selfplay_iteration,
                )
            if store is not None:
                # Index archives from earlier iterations, then give this round
                # a contiguous id range shared by every self-play process.
                store.refresh()
            buffered_before = len(replay)
            game_id_start = store.next_game_id if store is not None else 0
            already = 0
            if offset == 0 and resume is not None:
                already = _resume_selfplay_offset(
                    store, pending_progress, completed, config.self_play.num_games
                )
            remaining = config.self_play.num_games - already
            seed = config.seed + (iteration + offset) * 100_003
            if already:
                seed += already * _RESUME_SEED_STRIDE
            with self_play_progress(
                f"第 {completed} 轮自对弈", config.self_play.num_games
            ) as advance:
                advance(already)
                if remaining <= 0:
                    logger.info(
                        "iteration %d self-play: %d/%d games already saved; "
                        "skipping straight to training",
                        completed, already, config.self_play.num_games,
                    )
                    results = []
                elif config.self_play.backend == "cpp_onnx":
                    results = generate_games_cpp(
                        selfplay_network,
                        spec,
                        config.mcts,
                        config.self_play,
                        device=trainer.device,
                        seed=seed,
                        num_games=remaining,
                        model_path=(
                            selfplay_model_dir
                            / f"model_iter_{completed:04d}.onnx"
                        ),
                        progress=lambda done: advance(already + done),
                        game_store=store,
                        game_id_start=game_id_start,
                        iteration=completed,
                    )
                else:
                    results = generate_games(
                        selfplay_network,
                        spec,
                        config.mcts,
                        config.self_play,
                        device=trainer.device,
                        seed=seed,
                        num_games=remaining,
                        progress=lambda done, _total, _result: advance(already + done),
                        game_store=store,
                        game_id_start=game_id_start,
                        iteration=completed,
                    )
            self_play_seconds = time.monotonic() - started
            if store is not None:
                # Children write their own archives, so re-scan before training.
                store.refresh()
                for result in results:
                    if not result.persisted:
                        store.append_result(
                            store.next_game_id, result, iteration=completed
                        )
                total_samples = store.total_generated
            else:
                assert buffer is not None
                for result in results:
                    buffer.push(result.samples)
            self_play = _self_play_summary(results)
            total_games += len(results)
            selfplay_state = {
                "iteration": completed,
                "games": already + len(results),
            }
            if already:
                self_play["resumed_games"] = already
                logger.info(
                    "iteration %d resumed self-play: %d games already saved, "
                    "%d played now (%d/%d total)",
                    completed, already, len(results),
                    already + len(results), config.self_play.num_games,
                )
            new_samples = int(self_play["samples"])
            if buffer is not None:
                total_samples += new_samples
            replay_window = None
            if config.window.enabled:
                assert total_samples >= 0
                size = compute_window_size(
                    total_samples, config.replay_buffer_size, config.window
                )
                replay.trim_to(size)
                replay_window = {
                    "total_generated": total_samples,
                    "window_size": size,
                    "buffered": len(replay),
                    "evicted": buffered_before + new_samples - len(replay),
                    "sampling": config.window.sampling,
                    "keep_samples": config.window.keep_samples,
                    "estimated_bytes": replay.estimated_bytes(),
                    "backend": config.replay_backend,
                }
            else:
                # Single-process appends already enforce capacity; this is the
                # only enforcement after child processes wrote their own files.
                replay.trim_to(config.replay_buffer_size)
            logger.info(
                "iteration %d self-play: %d games, %d new samples "
                "(%d played, %d tree) in %.1fs, avg_semimoves=%.1f, "
                "avg_board_sq=%.1f, replay=%d",
                completed,
                len(results),
                self_play["samples"],
                self_play["played_samples"],
                self_play["tree_samples"],
                self_play_seconds,
                self_play["avg_semimoves"],
                self_play["avg_board_count_squared"],
                len(replay),
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

            if interrupts.stop_requested():
                # Self-play (and its file writes) is the expensive part.  Once
                # its in-flight games are done, do not start training on a
                # round the user asked to leave: the samples are already on
                # disk and the final checkpoint records the counters.
                stopped_early = True
                logger.warning(
                    "Ctrl+C: self-play stopped after %d/%d games; %d new "
                    "samples saved; skipping training and saving progress "
                    "before exit",
                    already + len(results),
                    config.self_play.num_games,
                    new_samples,
                )
                break

            trained: dict[str, object] = {
                "skipped": True,
                "train_counts": summarize(replay.train_counts),
            }
            if len(replay) >= config.min_replay_size:
                trainer.model.train()
                trained = _training_phase(
                    trainer,
                    replay,
                    config,
                    iteration=iteration + offset,
                    sink=sink,
                )
                trained.setdefault("skipped", False)
            else:
                logger.info(
                    "replay has %d samples (< %d); skipping training",
                    len(replay),
                    config.min_replay_size,
                )

            if replay_window is not None:
                replay_window["coverage"] = (
                    int(trained.get("samples", 0)) / max(1, len(replay))
                )
                logger.info("replay_window: %s", json.dumps(replay_window, sort_keys=True))

            export_path: str | None = None
            if config.export_interval > 0 and completed % config.export_interval == 0:
                exported = export_network(network, spec, paths.model_path, verify=True)
                export_path = str(exported)

            gate_result = None
            candidate_saved = False
            if gate is not None:
                if not trained["skipped"] and completed % config.gate_interval == 0:
                    # A promotion must always have a resumable training checkpoint.
                    extra = _checkpoint_extra(
                        config, completed, total_games, total_samples,
                        evaluation_state, accepted_state=gate.summary(),
                        shuffle_state=replay.shuffle_state,
                        selfplay_progress=selfplay_state,
                    )
                    _save_replay_state(replay.state_dict(), paths.replay)
                    _save_train_counts(replay, paths)
                    _save_checkpoint(trainer, paths.iteration_checkpoint(completed), extra)
                    candidate_saved = True
                gate_result = gate.evaluate(
                    network, completed, trained=not bool(trained["skipped"]),
                ).as_dict()

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
                "buffer_size": len(replay),
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
            if gate_result is not None:
                record["selfplay_model_iteration"] = selfplay_iteration
                record["gate"] = gate_result
            if replay_window is not None:
                record["replay_window"] = replay_window
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
                or candidate_saved
            )
            if save_now:
                extra = _checkpoint_extra(
                    config,
                    completed,
                    total_games,
                    total_samples,
                    evaluation_state,
                    accepted_state=gate.summary() if gate is not None else None,
                    shuffle_state=replay.shuffle_state,
                    selfplay_progress=selfplay_state,
                )
                periodic = paths.iteration_checkpoint(completed)
                if not candidate_saved:
                    _save_replay_state(replay.state_dict(), paths.replay)
                    _save_train_counts(replay, paths)
                _save_checkpoint(trainer, periodic, extra)
                _publish_latest(periodic, paths.latest)
                logger.info(
                    "saved iteration %d (%d samples in replay) to %s",
                    completed,
                    len(replay),
                    periodic,
                )

            logger.info(
                "iteration %d/%d | games=%d samples=%d buffer=%d "
                "loss=%s time=%.1fs",
                completed,
                total_iterations,
                self_play["games"],
                self_play["samples"],
                len(replay),
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
            accepted_state=gate.summary() if gate is not None else None,
            shuffle_state=replay.shuffle_state,
            selfplay_progress=selfplay_state,
        )
        _save_replay_state(replay.state_dict(), paths.replay)
        _save_train_counts(replay, paths)
        _save_checkpoint(trainer, paths.final, extra)
        _publish_latest(paths.final, paths.latest)
        logger.info("saved final checkpoint to %s", paths.final)

    summary["end_iteration"] = iteration
    summary["total_games"] = total_games
    summary["total_samples"] = total_samples
    summary["buffer_size"] = len(replay)
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

    gate = record.get("gate")
    if isinstance(gate, dict):
        _add_scalar(writer, "Gate/average_score", gate.get("average_score"), done)
        _add_scalar(writer, "Gate/accepted_iteration", gate.get("accepted_after"), done)

    replay_window = record.get("replay_window")
    if isinstance(replay_window, dict):
        for key in ("window_size", "buffered", "coverage"):
            _add_scalar(writer, f"Replay/{key}", replay_window.get(key), done)

    training = record.get("training", {})
    metrics = training.get("metrics", {}) if isinstance(training, dict) else {}
    for key, tag in (
        ("loss", "IterationLoss/total"),
        ("value_loss", "IterationLoss/value"),
        ("policy_loss", "IterationLoss/policy"),
    ):
        _add_scalar(writer, tag, metrics.get(key), done)

    train_counts = training.get("train_counts") if isinstance(training, dict) else None
    if isinstance(train_counts, dict):
        for key in ("trained", "untrained", "mean", "median", "p90", "max"):
            _add_scalar(writer, f"Replay/train_count_{key}", train_counts.get(key), done)

    self_play = record.get("self_play", {})
    if isinstance(self_play, dict):
        for key in (
            "samples", "played_samples", "tree_samples", "avg_board_count_squared"
        ):
            _add_scalar(writer, f"SelfPlay/{key}", self_play.get(key), done)
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
