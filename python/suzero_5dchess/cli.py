"""Command line entry point for 5D chess training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

from . import console_sink
from .archive_writer import ArchiveWriter
from .config import NetworkConfig, TrainConfig, debug_train_config
from .console_mode import disable_quick_edit
from .endgame_sampling import EndgameSamplingConfig
from .evaluation import load_network_checkpoint, play_match
from .export_onnx import export_network
from .network import build_network
from .pipeline import RunPaths, _read_checkpoint, run_training
from .protocol import VARIANTS, resolve_variant

logger = logging.getLogger("suzero_5dchess")


def _configure_logging(log_dir: Path | None = None) -> None:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    # The console is the only unreliable destination: a paused console must
    # never stall the trainer, so it is written from a background thread.
    stream = logging.StreamHandler(console_sink.install() or sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def _add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--debug", action="store_true", help="tiny CPU preset")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--gn5d-data-dir", default=None,
                        help="root directory for GN5D game and match archives")
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--onnx-output", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint path or 'latest'")

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="iterations per run; with --continuous this is the batch length",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="keep training after each batch of --iterations iterations: "
        "resume from the latest checkpoint and start the next batch without "
        "exiting. One Ctrl+C during self-play stops handing out new games, "
        "lets the in-flight games finish and saves their samples (skipping "
        "training); during training/evaluation it finishes the iteration. "
        "A second press aborts immediately",
    )
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-replay-size", type=int, default=None)
    parser.add_argument("--replay-size", type=int, default=None)
    parser.add_argument(
        "--replay-backend", choices=("file", "memory"), default=None,
        help="file: one NPZ per finished game, written immediately and trained "
        "from disk (default); memory: encode the whole round in RAM",
    )
    parser.add_argument(
        "--repeat-rows", dest="no_repeat_rows", action="store_false",
        default=None,
        help="let adjacent shuffle rounds draw the same rows (default: each "
        "round excludes the rows the previous round selected)",
    )
    parser.add_argument(
        "--no-repeat-rows", dest="no_repeat_rows", action="store_true",
        default=None,
        help="keep adjacent shuffle rounds disjoint (default)",
    )
    parser.add_argument(
        "--window", action="store_true",
        help="enable a dynamically growing replay window (default: disabled)",
    )
    parser.add_argument("--window-expand-per-row", type=float, default=None)
    parser.add_argument("--window-taper-exponent", type=float, default=None)
    parser.add_argument("--window-taper-scale", type=int, default=None,
                        help="taper scale in generated samples (default: --replay-size)")
    parser.add_argument("--window-max-samples", type=int, default=None)
    parser.add_argument("--window-keep-samples", type=int, default=None,
                        help="optional rows per file-shuffle generation (does not trim history)")
    parser.add_argument(
        "--window-sampling", choices=("shuffle", "uniform"), default=None,
        help="shuffle: finite NPZ file pass across epochs; uniform: in-memory sampling",
    )
    parser.add_argument(
        "--endgame-sampling", dest="endgame_sampling", action="store_true",
        default=None,
        help="repeat rows near the end of their game inside each replay round; "
        "every row is still included once (default: disabled)",
    )
    parser.add_argument(
        "--no-endgame-sampling", dest="endgame_sampling", action="store_false",
        default=None,
        help="disable endgame row weighting (default)",
    )
    parser.add_argument(
        "--endgame-weight", type=float, default=None,
        help="expected extra copies of a game's final position (default: 1.0)",
    )
    parser.add_argument(
        "--endgame-exponent", type=float, default=None,
        help="ramp exponent of the endgame weight; 1 is linear, larger values "
        "concentrate the boost in the last part of a game (default: 2.0)",
    )
    parser.add_argument("--epochs-per-iteration", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--value-loss-weight", type=float, default=None)
    parser.add_argument("--policy-loss-weight", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--export-interval", type=int, default=None)

    parser.add_argument("--c-puct", type=float, default=None)
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--dirichlet-epsilon", type=float, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--action-mlp-hidden", type=int, default=None)
    parser.add_argument("--max-params", type=int, default=None)

    parser.add_argument("--min-board-limit", type=int, default=None)
    parser.add_argument("--max-board-limit", type=int, default=None)
    parser.add_argument("--material-scale", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--temperature-final", type=float, default=None)
    parser.add_argument("--temperature-threshold", type=int, default=None)
    parser.add_argument("--no-playout-cap", action="store_true")
    parser.add_argument("--full-search-probability", type=float, default=None)
    parser.add_argument("--fast-sims", type=int, default=None)
    parser.add_argument(
        "--record-tree-positions", dest="record_tree_positions",
        action="store_true", default=None,
        help="reuse well-visited side positions from full-search trees (default; "
        "no extra playouts)",
    )
    parser.add_argument(
        "--no-record-tree-positions", dest="record_tree_positions",
        action="store_false",
        help="disable tree-position sampling",
    )
    parser.add_argument("--tree-min-visits", type=int, default=None)
    parser.add_argument("--tree-max-depth", type=int, default=None)
    parser.add_argument("--tree-max-per-turn", type=int, default=None,
                        help="maximum tree rows per full-search decision (small step)")
    parser.add_argument("--tree-sample-prob", type=float, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="concurrent self-play games (Python threads, or C++ game threads "
        "with --selfplay-backend cpp_onnx)",
    )
    parser.add_argument(
        "--selfplay-backend",
        choices=("python", "cpp_onnx"),
        default=None,
        help="self-play execution backend (default: python). cpp_onnx exports "
        "the live network to ONNX and runs PUCT + ONNX Runtime in C++",
    )
    parser.add_argument(
        "--cpp-onnx-provider",
        choices=("cpu", "cuda"),
        default=None,
        help="ONNX Runtime provider for the cpp_onnx backend (default: cpu). "
        "cuda requires onnxruntime-gpu and fails loudly if it is unavailable",
    )
    parser.add_argument(
        "--num-server-threads",
        type=int,
        default=None,
        help="C++ ONNX evaluator server threads for the cpp_onnx backend "
        "(default: 1). Each thread owns an ONNX session, so this is separate "
        "from --workers (concurrent games)",
    )
    parser.add_argument(
        "--selfplay-processes",
        type=int,
        default=None,
        help="number of self-play worker processes (default: 1). Values > 1 "
        "require --selfplay-backend cpp_onnx; --workers, "
        "--num-server-threads and --games stay total budgets shared across "
        "processes and must each be >= the process count",
    )
    parser.add_argument(
        "--record-games",
        action="store_true",
        help="write per-game .gn5d action-only records and per-iteration JSON to the log directory",
    )

    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--evaluation-games", type=int, default=None)
    parser.add_argument("--evaluation-sims", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--model-match-interval", type=int, default=None,
                        help="paired C++ matches every N global iterations; 0 disables")
    parser.add_argument("--model-match-lookback", type=int, default=None,
                        help="opponent is exactly N iterations older (not nearest checkpoint)")
    parser.add_argument("--model-match-games", type=int, default=None)
    parser.add_argument("--model-match-workers", type=int, default=None,
                        help="concurrent C++ games per opponent (default: 20); "
                        "independent of self-play workers; 1 runs sequentially")
    parser.add_argument("--model-match-sims", type=int, default=None)
    parser.add_argument(
        "--model-match-max-boards",
        type=int,
        default=None,
        help="board limit for periodic model matches (default: 25)",
    )
    parser.add_argument("--model-match-provider", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--model-match-reference", default=None,
                        help="additional fixed ONNX or trusted checkpoint opponent")

    parser.add_argument("--gate-interval", type=int, default=None,
                        help="candidate promotion matches every N iterations; 0 disables")
    parser.add_argument("--gate-games", type=int, default=None)
    parser.add_argument("--gate-workers", type=int, default=None,
                        help="concurrent C++ gate games (default: 20)")
    parser.add_argument("--gate-simulations", type=int, default=None)
    parser.add_argument("--gate-min-average-score", type=float, default=None,
                        help="promote when the candidate's mean game score "
                             "(forced material terminals included) reaches this value")
    parser.add_argument("--gate-min-finished", type=int, default=None)
    parser.add_argument("--gate-max-boards", type=int, default=None)
    parser.add_argument("--gate-provider", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--gate-temperature", type=float, default=None)

    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-jsonl", action="store_true")


def _build_config(args: argparse.Namespace) -> TrainConfig:
    config = debug_train_config() if args.debug else TrainConfig()
    if args.variant is not None:
        config.variant = args.variant
    if args.debug and args.checkpoint_dir is None:
        config.checkpoint_dir = (
            "runs/debug" if config.variant == "very_small"
            else f"runs/debug/{config.variant}"
        )

    # Keep the reference width on 8x8 Turn Zero boards without changing the
    # default 200k budget for Very Small. Explicit network flags still win.
    if config.variant == "standard_turn_zero":
        config.network.max_params = 300_000
        # Turn Zero's wider branch factor gets a deeper default self-play
        # search than the shared presets. The debug preset stays tiny, and an
        # explicit --sims still wins.
        if not args.debug:
            config.mcts.num_simulations = 1000
    network = config.network
    for name in ("d_model", "n_heads", "n_layers", "d_ff", "action_mlp_hidden", "max_params"):
        value = getattr(args, name)
        if value is not None:
            setattr(network, name, value)
    config.network = network

    scalar = {
        "device": "device",
        "seed": "seed",
        "checkpoint_dir": "checkpoint_dir",
        "log_dir": "log_dir",
        "gn5d_data_dir": "gn5d_data_dir",
        "tensorboard_dir": "tensorboard_dir",
        "onnx_output": "onnx_output",
    }
    for arg_name, field_name in scalar.items():
        value = getattr(args, arg_name)
        if value is not None:
            setattr(config, field_name, value)

    for field_name in (
        "batch_size",
        "min_replay_size",
        "replay_buffer_size",
        "replay_backend",
        "no_repeat_rows",
        "epochs_per_iteration",
        "steps_per_epoch",
        "learning_rate",
        "weight_decay",
        "value_loss_weight",
        "policy_loss_weight",
        "max_grad_norm",
        "save_interval",
        "export_interval",
        "evaluation_games",
        "evaluation_simulations",
        "evaluation_interval",
        "reference_checkpoint",
        "model_match_interval",
        "model_match_lookback",
        "model_match_games",
        "model_match_workers",
        "model_match_simulations",
        "model_match_max_boards",
        "model_match_provider",
        "model_match_reference",
        "gate_interval",
        "gate_games",
        "gate_workers",
        "gate_simulations",
        "gate_min_average_score",
        "gate_min_finished",
        "gate_max_boards",
        "gate_provider",
        "gate_temperature",
    ):
        arg_name = {
            "replay_buffer_size": "replay_size",
            "learning_rate": "lr",
            "evaluation_simulations": "evaluation_sims",
            "model_match_simulations": "model_match_sims",
        }.get(field_name, field_name)
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, field_name, value)

    if getattr(args, "window", False):
        config.window.enabled = True
    for field_name in (
        "expand_per_row", "taper_exponent", "taper_scale",
        "max_samples", "keep_samples", "sampling",
    ):
        value = getattr(args, f"window_{field_name}", None)
        if value is not None:
            setattr(config.window, field_name, value)

    if (
        args.endgame_sampling is not None
        or args.endgame_weight is not None
        or args.endgame_exponent is not None
    ):
        config.endgame = EndgameSamplingConfig(
            enabled=(
                config.endgame.enabled
                if args.endgame_sampling is None
                else args.endgame_sampling
            ),
            strength=(
                config.endgame.strength
                if args.endgame_weight is None
                else args.endgame_weight
            ),
            exponent=(
                config.endgame.exponent
                if args.endgame_exponent is None
                else args.endgame_exponent
            ),
        )

    if args.iterations is not None:
        config.num_iterations = args.iterations
    if args.evaluation:
        config.evaluation = True
    if args.no_tensorboard:
        config.tensorboard = False
    if args.no_jsonl:
        config.jsonl = False

    self_play = config.self_play
    for arg_name, field_name in (
        ("games", "num_games"),
        ("min_board_limit", "min_board_limit"),
        ("max_board_limit", "max_board_limit"),
        ("material_scale", "material_scale"),
        ("temperature", "temperature"),
        ("temperature_final", "temperature_final"),
        ("temperature_threshold", "temperature_threshold"),
        ("full_search_probability", "full_search_probability"),
        ("fast_sims", "fast_simulations"),
        ("tree_min_visits", "record_tree_min_visits"),
        ("tree_max_depth", "record_tree_max_depth"),
        ("tree_max_per_turn", "record_tree_max_per_turn"),
        ("tree_sample_prob", "record_tree_sample_probability"),
        ("workers", "workers"),
        ("selfplay_backend", "backend"),
        ("cpp_onnx_provider", "cpp_onnx_provider"),
        ("num_server_threads", "num_server_threads"),
        ("selfplay_processes", "processes"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(self_play, field_name, value)
    if args.no_playout_cap:
        self_play.playout_cap_randomization = False
    if args.record_games:
        self_play.record_games = True
    if args.record_tree_positions is not None:
        self_play.record_tree_positions = args.record_tree_positions
    config.self_play = self_play

    for arg_name, field_name in (
        ("sims", "num_simulations"),
        ("c_puct", "c_puct"),
        ("dirichlet_alpha", "dirichlet_alpha"),
        ("dirichlet_epsilon", "dirichlet_epsilon"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config.mcts, field_name, value)
    config.__post_init__()
    return config


def _resolve_resume(
    config: TrainConfig,
    resume: str | None,
    *,
    check_geometry: bool = False,
) -> Path | None:
    """Resolve ``--resume`` and adopt the checkpoint's model geometry.

    With ``check_geometry`` (the user passed explicit network flags), a
    mismatch raises instead of silently replacing those flags.
    """

    if resume is None:
        return None
    if resume == "latest":
        path = RunPaths.from_config(config).latest
        if not path.exists():
            raise SystemExit(f"no latest checkpoint at {path}")
    else:
        path = Path(resume)
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")

    extra = _read_checkpoint(path)
    saved_network = NetworkConfig(**extra.get("network_config", {}))
    saved_variant = str(extra.get("variant", config.variant))
    if config.variant != saved_variant:
        raise SystemExit(
            f"checkpoint variant {saved_variant!r} does not match --variant "
            f"{config.variant!r}"
        )
    if check_geometry and saved_network != config.network:
        raise SystemExit(
            "explicit network options do not match the checkpoint geometry; "
            "omit them when resuming"
        )
    config.network = saved_network
    return path


def _merge_summaries(
    combined: dict[str, object] | None, summary: dict[str, object]
) -> dict[str, object]:
    """Accumulate per-batch summaries into one run summary.

    The per-iteration records are extended; the cumulative counters simply
    take the newest batch's value because the trainer loads them from the
    checkpoint it resumed from.
    """

    if combined is None:
        return dict(summary)
    iterations = combined.get("iterations")
    if isinstance(iterations, list):
        iterations.extend(summary.get("iterations") or [])  # type: ignore[arg-type]
    for key in (
        "end_iteration",
        "total_games",
        "total_samples",
        "buffer_size",
        "onnx",
        "stopped_early",
    ):
        if key in summary:
            combined[key] = summary[key]
    return combined


def _run_training_batches(
    config: TrainConfig,
    resume_path: Path | None,
    paths: RunPaths,
    *,
    continuous: bool,
) -> dict[str, object]:
    """Run one batch, or keep resuming until the user asks to stop.

    Each batch is exactly ``config.num_iterations`` iterations and continues
    from the checkpoint the previous batch published as ``latest.pt``.  The
    first cooperative Ctrl+C sets ``stopped_early`` on the batch summary, which
    ends the loop instead of starting another batch.
    """

    if continuous and config.num_iterations <= 0:
        raise SystemExit("--continuous requires --iterations > 0")

    combined: dict[str, object] | None = None
    batch = 0
    archive_writer = ArchiveWriter() if config.self_play.record_games else None
    try:
        if archive_writer is not None:
            archive_writer.start()
        while True:
            batch += 1
            if batch > 1:
                logger.info(
                    "continuous: starting batch %d from %s", batch, resume_path
                )
            if archive_writer is None:
                summary = run_training(config, resume=resume_path)
            else:
                summary = run_training(
                    config,
                    resume=resume_path,
                    archive_writer=archive_writer,
                )
            combined = _merge_summaries(combined, summary)
            _report_summary(combined, paths.log_dir / "summary.json")
            if not continuous or summary.get("stopped_early"):
                return combined
            if not paths.latest.exists():
                logger.warning(
                    "continuous: batch %d wrote no checkpoint; stopping", batch
                )
                return combined
            resume_path = paths.latest
    finally:
        if archive_writer is not None:
            if sys.exc_info()[0] is None:
                archive_writer.close()
                logger.info(
                    "all queued game archives saved to %s",
                    paths.gn5d_log_dir / "games",
                )
            else:
                archive_writer.abort()


def cmd_train(args: argparse.Namespace) -> int:
    config = _build_config(args)
    geometry_flags = (
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "action_mlp_hidden",
        "max_params",
    )
    explicit_geometry = any(
        getattr(args, name, None) is not None for name in geometry_flags
    )
    resume_path = _resolve_resume(
        config, args.resume, check_geometry=explicit_geometry
    )
    paths = RunPaths.from_config(config)
    _configure_logging(paths.log_dir)
    try:
        _run_training_batches(
            config, resume_path, paths, continuous=args.continuous
        )
    except NotImplementedError as exc:
        # Keep unimplemented interface switches from dumping a traceback.
        raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
        # The first Ctrl+C is handled cooperatively inside run_training; this
        # is the second press, which abandons the in-flight iteration.
        print("aborted by Ctrl+C; the previous checkpoint is still valid", file=sys.stderr)
        return 130
    return 0


def _report_summary(summary: dict[str, object], path: Path) -> None:
    """Write the full run summary to disk and log a short recap.

    The JSON grows by one record per iteration, so dumping it to the console
    buries the last few log lines of a long run.  It lands in
    ``<log_dir>/summary.json`` instead, and the console gets three lines.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    iterations = summary.get("iterations") or []
    last = iterations[-1] if iterations else {}
    logger.info(
        "run finished: iterations %s->%s (%d completed) | games=%s samples=%s buffer=%s%s",
        summary.get("start_iteration"),
        summary.get("end_iteration"),
        len(iterations),  # type: ignore[arg-type]
        summary.get("total_games"),
        summary.get("total_samples"),
        summary.get("buffer_size"),
        " | stopped early" if summary.get("stopped_early") else "",
    )
    metrics = _last_metrics(last)
    if metrics:
        logger.info(
            "last iteration loss=%s policy=%s value=%s",
            metrics.get("loss"),
            metrics.get("policy_loss"),
            metrics.get("value_loss"),
        )
    logger.info("summary JSON: %s", path)


def _last_metrics(record: object) -> dict[str, object]:
    if not isinstance(record, dict):
        return {}
    training = record.get("training")
    if not isinstance(training, dict):
        return {}
    metrics = training.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def cmd_export(args: argparse.Namespace) -> int:
    spec = resolve_variant(args.variant)
    network = load_network_checkpoint(args.checkpoint, spec, "cpu")
    if args.d_model is not None:
        # The checkpoint's geometry wins; this flag is only a sanity check.
        if network.config.d_model != args.d_model:
            raise SystemExit(
                f"checkpoint d_model={network.config.d_model}, requested "
                f"{args.d_model}"
            )
    output = export_network(network, spec, args.output, verify=not args.no_verify)
    print(output)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    spec = resolve_variant(args.variant)
    network_a = load_network_checkpoint(args.checkpoint_a, spec, args.device)
    network_b = load_network_checkpoint(args.checkpoint_b, spec, args.device)
    from .config import MctsConfig

    result = play_match(
        network_a,
        network_b,
        spec,
        MctsConfig(num_simulations=args.sims, dirichlet_epsilon=0.0),
        games=args.games,
        device_a=args.device,
        device_b=args.device,
        seed=args.seed,
        board_limit=args.board_limit,
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    from .match import MatchPlayer, material_average_score, play_model_match

    def player(side: str) -> MatchPlayer:
        return MatchPlayer(
            model=getattr(args, f"model_{side}"),
            name=getattr(args, f"name_{side}"),
            simulations=getattr(args, f"sims_{side}"),
            c_puct=getattr(args, f"c_puct_{side}"),
            max_depth=getattr(args, f"max_depth_{side}"),
            temperature=getattr(args, f"temperature_{side}"),
            provider=getattr(args, f"provider_{side}"),
            cuda_device_id=getattr(args, f"cuda_device_{side}"),
            line_shift=getattr(args, f"line_shift_{side}"),
            reverse_move_order=getattr(args, f"reverse_moves_{side}"),
        )

    def progress(done: int, game: dict) -> None:
        status = "truncated" if game["truncated"] else game["terminal_reason"]
        print(f"match {done}/{args.games}: {len(game['moves'])} actions, {status}",
              file=sys.stderr, flush=True)

    if args.output is not None and Path(args.output).exists():
        raise SystemExit(f"output already exists: {args.output}")
    try:
        result = play_model_match(
            player("a"), player("b"), variant=args.variant, games=args.games,
            max_moves=args.max_moves, board_limit=args.board_limit,
            material_scale=args.material_scale, seed=args.seed, progress=progress,
            workers=args.workers,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
        print("match interrupted; incomplete series not reported", file=sys.stderr)
        return 130
    if args.output is not None:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    summary = {key: value for key, value in result.items() if key != "games"}
    summary["games_played"] = len(result["games"])
    material_scores = [material_average_score(result, index) for index in range(2)]
    summary["material_average_scores"] = material_scores
    if any(score is not None for score in material_scores):
        labels = result["participants"]
        formatted = " ".join(
            f"{labels[index]}={score:+.4f}" if score is not None
            else f"{labels[index]}=N/A"
            for index, score in enumerate(material_scores)
        )
        print(f"material average score: {formatted}", file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="suzero-5dc")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run AlphaZero iterations")
    _add_training_options(train)
    train.set_defaults(func=cmd_train)

    export = sub.add_parser("export", help="export a checkpoint to ONNX")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--variant", choices=sorted(VARIANTS), default="very_small")
    export.add_argument("--d-model", type=int, default=None)
    export.add_argument("--no-verify", action="store_true")
    export.set_defaults(func=cmd_export)

    evaluate = sub.add_parser("evaluate", help="play two checkpoints against each other")
    evaluate.add_argument("--checkpoint-a", required=True)
    evaluate.add_argument("--checkpoint-b", required=True)
    evaluate.add_argument("--games", type=int, default=20)
    evaluate.add_argument("--sims", type=int, default=50)
    evaluate.add_argument("--variant", choices=sorted(VARIANTS), default="very_small")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--seed", type=int, default=20261021)
    evaluate.add_argument(
        "--board-limit",
        type=int,
        default=None,
        help="board-count material limit (default: variant setting)",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    match = sub.add_parser("match", help="paired C++ matches of ONNX models or checkpoints")
    for side in ("a", "b"):
        match.add_argument(f"--model-{side}", required=True, help=".onnx (+ .json), .pt or .pth")
        match.add_argument(f"--name-{side}", default=side.upper())
        match.add_argument(f"--sims-{side}", type=int, default=100)
        match.add_argument(f"--c-puct-{side}", type=float, default=1.0)
        match.add_argument(f"--max-depth-{side}", type=int, default=128)
        match.add_argument(f"--temperature-{side}", type=float, default=0.0)
        match.add_argument(f"--provider-{side}", choices=("cpu", "cuda"), default="cpu")
        match.add_argument(f"--cuda-device-{side}", type=int, default=0)
        match.add_argument(f"--line-shift-{side}", type=int, default=None,
                           help="input encoding override (default: model metadata)")
        match.add_argument(f"--reverse-moves-{side}", action="store_true",
                           help="reverse non-SUBMIT local action order")
    match.add_argument("--variant", choices=sorted(VARIANTS), default="very_small")
    match.add_argument("--games", type=int, default=2, help="positive even number")
    match.add_argument("--workers", type=int, default=1,
                       help="concurrent C++ games (default: 1)")
    match.add_argument(
        "--max-moves",
        type=int,
        default=0,
        help="cap on all actions including SUBMIT; 0 disables the cap (default)",
    )
    match.add_argument("--board-limit", type=int, default=None)
    match.add_argument("--material-scale", type=float, default=2.0)
    match.add_argument("--seed", type=int, default=20261023)
    match.add_argument("--output", default=None, help="new JSON file for full results and move traces")
    match.set_defaults(func=cmd_match)
    return parser


def _console_protection() -> Callable[[], None] | None:
    """Optionally disable QuickEdit for unattended runs.

    Off by default: without QuickEdit the legacy console can no longer select
    or copy text with the mouse. ``console_sink`` already keeps a paused
    console from stalling training, so this is only for runs whose console is
    never read.
    """

    if os.environ.get("SUZERO_DISABLE_QUICK_EDIT", "") != "1":
        return None
    restore = disable_quick_edit()
    if restore is not None:
        # Logging is not configured yet, and this must stay visible anyway.
        print(
            "[INFO] QuickEdit disabled for this run (SUZERO_DISABLE_QUICK_EDIT=1); "
            "copy with Alt+Space > 编辑 > 标记",
            file=sys.stderr,
            flush=True,
        )
    return restore


def main(argv: list[str] | None = None) -> int:
    restore_console_mode = _console_protection()
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        return int(args.func(args))
    finally:
        if restore_console_mode is not None:
            restore_console_mode()


if __name__ == "__main__":
    sys.exit(main())
