"""Self-play through the C++ PUCT + ONNX Runtime bridge.

This is the ``cpp_onnx`` backend.  Every iteration:

1. export the live PyTorch network to ONNX (plus the sidecar metadata),
2. run several C++ game threads against that model (CPU or CUDA),
3. convert the returned arrays into :class:`PositionSample` objects so the
   rest of the trainer (replay buffer, objectives, checkpoints) is unchanged.

The heavy lifting — rules, search and network inference — stays in C++.  Only
the finished positions cross the Python boundary, which is why this backend
does not pay a Python round trip per playout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import MctsConfig, SelfPlayConfig
from .env import _5dchess, _require_binding
from .export_onnx import export_network
from .interrupts import stop_requested
from .network import SuzeroNetwork
from .protocol import VariantSpec
from .sample import PositionSample
from .sample_codec import planes_to_codes
from .self_play import SelfPlayResult

if TYPE_CHECKING:  # pragma: no cover - avoids a module cycle at runtime
    from .game_store import GameStore

logger = logging.getLogger(__name__)

# Keep the two newest exported models: ONNX Runtime may still hold a handle to
# the file it loaded, and a fresh path per iteration sidesteps that entirely.
_KEEP_MODELS = 2


def backend_available() -> bool:
    """True when the built extension exposes the self-play bridge."""

    return _5dchess is not None and hasattr(_5dchess, "run_selfplay")


def _export_model(
    network: SuzeroNetwork, spec: VariantSpec, model_path: Path
) -> Path:
    """Export the current network for the C++ evaluator.

    Verification is skipped deliberately: the C++ evaluator validates the ONNX
    signature and sidecar before the first inference, and a second CPU session
    here would only slow the iteration down.
    """

    return export_network(network, spec, model_path, verify=False)


def _prune_models(directory: Path, keep: int = _KEEP_MODELS) -> None:
    try:
        models = sorted(directory.glob("model_iter_*.onnx"))
    except OSError:
        return
    for stale in models[:-keep]:
        for path in (stale, Path(str(stale) + ".json")):
            try:
                path.unlink()
            except OSError:
                pass


def _samples_by_game(
    raw: dict[str, np.ndarray],
) -> dict[int, list[PositionSample]]:
    codes = raw.get("board_codes")
    planes = None if codes is not None else raw["board_planes"]
    l_coords = raw["l_coords"]
    t_coords = raw["t_coords"]
    used = raw["used_board_counts"]
    urgency = raw["urgency"]
    side = raw["side_to_move"]
    actions = raw["actions"]
    action_counts = raw["action_counts"]
    policy = raw["policy"]
    has_submit = raw["has_submit"]
    value = raw["value"]
    player = raw["player"]
    move_count = raw["move_count"]
    game_index = raw["game_index"]
    source = raw.get("sample_source")
    search_visits = raw.get("search_visits")
    if (source is None) != (search_visits is None):
        raise ValueError("self-play sample_source and search_visits must be supplied together")
    if source is not None:
        if (
            source.dtype != np.int64
            or source.shape != (len(used),)
            or not np.isin(source, (0, 1)).all()
            or search_visits.dtype != np.int64
            or search_visits.shape != (len(used),)
            or (search_visits < 0).any()
            or ((source == 1) & (search_visits == 0)).any()
        ):
            raise ValueError("invalid self-play tree-position metadata")

    board_source = codes if codes is not None else planes
    assert board_source is not None
    if codes is not None:
        flat = codes.ndim == 2 or "board_offsets" in raw or "action_offsets" in raw
        flat_boards = int(codes.shape[0]) if flat else 0
    else:
        flat = planes.ndim == 3 or "board_offsets" in raw or "action_offsets" in raw
        flat_boards = int(planes.shape[0]) if flat else 0
    if flat:
        for key, counts, length in (
            ("board_offsets", used, flat_boards),
            ("action_offsets", action_counts, actions.shape[0]),
        ):
            offsets = np.asarray(raw.get(key))
            if (
                offsets.dtype != np.int64
                or offsets.shape != (len(used) + 1,)
                or offsets[0] != 0
                or offsets[-1] != length
                or (np.diff(offsets) < 0).any()
                or not np.array_equal(np.diff(offsets), counts)
            ):
                raise ValueError(f"invalid self-play {key}")
        if (
            l_coords.shape != (flat_boards,)
            or t_coords.shape != (flat_boards,)
            or actions.shape != (actions.shape[0], 8)
            or policy.shape != (actions.shape[0],)
        ):
            raise ValueError("invalid flat self-play array shapes")

    grouped: dict[int, list[PositionSample]] = {}
    for i in range(len(used)):
        boards = int(used[i])
        submit = bool(has_submit[i])
        total = int(action_counts[i])
        candidates = total - int(submit)
        if candidates < 1:
            # The C++ loop never samples forced positions; guard anyway.
            continue
        if flat:
            board_start = int(raw["board_offsets"][i])
            action_start = int(raw["action_offsets"][i])
            board_row = board_source[board_start:board_start + boards]
            l_row = l_coords[board_start:board_start + boards]
            t_row = t_coords[board_start:board_start + boards]
            action_row = actions[action_start:action_start + candidates]
            policy_row = policy[action_start:action_start + total]
        else:
            # Accept older extensions and already-written worker payloads.
            board_row = board_source[i, :boards]
            l_row = l_coords[i, :boards]
            t_row = t_coords[i, :boards]
            action_row = actions[i, :candidates]
            policy_row = policy[i, :total]
        if codes is not None:
            board_codes = np.array(board_row, dtype=np.uint8)
        else:
            board_codes = planes_to_codes(board_row)
        sample = PositionSample(
            board_codes=board_codes,
            l_coords=np.array(l_row, dtype=np.int64),
            t_coords=np.array(t_row, dtype=np.int64),
            urgency=float(urgency[i]),
            side_to_move=int(side[i]),
            actions=np.array(action_row, dtype=np.int64),
            policy=np.array(policy_row, dtype=np.float32),
            has_submit=submit,
            value_target=float(value[i]),
            player=int(player[i]),
            move_count=int(move_count[i]),
            metadata=(
                {"tree_position": 1.0, "search_visits": float(search_visits[i])}
                if source is not None and source[i] == 1 else {}
            ),
        )
        grouped.setdefault(int(game_index[i]), []).append(sample)
    return grouped


def _result_from_entry(
    entry: dict[str, Any], samples: list[PositionSample]
) -> SelfPlayResult:
    """Build one result, retaining the sample count even without the rows.

    A streaming backend returns only game metadata to the parent, so the count
    embedded in ``entry["samples"]`` is what keeps summaries accurate.
    """

    count = len(samples) if samples else int(entry.get("samples", 0))
    return SelfPlayResult(
        samples=samples,
        white_outcome=float(entry["white_outcome"]),
        terminal_reason=str(entry["terminal_reason"]),
        semimoves=int(entry["semimoves"]),
        turns=int(entry["turns"]),
        board_limit=int(entry["board_limit"]),
        board_count=int(entry["board_count"]),
        final_player=int(entry["final_player"]),
        moves=list(entry.get("moves", [])),
        final_fen=str(entry.get("final_fen", "")),
        tree_samples=int(entry.get("tree_samples", 0)),
        sample_count=count,
    )


def _selfplay_results(raw: dict[str, Any]) -> list[SelfPlayResult]:
    """Convert one native result payload into per-game results."""

    entries = sorted(raw["games"], key=lambda item: int(item["index"]))
    if "used_board_counts" not in raw:
        # Streamed payload: samples already went to disk from the callback.
        return [_result_from_entry(entry, []) for entry in entries]
    grouped = _samples_by_game(raw)
    return [
        _result_from_entry(entry, grouped.get(int(entry["index"]), []))
        for entry in entries
    ]


def generate_games(
    network: SuzeroNetwork,
    spec: VariantSpec,
    mcts_config: MctsConfig,
    selfplay_config: SelfPlayConfig,
    *,
    device: str | object = "cpu",
    seed: int = 0,
    num_games: int | None = None,
    model_path: Path,
    progress: object | None = None,
    game_store: "GameStore | None" = None,
    game_id_start: int = 0,
    iteration: int = -1,
) -> list[SelfPlayResult]:
    """Play ``num_games`` games with the C++ bridge and return labelled games.

    ``model_path`` is the unique ONNX destination for this iteration; callers
    pass one path per iteration so a still-open model file is never replaced.

    With ``game_store`` set, the native bridge streams each finished game to
    Python, which encodes and writes it immediately.  The same path is used by
    human-facing backends and by the multiprocess children.
    """

    del device  # The C++ evaluator chooses its device via the provider.
    _require_binding()
    if not backend_available():
        raise RuntimeError(
            "the built _5dchess extension has no run_selfplay bridge; "
            "rebuild with -DSUZERO_BUILD_PYTHON_BINDINGS=ON"
        )

    games = selfplay_config.num_games if num_games is None else int(num_games)
    if games <= 0:
        return []

    if selfplay_config.record_games and getattr(_5dchess, "RECORD_FORMAT_VERSION", 0) != 1:
        raise RuntimeError(
            ".gn5d recording requires a rebuilt _5dchess extension; run "
            "cmake --build build --config Release --target suzero_5dchess_py"
        )
    if selfplay_config.record_tree_positions and getattr(
        _5dchess, "TREE_POSITION_FORMAT_VERSION", 0
    ) != 1:
        raise RuntimeError(
            "tree-position sampling requires a rebuilt _5dchess extension; run "
            "cmake --build build --config Release --target suzero_5dchess_py, "
            "or disable it with --no-record-tree-positions"
        )
    if game_store is not None and getattr(
        _5dchess, "GAME_STREAM_FORMAT_VERSION", 0
    ) != 1:
        raise RuntimeError(
            "per-game replay streaming requires a rebuilt _5dchess extension; "
            "run cmake --build build --config Release --target "
            "suzero_5dchess_py, or select --replay-backend memory"
        )
    model_path = _export_model(network, spec, model_path)
    metadata_path = Path(str(model_path) + ".json")
    if stop_requested():
        # Ctrl+C during the export must not start a fresh batch; the caller
        # saves the current progress and exits.
        logger.warning("Ctrl+C: self-play cancelled before any game started")
        _prune_models(model_path.parent)
        return []

    binding_kwargs: dict[str, object] = dict(
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        provider=selfplay_config.cpp_onnx_provider,
        cuda_device_id=0,
        variant_pgn=spec.pgn,
        board_side=spec.board_side,
        line_shift=spec.line_shift,
        num_games=games,
        num_threads=selfplay_config.workers,
        num_simulations=mcts_config.num_simulations,
        fast_simulations=selfplay_config.fast_simulations,
        playout_cap_randomization=selfplay_config.playout_cap_randomization,
        full_search_probability=selfplay_config.full_search_probability,
        c_puct=mcts_config.c_puct,
        dirichlet_alpha=mcts_config.dirichlet_alpha,
        dirichlet_epsilon=mcts_config.dirichlet_epsilon,
        max_depth=mcts_config.max_depth,
        temperature=selfplay_config.temperature,
        temperature_final=selfplay_config.temperature_final,
        temperature_threshold=selfplay_config.temperature_threshold,
        min_board_limit=selfplay_config.min_board_limit,
        max_board_limit=selfplay_config.max_board_limit,
        material_scale=selfplay_config.material_scale,
        max_batch_size=128,
        seed=int(seed) & 0xFFFFFFFFFFFFFFFF,
    )
    if selfplay_config.record_games:
        binding_kwargs["record_games"] = True
    if selfplay_config.record_tree_positions:
        for name in (
            "record_tree_positions", "record_tree_min_visits", "record_tree_max_depth",
            "record_tree_max_per_turn", "record_tree_sample_probability",
        ):
            binding_kwargs[name] = getattr(selfplay_config, name)
    requested_server_threads = selfplay_config.num_server_threads

    # ``processes > 1`` is a separate execution path: the model above is
    # exported once, then each child interpreter opens its own native queues
    # and ONNX session against that read-only model.  ``workers``,
    # ``num_server_threads`` and ``num_games`` stay totals and are split by
    # the child scheduler.  The original in-process binding call below is
    # untouched for the default ``processes == 1``.
    if int(getattr(selfplay_config, "processes", 1)) > 1:
        from .cpp_selfplay_processes import run_self_play_processes

        transport = None
        if game_store is not None:
            transport = game_store.child_transport(iteration)
            transport["game_id_start"] = int(game_id_start)
        runner_kwargs: dict[str, object] = {
            "processes": int(getattr(selfplay_config, "processes", 1)),
            "total_server_threads": int(requested_server_threads),
            "progress": progress if callable(progress) else None,
        }
        if transport is not None:
            runner_kwargs["replay_store"] = transport
        results = run_self_play_processes(binding_kwargs, **runner_kwargs)
        _prune_models(model_path.parent)
        return results

    def on_game_finished(done: int, _index: int) -> None:
        if callable(progress):
            progress(done)

    binding_kwargs["progress"] = (
        on_game_finished if callable(progress) else None
    )
    streamed: set[int] = set()
    if game_store is not None:
        def stream_game(done: int, index: int, raw: dict[str, Any]) -> None:
            del done
            converted = _selfplay_results(raw)
            if len(converted) != 1:
                raise RuntimeError(
                    "streamed self-play payload must contain exactly one game"
                )
            game_store.append_result(
                int(game_id_start) + int(index),
                converted[0],
                iteration=iteration,
            )
            streamed.add(int(index))

        binding_kwargs["stream_games"] = True
        binding_kwargs["on_game"] = stream_game
    # Older extensions built before num_server_threads was exposed do not
    # accept the keyword.  Omit it at the default so those builds keep
    # working, but never let a non-default choice be silently ignored.
    if requested_server_threads != 1:
        binding_kwargs["num_server_threads"] = requested_server_threads

    if stop_requested():
        logger.warning("Ctrl+C: self-play cancelled before any game started")
        _prune_models(model_path.parent)
        return []

    try:
        raw = _5dchess.run_selfplay(**binding_kwargs)
    except TypeError as exc:
        if (
            requested_server_threads != 1
            and "num_server_threads" in str(exc)
        ):
            raise RuntimeError(
                "the built _5dchess extension does not accept "
                f"num_server_threads={requested_server_threads}; rebuild it "
                "with -DSUZERO_BUILD_5DCHESS=ON "
                "-DSUZERO_BUILD_PYTHON_BINDINGS=ON to use that setting"
            ) from exc
        raise

    results = _selfplay_results(raw)
    if game_store is not None:
        for result_index, result in enumerate(results):
            if result_index in streamed:
                result.persisted = True
        if len(streamed) != len(results):
            # Without this, unstreamed samples would be silently dropped: the
            # array payload is omitted whenever streaming is requested.
            raise RuntimeError(
                f"native self-play streamed {len(streamed)} of {len(results)} "
                "games; refusing to continue with missing replay samples"
            )
    _log_throughput(raw, results, selfplay_config)
    _prune_models(model_path.parent)
    return results


def _log_throughput(
    raw: dict[str, Any],
    results: list[SelfPlayResult],
    selfplay_config: SelfPlayConfig,
) -> None:
    """Log samples/s and the batches served by the C++ evaluator.

    ``stats["rows"]`` and the derived ``avg_backend_batch`` count backend
    requests, including forced-only SUBMIT requests that may be resolved
    without a corresponding ONNX Runtime input row.  They are backend-side
    figures, not a measurement of the batch sizes actually fed to the
    GPU/ONNX Runtime. ``backend_rows/s`` includes intermediate search nodes;
    ``samples/s`` counts full-search roots and default-enabled tree positions.
    A small average backend batch can have several causes and should not be
    attributed to one.
    """

    stats = raw.get("stats", {})
    samples = sum(result.sample_count for result in results)
    total_seconds = float(stats.get("total_seconds", 0.0))
    setup_seconds = float(stats.get("setup_seconds", 0.0))
    play_seconds = max(total_seconds - setup_seconds, 0.0)
    batches = int(stats.get("batches", 0))
    rows = int(stats.get("rows", 0))
    avg_batch = rows / batches if batches else 0.0

    def rate(count: int) -> float:
        return count / play_seconds if play_seconds > 0.0 else 0.0

    tree_samples = sum(result.tree_samples for result in results)
    logger.info(
        "cpp self-play (%s): %d games, %d samples (%d played, %d tree) "
        "in %.1fs (setup %.1fs)",
        selfplay_config.cpp_onnx_provider,
        len(results),
        samples,
        samples - tree_samples,
        tree_samples,
        total_seconds,
        setup_seconds,
    )
    logger.info(
        "cpp self-play batching: %.1f samples/s, %.0f backend_rows/s | "
        "batches=%d rows=%d avg_backend_batch=%.1f max_backend_batch=%d/%d "
        "server_threads=%d",
        rate(samples),
        rate(rows),
        batches,
        rows,
        avg_batch,
        int(stats.get("max_observed_batch", 0)),
        int(stats.get("max_batch_size", 0)),
        int(stats.get("server_threads", 0)),
    )
