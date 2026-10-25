"""Independent C++ model matches; the training evaluator stays unchanged."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

from .env import GameEnv, _5dchess, _require_binding
from .protocol import VariantSpec, resolve_variant


@dataclass(frozen=True)
class MatchPlayer:
    """An ONNX model (+ sidecar) or a trusted local training checkpoint."""

    model: str | Path
    name: str = ""
    simulations: int = 100
    c_puct: float = 1.0
    max_depth: int = 128
    temperature: float = 0.0
    provider: str = "cpu"
    cuda_device_id: int = 0
    intra_op_threads: int = 1
    metadata: str | Path | None = None
    line_shift: int | None = None
    reverse_move_order: bool = False


@dataclass
class RecordableMatchResult:
    """Match game fields consumed by the shared GN5D record builder."""

    white_outcome: float
    terminal_reason: str
    semimoves: int
    turns: int
    board_limit: int
    board_count: int
    final_player: int
    moves: list[dict[str, object]]
    final_fen: str


def _decode_match_action(raw: object, index: int) -> tuple[bool, tuple[int, ...] | None]:
    if not isinstance(raw, str):
        raise ValueError(f"match game move {index}: action must be a string")
    fields = raw.split(":")
    if fields == ["5dc-action-v1", "SUBMIT"]:
        return True, None
    if len(fields) != 10 or fields[:2] != ["5dc-action-v1", "MOVE"]:
        raise ValueError(f"match game move {index}: unsupported action encoding")
    try:
        move = tuple(int(value) for value in fields[2:])
    except ValueError as exc:
        raise ValueError(f"match game move {index}: invalid move coordinate") from exc
    return False, move


def build_match_record_result(
    game: Mapping[str, object],
    spec: VariantSpec,
    *,
    board_limit: int,
    material_scale: float,
) -> RecordableMatchResult:
    """Replay one C++ match result into fields required by GN5D archiving."""

    env = GameEnv(spec, board_limit=board_limit, material_scale=material_scale)
    raw_moves = game.get("moves")
    if not isinstance(raw_moves, list):
        raise ValueError("match game moves must be a list")

    moves: list[dict[str, object]] = []
    semimoves = 0
    turns = 0
    for index, raw_move in enumerate(raw_moves):
        if not isinstance(raw_move, Mapping):
            raise ValueError(f"match game move {index}: expected an object")
        player = raw_move.get("player")
        if type(player) is not int or player not in (0, 1):
            raise ValueError(f"match game move {index}: invalid player")
        if player != env.current_player:
            raise ValueError(f"match game move {index}: player differs from native replay")
        is_submit, move = _decode_match_action(raw_move.get("action"), index)
        candidates = [
            action_id
            for action_id, (candidate_submit, candidate_move) in enumerate(env.actions)
            if candidate_submit == is_submit and (is_submit or candidate_move == move)
        ]
        if len(candidates) != 1:
            raise ValueError(f"match game move {index}: action is not uniquely legal")
        if is_submit:
            moves.append({"type": "submit", "player": player})
            turns += 1
        else:
            assert move is not None
            moves.append({
                "type": "move",
                "player": player,
                "from": list(move[:4]),
                "to": list(move[4:]),
                "promotion": "queen",
            })
            semimoves += 1
        env.apply_action(candidates[0])

    terminal = game.get("terminal")
    truncated = game.get("truncated")
    if type(terminal) is not bool or type(truncated) is not bool:
        raise ValueError("match game terminal flags must be booleans")
    if terminal != env.is_terminal or truncated != (not env.is_terminal):
        raise ValueError("match game terminal flags differ from native replay")
    if game.get("final_player") != env.current_player:
        raise ValueError("match game final player differs from native replay")
    if terminal and game.get("terminal_reason") != env.terminal_reason:
        raise ValueError("match game terminal reason differs from native replay")

    final_utility = game.get("final_utility")
    if terminal:
        if type(final_utility) not in (int, float) or not math.isclose(
            float(final_utility), env.terminal_utility, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("match game terminal utility differs from native replay")
        white_outcome = env.white_outcome
        terminal_reason = env.terminal_reason
    else:
        if final_utility is not None:
            raise ValueError("truncated match game must not have terminal utility")
        white_outcome = 0.0
        terminal_reason = "max_game_length"

    snapshot = env.record_snapshot()
    return RecordableMatchResult(
        white_outcome=white_outcome,
        terminal_reason=terminal_reason,
        semimoves=semimoves,
        turns=turns,
        board_limit=env.board_limit,
        board_count=env.board_count,
        final_player=env.current_player,
        moves=moves,
        final_fen=str(snapshot["fen"]),
    )


def material_average_score(result: dict, participant_index: int = 0) -> float | None:
    """Average utility for one participant over material-scored terminal games."""
    scores = [
        game["participant_utilities"][participant_index]
        for game in result["games"]
        if game["terminal"]
        and game["terminal_reason"] == "material"
        and game["participant_utilities"][participant_index] is not None
    ]
    return sum(scores) / len(scores) if scores else None


def _validate_player(player: MatchPlayer) -> None:
    if player.simulations <= 0 or player.max_depth <= 0:
        raise ValueError("simulations and max_depth must be positive")
    if not math.isfinite(player.c_puct) or player.c_puct < 0:
        raise ValueError("c_puct must be finite and non-negative")
    if not math.isfinite(player.temperature) or player.temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if player.provider not in ("cpu", "cuda"):
        raise ValueError("provider must be 'cpu' or 'cuda'")
    if player.cuda_device_id < 0 or player.intra_op_threads <= 0:
        raise ValueError("invalid CUDA device or intra-op thread count")
    path = Path(player.model)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in (".onnx", ".pt", ".pth"):
        raise ValueError("model must be .onnx, .pt or .pth")
    if path.suffix.lower() != ".onnx" and player.metadata is not None:
        raise ValueError("metadata override is only supported for ONNX models")


def _prepare_model(player: MatchPlayer, spec: VariantSpec, output: Path) -> MatchPlayer:
    if Path(player.model).suffix.lower() == ".onnx":
        return player
    # Export each checkpoint independently: architectures need not agree.
    from .evaluation import load_network_checkpoint
    from .export_onnx import export_network

    network = load_network_checkpoint(player.model, spec, "cpu")
    export_network(network, network.spec, output)
    return replace(player, model=output)


def _make_bot(player: MatchPlayer, spec: VariantSpec, board_limit: int, material_scale: float):
    metadata = Path(player.metadata or (str(player.model) + ".json"))
    info = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError("model metadata must be a JSON object")
    if info.get("variant", spec.name) != spec.name:
        raise ValueError("model metadata variant does not match the match rules")
    line_shift = player.line_shift
    if line_shift is None:
        line_shift = info.get("line_shift", spec.line_shift)
    if not isinstance(line_shift, int) or isinstance(line_shift, bool):
        raise ValueError("line_shift must be an integer")

    bot = _5dchess.MatchBot()
    bot.name = player.name
    bot.game.variant_pgn = spec.pgn
    bot.game.board_side = spec.board_side
    bot.game.board_squares = spec.board_squares
    bot.game.piece_channels = spec.piece_channels
    bot.game.line_shift = line_shift
    bot.game.board_limit = board_limit
    bot.game.material_scale = material_scale
    bot.evaluator.model_path = str(Path(player.model).resolve())
    bot.evaluator.metadata_path = str(metadata.resolve())
    bot.evaluator.provider = player.provider
    bot.evaluator.cuda_device_id = player.cuda_device_id
    bot.evaluator.intra_op_threads = player.intra_op_threads
    bot.evaluator.board_side = spec.board_side
    bot.evaluator.board_squares = spec.board_squares
    bot.evaluator.piece_channels = spec.piece_channels
    bot.search.max_playouts = player.simulations
    bot.search.max_depth = player.max_depth
    bot.search.c_puct = player.c_puct
    bot.search.temperature = player.temperature
    bot.search.temperature_early = player.temperature
    bot.reverse_move_order = player.reverse_move_order
    return bot


def play_model_match(
    a: MatchPlayer,
    b: MatchPlayer,
    *,
    variant: str | VariantSpec = "very_small",
    games: int = 2,
    max_moves: int = 0,
    board_limit: int | None = None,
    material_scale: float = 2.0,
    seed: int = 20261023,
    workers: int = 1,
    progress: Callable[[int, dict], None] | None = None,
) -> dict:
    """Play complete color-swapped pairs with independent search settings.

    Every small step, including SUBMIT, counts toward an explicitly enabled
    ``max_moves`` cap; zero leaves termination to the game rules. Truncated
    games have ``None`` utilities and never enter the draw count. Checkpoints
    are exported in a temporary directory, without modifying training files.
    ``workers`` caps concurrent C++ games; each model has its own batching
    evaluator. Results retain scheduled order. Progress runs on the calling
    thread with a completion count and the finished game's original index.
    Ctrl+C and progress callbacks are checked between game completions.
    """

    if games <= 0 or games % 2 != 0 or games > 2**31 - 1:
        raise ValueError("games must be a positive even 32-bit integer")
    if max_moves < 0 or max_moves > 2**31 - 1:
        raise ValueError("max_moves must be a non-negative 32-bit integer")
    if workers <= 0 or workers > 2**31 - 1:
        raise ValueError("workers must be a positive 32-bit integer")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must fit an unsigned 64-bit integer")
    if not math.isfinite(material_scale) or material_scale <= 0:
        raise ValueError("material_scale must be finite and positive")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    spec = resolve_variant(variant)
    limit = spec.default_board_limit if board_limit is None else board_limit
    if limit <= 0:
        raise ValueError("board_limit must be positive")
    for player in (a, b):
        _validate_player(player)
    _require_binding()
    if not hasattr(_5dchess, "run_match"):
        raise RuntimeError("rebuild suzero_5dchess_py to enable model matches")

    with tempfile.TemporaryDirectory(prefix="suzero-match-") as directory:
        players = [
            _prepare_model(replace(player, name=player.name or label), spec,
                           Path(directory) / f"{label}.onnx")
            for player, label in ((a, "A"), (b, "B"))
        ]
        bots = [_make_bot(player, spec, limit, material_scale) for player in players]
        return _5dchess.run_match(
            bots[0], bots[1], pairs=games // 2, max_moves=max_moves,
            seed=seed, progress=progress, num_threads=workers,
        )
