"""Training-time paired matches against exact-age and fixed opponents."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from .config import TrainConfig
from .env import _5dchess, _require_binding
from .evaluation import load_network_checkpoint
from .export_onnx import export_network
from .game_record import save_game_record
from .match import (
    MatchPlayer,
    build_match_record_result,
    material_average_score,
    play_model_match,
)
from .network import SuzeroNetwork
from .protocol import resolve_variant

logger = logging.getLogger(__name__)

# Periodic matches sample at a low temperature instead of taking a pure argmax.
# This keeps every measurement slightly noisy (and reproducible from the game
# seed) without the wide sampling of self-play's opening temperature.
_MATCH_TEMPERATURE = 0.1


def _win_rate(stats: dict[str, int]) -> float | None:
    """Return wins divided by normal terminal games, excluding truncations."""
    finished = stats["wins"] + stats["losses"] + stats["draws"]
    return stats["wins"] / finished if finished else None


def _win_rates_by_color(by_color: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, float | None]]:
    """Return per-participant white/black win rates in match input order."""
    return {
        participant: {
            color: _win_rate(by_color[index][color])
            for color in ("white", "black")
        }
        for index, participant in enumerate(("current", "opponent"))
    }


class PeriodicModelMatches:
    def __init__(self, config: TrainConfig, root: Path, log_dir: Path,
                 gn5d_log_dir: Path | None = None) -> None:
        self.config = config
        self.root = root
        self.directory = root / "matches" / "checkpoints"
        self.log_dir = (gn5d_log_dir or log_dir) / "matches"
        if config.model_match_interval:
            _require_binding()
            if not hasattr(_5dchess, "run_match"):
                raise RuntimeError("rebuild suzero_5dchess_py to enable periodic matches")
            reference = config.model_match_reference
            if reference:
                if not Path(reference).is_file():
                    raise FileNotFoundError(reference)
                if Path(reference).suffix.lower() != ".onnx":
                    # Fail before self-play if the requested checkpoint cannot
                    # represent these rules. Loading must not perturb training RNG.
                    with torch.random.fork_rng(devices=[]):
                        load_network_checkpoint(reference, resolve_variant(config.variant))

    def checkpoint(self, iteration: int) -> Path:
        return self.directory / f"agent_iter_{iteration:04d}.pt"

    def snapshot(self, network: SuzeroNetwork, iteration: int) -> None:
        if not self.config.model_match_interval:
            return
        path = self.checkpoint(iteration)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        torch.save({"model": network.state_dict(), "extra_state": {
            "variant": self.config.variant,
            "network_config": asdict(network.config),
            "iteration": iteration,
        }}, tmp)
        os.replace(tmp, path)

    def run(self, network: SuzeroNetwork, iteration: int) -> dict | None:
        cfg = self.config
        if not cfg.model_match_interval or iteration % cfg.model_match_interval:
            return None
        # Network construction during checkpoint export initializes parameters;
        # evaluation must not change the subsequent training random stream.
        with torch.random.fork_rng(devices=[]):
            return self._run(network, iteration)

    def _run(self, network: SuzeroNetwork, iteration: int) -> dict:
        cfg = self.config
        spec = resolve_variant(cfg.variant)
        target = iteration - cfg.model_match_lookback
        previous = self.checkpoint(target)
        if not previous.is_file():
            previous = self.root / f"agent_iter_{target:04d}.pt"
        opponents = []
        reports = {}
        if target >= 0 and previous.is_file():
            opponents.append(("previous", previous, target))
        else:
            reports["previous"] = {
                "skipped": True, "opponent_iteration": target,
                "reason": "exact historical checkpoint unavailable",
            }
            logger.warning("model match: iteration %d unavailable; skipping history only", target)
        if cfg.model_match_reference:
            opponents.append(("fixed", Path(cfg.model_match_reference), None))
        if not opponents:
            return reports

        import tempfile

        self.log_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="suzero-periodic-match-") as directory:
            current = export_network(network, network.spec, Path(directory) / "current.onnx")
            for index, (label, opponent, opponent_iteration) in enumerate(opponents):
                started = time.monotonic()
                logger.info("model match %s: %d games, %d workers, provider=%s",
                            label, cfg.model_match_games,
                            min(cfg.model_match_workers, cfg.model_match_games),
                            cfg.model_match_provider)
                result = play_model_match(
                    MatchPlayer(current, name=f"current_iter_{iteration}",
                                simulations=cfg.model_match_simulations,
                                c_puct=cfg.mcts.c_puct, max_depth=cfg.mcts.max_depth,
                                temperature=_MATCH_TEMPERATURE,
                                provider=cfg.model_match_provider),
                    MatchPlayer(opponent, name=label,
                                simulations=cfg.model_match_simulations,
                                c_puct=cfg.mcts.c_puct, max_depth=cfg.mcts.max_depth,
                                temperature=_MATCH_TEMPERATURE,
                                provider=cfg.model_match_provider),
                    variant=cfg.variant, games=cfg.model_match_games,
                    workers=cfg.model_match_workers,
                    max_moves=0,
                    board_limit=cfg.model_match_max_boards,
                    material_scale=cfg.self_play.material_scale,
                    seed=cfg.seed + iteration * 1009 + index,
                )
                gn5d_records = []
                for game_id, game in enumerate(result["games"], start=1):
                    record_result = build_match_record_result(
                        game,
                        spec,
                        board_limit=cfg.model_match_max_boards,
                        material_scale=cfg.self_play.material_scale,
                    )
                    gn5d_records.append(str(save_game_record(
                        record_result,
                        spec,
                        self.log_dir,
                        iteration=iteration,
                        game_id=game_id,
                        backend=f"model_match_{label}",
                        material_scale=cfg.self_play.material_scale,
                    )))
                total = result["totals"][0]
                win_rate = _win_rate(total)
                score_rate = (
                    (total["wins"] + 0.5 * total["draws"]) /
                    (total["wins"] + total["losses"] + total["draws"])
                    if total["wins"] + total["losses"] + total["draws"] else None
                )
                win_rates_by_color = _win_rates_by_color(result["by_color"])
                material_score = material_average_score(result)
                material_games = sum(
                    game["terminal"] and game["terminal_reason"] == "material"
                    for game in result["games"]
                )
                report = {
                    "iteration": iteration, "opponent": str(opponent),
                    "opponent_iteration": opponent_iteration,
                    "seconds": time.monotonic() - started,
                    "totals": result["totals"], "by_color": result["by_color"],
                    "win_rate": win_rate,
                    "win_rate_by_color": win_rates_by_color,
                    "score_rate": score_rate,
                    "material_average_score": material_score,
                    "material_games": material_games,
                    "games": cfg.model_match_games,
                    "workers": min(cfg.model_match_workers, cfg.model_match_games),
                    "simulations": cfg.model_match_simulations,
                    "temperature": _MATCH_TEMPERATURE,
                    "board_limit": cfg.model_match_max_boards,
                    "provider": cfg.model_match_provider,
                    "gn5d_records": gn5d_records,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                output = self.log_dir / f"iter_{iteration:04d}_{label}_{uuid4().hex}.json"
                with output.open("x", encoding="utf-8") as handle:
                    json.dump({"summary": report, "match": result}, handle,
                              ensure_ascii=False, indent=2)
                    handle.write("\n")
                report["output"] = str(output)
                reports[label] = report
                current_color_rates = win_rates_by_color["current"]
                opponent_color_rates = win_rates_by_color["opponent"]
                format_rate = lambda value: (
                    f"{value:.1%}" if value is not None else "N/A"
                )
                format_score = lambda value: (
                    f"{value:+.4f}" if value is not None else "N/A"
                )
                opponent_description = (
                    f"historical model at iteration {opponent_iteration}"
                    if opponent_iteration is not None else "fixed reference model"
                )
                logger.info(
                    "model match %s completed: current model at iteration %d vs %s",
                    label, iteration, opponent_description,
                )
                logger.info(
                    "  current model result: %d wins, %d losses, %d draws, "
                    "%d truncated games",
                    total["wins"], total["losses"], total["draws"],
                    total["truncated"],
                )
                logger.info(
                    "  win rate over normally finished games "
                    "(truncated games excluded): %s",
                    format_rate(win_rate),
                )
                logger.info(
                    "  win rate by color over normally finished games: "
                    "current model as White=%s, as Black=%s; "
                    "opponent as White=%s, as Black=%s",
                    format_rate(current_color_rates["white"]),
                    format_rate(current_color_rates["black"]),
                    format_rate(opponent_color_rates["white"]),
                    format_rate(opponent_color_rates["black"]),
                )
                logger.info(
                    "  average score over %d material-ended games "
                    "(current model perspective; positive means ahead): %s",
                    material_games, format_score(material_score),
                )
                logger.info("  %d GN5D records saved below %s",
                            len(gn5d_records), self.log_dir / "games")
                logger.info("  full match result: %s", output)
        return reports
