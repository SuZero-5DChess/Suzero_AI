"""Model-vs-model evaluation for 5D chess checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import MctsConfig, NetworkConfig
from .env import GameEnv
from .mcts import MCTS, TorchEvaluator
from .network import SuzeroNetwork, build_network
from .protocol import VariantSpec


@dataclass
class MatchResult:
    games: int = 0
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    a_score: float = 0.0
    terminal_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "draws": self.draws,
            "a_score": self.a_score,
            "terminal_reasons": dict(self.terminal_reasons),
        }


def load_network_checkpoint(
    path: str | Path,
    spec: VariantSpec,
    device: torch.device | str = "cpu",
) -> SuzeroNetwork:
    """Load a training checkpoint written by this package."""

    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        from .legacy_model import load_legacy_checkpoint

        return load_legacy_checkpoint(state, spec).to(device)
    if "model" not in state:
        raise ValueError(f"{path!r} is not a suzero_5dchess checkpoint")
    extra = state.get("extra_state", {})
    variant = extra.get("variant", spec.name)
    if variant != spec.name:
        raise ValueError(
            f"checkpoint variant {variant!r} does not match {spec.name!r}"
        )
    network_config = NetworkConfig(**extra.get("network_config", {}))
    network = build_network(network_config, spec)
    network.load_state_dict(state["model"])
    network.to(device)
    network.eval()
    return network


def _play_eval_game(
    white_network: SuzeroNetwork,
    black_network: SuzeroNetwork,
    spec: VariantSpec,
    mcts_config: MctsConfig,
    *,
    device_white: torch.device | str,
    device_black: torch.device | str,
    seed: int,
    board_limit: int | None,
) -> tuple[float, str]:
    rng = np.random.default_rng(seed)
    white_search = MCTS(
        TorchEvaluator(white_network, device_white), mcts_config, rng=rng
    )
    black_search = MCTS(
        TorchEvaluator(black_network, device_black), mcts_config, rng=rng
    )
    env = GameEnv(spec, board_limit=board_limit)
    while not env.is_terminal:
        if not env.actions:
            return (-1.0 if env.current_player == 0 else 1.0), "no_legal_action"
        if env.forced_submit:
            env.apply_action(0)
            continue
        search = white_search if env.current_player == 0 else black_search
        root = search.run(env, add_noise=False)
        action = search.choose_action(root, 0.0, rng)
        env.apply_action(action)
    return env.white_outcome, env.terminal_reason


def play_match(
    network_a: SuzeroNetwork,
    network_b: SuzeroNetwork,
    spec: VariantSpec,
    mcts_config: MctsConfig,
    *,
    games: int,
    device_a: torch.device | str = "cpu",
    device_b: torch.device | str = "cpu",
    seed: int = 20261021,
    board_limit: int | None = None,
) -> MatchResult:
    if games <= 0 or games % 2 != 0:
        raise ValueError("evaluation games must be a positive even number")
    if board_limit is not None and board_limit < 1:
        raise ValueError("evaluation board_limit must be positive")
    result = MatchResult(games=games)
    network_a.eval()
    network_b.eval()
    for index in range(games):
        a_is_white = index % 2 == 0
        if a_is_white:
            white, black = network_a, network_b
            device_white, device_black = device_a, device_b
        else:
            white, black = network_b, network_a
            device_white, device_black = device_b, device_a
        outcome, reason = _play_eval_game(
            white,
            black,
            spec,
            mcts_config,
            device_white=device_white,
            device_black=device_black,
            seed=seed + index * 7919,
            board_limit=board_limit,
        )
        result.terminal_reasons[reason] = result.terminal_reasons.get(reason, 0) + 1
        a_outcome = outcome if a_is_white else -outcome
        result.a_score += a_outcome
        if a_outcome > 0:
            result.a_wins += 1
        elif a_outcome < 0:
            result.b_wins += 1
        else:
            result.draws += 1
    result.a_score /= games
    return result
