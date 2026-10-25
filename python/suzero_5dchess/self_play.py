"""AlphaZero self-play for 5D chess."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import MctsConfig, SelfPlayConfig
from .env import GameEnv
from .mcts import MCTS, TorchEvaluator
from .network import SuzeroNetwork
from .protocol import VariantSpec
from .sample import PositionSample, make_sample


@dataclass
class SelfPlayResult:
    samples: list[PositionSample]
    white_outcome: float
    terminal_reason: str
    semimoves: int
    turns: int
    board_limit: int
    board_count: int
    final_player: int
    moves: list[dict[str, object]] = field(default_factory=list)
    final_fen: str = ""


def _sample_board_limit(
    spec: VariantSpec, config: SelfPlayConfig, rng: np.random.Generator
) -> int:
    low = max(1, config.min_board_limit)
    high = max(low, config.max_board_limit)
    return int(rng.integers(low, high + 1))


def play_game(
    network: SuzeroNetwork,
    spec: VariantSpec,
    mcts_config: MctsConfig,
    selfplay_config: SelfPlayConfig,
    *,
    device: torch.device | str = "cpu",
    seed: int = 0,
    board_limit: int | None = None,
) -> SelfPlayResult:
    """Play one game and return labelled samples.

    Full-search moves contribute a training sample and receive Dirichlet root
    noise.  Fast moves (playout-cap randomization) neither add noise nor
    produce a sample, exactly like the reference pipeline.

    The caller is responsible for putting ``network`` in eval mode (as
    :func:`generate_games` does); otherwise dropout would perturb search.
    """

    rng = np.random.default_rng(seed)
    evaluator = TorchEvaluator(network, device)
    search = MCTS(evaluator, mcts_config, rng=rng)
    limit = (
        _sample_board_limit(spec, selfplay_config, rng)
        if board_limit is None
        else int(board_limit)
    )
    env = GameEnv(
        spec,
        board_limit=limit,
        material_scale=selfplay_config.material_scale,
    )

    if selfplay_config.record_games:
        # Fail before an expensive game if the installed binding is too old.
        env.record_snapshot()
    samples: list[PositionSample] = []
    history: list[dict[str, object]] = []
    semimoves = 0
    turns = 0

    while not env.is_terminal:
        if not env.actions:
            break

        if env.forced_submit:
            if selfplay_config.record_games:
                history.append({
                    "type": "submit", "player": env.current_player,
                    "kind": "forced_submit", "turn": turns + 1,
                })
            env.apply_action(0)
            turns += 1
            continue

        use_full = (
            not selfplay_config.playout_cap_randomization
            or rng.random() < selfplay_config.full_search_probability
        )
        simulations = (
            mcts_config.num_simulations
            if use_full
            else selfplay_config.fast_simulations
        )
        root = search.run(
            env,
            add_noise=use_full,
            num_simulations=simulations,
        )
        policy = root.visit_policy()
        player = env.current_player

        if use_full:
            samples.append(make_sample(env, policy, player))

        temperature = (
            selfplay_config.temperature
            if semimoves < selfplay_config.temperature_threshold
            else selfplay_config.temperature_final
        )
        action_id = search.choose_action(root, temperature, rng)
        is_submit, move_key = env.actions[action_id]

        if selfplay_config.record_games:
            entry: dict[str, object] = {
                "type": "submit" if is_submit else "move",
                "kind": "search",
                "turn": turns,
                "semimove": semimoves,
                "player": player,
                "action_id": action_id,
                "is_submit": is_submit,
                "root_value": float(root.q),
                "full_search": use_full,
            }
            if not is_submit:
                assert move_key is not None
                entry.update({
                    "from": list(move_key[:4]),
                    "to": list(move_key[4:]),
                    "promotion": "queen",
                })
            history.append(entry)
        env.apply_action(action_id)
        if is_submit:
            turns += 1
        else:
            semimoves += 1

    if env.is_terminal:
        white_outcome = env.white_outcome
        terminal_reason = env.terminal_reason
    elif not env.actions:
        white_outcome = -1.0 if env.current_player == 0 else 1.0
        terminal_reason = "no_legal_action"
    else:
        white_outcome = 0.0
        terminal_reason = "aborted"

    for sample in samples:
        sample.assign_outcome(white_outcome)

    return SelfPlayResult(
        samples=samples,
        white_outcome=white_outcome,
        terminal_reason=terminal_reason,
        semimoves=semimoves,
        turns=turns,
        board_limit=limit,
        board_count=env.board_count,
        final_player=env.current_player,
        moves=history,
        final_fen=str(env.record_snapshot()["fen"]) if selfplay_config.record_games else "",
    )


def generate_games(
    network: SuzeroNetwork,
    spec: VariantSpec,
    mcts_config: MctsConfig,
    selfplay_config: SelfPlayConfig,
    *,
    device: torch.device | str = "cpu",
    seed: int = 0,
    num_games: int | None = None,
    progress: object | None = None,
) -> list[SelfPlayResult]:
    """Play games with the live network.

    ``SelfPlayConfig.workers`` runs games concurrently in threads.  PyTorch
    inference in ``eval`` mode is thread-safe and releases the GIL inside the
    kernels, while each game owns its environment, RNG and search tree.
    Inference is not batched across games, so the speed-up is modest; the
    option exists to overlap network calls with Python-side tree work.
    """

    games = selfplay_config.num_games if num_games is None else int(num_games)
    if games <= 0:
        return []
    was_training = network.training
    network.eval()

    def run_one(index: int) -> SelfPlayResult:
        return play_game(
            network,
            spec,
            mcts_config,
            selfplay_config,
            device=device,
            seed=seed + index * 9973,
        )

    try:
        if selfplay_config.workers <= 1 or games == 1:
            results = []
            for index in range(games):
                result = run_one(index)
                results.append(result)
                if callable(progress):
                    progress(index + 1, games, result)
            return results

        results: list[SelfPlayResult | None] = [None] * games
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(selfplay_config.workers, games)
        ) as pool:
            futures = {pool.submit(run_one, index): index for index in range(games)}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                result = future.result()
                results[index] = result
                completed += 1
                if callable(progress):
                    progress(completed, games, result)
        return [result for result in results if result is not None]
    finally:
        network.train(was_training)
