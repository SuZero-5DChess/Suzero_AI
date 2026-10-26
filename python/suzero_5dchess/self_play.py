"""AlphaZero self-play for 5D chess."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from .config import MctsConfig, SelfPlayConfig
from .env import GameEnv
from .interrupts import stop_requested
from .mcts import MCTS, TorchEvaluator
from .network import SuzeroNetwork
from .protocol import VariantSpec
from .sample import PositionSample, make_sample
from .tree_positions import extract_tree_positions

if TYPE_CHECKING:  # pragma: no cover - avoids a module cycle at runtime
    from .game_store import GameStore


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
    tree_samples: int = 0
    # Rows this game contributed.  A streamed result releases ``samples`` after
    # writing them to disk, so this stays the authoritative count.
    sample_count: int = field(default=-1, compare=False)
    # Set by a streaming writer once this game's samples are on disk, so the
    # pipeline can append only the games a backend could not stream.
    persisted: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            self.sample_count = len(self.samples)

    def release_samples(self) -> None:
        """Drop the in-RAM rows after they were persisted to the replay store."""

        self.samples = []


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
    noise. Tree positions (enabled by default) reuse full-search nodes with
    their own search values. Fast moves (playout-cap randomization) neither add
    noise nor produce samples.

    The caller is responsible for putting ``network`` in eval mode (as
    :func:`generate_games` does); otherwise dropout would perturb search.
    """

    rng = np.random.default_rng(seed)
    # Derive an independent stream without advancing the gameplay/search RNG.
    tree_rng = np.random.default_rng([seed, 0x54524545])
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
    tree_samples: list[PositionSample] = []
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
        if use_full and selfplay_config.record_tree_positions:
            tree_samples.extend(extract_tree_positions(
                root, action_id, selfplay_config, tree_rng,
            ))
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
    # Off-trajectory values belong to their own searches, not this game's result.
    samples.extend(tree_samples)

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
        tree_samples=len(tree_samples),
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
    game_store: "GameStore | None" = None,
    game_id_start: int = 0,
    iteration: int = -1,
) -> list[SelfPlayResult]:
    """Play games with the live network.

    ``SelfPlayConfig.workers`` runs games concurrently in threads.  PyTorch
    inference in ``eval`` mode is thread-safe and releases the GIL inside the
    kernels, while each game owns its environment, RNG and search tree.
    Inference is not batched across games, so the speed-up is modest; the
    option exists to overlap network calls with Python-side tree work.

    When ``game_store`` is supplied, every game is encoded and written to disk
    as soon as that game finishes, while other games keep running.
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

    def persist(index: int, result: SelfPlayResult) -> None:
        if game_store is not None:
            game_store.append_result(
                int(game_id_start) + index, result, iteration=iteration
            )

    try:
        if selfplay_config.workers <= 1 or games == 1:
            results = []
            for index in range(games):
                if stop_requested():
                    break
                result = run_one(index)
                persist(index, result)
                results.append(result)
                if callable(progress):
                    progress(index + 1, games, result)
            return results

        results: list[SelfPlayResult | None] = [None] * games
        max_workers = min(selfplay_config.workers, games)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as pool:
            futures: dict[concurrent.futures.Future, int] = {}
            next_index = 0
            completed = 0

            def submit_available() -> None:
                nonlocal next_index
                # Stop handing out games once the user asked to quit; the ones
                # already running still finish and are persisted.
                while (
                    len(futures) < max_workers
                    and next_index < games
                    and not stop_requested()
                ):
                    futures[pool.submit(run_one, next_index)] = next_index
                    next_index += 1

            submit_available()
            while futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    index = futures.pop(future)
                    result = future.result()
                    persist(index, result)
                    results[index] = result
                    completed += 1
                    if callable(progress):
                        progress(completed, games, result)
                submit_available()
        return [result for result in results if result is not None]
    finally:
        network.train(was_training)
