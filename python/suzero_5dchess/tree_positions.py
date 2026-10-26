"""Reuse well-visited off-trajectory MCTS positions without further search.

Like KataGo's ``recordTreePositionsRec``, a position is eligible only if its
player has always followed the most-visited edge on the path from the root.
Players need not alternate on edges: in 5D chess only SUBMIT changes the turn.
"""

from __future__ import annotations

import numpy as np

from .config import SelfPlayConfig
from .mcts import Node
from .sample import PositionSample, make_sample


_TRAVERSAL_BUDGET = 4096
_MAX_DEPTH = 64


def extract_tree_positions(
    root: Node,
    played_action: int,
    config: SelfPlayConfig,
    rng: np.random.Generator,
) -> list[PositionSample]:
    """Extract bounded search-labelled samples, excluding the played subtree.

    The caller supplies a separate RNG and calls this only after selecting an
    action from a full search. Neither the root nor forced-SUBMIT/terminal
    positions are training rows. Both node visits and outgoing visits must
    meet the threshold; policy targets never fall back to network priors.
    """

    samples: list[PositionSample] = []
    probability = config.record_tree_sample_probability
    max_samples = config.record_tree_max_per_turn
    max_depth = min(_MAX_DEPTH, config.record_tree_max_depth)
    if (
        not config.record_tree_positions
        or probability <= 0
        or max_samples <= 0
        or max_depth <= 0
    ):
        return samples

    min_visits = config.record_tree_min_visits
    # Global state deduplication also breaks every path cycle, including a
    # return to the root. Keep ineligible positions in this set as well.
    seen = {root.state.state_key}
    remaining = _TRAVERSAL_BUDGET - 1

    def walk(node: Node, depth: int, always_best: tuple[bool, bool]) -> None:
        nonlocal remaining
        if node.terminal or node.state.is_terminal or not node.expanded:
            return

        actions = node.state.actions
        if depth > 0 and always_best[node.player] and any(
            not is_submit for is_submit, _ in actions
        ):
            counts = np.zeros(len(actions), dtype=np.float64)
            for action, child in node.children.items():
                counts[action] = child.visits
            outgoing_visits = float(counts.sum())
            if outgoing_visits > 0 and outgoing_visits >= min_visits:
                if rng.random() < probability:
                    sample = make_sample(
                        node.state, (counts / outgoing_visits).astype(np.float32),
                        node.player,
                    )
                    sample.value_target = float(node.q)
                    sample.metadata = {
                        "tree_position": 1.0,
                        "search_visits": float(node.visits),
                    }
                    samples.append(sample)

        if (
            depth >= max_depth
            or len(samples) >= max_samples
            or remaining <= 0
            or not node.children
        ):
            return

        best_action = min(
            node.children, key=lambda action: (-node.children[action].visits, action)
        )
        for action, child in sorted(node.children.items()):
            if remaining <= 0 or len(samples) >= max_samples:
                break
            remaining -= 1
            if depth == 0 and action == played_action:
                continue
            if child.visits < min_visits:
                continue
            flags = list(always_best)
            if action != best_action:
                flags[node.player] = False
            if not any(flags):
                continue
            key = child.state.state_key
            if key in seen:
                continue
            seen.add(key)
            walk(child, depth + 1, (flags[0], flags[1]))

    walk(root, 0, (True, True))
    return samples
