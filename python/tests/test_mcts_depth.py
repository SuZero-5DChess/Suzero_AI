"""Depth-boundary regressions using a tiny game, without the rules extension."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from suzero_5dchess.config import MctsConfig
from suzero_5dchess.mcts import MCTS


class ChainState:
    def __init__(self, remaining: int, switch_player: bool = False, player: int = 0):
        self.remaining = remaining
        self.switch_player = switch_player
        self.current_player = player

    def clone(self):
        return ChainState(self.remaining, self.switch_player, self.current_player)

    @property
    def is_terminal(self):
        return self.remaining == 0

    @property
    def terminal_utility(self):
        return -1.0 if self.switch_player and self.current_player == 1 else 1.0

    @property
    def actions(self):
        return [] if self.is_terminal else [(False, (0,) * 8)]

    def apply_action(self, action):
        assert action == 0 and not self.is_terminal
        self.remaining -= 1
        if self.switch_player:
            self.current_player = 1 - self.current_player


class ConstantEvaluator:
    def __init__(self, value=0.25):
        self.value = value
        self.calls = 0

    def evaluate(self, state):
        assert not state.is_terminal
        self.calls += 1
        return np.ones(1, dtype=np.float32), self.value


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("switch_player", [False, True])
def test_terminal_at_depth_limit_is_visited_and_backed_up(depth, switch_player):
    evaluator = ConstantEvaluator()
    search = MCTS(evaluator, MctsConfig(max_depth=depth, num_simulations=10))
    root = search.run(ChainState(depth, switch_player))

    node = root
    for _ in range(depth):
        node = node.children[0]
    assert node.terminal
    assert node.visits > 0
    assert node.q == pytest.approx(node.terminal_value)
    # The final simulation reaches the terminal; check its root-relative value.
    before = root.value_sum
    search._simulate(root)
    assert root.value_sum - before == pytest.approx(1.0)
    assert root.visits == 11
    if depth == 1:
        assert node.visits == 11
        assert root.q == pytest.approx(1.0)
    np.testing.assert_array_equal(root.visit_policy(), [1.0])


@pytest.mark.parametrize("switch_player", [False, True])
def test_nonterminal_depth_limit_uses_leaf_estimate_without_another_edge(switch_player):
    evaluator = ConstantEvaluator()
    search = MCTS(evaluator, MctsConfig(max_depth=1, num_simulations=10))
    root = search.run(ChainState(3, switch_player))
    child = root.children[0]

    assert root.visits == child.visits == 10
    assert child.q == pytest.approx(0.25)
    assert root.q == pytest.approx(-0.25 if switch_player else 0.25)
    assert child.children == {}
    assert evaluator.calls == 2
