"""Off-trajectory tree targets and Python self-play isolation regressions."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from suzero_5dchess.config import MctsConfig, SelfPlayConfig
from suzero_5dchess.env import EncodedPosition
from suzero_5dchess.mcts import Node
from suzero_5dchess.protocol import VERY_SMALL
from suzero_5dchess.tree_positions import extract_tree_positions


class TreeState:
    def __init__(self, tag, player, action_count=3, *, terminal=False, forced=False):
        self.tag = tag
        self.key = str(tag)
        self.key_reads = 0
        self.current_player = player
        self.is_terminal = terminal
        self.move_count = tag
        if forced:
            self.actions = ((True, None),)
        elif terminal:
            self.actions = ()
        else:
            self.actions = tuple(
                (False, (0, 0, 0, 0, 1, 0, 0, i))
                for i in range(action_count - 1)
            ) + ((True, None),)

    @property
    def state_key(self):
        self.key_reads += 1
        return self.key

    def encode(self):
        return EncodedPosition(
            board_planes=np.zeros((1, 12, 16), dtype=np.float32),
            l_coords=np.array([0], dtype=np.int64),
            t_coords=np.array([self.tag], dtype=np.int64),
            urgency=0.0, side_to_move=self.current_player, num_boards=1,
            board_side=4, board_squares=16, piece_channels=12,
        )


def node(tag, player=1, visits=100, *, terminal=False, forced=False, action_count=3):
    state = TreeState(tag, player, action_count, terminal=terminal, forced=forced)
    return Node(
        state, player, priors=np.full(len(state.actions), 1 / max(1, len(state.actions)),
                                     dtype=np.float32),
        visits=visits, value_sum=visits * 0.375,
        terminal=terminal, expanded=True,
    )


def target(tag, player=1, visits=100, counts=(8, 2, 0)):
    result = node(tag, player, visits, action_count=len(counts))
    result.children = {
        action: node(tag * 100 + action + 10000, player, count, terminal=True)
        for action, count in enumerate(counts)
    }
    return result


def root_with(*children):
    root = node(0, player=0, action_count=len(children) + 2)
    root.children = dict(enumerate(children))
    played_action = len(children)
    root.children[played_action] = node(99999, visits=200, terminal=True)
    return root, played_action


def config(**kwargs):
    return SelfPlayConfig(**{
        "record_tree_positions": True,
        "record_tree_min_visits": 4,
        "record_tree_max_depth": 5,
        "record_tree_max_per_turn": 20,
        "record_tree_sample_probability": 1.0,
        **kwargs,
    })


def extract(root, played_action, cfg=None, seed=7):
    return extract_tree_positions(root, played_action, cfg or config(),
                                  np.random.default_rng(seed))


def tags(samples):
    return [int(sample.t_coords[0]) for sample in samples]


def test_raw_visit_policy_search_value_and_metadata():
    candidate = target(1, player=1)
    candidate.priors = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    root, played = root_with(candidate)
    samples = extract(root, played)
    assert tags(samples) == [1]
    sample = samples[0]
    np.testing.assert_allclose(sample.policy, [0.8, 0.2, 0.0])
    assert sample.policy.dtype == np.float32
    assert sample.has_submit and sample.candidate_count == 2
    assert sample.player == 1
    assert sample.value_target == pytest.approx(0.375)  # no white-relative sign flip
    assert sample.metadata == {"tree_position": 1.0, "search_visits": 100.0}
    np.testing.assert_array_equal(candidate.priors, [0.0, 0.0, 1.0])
    assert candidate.visits == 100 and candidate.value_sum == 37.5


def test_root_and_entire_played_action_subtree_are_excluded():
    played_node = target(2)
    played_node.children[0] = target(3, visits=8, counts=(4, 0, 0))
    root, played = root_with(target(1))
    root.children[played] = played_node
    assert tags(extract(root, played)) == [1]
    assert played_node.state.key_reads == 0
    assert played_node.children[0].state.key_reads == 0


def test_always_best_flags_follow_players_not_depth():
    # White deviates at the root, then makes another white move. Black has
    # still played only best moves, even though it first appears at depth 2.
    side = node(1, player=0)
    black_best = target(2, player=1, visits=80, counts=(20, 60, 0))
    black_other = target(3, player=1, visits=40)
    side.children = {0: black_best, 1: black_other}
    black_best.children[0] = target(4, player=1, visits=20)
    black_best.children[1] = target(5, player=0, visits=60)
    root, played = root_with(side)
    assert tags(extract(root, played)) == [2, 3]
    # The non-best black edge invalidates black too, so its subtree is pruned.
    assert black_best.children[0].state.key_reads == 0


def test_best_visit_ties_choose_smallest_action_id():
    first = target(1, player=0)
    second = target(2, player=0)
    root, played = root_with(first, second)
    root.children[played].visits = 50
    # Reverse insertion order to ensure ties are action-id based.
    root.children = dict(reversed(list(root.children.items())))
    assert tags(extract(root, played)) == [1]


@pytest.mark.parametrize("reason", [
    "node_visits", "outgoing_visits", "zero_visits", "forced_submit",
    "terminal", "state_terminal", "unexpanded",
])
def test_weak_or_untrainable_nodes_are_not_samples(reason):
    candidate = target(1)
    if reason == "node_visits":
        candidate.visits = 3
    elif reason == "outgoing_visits":
        candidate = target(1, counts=(2, 1, 0))
    elif reason == "zero_visits":
        candidate = target(1, counts=(0, 0, 0))
    elif reason == "forced_submit":
        candidate = node(1, forced=True)
        candidate.children[0] = node(2, terminal=True)
    elif reason == "terminal":
        candidate.terminal = True
    elif reason == "state_terminal":
        candidate.state.is_terminal = True
    elif reason == "unexpanded":
        candidate.expanded = False
    root, played = root_with(candidate)
    assert not extract(root, played)


def test_forced_submit_can_lead_to_an_eligible_position():
    forced = node(1, player=0, forced=True)
    forced.children[0] = target(2, player=1)
    root, played = root_with(forced)
    assert tags(extract(root, played)) == [2]


def test_depth_and_accepted_count_limits():
    first = target(1)
    second = target(2, visits=8, counts=(4, 0, 0))
    first.children[0] = second
    root, played = root_with(first, target(3))
    assert tags(extract(root, played, config(record_tree_max_depth=1))) == [1, 3]
    assert tags(extract(root, played, config(record_tree_max_depth=2))) == [1, 2, 3]
    assert tags(extract(root, played, config(record_tree_max_per_turn=1))) == [1]
    assert tags(extract(root, played, config(record_tree_max_per_turn=2))) == [1, 2]


def test_depth_is_hard_bounded_even_if_config_is_mutated():
    first = node(1)
    current = first
    for depth in range(2, 72):
        child = node(depth)
        current.children = {0: child}
        current = child
    root, played = root_with(first)
    cfg = config(record_tree_max_depth=64, record_tree_max_per_turn=100)
    cfg.record_tree_max_depth = 1000
    result = extract(root, played, cfg)
    assert tags(result) == list(range(1, 65))


def test_state_deduplication_includes_root_and_breaks_cycles():
    first = target(1)
    duplicate = target(1)
    root_copy = target(0)
    first.children[0] = root_copy  # distinct Node, same exact state key as root
    root, played = root_with(first, duplicate, target(2))
    root.children[2].children[0] = root  # actual object/path cycle
    assert tags(extract(root, played)) == [1, 2]
    assert duplicate.state.key_reads == 1
    assert root.state.key_reads == 2  # initial seed plus the attempted cycle


def test_probability_is_reproducible_and_never_falls_back_to_priors():
    root, played = root_with(*(target(i) for i in range(1, 13)))
    cfg = config(record_tree_sample_probability=0.5)
    selected = tags(extract(root, played, cfg, seed=17))
    assert selected == tags(extract(root, played, cfg, seed=17))
    assert 0 < len(selected) < 12
    assert not extract(root, played, replace(cfg, record_tree_sample_probability=0))
    assert not extract(root, played, replace(cfg, record_tree_positions=False))
    assert len(extract(root, played, replace(cfg, record_tree_sample_probability=1))) == 12


def test_hard_traversal_budget_bounds_even_rejected_nodes():
    # None of these terminal children can yield a row, so the acceptance cap
    # cannot bound the walk. State-key reads expose how many nodes it visits.
    children = [node(i, terminal=True) for i in range(1, 5001)]
    root, played = root_with(*children)
    assert not extract(root, played)
    reads = root.state.key_reads + sum(child.state.key_reads for child in children)
    assert reads == 4096
    assert children[-1].state.key_reads == 0


def test_selfplay_result_tree_count_defaults_to_zero():
    from suzero_5dchess.self_play import SelfPlayResult

    result = SelfPlayResult([], 0.0, "test", 0, 0, 1, 1, 0)
    assert result.tree_samples == 0


def test_real_selfplay_keeps_trajectory_main_rows_and_evaluation_count(monkeypatch):
    pytest.importorskip("suzero_5dchess._5dchess")
    from suzero_5dchess import self_play

    class CountingEvaluator:
        def __init__(self, *args):
            self.calls = 0

        def evaluate(self, state):
            self.calls += 1
            return np.full(len(state.actions), 1 / len(state.actions), dtype=np.float32), 0.375

    evaluators = []

    def evaluator_factory(*args):
        result = CountingEvaluator()
        evaluators.append(result)
        return result

    monkeypatch.setattr(self_play, "TorchEvaluator", evaluator_factory)
    extracted_values = []
    original_extract = self_play.extract_tree_positions

    def capture_extract(*args):
        samples = original_extract(*args)
        extracted_values.extend(sample.value_target for sample in samples)
        return samples

    monkeypatch.setattr(self_play, "extract_tree_positions", capture_extract)
    cfg = config(
        min_board_limit=4, max_board_limit=4, record_games=True,
        record_tree_min_visits=1, record_tree_max_per_turn=4,
        full_search_probability=0.6, fast_simulations=2,
    )
    mcts = MctsConfig(num_simulations=96, max_depth=16)
    plain = self_play.play_game(object(), VERY_SMALL, mcts,
                               replace(cfg, record_tree_positions=False), seed=41)
    enabled = self_play.play_game(object(), VERY_SMALL, mcts, cfg, seed=41)
    assert enabled.tree_samples > 0
    assert plain.tree_samples == 0
    assert enabled.moves == plain.moves  # archives contain actual moves only
    assert enabled.final_fen == plain.final_fen
    for key in ("white_outcome", "terminal_reason", "semimoves", "turns",
                "board_limit", "board_count", "final_player"):
        assert getattr(enabled, key) == getattr(plain, key)
    assert evaluators[0].calls == evaluators[1].calls  # extraction never evaluates
    assert any(not move.get("full_search", True) for move in enabled.moves)
    main = [sample for sample in enabled.samples if not sample.metadata.get("tree_position")]
    sides = [sample for sample in enabled.samples if sample.metadata.get("tree_position")]
    assert len(main) == len(plain.samples)
    assert len(sides) == enabled.tree_samples
    assert [sample.value_target for sample in sides] == extracted_values
    assert enabled.samples == main + sides
    for before, after in zip(plain.samples, main):
        assert before.metadata == after.metadata == {}
        for key in ("board_codes", "l_coords", "t_coords", "actions", "policy"):
            np.testing.assert_array_equal(getattr(before, key), getattr(after, key))
        for key in ("value_target", "player", "move_count", "urgency", "side_to_move"):
            assert getattr(before, key) == getattr(after, key)
    assert any(sample.value_target != (enabled.white_outcome if sample.player == 0
                                      else -enabled.white_outcome) for sample in sides)
    assert all(sample.metadata["search_visits"] >= 1 for sample in sides)


def test_fast_searches_do_not_extract_tree_rows(monkeypatch):
    pytest.importorskip("suzero_5dchess._5dchess")
    from suzero_5dchess import self_play

    class UniformEvaluator:
        def __init__(self, *args):
            pass

        def evaluate(self, state):
            return np.full(len(state.actions), 1 / len(state.actions), dtype=np.float32), 0.0

    def unexpected_extract(*args):
        pytest.fail("fast searches must not extract tree targets")

    monkeypatch.setattr(self_play, "TorchEvaluator", UniformEvaluator)
    monkeypatch.setattr(self_play, "extract_tree_positions", unexpected_extract)
    result = self_play.play_game(
        object(), VERY_SMALL, MctsConfig(num_simulations=2, max_depth=4),
        config(min_board_limit=2, max_board_limit=2, full_search_probability=0,
               fast_simulations=2), seed=12,
    )
    assert not result.samples and result.tree_samples == 0
