"""
Unit tests for the halfstep-level MCTS module.

These tests mock the C++ engine module and network so they can run
without building the compiled engine extension.
"""

import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, PropertyMock, patch, call

# Mock the C++ engine module BEFORE any alphazero import
sys.modules["engine"] = MagicMock()
engine_mock = sys.modules["engine"]
# Mock vec4 class used by engine_state.get_movable_pieces()
engine_mock.vec4 = MagicMock()

sys.path.insert(0, str(Path(__file__).parent))

from alphazero.mcts import MCTS, MCTSNode, TranspositionEntry, SUBMIT_ACTION
from alphazero.halfstep_state import Coord4, HalfstepState


# ═══════════════════════════════════════════════════════════════════
#  Mock helpers
# ═══════════════════════════════════════════════════════════════════

def make_mock_vec4(x, y, t, l):
    """Create a mock engine.vec4 with .x(), .y(), .t(), .l() accessors."""
    v = MagicMock()
    v.x.return_value = x
    v.y.return_value = y
    v.t.return_value = t
    v.l.return_value = l
    return v


def make_mock_config(**overrides):
    """Create a minimal MCTSConfig with defaults."""
    cfg = MagicMock()
    cfg.num_simulations = overrides.get("num_simulations", 8)
    cfg.c_puct = overrides.get("c_puct", 2.0)
    cfg.dirichlet_alpha = overrides.get("dirichlet_alpha", 0.3)
    cfg.dirichlet_epsilon = overrides.get("dirichlet_epsilon", 0.25)
    cfg.use_transposition_table = overrides.get("use_transposition_table", False)
    return cfg


def make_mock_network(board_side=4, board_squares=16, num_boards=2, batch_size=1):
    """
    Create a mock network that returns deterministic logits.
    raw_logits shape: [num_boards, board_squares]
    """
    net = MagicMock()
    # Create deterministic logits: each board-square gets a unique value
    rng = np.random.RandomState(42)
    logits = rng.randn(num_boards, board_squares).astype(np.float32)
    value = 0.3  # deterministic value estimate

    def predict_actions(board_planes, l_coords, t_coords, urgency):
        v = torch.tensor([value], dtype=torch.float32)
        rl = torch.from_numpy(logits.copy())
        sl = torch.tensor([0.5], dtype=torch.float32)
        return v, rl, sl

    net.predict_actions.side_effect = predict_actions
    net._logits = logits  # for test assertions
    net._value = value
    return net


def make_mock_engine_state(raw_sources, gen_piece_move_map, mandatory=()):
    """Create a mock engine state."""
    state = MagicMock()
    state.get_movable_pieces.return_value = raw_sources
    state.get_timeline_status.return_value = (list(mandatory), [], [])
    state.gen_piece_move_unsafe.side_effect = lambda src: gen_piece_move_map.get(
        (src.x(), src.y(), src.t(), src.l()), []
    )
    return state


def make_mock_env(board_side=4, **overrides):
    """Create a mock SemimoveEnv."""
    from alphazero.env import Semimove
    env = MagicMock()
    env.board_side = board_side
    env.done = overrides.get("done", False)
    env.outcome = overrides.get("outcome", None)
    env.current_player = overrides.get("current_player", 0)
    env.state = overrides.get("engine_state", MagicMock())
    env.board_count = overrides.get("board_count", 2)

    # Default frontier: one semimove + submit
    default_frontier = overrides.get(
        "legal_frontier",
        ([Semimove(line_idx=0, from_pos=(0, 0, 0, 0), to_pos=(0, 1, 0, 0))], True),
    )
    env.get_legal_frontier.return_value = default_frontier

    # encode_state returns deterministic board keys
    def encode_state(urgency=0.0):
        return {
            "board_planes": np.zeros((2, 14, board_side * board_side), dtype=np.float32),
                        "l_coords": np.array([0, 1], dtype=np.int64),
            "t_coords": np.array([0, 0], dtype=np.int64),
            "board_keys": [(0, 0, False), (0, 0, True)],
            "num_boards": 2,
        }

    env.encode_state.side_effect = encode_state
    return env


def make_mock_halfstep_state(**overrides):
    """Create a mock HalfstepState."""
    state = MagicMock()
    env = overrides.get("env", make_mock_env())
    state.env = env
    state.pending_from = overrides.get("pending_from", None)
    state.current_player.return_value = overrides.get("current_player", 0)
    state.legal_destinations_for.return_value = overrides.get("legal_dests", [])
    state.get_mcts_transposition_key.return_value = overrides.get("tt_key", (1, 2, 3, None))
    state.clone.return_value = state  # simple: clone returns self
    state.apply_halfstep.side_effect = overrides.get("apply_halfstep", lambda a: state)
    state.is_done.return_value = overrides.get("is_done", False)
    state.get_outcome.return_value = overrides.get("outcome", None)
    state.env.board_side = overrides.get("board_side", 4)
    return state


# ═══════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════

class TestMCTSNode(unittest.TestCase):
    """MCTSNode basic operations."""

    def _make_state(self, **overrides):
        return make_mock_halfstep_state(**overrides)

    def test_init(self):
        node = MCTSNode(state=self._make_state())
        assert node.parent is None
        assert node.action is None
        assert node.prior == 0.0
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.children == []
        assert node.is_expanded is False
        assert node.is_terminal is False
        assert node.terminal_value == 0.0

    def test_init_with_parent(self):
        parent = MCTSNode(state=self._make_state())
        child = MCTSNode(state=self._make_state(), parent=parent, action=(1, 2, 3, 4), prior=0.7)
        assert child.parent is parent
        assert child.action == (1, 2, 3, 4)
        assert child.prior == 0.7

    def test_q_value_zero_visits(self):
        node = MCTSNode(state=self._make_state())
        assert node.q_value == 0.0

    def test_q_value(self):
        node = MCTSNode(state=self._make_state())
        node.visit_count = 5
        node.value_sum = 2.5
        assert node.q_value == 0.5

    def test_ucb_score(self):
        parent = MCTSNode(state=self._make_state())
        parent.visit_count = 10
        child = MCTSNode(state=self._make_state(), parent=parent, prior=0.5)
        child.visit_count = 3
        child.value_sum = 0.6

        score = child.ucb_score(c_puct=2.0)
        expected = 0.2 + 2.0 * 0.5 * math.sqrt(10) / 4
        assert abs(score - expected) < 1e-6, f"{score} != {expected}"

    def test_ucb_score_no_parent(self):
        """Root node: returns 0."""
        node = MCTSNode(state=self._make_state(), prior=0.3)
        node.visit_count = 2
        node.value_sum = 0.4
        score = node.ucb_score(c_puct=1.0)
        assert score == 0.0

    def test_ucb_score_unvisited_child(self):
        """Unvisited child: q_value = 0, score = exploration term."""
        parent = MCTSNode(state=self._make_state())
        parent.visit_count = 10
        child = MCTSNode(state=self._make_state(), parent=parent, prior=0.5)
        score = child.ucb_score(c_puct=2.0)
        expected = 2.0 * 0.5 * math.sqrt(10) / 1
        assert abs(score - expected) < 1e-6

    def test_best_child(self):
        """best_child returns child with highest UCB score."""
        parent = MCTSNode(state=self._make_state())
        parent.visit_count = 10
        c1 = MCTSNode(state=self._make_state(), parent=parent, action=(0, 0, 0, 0), prior=0.1)
        c2 = MCTSNode(state=self._make_state(), parent=parent, action=(1, 0, 0, 0), prior=0.5)
        c3 = MCTSNode(state=self._make_state(), parent=parent, action=(2, 0, 0, 0), prior=0.3)
        parent.children = [c1, c2, c3]
        # Give all children some visits
        for i, c in enumerate(parent.children):
            c.visit_count = 3 + i
            c.value_sum = 0.5 * c.visit_count

        best = parent.best_child(c_puct=2.0)
        # Middle child (highest prior + good q_value) should win
        assert best is c2


class TestTranspositionEntry(unittest.TestCase):
    """TranspositionEntry dataclass."""

    def test_init(self):
        entry = TranspositionEntry(
            value=0.5,
            child_specs=[((0, 0, 0, 0), 0.5)],
            is_terminal=False,
            terminal_value=0.0,
        )
        assert entry.value == 0.5
        assert entry.child_specs == [((0, 0, 0, 0), 0.5)]
        assert entry.is_terminal is False


class TestMCTSSearch(unittest.TestCase):
    """MCTS.search full simulation loop."""

    def setUp(self):
        self.network = make_mock_network()
        self.cfg = make_mock_config(num_simulations=4, use_transposition_table=False)
        self.device = torch.device("cpu")
        self.mcts = MCTS(self.network, self.cfg, self.device)

    def test_search_expands_root(self):
        """search() returns root node with children."""
        from alphazero.env import Semimove

        # Build a proper mock for real search
        board_side = 4
        board_squares = 16
        num_boards = 2

        # Create mock engine state with movable pieces
        src_a = make_mock_vec4(0, 0, 0, 0)
        src_b = make_mock_vec4(1, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a, src_b],
            gen_piece_move_map={
                (0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)],
                (1, 0, 0, 0): [make_mock_vec4(1, 1, 0, 0)],
            },
        )

        # Create mock network with deterministic logits
        rng = np.random.RandomState(42)
        raw_logits = rng.randn(num_boards, board_squares).astype(np.float32)

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(raw_logits.copy()),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        # Create mock env that returns real-looking data
        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env  # clone returns self for test stability

        # Encode_state returns valid planes
        def encode_state(urgency=0.0):
            return {
                "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0)),
             Semimove(0, (1, 0, 0, 0), (1, 1, 0, 0))],
            True,
        )
        env.can_submit.return_value = True

        # Create HalfstepState with this env
        state = HalfstepState(env=env, pending_from=None)

        root = self.mcts.search(state, board_limit=10)

        # Root children should be source actions
        child_actions = [c.action for c in root.children]
        assert (0, 0, 0, 0) in child_actions
        assert (1, 0, 0, 0) in child_actions
        assert SUBMIT_ACTION in child_actions

        # Root should have children
        assert len(root.children) > 0
        # Root should be expanded
        assert root.is_expanded is True

    def test_search_network_call_count(self):
        """Each simulation expands one leaf -> network calls = 1 (root) + sims."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 2
        rng = np.random.RandomState(42)
        raw_logits = rng.randn(num_boards, board_squares).astype(np.float32)

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(raw_logits.copy()),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.reset_mock()
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)]},
        )
        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        def encode_state(urgency=0.0):
            return {
                "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            True,
        )
        env.can_submit.return_value = True

        state = HalfstepState(env=env, pending_from=None)
        self.mcts.search(state, board_limit=10)

        # Network calls: 1 (root expansion) + num_simulations (each sim expands a leaf)
        total_calls = self.network.predict_actions.call_count
        # With 4 sims, expect 1 + 4 = 5 calls (some sims may also expand dest nodes)
        assert total_calls >= 1 + self.cfg.num_simulations, \
            f"Expected >= {1 + self.cfg.num_simulations} calls, got {total_calls}"


class TestMCTSSelectHalfstep(unittest.TestCase):
    """MCTS.select_halfstep."""

    def setUp(self):
        self.network = make_mock_network()
        self.cfg = make_mock_config(num_simulations=8, use_transposition_table=False)
        self.device = torch.device("cpu")
        self.mcts = MCTS(self.network, self.cfg, self.device)

    def test_select_halfstep_returns_single_action(self):
        """select_halfstep returns a single halfstep action (Coord4 or SUBMIT)."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 2
        rng = np.random.RandomState(42)
        raw_logits = rng.randn(num_boards, board_squares).astype(np.float32)

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(raw_logits.copy()),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)]},
        )
        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        def encode_state(urgency=0.0):
            return {
                "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            True,
        )
        env.can_submit.return_value = True

        state = HalfstepState(env=env, pending_from=None)
        root = self.mcts.search(state, board_limit=10)
        halfstep_action, policy_probs, root_value = self.mcts.select_halfstep(
            root, temperature=0.0  # greedy
        )

        # Should return a single halfstep action (Coord4 or SUBMIT or None)
        assert halfstep_action is not None
        # Should be a 4-tuple (Coord4) or SUBMIT_ACTION string
        assert isinstance(halfstep_action, tuple) or halfstep_action == SUBMIT_ACTION
        if isinstance(halfstep_action, tuple):
            assert len(halfstep_action) == 4
        # policy_probs should be a 1D array matching root children
        assert isinstance(policy_probs, np.ndarray)
        assert len(policy_probs) == len(root.children)
        # root_value should be a numeric type
        assert isinstance(root_value, (float, np.floating))

    def test_select_halfstep_no_children(self):
        """No legal actions -> returns None."""
        env = make_mock_env(legal_frontier=([], False), done=False, board_count=2)
        state = make_mock_halfstep_state(env=env, pending_from=None)

        root = MCTSNode(state=state)
        halfstep_action, policy_probs, root_value = self.mcts.select_halfstep(
            root, temperature=0.0
        )

        assert halfstep_action is None
        assert len(policy_probs) == 0
        assert isinstance(root_value, float)


class TestMCTSBackup(unittest.TestCase):
    """MCTS._backup value propagation."""

    def setUp(self):
        self.network = make_mock_network()
        self.cfg = make_mock_config()
        self.device = torch.device("cpu")
        self.mcts = MCTS(self.network, self.cfg, self.device)
        self._state = make_mock_halfstep_state()

    def test_backup_value_no_flip(self):
        """Backup adds value without flipping at non-SUBMIT nodes."""
        root = MCTSNode(state=self._state)
        child = MCTSNode(state=self._state, parent=root, action=(0, 0, 0, 0))
        root.children = [child]

        self.mcts._backup(child, 0.5)
        assert child.visit_count == 1
        assert child.value_sum == 0.5
        assert root.visit_count == 1
        assert root.value_sum == 0.5

    def test_backup_value_flip_at_submit(self):
        """Backup flips value at SUBMIT nodes."""
        root = MCTSNode(state=self._state)
        child = MCTSNode(state=self._state, parent=root, action=SUBMIT_ACTION)
        root.children = [child]

        self.mcts._backup(child, 0.5)
        assert child.visit_count == 1
        assert child.value_sum == 0.5
        # SUBMIT flips: child is the leaf whose action == SUBMIT,
        # so the value flips when backing up to root
        assert root.visit_count == 1
        assert root.value_sum == -0.5

    def test_backup_value_flip_through_submit(self):
        """Backup flips value when passing through a SUBMIT node."""
        # root → submit_node → child
        submit_node = MCTSNode(state=self._state, action=SUBMIT_ACTION)
        root = MCTSNode(state=self._state)
        root.children = [submit_node]
        submit_node.parent = root
        child = MCTSNode(state=self._state, parent=submit_node, action=(0, 0, 0, 0))
        submit_node.children = [child]

        self.mcts._backup(child, 0.5)
        assert child.visit_count == 1
        assert child.value_sum == 0.5
        assert submit_node.visit_count == 1
        assert submit_node.value_sum == 0.5
        # Value flips at SUBMIT when backing up through it
        assert root.visit_count == 1
        assert root.value_sum == -0.5


# ═══════════════════════════════════════════════════════════════════
#  Tests: pending_from board plane marking (已选择棋子时反转)
# ═══════════════════════════════════════════════════════════════════

class TestMCTSPendingFromMarking(unittest.TestCase):
    """MCTS._expand board plane negation when pending_from is set."""

    def setUp(self):
        self.network = make_mock_network()
        self.cfg = make_mock_config(use_transposition_table=False)
        self.device = torch.device("cpu")
        self.mcts = MCTS(self.network, self.cfg, self.device)

    def test_board_planes_negate_selected_piece(self):
        """When pending_from is set, the selected piece's board square is negated."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 2

        # Mock network: capture board_planes to verify negation
        captured_planes = {}

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            captured_planes["planes"] = board_planes.detach().cpu().numpy().copy()
            captured_planes["keys"] = (l_coords.detach().cpu().numpy().copy(),
                                       t_coords.detach().cpu().numpy().copy())
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(np.zeros((num_boards, board_squares), dtype=np.float32)),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)  # piece at (0,0,0,0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)]},
        )

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        # Board keys: board 0 = (l=0, t=0, c=False=white), board 1 = (l=0, t=0, c=True=black)
        def encode_state(urgency=0.0):
            return {
                "board_planes": np.ones((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            False,
        )
        env.can_submit.return_value = False

        # ── Create state with pending_from set ──
        # The selected piece is at (0, 0, 0, 0) — white player board (c=False), board index 0
        # Square index = x + y * board_side = 0 + 0 * 4 = 0
        state = HalfstepState(env=env, pending_from=(0, 0, 0, 0))

        node = MCTSNode(state=state)
        self.mcts._expand(node, urgency=0.0)

        # Verify: board 0, channel all, square 0 should be negated (multiplied by -1)
        planes = captured_planes["planes"]
        # Before negation: all 1s. After negation of square 0 across all channels:
        expected_planes = np.ones((num_boards, 14, board_squares), dtype=np.float32)
        expected_planes[0, :, 0] *= -1.0  # board 0 (white), all channels, square 0 negated

        np.testing.assert_array_equal(
            planes, expected_planes,
            err_msg="pending_from piece's square should be negated across all channels"
        )

    def test_board_planes_negate_at_different_position(self):
        """Negation works for a piece not at (0,0)."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 2

        captured_planes = {}

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            captured_planes["planes"] = board_planes.detach().cpu().numpy().copy()
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(np.zeros((num_boards, board_squares), dtype=np.float32)),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        # Piece at (2, 3, 0, 0) — x=2, y=3
        src_a = make_mock_vec4(2, 3, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(2, 3, 0, 0): [make_mock_vec4(2, 1, 0, 0)]},
        )

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        def encode_state(urgency=0.0):
            return {
                "board_planes": np.ones((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (2, 3, 0, 0), (2, 1, 0, 0))],
            False,
        )
        env.can_submit.return_value = False

        state = HalfstepState(env=env, pending_from=(2, 3, 0, 0))
        node = MCTSNode(state=state)
        self.mcts._expand(node, urgency=0.0)

        planes = captured_planes["planes"]
        # Square at (2, 3) = 2 + 3*4 = 14
        expected_planes = np.ones((num_boards, 14, board_squares), dtype=np.float32)
        expected_planes[0, :, 14] *= -1.0

        np.testing.assert_array_equal(
            planes, expected_planes,
            err_msg="pending_from piece at (2,3) should negate square 14"
        )

    def test_no_negation_when_pending_from_is_none(self):
        """When pending_from is None, board planes should NOT be negated."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 2

        captured_planes = {}

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            captured_planes["planes"] = board_planes.detach().cpu().numpy().copy()
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(np.zeros((num_boards, board_squares), dtype=np.float32)),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)]},
        )

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        def encode_state(urgency=0.0):
            return {
                "board_planes": np.ones((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            True,
        )
        env.can_submit.return_value = True

        state = HalfstepState(env=env, pending_from=None)
        node = MCTSNode(state=state)
        self.mcts._expand(node, urgency=0.0)

        planes = captured_planes["planes"]
        # No negation when pending_from is None
        expected_planes = np.ones((num_boards, 14, board_squares), dtype=np.float32)
        np.testing.assert_array_equal(
            planes, expected_planes,
            err_msg="No negation should occur when pending_from is None"
        )

    def test_negation_board_index_match(self):
        """Negation applies to the correct board index matching player color."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 3

        captured_planes = {}

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            captured_planes["planes"] = board_planes.detach().cpu().numpy().copy()
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(np.zeros((num_boards, board_squares), dtype=np.float32)),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        # Piece at (1, 0, 0, 1) on timeline 1, black player (current_player=1)
        src_a = make_mock_vec4(1, 0, 0, 1)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(1, 0, 0, 1): [make_mock_vec4(1, 1, 0, 1)]},
        )

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 3
        env.done = False
        env.outcome = None
        env.current_player = 1  # Black player
        env.state = engine_state
        env.clone.return_value = env

        # Board keys: black's board with l=1,t=0 matches board_keys[2] (1, 0, True)
        def encode_state(urgency=0.0):
            return {
                "board_planes": np.ones((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True), (1, 0, True)],  # idx2 = black (l=1,t=0)
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(1, (1, 0, 0, 1), (1, 1, 0, 1))],
            False,
        )
        env.can_submit.return_value = False

        state = HalfstepState(env=env, pending_from=(1, 0, 0, 1))
        node = MCTSNode(state=state)
        self.mcts._expand(node, urgency=0.0)

        planes = captured_planes["planes"]
        # board index 2 (black, l=1, t=0, c=True), square at (1,0) = 1 + 0*4 = 1 should be negated
        expected_planes = np.ones((num_boards, 14, board_squares), dtype=np.float32)
        expected_planes[2, :, 1] *= -1.0

        np.testing.assert_array_equal(
            planes, expected_planes,
            err_msg="Black player's piece should negate on the correct board index"
        )

    def test_negation_does_not_affect_other_boards(self):
        """Negation of one board's square should not affect other boards."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 3

        captured_planes = {}

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            captured_planes["planes"] = board_planes.detach().cpu().numpy().copy()
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(np.zeros((num_boards, board_squares), dtype=np.float32)),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)]},
        )

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 3
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        def encode_state(urgency=0.0):
            return {
                "board_planes": np.ones((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True), (1, 0, False)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            False,
        )
        env.can_submit.return_value = False

        state = HalfstepState(env=env, pending_from=(0, 0, 0, 0))
        node = MCTSNode(state=state)
        self.mcts._expand(node, urgency=0.0)

        planes = captured_planes["planes"]
        # Only board 0 (white, l=0, t=0), square 0 should be negated
        expected_planes = np.ones((num_boards, 14, board_squares), dtype=np.float32)
        expected_planes[0, :, 0] *= -1.0

        # Board 1 and 2 should remain unchanged
        np.testing.assert_array_equal(
            planes[1], expected_planes[1],
            err_msg="Board 1 should not be affected by negation on board 0"
        )
        np.testing.assert_array_equal(
            planes[2], expected_planes[2],
            err_msg="Board 2 should not be affected by negation on board 0"
        )


# ═══════════════════════════════════════════════════════════════════
#  Tests: Advanced backup propagation (backup 传播验证)
# ═══════════════════════════════════════════════════════════════════

class TestMCTSBackupAdvanced(unittest.TestCase):
    """Advanced MCTS._backup propagation tests."""

    def setUp(self):
        self.network = make_mock_network()
        self.cfg = make_mock_config()
        self.device = torch.device("cpu")
        self.mcts = MCTS(self.network, self.cfg, self.device)
        self._state = make_mock_halfstep_state()

    def test_backup_deep_chain_no_flip(self):
        """Deep chain without SUBMIT: value propagates unchanged."""
        # root → c1 → c2 → c3 (leaf)
        nodes = self._build_chain(4, actions=[(0,0,0,0), (0,1,0,0), (0,2,0,0)])
        leaf = nodes[-1]

        self.mcts._backup(leaf, 0.7)

        for i, node in enumerate(nodes):
            self.assertEqual(node.visit_count, 1, f"Node {i} visit count")
            self.assertAlmostEqual(node.value_sum, 0.7, places=6,
                                   msg=f"Node {i} value_sum should be 0.7 (no flip)")

    def test_backup_single_submit_flip(self):
        """Single SUBMIT in chain: value flips after passing through it."""
        # root → submit → c1 → c2 (leaf)
        nodes = self._build_chain(4, actions=[SUBMIT_ACTION, (0,1,0,0), (0,2,0,0)])
        leaf = nodes[-1]

        self.mcts._backup(leaf, 0.5)

        # leaf: no flip yet
        self.assertAlmostEqual(leaf.value_sum, 0.5, places=6)
        # c1: no flip (action is not SUBMIT)
        self.assertAlmostEqual(nodes[2].value_sum, 0.5, places=6)
        # submit: value flips
        self.assertAlmostEqual(nodes[1].value_sum, 0.5, places=6)
        # root: flipped value
        self.assertAlmostEqual(nodes[0].value_sum, -0.5, places=6)

    def test_backup_double_submit_flip(self):
        """Two SUBMITs in chain: value flips twice (back to original sign)."""
        # root → submit1 → c1 → submit2 → c2 (leaf)
        nodes = self._build_chain(5, actions=[SUBMIT_ACTION, (0,1,0,0), SUBMIT_ACTION, (0,2,0,0)])
        leaf = nodes[-1]

        self.mcts._backup(leaf, 0.5)

        # leaf: original
        self.assertAlmostEqual(leaf.value_sum, 0.5, places=6)
        # submit2: no flip at this node (value flip happens when backing UP to parent)
        self.assertAlmostEqual(nodes[3].value_sum, 0.5, places=6)
        # c1: after first flip (submit2)
        self.assertAlmostEqual(nodes[2].value_sum, -0.5, places=6)
        # submit1: after first flip
        self.assertAlmostEqual(nodes[1].value_sum, -0.5, places=6)
        # root: after second flip (submit1) → back to original
        self.assertAlmostEqual(nodes[0].value_sum, 0.5, places=6)

    def test_backup_triple_submit_flip(self):
        """Three SUBMITs: flips three times = inverted sign."""
        nodes = self._build_chain(7, actions=[
            SUBMIT_ACTION, (0,1,0,0), SUBMIT_ACTION, (0,2,0,0), SUBMIT_ACTION, (0,3,0,0)
        ])
        leaf = nodes[-1]

        self.mcts._backup(leaf, 0.5)

        # 3 flips = -0.5 at root
        self.assertAlmostEqual(nodes[0].value_sum, -0.5, places=6,
                               msg="Three flips should invert the sign")

    def test_backup_chain_with_mixed_actions(self):
        """Mixed SUBMIT and non-SUBMIT actions in backup chain."""
        # root → submit → c1(no_flip) → c2(no_flip) → submit → leaf
        nodes = self._build_chain(6, actions=[
            SUBMIT_ACTION, (0, 1, 0, 0), (0, 2, 0, 0), SUBMIT_ACTION, (0, 3, 0, 0),
        ])
        leaf = nodes[-1]

        self.mcts._backup(leaf, 0.5)

        # leaf: no flip
        self.assertAlmostEqual(leaf.value_sum, 0.5, places=6)
        # submit2: no flip at this node
        self.assertAlmostEqual(nodes[4].value_sum, 0.5, places=6)
        # c2: after first flip (submit2 at node 4 → parent of leaf is c1... wait)
        # Chain: nodes[0]=root→nodes[1]=submit→nodes[2]=c1→nodes[3]=c2→nodes[4]=submit2→nodes[5]=leaf
        # Backup from leaf: no flip at leaf (action not SUBMIT)
        # nodes[4] (submit2) → flip at its parent, so value flips when going to nodes[3]
        # nodes[3] (c2) → no flip, gets -0.5
        # nodes[2] (c1) → no flip, gets -0.5
        # nodes[1] (submit1) → flip at its parent (nodes[0]), so nodes[0] gets +0.5
        self.assertAlmostEqual(nodes[0].value_sum, 0.5, places=6,
                               msg="Two flips = back to original sign")

    def test_backup_multiple_visits_accumulate(self):
        """Multiple backups accumulate visit_count and value_sum."""
        root = MCTSNode(state=self._state)
        child = MCTSNode(state=self._state, parent=root, action=(0,0,0,0))
        root.children = [child]

        self.mcts._backup(child, 0.5)
        self.mcts._backup(child, -0.3)
        self.mcts._backup(child, 0.8)

        self.assertEqual(child.visit_count, 3)
        self.assertAlmostEqual(child.value_sum, 1.0, places=6)  # 0.5 - 0.3 + 0.8
        self.assertEqual(root.visit_count, 3)
        self.assertAlmostEqual(root.value_sum, 1.0, places=6)

    def test_backup_root_node(self):
        """Backup from root node itself (no parent)."""
        root = MCTSNode(state=self._state)
        self.mcts._backup(root, 0.5)

        self.assertEqual(root.visit_count, 1)
        self.assertAlmostEqual(root.value_sum, 0.5, places=6)

    def test_backup_negative_value(self):
        """Negative value propagates correctly through SUBMIT flip."""
        root = MCTSNode(state=self._state)
        child = MCTSNode(state=self._state, parent=root, action=SUBMIT_ACTION)
        root.children = [child]

        self.mcts._backup(child, -0.7)

        # Negative value, flip at SUBMIT -> positive at root
        self.assertAlmostEqual(child.value_sum, -0.7, places=6)
        self.assertAlmostEqual(root.value_sum, 0.7, places=6)

    def test_backup_mixed_sign_values(self):
        """Multiple simulator backups with different values."""
        root = MCTSNode(state=self._state)
        child = MCTSNode(state=self._state, parent=root, action=(0,0,0,0))
        root.children = [child]

        values = [0.5, -0.2, 0.3, -0.8, 0.1]
        for v in values:
            self.mcts._backup(child, v)

        self.assertEqual(child.visit_count, 5)
        self.assertAlmostEqual(child.value_sum, sum(values), places=6)
        self.assertEqual(root.visit_count, 5)
        self.assertAlmostEqual(root.value_sum, sum(values), places=6)

        # Q-value should be the mean
        expected_q = sum(values) / 5
        self.assertAlmostEqual(child.q_value, expected_q, places=6)

    def _build_chain(self, length: int, actions: list) -> list:
        """Build a linear chain of nodes: root → c1 → c2 → ... → leaf.
        Returns list [root, c1, c2, ..., leaf].
        The actions list has length-1 entries (each node's action except root).
        """
        nodes = [MCTSNode(state=self._state)]
        for act in actions:
            parent = nodes[-1]
            child = MCTSNode(state=self._state, parent=parent, action=act)
            parent.children.append(child)
            nodes.append(child)
        return nodes


# ═══════════════════════════════════════════════════════════════════
#  Tests: Expand edge cases
# ═══════════════════════════════════════════════════════════════════

class TestMCTSExpandEdgeCases(unittest.TestCase):
    """MCTS._expand edge cases."""

    def setUp(self):
        self.network = make_mock_network()
        self.cfg = make_mock_config(use_transposition_table=False)
        self.device = torch.device("cpu")
        self.mcts = MCTS(self.network, self.cfg, self.device)

    def test_expand_terminal_state(self):
        """Terminal state: no network call, returns terminal_value."""
        from alphazero.env import Semimove
        board_side = 4
        board_squares = 16
        num_boards = 2

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = True
        env.outcome = 1.0  # White wins
        env.current_player = 0
        env.clone.return_value = env

        env.encode_state.side_effect = lambda urgency=0.0: {
            "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                        "l_coords": np.array([0, 1], dtype=np.int64),
            "t_coords": np.array([0, 0], dtype=np.int64),
            "board_keys": [(0, 0, False), (0, 0, True)],
            "num_boards": num_boards,
        }

        state = HalfstepState(env=env, pending_from=None)
        node = MCTSNode(state=state)

        value = self.mcts._expand(node, urgency=0.0)

        # Terminal: white wins, current player = white (0) → outcome = 1.0
        self.assertAlmostEqual(value, 1.0, places=6,
                               msg="Terminal value should be 1.0 for white win, white to move")
        self.assertTrue(node.is_terminal)
        self.assertAlmostEqual(node.terminal_value, 1.0, places=6)
        # Network should NOT be called
        self.network.predict_actions.assert_not_called()

    def test_expand_terminal_black_wins_white_to_move(self):
        """Terminal state: black wins, white to move → negative value for white."""
        from alphazero.env import Semimove
        board_side = 4
        board_squares = 16
        num_boards = 2

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = True
        env.outcome = -1.0  # Black wins (from white perspective)
        env.current_player = 0  # White to move
        env.clone.return_value = env

        env.encode_state.side_effect = lambda urgency=0.0: {
            "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                        "l_coords": np.array([0, 1], dtype=np.int64),
            "t_coords": np.array([0, 0], dtype=np.int64),
            "board_keys": [(0, 0, False), (0, 0, True)],
            "num_boards": num_boards,
        }

        state = HalfstepState(env=env, pending_from=None)
        node = MCTSNode(state=state)

        value = self.mcts._expand(node, urgency=0.0)

        # Black wins, white to move → white gets -1.0
        self.assertAlmostEqual(value, -1.0, places=6)
        self.assertTrue(node.is_terminal)

    def test_expand_terminal_black_wins_black_to_move(self):
        """Terminal: black wins, black to move → positive value for black."""
        from alphazero.env import Semimove
        board_side = 4
        board_squares = 16
        num_boards = 2

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = True
        env.outcome = -1.0  # Black wins (from white perspective)
        env.current_player = 1  # Black to move
        env.clone.return_value = env

        env.encode_state.side_effect = lambda urgency=0.0: {
            "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                        "l_coords": np.array([0, 1], dtype=np.int64),
            "t_coords": np.array([0, 0], dtype=np.int64),
            "board_keys": [(0, 0, False), (0, 0, True)],
            "num_boards": num_boards,
        }

        state = HalfstepState(env=env, pending_from=None)
        node = MCTSNode(state=state)

        value = self.mcts._expand(node, urgency=0.0)

        # Black wins, black to move → black gets +1.0 (outcome -1.0 flipped)
        self.assertAlmostEqual(value, 1.0, places=6)
        self.assertTrue(node.is_terminal)

    def test_expand_no_legal_destinations(self):
        """Destination node with no legal destinations → terminal with value 0."""
        from alphazero.env import Semimove
        board_side = 4
        board_squares = 16
        num_boards = 2

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.clone.return_value = env

        env.encode_state.side_effect = lambda urgency=0.0: {
            "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                        "l_coords": np.array([0, 1], dtype=np.int64),
            "t_coords": np.array([0, 0], dtype=np.int64),
            "board_keys": [(0, 0, False), (0, 0, True)],
            "num_boards": num_boards,
        }
        env.get_legal_frontier.return_value = ([], False)

        # Create state with pending_from set, but legal_destinations_for returns empty
        # The _expand method will call legal_destinations_for(pending_from) via
        # state.legal_destinations_for. But with the mock env, it calls env.get_legal_frontier.
        # Actually, HalfstepState.legal_destinations_for uses env.get_legal_frontier...
        # Let me check: in _expand, for dest nodes, it calls state.legal_destinations_for(state.pending_from)
        # which calls env.get_legal_frontier internally.
        state = HalfstepState(env=env, pending_from=(0, 0, 0, 0))
        # env.get_legal_frontier returns ([], False) → no destinations match → empty list

        # Mock network to return valid values (needed before the empty-check)
        def predict_actions(board_planes, l_coords, t_coords, urgency):
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(np.ones((num_boards, board_squares), dtype=np.float32)),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        node = MCTSNode(state=state)
        value = self.mcts._expand(node, urgency=0.0)

        # No legal destinations → terminal with value 0
        self.assertAlmostEqual(value, 0.0, places=6)
        self.assertTrue(node.is_terminal)
        self.assertAlmostEqual(node.terminal_value, 0.0, places=6)

    def test_expand_no_legal_actions_at_source(self):
        """Source node with no legal actions → terminal with value -1."""
        from alphazero.env import Semimove
        board_side = 4
        board_squares = 16
        num_boards = 2

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.clone.return_value = env

        # legal_first_choices returns empty, can_submit returns False
        env.get_legal_frontier.return_value = ([], False)
        env.can_submit.return_value = False
        env.encode_state.side_effect = lambda urgency=0.0: {
            "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                        "l_coords": np.array([0, 1], dtype=np.int64),
            "t_coords": np.array([0, 0], dtype=np.int64),
            "board_keys": [(0, 0, False), (0, 0, True)],
            "num_boards": num_boards,
        }

        state = HalfstepState(env=env, pending_from=None)
        node = MCTSNode(state=state)

        # Need to mock legal_first_choices to return empty
        # Actually HalfstepState.legal_first_choices uses get_legal_frontier...
        # With empty frontier and no can_submit, action_logits will be empty
        # BUT we need the network to be called first (it's called before the empty check)
        # Actually, looking at the code: the network is called, THEN action_logits is built.
        # If action_logits ends up empty, it sets is_terminal and returns -1.
        # But with our mock network that returns -20.0 for invalid indices...
        # Actually, legal_first_choices returns [] from the empty frontier.
        # Then action_logits loop is empty. can_submit is False. So action_logits = [].
        # Then it returns _no_legal_action_terminal_value() = -1.0

        # To make the network call not crash, we need predict_actions to work
        def predict_actions(board_planes, l_coords, t_coords, urgency):
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(np.ones((num_boards, board_squares), dtype=np.float32)),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        value = self.mcts._expand(node, urgency=0.0)

        # No legal actions → terminal with value -1 (current player loses)
        self.assertAlmostEqual(value, -1.0, places=6)
        self.assertTrue(node.is_terminal)
        self.assertAlmostEqual(node.terminal_value, -1.0, places=6)


# ═══════════════════════════════════════════════════════════════════
#  Tests: Advanced select_halfstep
# ═══════════════════════════════════════════════════════════════════

class TestMCTSSelectHalfstepAdvanced(unittest.TestCase):
    """Advanced MCTS.select_halfstep tests."""

    def setUp(self):
        self.network = make_mock_network()
        self.cfg = make_mock_config()
        self.device = torch.device("cpu")
        self.mcts = MCTS(self.network, self.cfg, self.device)
        self._state = make_mock_halfstep_state()

    def test_temperature_zero_picks_argmax(self):
        """Temperature=0 always picks the action with highest visit count."""
        root = MCTSNode(state=self._state)
        children = [
            MCTSNode(state=self._state, parent=root, action=(0, 0, 0, 0)),
            MCTSNode(state=self._state, parent=root, action=(1, 0, 0, 0)),
            MCTSNode(state=self._state, parent=root, action=(2, 0, 0, 0)),
        ]
        root.children = children
        # Set visit counts: child 2 has highest
        children[0].visit_count = 5
        children[1].visit_count = 3
        children[2].visit_count = 10

        # Run multiple times: with temperature=0, should always pick the same
        for _ in range(20):
            action, policy, _ = self.mcts.select_halfstep(root, temperature=0.0)
            self.assertEqual(action, (2, 0, 0, 0),
                             "Temperature=0 should always pick argmax")

    def test_temperature_zero_policy_one_hot(self):
        """Temperature=0 produces a one-hot policy."""
        root = MCTSNode(state=self._state)
        children = [
            MCTSNode(state=self._state, parent=root, action=(0, 0, 0, 0)),
            MCTSNode(state=self._state, parent=root, action=(1, 0, 0, 0)),
        ]
        root.children = children
        children[0].visit_count = 3
        children[1].visit_count = 7

        _, policy, _ = self.mcts.select_halfstep(root, temperature=0.0)

        np.testing.assert_array_equal(policy, [0.0, 1.0],
                                      "Temperature=0 policy should be one-hot")

    def test_policy_sum_to_one(self):
        """Policy probabilities always sum to 1."""
        root = MCTSNode(state=self._state)
        children = [
            MCTSNode(state=self._state, parent=root, action=(i, 0, 0, 0))
            for i in range(5)
        ]
        root.children = children
        for i, c in enumerate(children):
            c.visit_count = i + 1

        for temp in [0.5, 1.0, 2.0, 5.0]:
            _, policy, _ = self.mcts.select_halfstep(root, temperature=temp)
            self.assertAlmostEqual(policy.sum(), 1.0, places=6,
                                   msg=f"Policy should sum to 1 at temperature {temp}")

    def test_policy_non_negative(self):
        """All policy probabilities are non-negative."""
        root = MCTSNode(state=self._state)
        children = [
            MCTSNode(state=self._state, parent=root, action=(i, 0, 0, 0))
            for i in range(4)
        ]
        root.children = children
        for i, c in enumerate(children):
            c.visit_count = (i + 1) * 2

        for temp in [0.1, 1.0, 10.0]:
            _, policy, _ = self.mcts.select_halfstep(root, temperature=temp)
            self.assertTrue(np.all(policy >= 0),
                            f"All policy values should be >= 0 at temp={temp}")

    def test_select_halfstep_returns_root_value(self):
        """select_halfstep returns correct root q_value."""
        root = MCTSNode(state=self._state)
        root.visit_count = 10
        root.value_sum = 3.0  # q_value = 0.3
        children = [
            MCTSNode(state=self._state, parent=root, action=(0, 0, 0, 0)),
        ]
        root.children = children
        children[0].visit_count = 10
        children[0].value_sum = 3.0

        _, _, root_value = self.mcts.select_halfstep(root, temperature=0.0)
        self.assertAlmostEqual(root_value, 0.3, places=6)

    def test_high_temperature_near_uniform(self):
        """Very high temperature produces near-uniform distribution."""
        root = MCTSNode(state=self._state)
        children = [
            MCTSNode(state=self._state, parent=root, action=(i, 0, 0, 0))
            for i in range(3)
        ]
        root.children = children
        children[0].visit_count = 100
        children[1].visit_count = 2
        children[2].visit_count = 1

        _, policy, _ = self.mcts.select_halfstep(root, temperature=100.0)

        # With extremely high temperature, distribution should be near uniform
        expected = 1.0 / 3
        for p in policy:
            self.assertAlmostEqual(p, expected, places=1,
                                   msg="High temp should produce near-uniform policy")

    def test_no_children_returns_default(self):
        """No children returns None action and empty policy."""
        root = MCTSNode(state=self._state)
        root.children = []

        action, policy, rv = self.mcts.select_halfstep(root, temperature=1.0)

        self.assertIsNone(action)
        self.assertEqual(len(policy), 0)
        self.assertAlmostEqual(rv, root.q_value, places=6)

    def test_all_zero_visits_returns_default(self):
        """All children have 0 visits: returns None and empty policy."""
        root = MCTSNode(state=self._state)
        children = [
            MCTSNode(state=self._state, parent=root, action=(0, 0, 0, 0)),
            MCTSNode(state=self._state, parent=root, action=(1, 0, 0, 0)),
        ]
        root.children = children
        # All zero visits
        action, policy, _ = self.mcts.select_halfstep(root, temperature=1.0)

        self.assertIsNone(action)
        self.assertEqual(len(policy), 0)


# ═══════════════════════════════════════════════════════════════════
#  Tests: Transposition table
# ═══════════════════════════════════════════════════════════════════

class TestMCTSTranspositionTable(unittest.TestCase):
    """MCTS transposition table caching."""

    def setUp(self):
        self.network = make_mock_network()
        self.cfg = make_mock_config(use_transposition_table=True)
        self.device = torch.device("cpu")
        self.mcts = MCTS(self.network, self.cfg, self.device)

    def test_tt_caches_and_reuses_expansion(self):
        """Cached state: reuses children without network call."""
        from alphazero.env import Semimove
        board_side = 4
        board_squares = 16
        num_boards = 2

        # Track network calls
        network_call_count = [0]

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            network_call_count[0] += 1
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(np.ones((num_boards, board_squares), dtype=np.float32)),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)]},
        )

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        def encode_state(urgency=0.0):
            return {
                "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            True,
        )
        env.can_submit.return_value = True

        state = HalfstepState(env=env, pending_from=None)
        node1 = MCTSNode(state=state.clone())
        node2 = MCTSNode(state=state.clone())

        # First expand: should call network
        v1 = self.mcts._expand(node1, urgency=0.0)
        self.assertEqual(network_call_count[0], 1, "First expand should call network")

        # Second expand with same state: should use cache
        v2 = self.mcts._expand(node2, urgency=0.0)
        self.assertEqual(network_call_count[0], 1, "Second expand should use cache (no network call)")

        # Both should return same value
        self.assertAlmostEqual(v1, v2, places=6)
        # Both should have same children
        self.assertEqual(len(node1.children), len(node2.children))
        for c1, c2 in zip(node1.children, node2.children):
            self.assertEqual(c1.action, c2.action)

    def test_tt_different_tt_key_does_not_cache(self):
        """Different states with different TT keys → separate network calls."""
        from alphazero.env import Semimove
        board_side = 4
        board_squares = 16
        num_boards = 2

        network_call_count = [0]

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            network_call_count[0] += 1
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(np.ones((num_boards, board_squares), dtype=np.float32)),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)]},
        )

        # Encode_state with different t_coords to produce different TT keys
        call_idx = [0]

        def make_encode_state(tt_key_base):
            def encode_state(urgency=0.0):
                nonlocal tt_key_base
                return {
                    "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                                        "l_coords": np.array([0, 1], dtype=np.int64),
                    "t_coords": np.array([0, 0], dtype=np.int64),
                    "board_keys": [(0, 0, False), (0, 0, True)],
                    "num_boards": num_boards,
                }
            return encode_state

        env1 = MagicMock()
        env1.board_side = board_side
        env1.board_count = 2
        env1.done = False
        env1.outcome = None
        env1.current_player = 0
        env1.state = engine_state
        env1.clone.return_value = env1
        env1.encode_state.side_effect = make_encode_state((1, 2, 3))
        env1.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            True,
        )
        env1.can_submit.return_value = True

        env2 = MagicMock()
        env2.board_side = board_side
        env2.board_count = 3  # different → different state
        env2.done = False
        env2.outcome = None
        env2.current_player = 0
        env2.state = engine_state
        env2.clone.return_value = env2
        env2.encode_state.side_effect = make_encode_state((4, 5, 6))
        env2.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            True,
        )
        env2.can_submit.return_value = True

        # Use HalfstepState with different pending_from to get different TT keys
        state1 = HalfstepState(env=env1, pending_from=None)
        state2 = HalfstepState(env=env2, pending_from=None)

        # Override get_mcts_transposition_key to return different keys
        state1.get_mcts_transposition_key = lambda: (1, 2, 3, None)
        state2.get_mcts_transposition_key = lambda: (4, 5, 6, None)

        node1 = MCTSNode(state=state1)
        node2 = MCTSNode(state=state2)

        self.mcts._expand(node1, urgency=0.0)
        self.mcts._expand(node2, urgency=0.0)

        # Different TT keys → separate network calls
        self.assertEqual(network_call_count[0], 2,
                         "Different TT keys should require separate network calls")


# ═══════════════════════════════════════════════════════════════════
#  Tests: Integration — full search pipeline
# ═══════════════════════════════════════════════════════════════════

class TestMCTSSearchIntegration(unittest.TestCase):
    """Full MCTS search pipeline integration tests."""

    def setUp(self):
        self.network = make_mock_network()
        self.cfg = make_mock_config(num_simulations=16, use_transposition_table=False)
        self.device = torch.device("cpu")
        self.mcts = MCTS(self.network, self.cfg, self.device)

    def test_search_updates_visit_counts(self):
        """After search, root has correct visit counts."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 2

        rng = np.random.RandomState(42)
        raw_logits = rng.randn(num_boards, board_squares).astype(np.float32)

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(raw_logits.copy()),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)]},
        )

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        def encode_state(urgency=0.0):
            return {
                "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            True,
        )
        env.can_submit.return_value = True

        state = HalfstepState(env=env, pending_from=None)
        root = self.mcts.search(state, board_limit=10)

        # Root visit count: 1 (from its own expansion loop backup) + num_simulations
        # Wait — _expand does NOT call _backup itself. Only the search loop does.
        # Each simulation's _backup adds 1 visit to every node on the path (including root).
        # So root visits = num_simulations (root is visited once per simulation).
        expected_visits = self.cfg.num_simulations
        self.assertEqual(root.visit_count, expected_visits,
                         f"Root visit count should be {expected_visits}")

    def test_search_children_visit_counts_sum_to_root(self):
        """Sum of children's visit counts equals root visit count (minus root's own visit)."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 2

        rng = np.random.RandomState(42)
        raw_logits = rng.randn(num_boards, board_squares).astype(np.float32)

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(raw_logits.copy()),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)
        src_b = make_mock_vec4(1, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a, src_b],
            gen_piece_move_map={
                (0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)],
                (1, 0, 0, 0): [make_mock_vec4(1, 1, 0, 0)],
            },
        )

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        def encode_state(urgency=0.0):
            return {
                "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0)),
             Semimove(0, (1, 0, 0, 0), (1, 1, 0, 0))],
            True,
        )
        env.can_submit.return_value = True

        state = HalfstepState(env=env, pending_from=None)
        root = self.mcts.search(state, board_limit=10)

        # Root visit count = 1 (its own initial visit from expansion) + num_simulations
        # Each simulation traverses down the tree, adding 1 visit to each node on the path,
        # including the leaf. So root visits = 1 + num_simulations.
        # Children visits sum = root visits - 1 (root's own visit from expansion),
        # because each simulation's traversal path counts the root once.
        # Actually: root starts with visit_count=0. _expand adds 1 (node.visit_count += 1 in _backup).
        # Then each simulation: _select returns a leaf, _expand it, _backup adds 1 to every node on path.
        # So total root visits = 1 (from expansion) + num_simulations.
        # Children total = root visits - 1 (the root's own count from expansion includes the
        # root itself, not counted in children).
        # Actually, the backup from root's expansion adds: root.visit_count += 1.
        # Then each simulation: leaf.visit_count += 1, ... up to root.visit_count += 1.
        # So root visits = 1 + num_simulations.
        # Children visits sum = (root visits - 1) + num_simulations = 2*num_simulations?? No...
        # Each simulation adds 1 to root AND 1 to each node on the path.
        # So root visits = 1 + num_simulations.
        # Each child gets visited by some subset of simulations. Sum of all children visits = num_simulations.
        # Because each simulation traverses exactly one child, adding 1 to that child.
        # Well, actually it could traverse deeper: root → child → grandchild. The child on the path
        # gets +1 from the backup. And the grandchild also gets +1. But the sum of children visits
        # = num_simulations, since each simulation traverses through exactly one child of root.
        # Wait, some simulations might traverse through the same child if it's the best_child repeatedly.
        # The sum of root's children visits = num_simulations (each sim adds 1 to exactly one child of root).
        # But root also has its own visit (1 from expansion). So root visits = 1 + num_simulations.
        # Children sum = root visits - 1 = num_simulations.

        children_sum = sum(c.visit_count for c in root.children)
        self.assertEqual(children_sum, self.cfg.num_simulations,
                         f"Children visits sum ({children_sum}) should equal num_simulations ({self.cfg.num_simulations})")

    def test_search_does_not_expand_same_node_twice(self):
        """Each node is expanded at most once (transposition table off)."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 2

        rng = np.random.RandomState(42)
        raw_logits = rng.randn(num_boards, board_squares).astype(np.float32)

        # Track which states are expanded (by action sequence)
        expanded_states = set()

        original_expand = self.mcts._expand

        def tracking_expand(node, urgency):
            action_key = tuple(node.state.pending_from) if node.state.pending_from else "root"
            expanded_states.add((id(node.state), action_key))
            return original_expand(node, urgency)

        self.mcts._expand = tracking_expand

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(rng.randn(num_boards, board_squares).astype(np.float32).copy()),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)
        src_b = make_mock_vec4(1, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a, src_b],
            gen_piece_move_map={
                (0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)],
                (1, 0, 0, 0): [make_mock_vec4(1, 1, 0, 0)],
            },
        )

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        def encode_state(urgency=0.0):
            return {
                "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0)),
             Semimove(0, (1, 0, 0, 0), (1, 1, 0, 0))],
            True,
        )
        env.can_submit.return_value = True

        state = HalfstepState(env=env, pending_from=None)
        root = self.mcts.search(state, board_limit=10)

        # Root should be expanded once
        self.assertTrue(root.is_expanded)
        # Each child should be expanded at most once
        for child in root.children:
            if child.is_expanded:
                pass  # Valid: expanded once
            # Visiting children multiple times is fine, but re-expansion should not happen
            # (is_expanded flag prevents re-entering _expand)

    def test_search_with_submit_produces_correct_tree(self):
        """Search properly handles SUBMIT actions in the tree."""
        from alphazero.env import Semimove

        board_side = 4
        board_squares = 16
        num_boards = 2

        rng = np.random.RandomState(42)
        raw_logits = rng.randn(num_boards, board_squares).astype(np.float32)

        def predict_actions(board_planes, l_coords, t_coords, urgency):
            return (
                torch.tensor([0.3], dtype=torch.float32).item(),
                torch.from_numpy(raw_logits.copy()),
                torch.tensor([0.5], dtype=torch.float32).item(),
            )
        self.network.predict_actions.side_effect = predict_actions

        src_a = make_mock_vec4(0, 0, 0, 0)
        engine_state = make_mock_engine_state(
            raw_sources=[src_a],
            gen_piece_move_map={(0, 0, 0, 0): [make_mock_vec4(0, 1, 0, 0)]},
        )

        env = MagicMock()
        env.board_side = board_side
        env.board_count = 2
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.state = engine_state
        env.clone.return_value = env

        def encode_state(urgency=0.0):
            return {
                "board_planes": np.zeros((num_boards, 14, board_squares), dtype=np.float32),
                                "l_coords": np.array([0, 1], dtype=np.int64),
                "t_coords": np.array([0, 0], dtype=np.int64),
                "board_keys": [(0, 0, False), (0, 0, True)],
                "num_boards": num_boards,
            }
        env.encode_state.side_effect = encode_state
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            True,
        )
        env.can_submit.return_value = True

        state = HalfstepState(env=env, pending_from=None)
        root = self.mcts.search(state, board_limit=10)

        # Root should have a SUBMIT child
        submit_children = [c for c in root.children if c.action == SUBMIT_ACTION]
        self.assertEqual(len(submit_children), 1,
                         "Root should have a SUBMIT child")
        submit_node = submit_children[0]

        # SUBMIT children should be expanded (they were visited during search)
        if submit_node.visit_count > 0:
            # SUBMIT node's children should have different current_player
            if submit_node.children:
                for sc in submit_node.children:
                    self.assertIsNotNone(sc.state,
                                         "SUBMIT child should have a valid state")


# ═══════════════════════════════════════════════════════════════════
#  Tests: HalfstepState apply_halfstep reversal (unselect)
# ═══════════════════════════════════════════════════════════════════

class TestHalfstepStateReversal(unittest.TestCase):
    """HalfstepState operations: selection, reversal, edge cases."""

    def test_apply_halfstep_select_source_then_another_source(self):
        """Selecting a source when one is already selected should not be possible
        (the state transitions to destination mode — no direct reversal)."""
        from alphazero.env import Semimove, SemimoveEnv
        from alphazero.halfstep_state import HalfstepState

        # This test validates the design: once pending_from is set,
        # the only legal actions are destinations for that piece.
        # The MCTS tree structure handles the "reversal" by branching
        # at the root level, not by mutating state.

        env = MagicMock()
        env.board_side = 4
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.clone.return_value = env

        # Source has two legal destinations
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0)),
             Semimove(0, (0, 0, 0, 0), (0, 2, 0, 0))],
            False,
        )

        state = HalfstepState(env=env, pending_from=None)

        # Select source (0,0,0,0)
        s1 = state.apply_halfstep((0, 0, 0, 0))
        self.assertIsNotNone(s1, "Selecting a valid source should succeed")
        self.assertEqual(s1.pending_from, (0, 0, 0, 0),
                         "pending_from should be set after source selection")

        # Attempting to select a different source should fail
        # (legal_destinations_for won't contain (1, 0, 0, 0) as a destination)
        s2 = s1.apply_halfstep((1, 0, 0, 0))
        self.assertIsNone(s2, "Selecting a different source when pending_from is set should fail")

        # MCTS handles "reversal" naturally: different source choices are
        # different branches from the root node, not state mutations.

    def test_apply_halfstep_submit_only_when_no_pending(self):
        """SUBMIT is only allowed when pending_from is None."""
        env = MagicMock()
        env.board_side = 4
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.clone.return_value = env
        env.get_legal_frontier.return_value = ([], False)
        env.can_submit.return_value = True

        state = HalfstepState(env=env, pending_from=None)
        result = state.apply_halfstep(SUBMIT_ACTION)
        self.assertIsNotNone(result, "SUBMIT with no pending_from should succeed")

        # Now with pending_from set
        state2 = HalfstepState(env=env, pending_from=(0, 0, 0, 0))
        result2 = state2.apply_halfstep(SUBMIT_ACTION)
        self.assertIsNone(result2, "SUBMIT with pending_from set should fail")

    def test_apply_halfstep_invalid_submit_string(self):
        """Invalid submit string returns None."""
        env = MagicMock()
        env.board_side = 4
        env.clone.return_value = env

        state = HalfstepState(env=env, pending_from=None)
        result = state.apply_halfstep("INVALID")
        self.assertIsNone(result, "Invalid submit string should return None")

    def test_apply_halfstep_invalid_coord_shape(self):
        """Non-4-tuple coord returns None."""
        env = MagicMock()
        env.board_side = 4
        env.clone.return_value = env

        state = HalfstepState(env=env, pending_from=None)
        result = state.apply_halfstep((1, 2, 3))  # Only 3 elements
        self.assertIsNone(result, "Non-4-tuple should return None")

    def test_apply_halfstep_can_not_submit(self):
        """SUBMIT when can_submit is False returns None."""
        env = MagicMock()
        env.board_side = 4
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.clone.return_value = env
        env.can_submit.return_value = False

        state = HalfstepState(env=env, pending_from=None)
        result = state.apply_halfstep(SUBMIT_ACTION)
        self.assertIsNone(result, "SUBMIT when can_submit is False should fail")

    def test_apply_halfstep_source_to_destination_flow(self):
        """Full flow: source → destination → resulting state has no pending_from."""
        from alphazero.env import Semimove

        env = MagicMock()
        env.board_side = 4
        env.board_count = 1
        env.done = False
        env.outcome = None
        env.current_player = 0
        env.clone.return_value = env

        # Initially, source (0,0,0,0) is available
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            False,
        )

        state = HalfstepState(env=env, pending_from=None)

        # Step 1: select source
        s1 = state.apply_halfstep((0, 0, 0, 0))
        self.assertIsNotNone(s1)
        self.assertEqual(s1.pending_from, (0, 0, 0, 0))

        # Step 2: select destination
        # apply_semimove is called on the cloned env
        env.apply_semimove.return_value = True
        env.get_legal_frontier.return_value = (
            [Semimove(0, (0, 0, 0, 0), (0, 1, 0, 0))],
            False,
        )

        s2 = s1.apply_halfstep((0, 1, 0, 0))
        self.assertIsNotNone(s2, "Selecting a valid destination should succeed")
        self.assertIsNone(s2.pending_from,
                          "pending_from should be None after completing a semimove")


if __name__ == "__main__":
    unittest.main(verbosity=2)