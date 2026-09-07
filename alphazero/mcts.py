# alphazero/mcts.py
"""
Standard AlphaZero MCTS for halfstep-level 5D Chess.

A halfstep is either selecting a source piece (src_coord) or selecting a
destination (dst_coord). Two halfsteps form one semimove (from → to).
A SUBMIT action ends the turn.

The tree is built one halfstep at a time: every node is a regular MCTS node
holding a HalfstepState, and every edge is one halfstep action. There is no
special source/destination node type — the state itself knows what legal
actions are available (via pending_from).
"""

import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .config import MCTSConfig
from .halfstep_state import HalfstepState, Coord4, SUBMIT_ACTION


@dataclass(slots=True)
class TranspositionEntry:
    """Cached expansion result for one state."""
    value: float
    child_specs: list[Tuple]  # (action, prior) tuples
    is_terminal: bool
    terminal_value: float = 0.0


class MCTSNode:
    """Standard AlphaZero MCTS node.

    Each node stores the full game state. The tree is uniform — every node
    plays the same role, and the state determines what legal actions exist.
    """

    __slots__ = [
        "parent", "action", "state", "prior",
        "visit_count", "value_sum",
        "children", "is_expanded", "is_terminal", "terminal_value",
    ]

    def __init__(self, state: HalfstepState, parent: Optional["MCTSNode"] = None,
                 action=None, prior: float = 0.0):
        self.parent = parent
        self.action = action          # halfstep action or SUBMIT that led here
        self.state = state            # the game state at this node
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: list[MCTSNode] = []
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, c_puct: float) -> float:
        """PUCT score = Q + c_puct * prior * sqrt(N_parent) / (1 + N_child)."""
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct: float) -> "MCTSNode":
        def key(c):
            q = c.q_value
            # 子节点行棋方与父节点不同 → 经过了回合边界 → Q 是对方视角，取负
            if c.state.current_player() != self.state.current_player():
                q = -q
            exploration = c_puct * c.prior * math.sqrt(self.visit_count) / (1 + c.visit_count)
            return q + exploration
        return max(self.children, key=key)


class MCTS:
    """Standard AlphaZero MCTS engine.

    Usage:
        mcts = MCTS(network, cfg, device)
        root = mcts.search(root_state, board_limit)
        halfstep_action, policy_probs, root_value = mcts.select_halfstep(root, temperature)
    """

    def __init__(self, network, cfg: MCTSConfig, device: torch.device):
        self.network = network
        self.cfg = cfg
        self.device = device
        self._transposition_table: dict[tuple, TranspositionEntry] = {}

    # ── Public API ───────────────────────────────────────────────────────

    def search(self, root_state: HalfstepState, board_limit: int,
               use_full_search: bool = True) -> MCTSNode:
        """Run MCTS from root_state. Returns the expanded root node.

        Args:
            root_state: The state to search from.
            board_limit: Board count limit for forced termination.
            use_full_search: If True (default), use cfg.num_simulations and add
                Dirichlet noise. If False, use cfg.playout_cap_randomization_fast_sims
                and skip Dirichlet noise — used for fast-search turns in playout cap
                randomization (Wu 2020, arXiv:1902.10565).
        """
        self._transposition_table.clear()
        root = MCTSNode(state=root_state.clone())
        self._expand(root, board_limit)

        # Add Dirichlet noise to root children (full search only; fast search
        # skips noise to maximize strength per the paper).
        if use_full_search and root.children:
            noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(root.children))
            eps = self.cfg.dirichlet_epsilon
            for child, n in zip(root.children, noise):
                child.prior = (1 - eps) * child.prior + eps * n

        num_sims = (self.cfg.num_simulations if use_full_search
                     else self.cfg.playout_cap_randomization_fast_sims)
        for _ in range(num_sims):
            leaf = self._select(root)

            if leaf.is_terminal:
                value = leaf.terminal_value
            else:
                value = self._expand(leaf, board_limit)

            self._backup(leaf, value)

        return root

    def select_halfstep(self, root: MCTSNode, temperature: float = 1.0) -> tuple:
        """Select a single halfstep action from the root's visit distribution.

        Returns:
            halfstep_action: Coord4 or SUBMIT_ACTION or None
            policy_probs: [A] raw visit-count distribution (for training target)
            root_value: float
        """
        children = root.children
        if not children:
            return None, np.zeros((0,), dtype=np.float32), root.q_value

        visits = np.array([c.visit_count for c in children], dtype=np.float64)
        total_visits = visits.sum()
        if total_visits <= 0:
            return None, np.zeros((0,), dtype=np.float32), root.q_value

        # ── Raw policy (for training target): normalized visit counts ──
        raw_policy = visits / total_visits

        # ── Sampling policy (temperature sharpened, for action selection) ──
        if temperature < 1e-3:
            sampling_policy = np.zeros_like(visits)
            sampling_policy[visits.argmax()] = 1.0
        else:
            scaled = visits.copy()
            if temperature != 1.0:
                scaled = np.power(scaled, 1.0 / temperature)
            sampling_policy = scaled / scaled.sum()

        # ── Select action using sampling policy ──
        if temperature < 1e-3:
            idx = int(visits.argmax())
        else:
            idx = int(np.random.choice(len(children), p=sampling_policy))

        return children[idx].action, raw_policy.astype(np.float32), root.q_value

    # ── Internal: selection, expansion, backup ───────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse tree following PUCT until reaching a leaf node."""
        while node.is_expanded and not node.is_terminal and node.children:
            node = node.best_child(self.cfg.c_puct)
        return node

    def _expand(self, node: MCTSNode, board_limit: int) -> float:
        """Evaluate leaf node with network and create child nodes.

        Args:
            node: leaf node to expand.
            board_limit: board count limit for forced termination, used to
                         compute urgency (boards remaining) from the leaf's
                         actual board count — each leaf may have a different
                         number of boards.

        Returns the value estimate for this node (current-player perspective).
        """
        state = node.state

        # Urgency: boards remaining until forced end, computed from the leaf's
        # actual board count (which may differ from the root's if new timelines
        # were opened during search).
        board_count = node.state.env.board_count
        urgency = max(0.0, float(board_limit - board_count))

        # Terminal check
        if state.is_done():
            outcome = state.get_outcome() or 0.0
            node.is_terminal = True
            node.terminal_value = self._white_to_current_player_value(
                outcome, state.current_player()
            )
            return node.terminal_value

        # ── Transposition table lookup ──
        tt_key = None
        if self.cfg.use_transposition_table:
            tt_key = state.get_mcts_transposition_key()
            cached = self._transposition_table.get(tt_key)
            if cached is not None:
                node.is_expanded = True
                node.is_terminal = cached.is_terminal
                node.terminal_value = cached.terminal_value
                if not cached.is_terminal:
                    for action, prior in cached.child_specs:
                        child_state = state.apply_halfstep(action)
                        if child_state is not None:
                            node.children.append(
                                MCTSNode(state=child_state, parent=node, action=action, prior=prior)
                            )
                return cached.value

        # ── Encode state for network ──
        encoded = state.env.encode_state(
            urgency=urgency,
            pending_from=state.pending_from,
        )
        board_keys = encoded["board_keys"]

        # Network forward
        board_planes = torch.from_numpy(encoded["board_planes"]).to(self.device)
        l_coords = torch.from_numpy(encoded["l_coords"]).to(self.device)
        t_coords = torch.from_numpy(encoded["t_coords"]).to(self.device)
        urg_tensor = torch.tensor([urgency], dtype=torch.long, device=self.device)
        stm = encoded.get("current_player", 0)

        with torch.no_grad():
            stm_tensor = torch.tensor([stm], dtype=torch.long, device=self.device)
            v, sl, rl = self.network.forward(
                board_planes.unsqueeze(0),
                l_coords.unsqueeze(0),
                t_coords.unsqueeze(0),
                urg_tensor,
                side_to_move=stm_tensor,
            )
            value = v.item()
            raw_logits = rl.squeeze(0)
            submit_logit = sl.item()

        raw_logits_np = raw_logits.detach().cpu().numpy().astype(np.float32)
        N_boards = raw_logits_np.shape[0]
        board_sq = raw_logits_np.shape[1]

        # ── Get legal halfstep actions and their logits ──
        if state.pending_from is None:
            # Source node: unique source squares + submit
            sources = state.legal_first_choices()
            can_submit = state.env.can_submit()
            action_logits: list[tuple] = []
            for src in sources:
                b_idx = self._find_board_index(board_keys, src, state.current_player())
                fx, fy = src[0], src[1]
                sq = fx + fy * state.env.board_side
                logit = raw_logits_np[b_idx, sq] \
                    if (0 <= b_idx < N_boards and 0 <= sq < board_sq) else -20.0
                action_logits.append((src, logit))
            if can_submit:
                action_logits.append((SUBMIT_ACTION, float(submit_logit)))
        else:
            # Destination node: legal destinations for selected piece
            dests = state.legal_destinations_for(state.pending_from)
            if not dests:
                node.is_terminal = True
                node.terminal_value = 0.0
                return 0.0
            action_logits = []
            for dst in dests:
                b_idx = self._find_board_index(board_keys, dst, state.current_player())
                tx, ty = dst[0], dst[1]
                sq = tx + ty * state.env.board_side
                logit = raw_logits_np[b_idx, sq] \
                    if (0 <= b_idx < N_boards and 0 <= sq < board_sq) else -20.0
                action_logits.append((dst, logit))

        if not action_logits:
            node.is_terminal = True
            node.terminal_value = self._no_legal_action_terminal_value()
            return node.terminal_value

        # ── Softmax → priors ──
        logits_arr = np.array([l for _, l in action_logits], dtype=np.float32)
        logits_arr = np.clip(logits_arr, -20.0, 20.0)
        logits_arr -= logits_arr.max()
        exp_l = np.exp(logits_arr)
        priors = exp_l / (exp_l.sum() + 1e-8)

        # ── Create children ──
        node.is_expanded = True
        for (action, _), prior in zip(action_logits, priors):
            child_state = state.apply_halfstep(action)
            if child_state is not None:
                node.children.append(
                    MCTSNode(state=child_state, parent=node, action=action, prior=prior)
                )

        # ── Cache in transposition table ──
        if tt_key is not None:
            self._transposition_table[tt_key] = TranspositionEntry(
                value=value,
                child_specs=[(c.action, c.prior) for c in node.children],
                is_terminal=False,
            )

        return value

    def _backup(self, node: MCTSNode, value: float):
        """Propagate value up the tree.

        Value flips at SUBMIT nodes (turn boundary → player changes).
        """
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            if node.action == SUBMIT_ACTION:
                value = -value
            node = node.parent

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _white_to_current_player_value(outcome_white: float, current_player: int) -> float:
        return outcome_white if current_player == 0 else -outcome_white

    @staticmethod
    def _no_legal_action_terminal_value() -> float:
        return -1.0

    def _find_board_index(self, board_keys: list[tuple], coord: Coord4, player: int) -> int:
        target_l = coord[3]
        target_t = coord[2]
        target_c = bool(player)
        for i, (l, t, c) in enumerate(board_keys):
            if l == target_l and t == target_t and c == target_c:
                return i
        for i, (l, t, c) in enumerate(board_keys):
            if l == target_l and t == target_t:
                return i
        return -1