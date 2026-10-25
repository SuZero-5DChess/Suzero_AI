"""PUCT search over small-step 5D chess.

This is a compact tree search, not a DAG: repeated positions are not merged and
the caller bounds depth.  It mirrors the AlphaZero semantics used by the C++
search adapter: values are current-player relative, signs flip only when the
player actually changes, SUBMIT is auto-played when forced, and the policy
target is the raw root visit distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from .config import MctsConfig
from .env import GameEnv
from .network import SuzeroNetwork
from .protocol import VariantSpec


class Evaluator(Protocol):
    def evaluate(self, env: GameEnv) -> tuple[np.ndarray, float]:
        """Return (priors over env.actions, value for the player to move)."""


class TorchEvaluator:
    """Run the Python network on one position for MCTS."""

    def __init__(
        self, network: SuzeroNetwork, device: torch.device | str = "cpu"
    ) -> None:
        self.network = network
        self.device = torch.device(device)

    @torch.inference_mode()
    def evaluate(self, env: GameEnv) -> tuple[np.ndarray, float]:
        actions = env.actions
        has_submit = bool(actions) and bool(actions[-1][0])
        non_submit = actions[:-1] if has_submit else actions
        q = len(non_submit)
        if q < 1:
            raise ValueError("the evaluator requires at least one non-submit action")

        encoded = env.encode()
        device = self.device
        board_planes = torch.from_numpy(encoded.board_planes).unsqueeze(0).to(device)
        l_coords = torch.from_numpy(encoded.l_coords).unsqueeze(0).to(device)
        t_coords = torch.from_numpy(encoded.t_coords).unsqueeze(0).to(device)
        used = torch.tensor(
            [encoded.num_boards], dtype=torch.int64, device=device
        )
        urgency = torch.tensor(
            [encoded.urgency], dtype=torch.float32, device=device
        )
        side = torch.tensor([encoded.side_to_move], dtype=torch.int64, device=device)
        actions_np = np.asarray(
            [action[1] for action in non_submit], dtype=np.int64
        ).reshape(q, 8)
        action_tensor = torch.from_numpy(actions_np).unsqueeze(0).to(device)

        value, submit_logit, other_logits = self.network(
            board_planes,
            l_coords,
            t_coords,
            used,
            urgency,
            side,
            action_tensor,
        )

        logits = other_logits[0].to(torch.float32).cpu().numpy().copy()
        if has_submit:
            logits = np.concatenate(
                (logits, submit_logit[0].to(torch.float32).cpu().numpy())
            )
        logits = np.clip(logits, -20.0, 20.0)
        logits = logits - logits.max()
        weights = np.exp(logits)
        priors = (weights / weights.sum()).astype(np.float32)
        value_scalar = float(value[0, 0].to(torch.float32).cpu())
        if not np.isfinite(value_scalar) or not np.isfinite(priors).all():
            raise RuntimeError("network produced a non-finite value or policy")
        return priors, value_scalar


@dataclass
class Node:
    state: GameEnv
    player: int
    priors: np.ndarray | None = None
    children: dict[int, "Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    terminal: bool = False
    terminal_value: float = 0.0
    expanded: bool = False

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def action_count(self) -> int:
        return 0 if self.priors is None else int(self.priors.shape[0])

    def visit_policy(self) -> np.ndarray:
        if self.priors is None:
            raise ValueError("node has not been expanded")
        counts = np.zeros(self.action_count, dtype=np.float64)
        for action, child in self.children.items():
            counts[action] = child.visits
        total = counts.sum()
        if total <= 0:
            return np.array(self.priors, dtype=np.float32, copy=True)
        return (counts / total).astype(np.float32)


class MCTS:
    """Serial PUCT search for one root position."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: MctsConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or MctsConfig()
        self.rng = rng or np.random.default_rng()
        self.root: Node | None = None

    # --------------------------------------------------------------- search
    def run(
        self,
        state: GameEnv,
        *,
        add_noise: bool = False,
        num_simulations: int | None = None,
    ) -> Node:
        root_state = state.clone()
        if root_state.is_terminal:
            raise ValueError("cannot search from a terminal position")
        root = Node(root_state, root_state.current_player)
        self.root = root
        self._expand(root)

        if add_noise and self.config.dirichlet_epsilon > 0 and root.action_count > 1:
            noise = self.rng.dirichlet(
                np.full(root.action_count, self.config.dirichlet_alpha)
            )
            epsilon = self.config.dirichlet_epsilon
            root.priors = (
                (1.0 - epsilon) * root.priors + epsilon * noise
            ).astype(np.float32)

        simulations = (
            self.config.num_simulations if num_simulations is None else num_simulations
        )
        if simulations <= 0:
            raise ValueError("num_simulations must be positive")
        for _ in range(simulations):
            self._simulate(root)
        return root

    # ------------------------------------------------------------- expansion
    def _expand(self, node: Node) -> float:
        if node.expanded:
            # Terminal values live in ``terminal_value``; they are not
            # accumulated into ``value_sum`` until a playout visits the node.
            return node.terminal_value if node.terminal else node.q

        state = node.state
        if state.is_terminal:
            node.terminal = True
            node.terminal_value = state.terminal_utility
            node.expanded = True
            return node.terminal_value

        actions = state.actions
        if not actions:
            # The engine normally turns this into a terminal state already.
            node.terminal = True
            node.terminal_value = -1.0
            node.expanded = True
            return -1.0

        if len(actions) == 1 and actions[0][0]:
            # Forced SUBMIT: no network call.  The node inherits the successor
            # value (converted to this node's player perspective).
            node.priors = np.ones(1, dtype=np.float32)
            node.expanded = True
            child = self._make_child(node, 0)
            node.children[0] = child
            child_value = self._expand(child)
            return child_value if child.player == node.player else -child_value

        priors, value = self.evaluator.evaluate(state)
        expected = len(actions)
        if priors.shape != (expected,):
            raise ValueError(
                f"evaluator returned {priors.shape[0]} priors for {expected} actions"
            )
        node.priors = priors.astype(np.float32, copy=False)
        node.expanded = True
        return float(value)

    def _make_child(self, node: Node, action: int) -> Node:
        state = node.state.clone()
        state.apply_action(action)
        player = state.current_player
        child = Node(state, player)
        if state.is_terminal:
            child.terminal = True
            child.terminal_value = state.terminal_utility
            child.expanded = True
        return child

    # ------------------------------------------------------------- selection
    def _select_action(self, node: Node) -> int:
        assert node.priors is not None
        total = sum(child.visits for child in node.children.values())
        sqrt_total = math.sqrt(max(1, total))
        best_action = 0
        best_score = -math.inf
        for action, prior in enumerate(node.priors):
            child = node.children.get(action)
            if child is None:
                q = self.config.q_init
                visits = 0
            elif child.visits == 0:
                # The child has never been visited, but a terminal value is
                # already exact and should not be replaced by the FPU value.
                q = child.terminal_value if child.terminal else self.config.q_init
                visits = 0
            else:
                # ``child.q`` is from the child's player perspective; flip it
                # when this edge changes the player (SUBMIT in 5D chess).
                q = child.q if child.player == node.player else -child.q
                visits = child.visits
            score = q + self.config.c_puct * float(prior) * sqrt_total / (
                1 + visits
            )
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _simulate(self, root: Node) -> float:
        node = root
        path: list[Node] = []
        value = 0.0
        value_player = node.player
        # max_depth counts edges; include the node reached by the last edge.
        for depth in range(self.config.max_depth + 1):
            path.append(node)
            if node.terminal:
                value = node.terminal_value
                value_player = node.player
                break
            if not node.expanded or depth == self.config.max_depth:
                value = self._expand(node)
                value_player = node.player
                break
            action = self._select_action(node)
            child = node.children.get(action)
            if child is None:
                child = self._make_child(node, action)
                node.children[action] = child
            node = child

        for visited in path:
            visited.visits += 1
            visited.value_sum += (
                value if visited.player == value_player else -value
            )
        return value

    # --------------------------------------------------------------- outputs
    def choose_action(
        self, root: Node, temperature: float, rng: np.random.Generator | None = None
    ) -> int:
        policy = root.visit_policy()
        if temperature < 1e-3:
            return int(np.argmax(policy))
        generator = rng or self.rng
        logits = np.log(np.maximum(policy, 1e-12)) / temperature
        logits -= logits.max()
        weights = np.exp(logits)
        probabilities = weights / weights.sum()
        return int(generator.choice(policy.shape[0], p=probabilities))
