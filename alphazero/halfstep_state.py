# alphazero/halfstep_state.py
"""
Halfstep-level state object for MCTS nodes.

A halfstep is one coordinate selection: either picking a source piece
or picking a destination. Two halfsteps form one semimove (from→to).

This module provides HalfstepState which wraps SemimoveEnv and records a
pending source selection (pending_from). apply_halfstep returns a new
HalfstepState (cloning underlying env) to allow immutable-style use in
MCTS nodes.

This file is independent from the previous procedural semimove_api and can
be imported by MCTS or agents.
"""
from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Union

from .env import SemimoveEnv, Semimove, RULE_CAPTURE_KING, SUBMIT_ACTION

Coord4 = Tuple[int, int, int, int]
HalfstepAction = Union[Coord4, str]


class HalfstepState:
    """State object for halfstep-level decisions used as MCTS node state.

    Semantics:
      - pending_from: Coord4 if a source has been selected and we're awaiting
        a destination; otherwise None.
      - env: SemimoveEnv instance representing authoritative game state.

    Methods are "functional" in style: apply_halfstep returns a new
    HalfstepState (with cloned env) when state changes. read-only queries
    do not mutate internal env.
    """

    def __init__(self, env: SemimoveEnv, pending_from: Optional[Coord4] = None):
        self.env = env
        self.pending_from: Optional[Coord4] = None if pending_from is None else tuple(pending_from)

    @staticmethod
    def create(variant_pgn: str, board_limit: int = 999) -> "HalfstepState":
        env = SemimoveEnv(variant_pgn, board_limit=board_limit, rules_mode=RULE_CAPTURE_KING)
        env.reset()
        return HalfstepState(env)

    def clone(self) -> "HalfstepState":
        return HalfstepState(self.env.clone(), pending_from=self.pending_from)

    def is_waiting_for_destination(self) -> bool:
        return self.pending_from is not None

    def legal_first_choices(self) -> List[Coord4]:
        """Return unique candidate source coordinates for initial halfstep."""
        frontier, _ = self.env.get_legal_frontier()
        seen = set()
        res: List[Coord4] = []
        for sm in frontier:
            k = tuple(sm.from_pos)
            if k not in seen:
                seen.add(k)
                res.append(k)
        return res

    def legal_destinations_for(self, from_pos: Coord4) -> List[Coord4]:
        """Return destination coordinates that form a legal semimove with from_pos."""
        frontier, _ = self.env.get_legal_frontier()
        res: List[Coord4] = []
        for sm in frontier:
            if tuple(sm.from_pos) == tuple(from_pos):
                res.append(tuple(sm.to_pos))
        return res

    def list_halfsteps(self) -> Dict[str, Union[List[Coord4], bool, Optional[Coord4]]]:
        """Return candidate coordinates and submit availability.

        Note: when pending_from is None, coords are candidate sources; when
        pending_from is set, coords include candidate destinations for that
        source (and possibly others) �� caller can filter using
        legal_destinations_for.
        """
        frontier, can_submit = self.env.get_legal_frontier()
        coords: List[Coord4] = []
        for sm in frontier:
            coords.append(tuple(sm.from_pos))
            coords.append(tuple(sm.to_pos))
        # deduplicate
        seen = set()
        uniq: List[Coord4] = []
        for c in coords:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return {"coords": uniq, "can_submit": bool(can_submit), "pending_from": self.pending_from}

    def apply_halfstep(self, half: HalfstepAction) -> Optional["HalfstepState"]:
        """Apply a halfstep and return a new HalfstepState on success.

        - SUBMIT: allowed only when not waiting for destination; returns new
          state after submit().
        - Coord4 when pending_from is None: sets pending_from if legal.
        - Coord4 when pending_from is set: attempts to apply semimove
          (pending_from -> coord) via SemimoveEnv.apply_semimove.
        """
        # submit handling
        if isinstance(half, str):
            if half != SUBMIT_ACTION and half != "SUBMIT":
                return None
            if self.pending_from is not None:
                return None
            if not self.env.can_submit():
                return None
            new_env = self.env.clone()
            new_env.submit_turn(assume_legal=True)
            return HalfstepState(new_env, pending_from=None)

        # coordinate handling
        if not (isinstance(half, tuple) and len(half) == 4):
            return None
        coord = tuple(half)

        if self.pending_from is None:
            # selecting a source
            if coord not in self.legal_first_choices():
                return None
            new_state = self.clone()
            new_state.pending_from = coord
            return new_state
        else:
            # selecting destination
            from_pos = self.pending_from
            if coord not in self.legal_destinations_for(from_pos):
                return None
            sm = Semimove(line_idx=from_pos[3], from_pos=from_pos, to_pos=coord)
            new_env = self.env.clone()
            applied = new_env.apply_semimove(sm, validate=True)
            if not applied:
                return None
            return HalfstepState(new_env, pending_from=None)

    def encode(self, urgency: float = 0.0) -> Dict:
        enc = self.env.encode_state(urgency=urgency)
        enc["pending_from"] = self.pending_from
        return enc

    def current_player(self) -> int:
        return int(self.env.current_player)

    def is_done(self) -> bool:
        return bool(self.env.done)

    def get_outcome(self) -> Optional[float]:
        return self.env.outcome

    def get_mcts_transposition_key(self) -> tuple:
        """Hashable key for MCTS transposition table."""
        # Delegate to env's method, but also include pending_from so that
        # source-selection and destination-selection nodes are distinct.
        base = self.env.get_mcts_transposition_key()
        return (base, self.pending_from)

    def __repr__(self) -> str:
        try:
            boards = len(self.env.get_boards())
        except Exception:
            boards = -1
        return f"HalfstepState(pending_from={self.pending_from}, boards={boards})"


class HalfstepGame:
    """Mutable halfstep-level game wrapper.

    Encapsulates a SemimoveEnv together with the pending_from state,
    providing step() to apply one halfstep action at a time.

    Usage:
        game = HalfstepGame(env)
        while not game.done:
            # MCTS search on current halfstep state
            mcts_state = game.make_mcts_state()
            action, policy, value = mcts.select_halfstep(root, temperature)

            if action is None:
                break

            # Record training data (before mutating env)
            ...

            game.step(action)   # applies the halfstep, mutates env
    """

    def __init__(self, env: "SemimoveEnv"):
        self.env = env
        self.pending_from: Optional[Coord4] = None

    @property
    def done(self) -> bool:
        return bool(self.env.done)

    @property
    def current_player(self) -> int:
        return int(self.env.current_player)

    @property
    def board_count(self) -> int:
        return self.env.board_count

    @property
    def total_semimoves(self) -> int:
        return self.env.total_semimoves

    def step(self, halfstep_action: HalfstepAction) -> None:
        """Apply one halfstep action, mutating env as needed."""
        if halfstep_action == SUBMIT_ACTION:
            assert self.pending_from is None, \
                "SUBMIT only allowed when pending_from is None"
            self.env.submit_turn(assume_legal=True)
        elif self.pending_from is None:
            # Source selection: just remember the piece, env unchanged
            self.pending_from = halfstep_action
        else:
            # Destination selection: complete the semimove, update env
            from_pos = self.pending_from
            sm = Semimove(line_idx=from_pos[3], from_pos=from_pos, to_pos=halfstep_action)
            self.env.apply_semimove(sm)
            self.pending_from = None

    def make_mcts_state(self) -> "HalfstepState":
        """Create a HalfstepState (for MCTS search) from current state."""
        return HalfstepState(self.env, pending_from=self.pending_from)


__all__ = ["HalfstepState", "HalfstepGame", "HalfstepAction", "Coord4", "SUBMIT_ACTION"]
