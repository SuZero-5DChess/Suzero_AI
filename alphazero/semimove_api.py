# alphazero/semimove_api.py
"""
Lightweight semimove-level abstraction API on top of SemimoveEnv.
Provides a small, engine-agnostic surface where a semimove (4D coords)
and submit are treated as equal first-class actions. By default this
wrapper disables engine "strict" check-based legality and uses the
capture-king style rules (i.e. no check/checkmate termination).

This module does not modify the engine; it composes the existing
SemimoveEnv and exposes simple helper functions expected by external
agents or evaluation scripts.
"""
from typing import List, Dict, Tuple, Union, Optional

from .env import SemimoveEnv, Semimove, RULE_CAPTURE_KING, SUBMIT_ACTION

Coord4 = Tuple[int, int, int, int]
ActionDict = Dict[str, Union[str, Coord4, int]]


def create_env(variant_pgn: str, board_limit: int = 999) -> SemimoveEnv:
    """Create a SemimoveEnv using capture-king rules (no check/mate).

    Args:
        variant_pgn: PGN headers / variant description.
        board_limit: maximum number of boards before forced termination.
    Returns:
        An initialized SemimoveEnv.
    """
    env = SemimoveEnv(variant_pgn, board_limit=board_limit, rules_mode=RULE_CAPTURE_KING)
    env.reset()
    return env


def list_actions(env: SemimoveEnv) -> List[ActionDict]:
    """Return available actions in a simple serializable format.

    Each action is either:
      - {'type': 'submit'}
      - {'type': 'semimove', 'from': (x,y,t,l), 'to': (x,y,t,l), 'line_idx': l}

    The returned list is already filtered using the env's lexicographic
    ordering and rules_mode.
    """
    semis = env.get_legal_semimoves()
    actions: List[ActionDict] = []
    for sm in semis:
        actions.append({
            "type": "semimove",
            "from": tuple(sm.from_pos),
            "to": tuple(sm.to_pos),
            "line_idx": int(sm.line_idx),
        })
    if env.can_submit():
        actions.append({"type": "submit"})
    return actions


def apply_action(env: SemimoveEnv, action: ActionDict) -> bool:
    """Apply an action to the env.

    If action is submit, this will call submit_turn(assume_legal=True)
    and return True on success. For semimoves, apply_semimove() is used.

    Returns True if the action was applied, False otherwise.
    """
    if action.get("type") == "submit":
        # assume legal to match typical agent behaviour
        res = env.submit_turn(assume_legal=True)
        # submit_turn returns outcome or None; treat non-None as successful
        return res is not None or not env.uses_strict_legal_enumeration

    if action.get("type") == "semimove":
        from_pos = action.get("from")
        to_pos = action.get("to")
        if not (isinstance(from_pos, tuple) and isinstance(to_pos, tuple)):
            return False
        sm = Semimove(line_idx=int(action.get("line_idx", from_pos[3])),
                      from_pos=tuple(from_pos),
                      to_pos=tuple(to_pos))
        return env.apply_semimove(sm, validate=True)

    return False


def encode_state(env: SemimoveEnv, urgency: float = 0.0) -> Dict:
    """Proxy to env.encode_state; returns the same dict used by the
    policy/value network input pipeline.
    """
    return env.encode_state(urgency=urgency)


def is_done(env: SemimoveEnv) -> bool:
    return bool(env.done)


def get_outcome(env: SemimoveEnv) -> Optional[float]:
    return env.outcome


def current_player(env: SemimoveEnv) -> int:
    return int(env.current_player)


# Exported names
__all__ = [
    "create_env",
    "list_actions",
    "apply_action",
    "encode_state",
    "is_done",
    "get_outcome",
    "current_player",
    "Semimove",
    "SUBMIT_ACTION",
]
