"""Python wrapper around the C++ 5D chess rules adapter.

The compiled ``_5dchess`` module is built by CMake with
``-DSUZERO_BUILD_PYTHON_BINDINGS=ON`` and placed next to this file.  Everything
in this wrapper is deliberately thin: rules live in C++, training lives in
Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import VariantSpec, resolve_variant

try:  # pragma: no cover - exercised through the import error path
    from . import _5dchess
except ImportError as _exc:  # pragma: no cover
    _5dchess = None
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None


Action = tuple[bool, tuple[int, int, int, int, int, int, int, int] | None]


@dataclass(frozen=True)
class EncodedPosition:
    board_planes: np.ndarray  # [N, 12, S] float32
    l_coords: np.ndarray  # [N] int64, already shifted by line_shift
    t_coords: np.ndarray  # [N] int64
    urgency: float
    side_to_move: int
    num_boards: int
    board_side: int
    board_squares: int
    piece_channels: int

    def as_dict(self) -> dict[str, object]:
        return {
            "board_planes": self.board_planes,
            "l_coords": self.l_coords,
            "t_coords": self.t_coords,
            "urgency": self.urgency,
            "side_to_move": self.side_to_move,
            "num_boards": self.num_boards,
        }


def _require_binding() -> None:
    if _5dchess is None:
        raise RuntimeError(
            "the C++ 5D chess binding is not built. Configure with "
            "-DSUZERO_BUILD_5DCHESS=ON -DSUZERO_BUILD_PYTHON_BINDINGS=ON "
            "and build the suzero_5dchess_py target. "
            f"Original import error: {_IMPORT_ERROR}"
        )


class GameEnv:
    """A mutable position with cached action enumeration."""

    def __init__(
        self,
        variant: str | VariantSpec = "very_small",
        *,
        board_limit: int | None = None,
        material_scale: float = 2.0,
        state: object | None = None,
    ) -> None:
        _require_binding()
        self.spec = resolve_variant(variant)
        self.material_scale = float(material_scale)
        if state is None:
            limit = (
                self.spec.default_board_limit if board_limit is None else board_limit
            )
            self._state = _5dchess.GameState(
                self.spec.pgn,
                self.spec.board_side,
                self.spec.line_shift,
                int(limit),
                self.material_scale,
            )
        else:
            self._state = state
        self._actions: tuple[Action, ...] | None = None

    # --------------------------------------------------------------- copying
    def clone(self) -> "GameEnv":
        return GameEnv(
            self.spec,
            material_scale=self.material_scale,
            state=self._state.clone(),
        )

    # --------------------------------------------------------------- actions
    @property
    def actions(self) -> tuple[Action, ...]:
        if self._actions is None:
            raw = self._state.actions()
            self._actions = tuple(
                (
                    bool(is_submit),
                    None
                    if move is None
                    else tuple(int(value) for value in move),  # type: ignore[arg-type]
                )
                for is_submit, move in raw
            )
        return self._actions

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def has_submit(self) -> bool:
        return bool(self.actions) and bool(self.actions[-1][0])

    @property
    def forced_submit(self) -> bool:
        return len(self.actions) == 1 and self.actions[0][0]

    def apply_action(self, action_id: int) -> None:
        self._state.apply_action(int(action_id))
        self._actions = None

    # ----------------------------------------------------------------- state
    @property
    def current_player(self) -> int:
        return int(self._state.current_player)

    @property
    def is_terminal(self) -> bool:
        return bool(self._state.is_terminal)

    @property
    def terminal_utility(self) -> float:
        return float(self._state.terminal_utility)

    @property
    def white_outcome(self) -> float:
        if not self.is_terminal:
            raise ValueError("white_outcome is only defined for terminal states")
        utility = self.terminal_utility
        return utility if self.current_player == 0 else -utility

    @property
    def terminal_reason(self) -> str:
        return str(self._state.terminal_reason)

    @property
    def board_count(self) -> int:
        return int(self._state.board_count)

    @property
    def board_limit(self) -> int:
        return int(self._state.board_limit)

    @property
    def urgency(self) -> float:
        return float(self._state.urgency)

    @property
    def move_count(self) -> int:
        return int(self._state.move_count)

    @property
    def state_key(self) -> str:
        """Exact identity; expensive to compute and only used for diagnostics."""

        return str(self._state.state_key)

    def record_snapshot(self) -> dict[str, object]:
        """Native board history and FEN for lossless game archives, not NN planes."""

        if not hasattr(self._state, "record_snapshot"):
            raise RuntimeError(
                ".gn5d recording requires a rebuilt _5dchess extension; run "
                "cmake --build build --config Release --target suzero_5dchess_py"
            )
        return self._state.record_snapshot()

    def encode(self) -> EncodedPosition:
        raw = self._state.encode()
        return EncodedPosition(
            board_planes=np.asarray(raw["board_planes"], dtype=np.float32),
            l_coords=np.asarray(raw["l_coords"], dtype=np.int64),
            t_coords=np.asarray(raw["t_coords"], dtype=np.int64),
            urgency=float(raw["urgency"]),
            side_to_move=int(raw["side_to_move"]),
            num_boards=int(raw["num_boards"]),
            board_side=int(raw["board_side"]),
            board_squares=int(raw["board_squares"]),
            piece_channels=int(raw["piece_channels"]),
        )
