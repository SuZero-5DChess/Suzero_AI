"""Wire protocol constants shared by training, export and the C++ runner.

The numbers here are an external ABI: the ONNX sidecar and the C++ evaluator
reject models whose protocol or board geometry does not match.  Keep this
module free of torch so that lightweight tools can import it.
"""

from __future__ import annotations

from dataclasses import dataclass

IO_PROTOCOL = "smallstep_movekey_v1"
ACTION_ENCODING = "smallstep_v1"
ACTION_INPUT_ENCODING = "movekey_int64_v1"
SEARCH_ALGORITHM = "dag_mcgs_v1"
CPP_BINARY_VERSION = 7

MOVE_KEY_FIELDS = 8
PIECE_CHANNELS = 12

ONNX_INPUT_NAMES = (
    "board_planes",
    "l_coords",
    "t_coords",
    "used_board_counts",
    "urgency",
    "side_to_move",
    "actions",
)
ONNX_OUTPUT_NAMES = ("value", "submit_logit", "other_logits")


@dataclass(frozen=True)
class VariantSpec:
    """Rules and geometry for one 5D chess variant."""

    name: str
    pgn: str
    board_side: int = 4
    line_shift: int = 5
    default_board_limit: int = 25
    min_board_limit: int = 15
    max_board_limit: int = 25

    @property
    def board_squares(self) -> int:
        return self.board_side * self.board_side

    @property
    def piece_channels(self) -> int:
        return PIECE_CHANNELS

    def config_dict(self) -> dict[str, object]:
        return {
            "variant": self.name,
            "variant_pgn": self.pgn,
            "board_side": self.board_side,
            "board_squares": self.board_squares,
            "piece_channels": self.piece_channels,
            "line_shift": self.line_shift,
            "board_limit": self.default_board_limit,
            "min_board_limit": self.min_board_limit,
            "max_board_limit": self.max_board_limit,
        }


VERY_SMALL = VariantSpec(
    name="very_small",
    pgn='[Board "Very Small - Open"]\n[Mode "5D"]\n',
    board_side=4,
    line_shift=5,
    default_board_limit=25,
    min_board_limit=15,
    max_board_limit=25,
)

STANDARD = VariantSpec(
    name="standard",
    pgn='[Board "Standard"]\n[Mode "5D"]\n',
    board_side=8,
    line_shift=5,
    default_board_limit=25,
    min_board_limit=15,
    max_board_limit=25,
)

STANDARD_TURN_ZERO = VariantSpec(
    name="standard_turn_zero",
    pgn='[Board "Standard - Turn Zero"]\n[Mode "5D"]\n',
    board_side=8,
    line_shift=5,
    default_board_limit=25,
    min_board_limit=15,
    max_board_limit=25,
)

VARIANTS: dict[str, VariantSpec] = {
    VERY_SMALL.name: VERY_SMALL,
    STANDARD.name: STANDARD,
    STANDARD_TURN_ZERO.name: STANDARD_TURN_ZERO,
}


def resolve_variant(variant: str | VariantSpec) -> VariantSpec:
    if isinstance(variant, VariantSpec):
        return variant
    try:
        return VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(
            f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}"
        ) from exc
