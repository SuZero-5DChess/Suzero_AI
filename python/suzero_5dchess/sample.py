"""Self-play sample representation, validation and collation.

Action ordering is part of the ONNX ABI:

* ``actions`` holds only non-submit MoveKeys, in the exact order used by the
  network's ``other_logits``;
* the optional SUBMIT slot comes last in ``policy`` and is represented by the
  separate ``submit_logit`` output.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import isqrt
from types import MappingProxyType

import numpy as np

from .env import EncodedPosition, GameEnv
from .protocol import MOVE_KEY_FIELDS, VariantSpec
from .sample_codec import codes_to_planes, planes_to_codes


@dataclass
class PositionSample:
    board_codes: np.ndarray  # [N, S] uint8, 0=empty, c+1=piece channel c
    l_coords: np.ndarray  # [N] int64
    t_coords: np.ndarray  # [N] int64
    urgency: float
    side_to_move: int
    actions: np.ndarray  # [q, 8] int64, non-submit only
    policy: np.ndarray  # [q + has_submit] float32
    has_submit: bool
    value_target: float = 0.0
    player: int = 0
    move_count: int = 0
    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def board_planes(self) -> np.ndarray:
        """Expand the compact codes to float32 ``[N, C, S]`` planes on demand.

        This is a compatibility/convenience view for diagnostics and tools; the
        training path rebuilds padded batches directly from ``board_codes``.
        """

        return codes_to_planes(self.board_codes)

    @property
    def num_boards(self) -> int:
        return int(self.board_codes.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.actions.shape[0])

    @property
    def policy_size(self) -> int:
        return int(self.policy.shape[0])

    def assign_outcome(self, white_outcome: float) -> None:
        self.value_target = (
            float(white_outcome) if self.player == 0 else -float(white_outcome)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "board_codes": self.board_codes,
            "l_coords": self.l_coords,
            "t_coords": self.t_coords,
            "urgency": self.urgency,
            "side_to_move": self.side_to_move,
            "actions": self.actions,
            "policy": self.policy,
            "has_submit": self.has_submit,
            "value_target": self.value_target,
            "player": self.player,
            "move_count": self.move_count,
        }


def make_sample(env: GameEnv, policy: np.ndarray, player: int) -> PositionSample:
    """Capture the pre-action position and a root visit distribution."""

    actions = env.actions
    has_submit = bool(actions) and bool(actions[-1][0])
    non_submit = actions[:-1] if has_submit else actions
    q = len(non_submit)
    if q < 1:
        raise ValueError("cannot sample a submit-only (forced) position")

    policy = np.asarray(policy, dtype=np.float32)
    expected = q + int(has_submit)
    if policy.shape != (expected,):
        raise ValueError(
            f"policy has shape {policy.shape}, expected ({expected},) for "
            f"q={q}, has_submit={has_submit}"
        )

    encoded: EncodedPosition = env.encode()
    return PositionSample(
        board_codes=planes_to_codes(encoded.board_planes),
        l_coords=np.array(encoded.l_coords, dtype=np.int64, copy=True),
        t_coords=np.array(encoded.t_coords, dtype=np.int64, copy=True),
        urgency=float(encoded.urgency),
        side_to_move=int(encoded.side_to_move),
        actions=np.asarray(
            [action[1] for action in non_submit], dtype=np.int64
        ).reshape(q, MOVE_KEY_FIELDS),
        policy=policy,
        has_submit=has_submit,
        player=int(player),
        move_count=int(env.move_count),
    )


class _ReplaySample(PositionSample):
    """Validated snapshot; arrays are backed by immutable bytes, not aliases."""

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("replay samples are immutable")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("replay samples are immutable")
        super().__delattr__(name)


def _immutable(array: np.ndarray) -> np.ndarray:
    """Back an array with ``bytes`` so it cannot be made writeable again."""

    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def freeze_sample(
    sample: PositionSample, spec: VariantSpec | None = None
) -> PositionSample:
    """Copy and validate at the replay boundary, after outcome assignment."""

    if not isinstance(sample, PositionSample):
        raise TypeError("samples must be PositionSample instances")
    arrays = {"board_codes", "l_coords", "t_coords", "actions", "policy"}
    record = {}
    for item in fields(PositionSample):
        value = getattr(sample, item.name)
        if item.name in arrays:
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError("replay arrays cannot contain objects")
            value = _immutable(array)
        elif item.name == "metadata":
            value = MappingProxyType({key: float(weight) for key, weight in value.items()})
        else:
            # Detach NumPy scalar / zero-dimensional-array aliases as well.
            value = np.asarray(value).item()
        record[item.name] = value
    snapshot = _ReplaySample(**record)
    if spec is None:
        codes = snapshot.board_codes
        squares = codes.shape[1] if codes.ndim == 2 else 0
        spec = VariantSpec(name="replay", pgn="", board_side=isqrt(squares))
    validate_sample(snapshot, spec)
    snapshot._sealed = True
    return snapshot


def seal_decoded(
    record: dict[str, object], spec: VariantSpec | None = None
) -> PositionSample:
    """Validate decoded fields and back their arrays with immutable bytes."""

    arrays = {"board_codes", "l_coords", "t_coords", "actions", "policy"}
    sealed: dict[str, object] = {}
    for item in fields(PositionSample):
        value = record[item.name]
        if item.name in arrays:
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError("replay arrays cannot contain objects")
            sealed[item.name] = _immutable(array)
        elif item.name == "metadata":
            sealed[item.name] = MappingProxyType(
                {str(key): float(weight) for key, weight in dict(value).items()}
            )
        else:
            sealed[item.name] = np.asarray(value).item()
    snapshot = _ReplaySample(**sealed)
    if spec is None:
        codes = snapshot.board_codes
        squares = codes.shape[1] if codes.ndim == 2 else 0
        spec = VariantSpec(name="replay", pgn="", board_side=isqrt(squares))
    _validate_sample(snapshot, spec, contents=False)
    snapshot._sealed = True
    return snapshot


def validate_sample(sample: PositionSample, spec: VariantSpec) -> None:
    _validate_sample(sample, spec, contents=True)


def _validate_sample(
    sample: PositionSample, spec: VariantSpec, *, contents: bool
) -> None:
    codes = np.asarray(sample.board_codes)
    if codes.dtype != np.uint8 or codes.ndim != 2:
        raise ValueError("board_codes must be uint8 [N, S]")
    n, squares = codes.shape
    if n < 1 or squares != spec.board_squares:
        raise ValueError(
            f"board_codes shape {codes.shape} does not match "
            f"[N, {spec.board_squares}]"
        )
    if contents and codes.size and int(codes.max()) > spec.piece_channels:
        raise ValueError("a board code exceeds the piece channel count")

    if sample.l_coords.shape != (n,) or sample.t_coords.shape != (n,):
        raise ValueError("coordinate arrays must have shape [N]")
    if not np.issubdtype(sample.l_coords.dtype, np.integer):
        raise ValueError("l_coords must be integer")
    if not np.issubdtype(sample.t_coords.dtype, np.integer):
        raise ValueError("t_coords must be integer")

    if not np.isfinite(sample.urgency) or sample.urgency < 0:
        raise ValueError("urgency must be finite and non-negative")
    if sample.side_to_move not in (0, 1):
        raise ValueError("side_to_move must be 0 or 1")

    actions = np.asarray(sample.actions)
    if actions.dtype != np.int64 or actions.ndim != 2:
        raise ValueError("actions must be int64 [q, 8]")
    q = actions.shape[0]
    if actions.shape[1] != MOVE_KEY_FIELDS or q < 1:
        raise ValueError("actions must be non-empty [q, 8]")
    if contents:
        if (actions[:, 0] < 0).any() or (actions[:, 0] >= spec.board_side).any():
            raise ValueError("source x coordinate out of range")
        if (actions[:, 1] < 0).any() or (actions[:, 1] >= spec.board_side).any():
            raise ValueError("source y coordinate out of range")
        if (actions[:, 4] < 0).any() or (actions[:, 4] >= spec.board_side).any():
            raise ValueError("destination x coordinate out of range")
        if (actions[:, 5] < 0).any() or (actions[:, 5] >= spec.board_side).any():
            raise ValueError("destination y coordinate out of range")
        if (actions[:, 2] < 0).any() or (actions[:, 6] < 0).any():
            raise ValueError("timeline t coordinates must be non-negative")
        # Timeline indices l are signed: branching can create negative lines.
        if not np.any(actions[:, :4] != actions[:, 4:], axis=1).all():
            raise ValueError("a move cannot have identical source and destination")
        unique = np.unique(actions, axis=0)
        if unique.shape[0] != q:
            raise ValueError("duplicate MoveKeys in a sample")

    policy = np.asarray(sample.policy)
    expected = q + int(sample.has_submit)
    if policy.dtype != np.float32 or policy.shape != (expected,):
        raise ValueError(
            f"policy must be float32 ({expected},), got {policy.dtype} "
            f"{policy.shape}"
        )
    if contents:
        if not np.isfinite(policy).all() or (policy < -1e-6).any():
            raise ValueError("policy must be finite and non-negative")
        total = float(policy.sum())
        if total <= 0.0 or abs(total - 1.0) > 1e-3:
            raise ValueError(f"policy must sum to 1, got {total}")

    if not np.isfinite(sample.value_target) or abs(sample.value_target) > 1.0:
        raise ValueError("value_target must be finite and in [-1, 1]")
    if sample.player not in (0, 1):
        raise ValueError("player must be 0 or 1")


def collate_samples(
    samples: list[PositionSample], spec: VariantSpec
) -> dict[str, np.ndarray]:
    """Pad a batch and align policy slots with the ONNX action layout."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    for sample in samples:
        if not isinstance(sample, PositionSample):
            raise TypeError("samples must be PositionSample instances")
        if isinstance(sample, _ReplaySample) and getattr(sample, "_sealed", False):
            _validate_sample(sample, spec, contents=False)
        else:
            validate_sample(sample, spec)

    batch = len(samples)
    max_boards = max(sample.num_boards for sample in samples)
    max_q = max(sample.candidate_count for sample in samples)
    max_policy = max_q + 1  # optional submit slot is always the last column

    planes = np.zeros(
        (batch, max_boards, spec.piece_channels, spec.board_squares),
        dtype=np.float32,
    )
    l_coords = np.zeros((batch, max_boards), dtype=np.int64)
    t_coords = np.zeros((batch, max_boards), dtype=np.int64)
    used = np.zeros((batch,), dtype=np.int64)
    urgency = np.zeros((batch,), dtype=np.float32)
    side = np.zeros((batch,), dtype=np.int64)
    actions = np.zeros((batch, max_q, MOVE_KEY_FIELDS), dtype=np.int64)
    policy = np.zeros((batch, max_policy), dtype=np.float32)
    policy_mask = np.zeros((batch, max_policy), dtype=bool)
    value = np.zeros((batch, 1), dtype=np.float32)

    for i, sample in enumerate(samples):
        n = sample.num_boards
        q = sample.candidate_count
        codes = np.asarray(sample.board_codes)
        boards, positions = np.nonzero(codes)
        planes[i, boards, codes[boards, positions] - 1, positions] = 1.0
        l_coords[i, :n] = sample.l_coords
        t_coords[i, :n] = sample.t_coords
        used[i] = n
        urgency[i] = sample.urgency
        side[i] = sample.side_to_move
        actions[i, :q] = sample.actions
        policy[i, :q] = sample.policy[:q]
        policy_mask[i, :q] = True
        if sample.has_submit:
            policy[i, max_q] = sample.policy[q]
            policy_mask[i, max_q] = True
        value[i, 0] = sample.value_target

    return {
        "board_planes": planes,
        "l_coords": l_coords,
        "t_coords": t_coords,
        "used_board_counts": used,
        "urgency": urgency,
        "side_to_move": side,
        "actions": actions,
        "policy": policy,
        "policy_mask": policy_mask,
        "value_target": value,
    }
