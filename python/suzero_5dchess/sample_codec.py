"""Compact, versioned self-play sample codec.

A ``PositionSample`` is stored as one zlib-compressed little-endian record.
The on-disk/in-memory representation is deliberately independent of the ONNX
input ABI: callers rebuild float32 ``[N, C, S]`` planes only when collating a
training batch.

Record layout (before zlib)::

    magic/version  uint8   = 1
    flags          uint8   side|player|submit|metadata|packed-actions
    num_boards     uint16
    candidates     uint16   non-submit action count q
    policy_len     uint16   q + has_submit
    squares        uint16   board_side * board_side
    move_count     uint16
    metadata_len   uint16
    urgency        float32
    value_target   float16
    board_codes    uint8   [N, S], 0 = empty, c+1 = piece channel c
    l_coords       int16   [N]
    t_coords       uint16  [N]
    actions        uint8   [q, 6] packed MoveKeys, or int16 [q, 8]
    policy         float16 [policy_len]
    metadata       utf-8 JSON object

Packed actions use 3 bytes per half-key: x (4 bits), y (4 bits), t (8 bits),
l (8 bits, two's complement).  The fallback int16 form covers long games and
larger boards without loss.  Piece codes are exact: the rules guarantee at
most one piece per square, so the 12-channel one-hot plane is pure redundancy.
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import Any, Mapping

import numpy as np

from .protocol import PIECE_CHANNELS

CODEC_VERSION = 1
COMPRESSION_LEVEL = 6

_HEADER = struct.Struct("<BBHHHHHHfe")

_FLAG_SIDE = 1
_FLAG_PLAYER = 2
_FLAG_SUBMIT = 4
_FLAG_METADATA = 8
_FLAG_PACKED_ACTIONS = 16

_MAX_PIECE_CODE = PIECE_CHANNELS
_PACKED_ACTION_BYTES = 6
_INT16_ACTION_BYTES = 16


# --------------------------------------------------------------- board codes


def planes_to_codes(planes: Any, *, validate: bool = True) -> np.ndarray:
    """Collapse ``[N, 12, S]`` one-hot float planes to ``[N, S]`` uint8 codes."""

    array = np.asarray(planes)
    if array.ndim != 3:
        raise ValueError(f"board planes must be [N, C, S], got {array.shape}")
    n, channels, squares = array.shape
    if channels != PIECE_CHANNELS:
        raise ValueError(
            f"board planes must have {PIECE_CHANNELS} channels, got {channels}"
        )
    if n < 1 or squares < 1:
        raise ValueError("board planes must contain at least one board")
    if array.dtype != np.float32:
        array = array.astype(np.float32)

    if validate:
        if not np.isfinite(array).all():
            raise ValueError("board planes contain non-finite values")
        if array.size and not np.all((array == 0.0) | (array == 1.0)):
            raise ValueError("board planes must be binary")
        if (array.sum(axis=1) > 1.0).any():
            raise ValueError("a square may hold at most one piece")

    channel = array.argmax(axis=1)
    present = array.max(axis=1) > 0.0
    return np.where(present, channel + 1, 0).astype(np.uint8)


def codes_to_planes(codes: Any, channels: int = PIECE_CHANNELS) -> np.ndarray:
    """Expand ``[N, S]`` uint8 piece codes back to ``[N, C, S]`` float planes."""

    array = np.asarray(codes)
    if array.ndim != 2:
        raise ValueError(f"board codes must be [N, S], got {array.shape}")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("board codes must be integers")
        if array.size and (array.min() < 0 or array.max() > channels):
            raise ValueError("board codes are out of range")
        array = array.astype(np.uint8)
    if array.size and int(array.max()) > channels:
        raise ValueError("board codes are out of range")

    n, squares = array.shape
    planes = np.zeros((n, channels, squares), dtype=np.float32)
    boards, positions = np.nonzero(array)
    plane_channels = array[boards, positions].astype(np.intp) - 1
    planes[boards, plane_channels, positions] = 1.0
    return planes


# ------------------------------------------------------------------- actions


def _pack_actions(actions: Any) -> tuple[bytes, bool]:
    """Return ``(payload, packed)``; packed uses 6 bytes per MoveKey."""

    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[1] != 8:
        raise ValueError(f"actions must be [q, 8], got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("actions must be integers")
    if array.size == 0:
        return b"", True

    values = array.astype(np.int64, copy=False)
    low = values.min(axis=0)
    high = values.max(axis=0)
    x_y_t_ok = (
        low[0] >= 0
        and high[0] < 16
        and low[1] >= 0
        and high[1] < 16
        and low[4] >= 0
        and high[4] < 16
        and low[5] >= 0
        and high[5] < 16
        and low[2] >= 0
        and high[2] < 256
        and low[6] >= 0
        and high[6] < 256
    )
    l_ok = (
        low[3] >= -128
        and high[3] < 128
        and low[7] >= -128
        and high[7] < 128
    )
    if x_y_t_ok and l_ok:
        source = (
            values[:, 0]
            | (values[:, 1] << 4)
            | (values[:, 2] << 8)
            | ((values[:, 3] & 0xFF) << 16)
        )
        target = (
            values[:, 4]
            | (values[:, 5] << 4)
            | (values[:, 6] << 8)
            | ((values[:, 7] & 0xFF) << 16)
        )
        packed = np.empty((values.shape[0], _PACKED_ACTION_BYTES), dtype=np.uint8)
        for offset, word in ((0, source), (3, target)):
            packed[:, offset] = word & 0xFF
            packed[:, offset + 1] = (word >> 8) & 0xFF
            packed[:, offset + 2] = (word >> 16) & 0xFF
        return packed.tobytes(), True

    if values.min() < -32768 or values.max() > 32767:
        raise ValueError("action coordinates do not fit int16")
    return values.astype("<i2").tobytes(), False


def _unpack_actions(payload: bytes, candidates: int, packed: bool) -> np.ndarray:
    if packed:
        expected = candidates * _PACKED_ACTION_BYTES
        if len(payload) != expected:
            raise ValueError("packed action payload has the wrong length")
        if candidates == 0:
            return np.zeros((0, 8), dtype=np.int64)
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(candidates, 6)
        result = np.empty((candidates, 8), dtype=np.int64)
        for offset, first in ((0, 0), (3, 4)):
            word = (
                raw[:, offset].astype(np.int64)
                | (raw[:, offset + 1].astype(np.int64) << 8)
                | (raw[:, offset + 2].astype(np.int64) << 16)
            )
            result[:, first] = word & 0xF
            result[:, first + 1] = (word >> 4) & 0xF
            result[:, first + 2] = (word >> 8) & 0xFF
            line = (word >> 16) & 0xFF
            result[:, first + 3] = np.where(line >= 128, line - 256, line)
        return result

    expected = candidates * _INT16_ACTION_BYTES
    if len(payload) != expected:
        raise ValueError("action payload has the wrong length")
    if candidates == 0:
        return np.zeros((0, 8), dtype=np.int64)
    return np.frombuffer(payload, dtype="<i2").reshape(candidates, 8).astype(np.int64)


# ------------------------------------------------------------------ metadata


def _encode_metadata(metadata: Mapping[str, float] | None) -> bytes:
    if not metadata:
        return b""
    document: dict[str, float] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("metadata values must be finite")
        document[key] = number
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _decode_metadata(payload: bytes) -> dict[str, float]:
    if not payload:
        return {}
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"sample metadata is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("sample metadata must be a JSON object")
    return {str(key): float(value) for key, value in document.items()}


# -------------------------------------------------------------------- codec


def _unsigned(value: Any, label: str, limit: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number < 0 or number > limit:
        raise ValueError(f"{label} {number} does not fit uint16")
    return number


def _coordinate_array(value: Any, label: str, low: int, high: int, dtype: str) -> bytes:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{label} must be integers")
    if array.size and (array.min() < low or array.max() > high):
        raise ValueError(f"{label} values are out of range")
    return array.astype(dtype).tobytes()


def encode_sample(sample: Any) -> bytes:
    """Encode one ``PositionSample`` (or duck-typed equivalent) to bytes."""

    codes = np.asarray(sample.board_codes)
    if codes.ndim != 2:
        raise ValueError(f"board codes must be [N, S], got {codes.shape}")
    if codes.dtype != np.uint8:
        if not np.issubdtype(codes.dtype, np.integer):
            raise ValueError("board codes must be integers")
        if codes.size and (codes.min() < 0 or codes.max() > _MAX_PIECE_CODE):
            raise ValueError("board codes are out of range")
        codes = codes.astype(np.uint8)
    if codes.size and int(codes.max()) > _MAX_PIECE_CODE:
        raise ValueError("board codes are out of range")
    boards, squares = codes.shape
    if boards < 1 or squares < 1:
        raise ValueError("a sample needs at least one board")

    actions = np.asarray(sample.actions)
    policy = np.asarray(sample.policy)
    if policy.ndim != 1:
        raise ValueError(f"policy must be [policy_len], got {policy.shape}")
    candidates = int(actions.shape[0]) if actions.ndim == 2 else 0
    has_submit = bool(sample.has_submit)
    policy_len = int(policy.shape[0])
    if policy_len != candidates + int(has_submit):
        raise ValueError(
            f"policy length {policy_len} does not match {candidates} actions"
        )

    action_payload, packed = _pack_actions(actions)
    metadata = _encode_metadata(getattr(sample, "metadata", None))

    side = 1 if int(sample.side_to_move) else 0
    player = 1 if int(sample.player) else 0
    flags = (
        (_FLAG_SIDE if side else 0)
        | (_FLAG_PLAYER if player else 0)
        | (_FLAG_SUBMIT if has_submit else 0)
        | (_FLAG_METADATA if metadata else 0)
        | (_FLAG_PACKED_ACTIONS if packed else 0)
    )

    urgency = float(sample.urgency)
    value = float(sample.value_target)
    if not np.isfinite(urgency) or not np.isfinite(value):
        raise ValueError("sample scalars must be finite")

    header = _HEADER.pack(
        CODEC_VERSION,
        flags,
        _unsigned(boards, "num_boards", 0xFFFF),
        _unsigned(candidates, "candidate count", 0xFFFF),
        _unsigned(policy_len, "policy length", 0xFFFF),
        _unsigned(squares, "board squares", 0xFFFF),
        _unsigned(getattr(sample, "move_count", 0), "move_count", 0xFFFF),
        _unsigned(len(metadata), "metadata length", 0xFFFF),
        urgency,
        value,
    )

    payload = b"".join(
        (
            header,
            codes.tobytes(),
            _coordinate_array(sample.l_coords, "l_coords", -32768, 32767, "<i2"),
            _coordinate_array(sample.t_coords, "t_coords", 0, 65535, "<u2"),
            action_payload,
            policy.astype("<f2").tobytes(),
            metadata,
        )
    )
    return zlib.compress(payload, COMPRESSION_LEVEL)


def peek_move_count(blob: bytes) -> int:
    """Read ``move_count`` without inflating the whole record.

    Replay weighting needs one integer per row.  Inflating only the fixed
    header keeps that pass far cheaper than a full :func:`decode_sample`,
    which would also rebuild the board, actions and policy arrays.
    """

    decompressor = zlib.decompressobj()
    try:
        head = decompressor.decompress(blob, _HEADER.size)
    except zlib.error as exc:
        raise ValueError(f"sample record is not valid zlib data: {exc}") from exc
    if len(head) < _HEADER.size:
        raise ValueError("sample record is truncated")
    (version, _flags, _boards, _candidates, _policy, _squares, move_count,
     _metadata_len, _urgency, _value) = _HEADER.unpack_from(head, 0)
    if version != CODEC_VERSION:
        raise ValueError(f"unsupported sample codec version {version}")
    return int(move_count)


def decode_sample(blob: bytes) -> dict[str, object]:
    """Decode one codec blob into a dictionary of ``PositionSample`` fields."""

    try:
        payload = zlib.decompress(blob)
    except zlib.error as exc:
        raise ValueError(f"sample record is not valid zlib data: {exc}") from exc
    if len(payload) < _HEADER.size:
        raise ValueError("sample record is truncated")

    (
        version,
        flags,
        boards,
        candidates,
        policy_len,
        squares,
        move_count,
        metadata_len,
        urgency,
        value,
    ) = _HEADER.unpack_from(payload, 0)
    if version != CODEC_VERSION:
        raise ValueError(f"unsupported sample codec version {version}")
    if boards < 1 or squares < 1:
        raise ValueError("sample record has no boards")
    has_submit = bool(flags & _FLAG_SUBMIT)
    if policy_len != candidates + int(has_submit):
        raise ValueError("sample record has an inconsistent policy length")
    packed = bool(flags & _FLAG_PACKED_ACTIONS)

    offset = _HEADER.size
    code_bytes = boards * squares
    action_bytes = candidates * (
        _PACKED_ACTION_BYTES if packed else _INT16_ACTION_BYTES
    )
    expected = (
        code_bytes
        + 2 * boards
        + 2 * boards
        + action_bytes
        + 2 * policy_len
        + metadata_len
    )
    if len(payload) != offset + expected:
        raise ValueError(
            f"sample record has {len(payload) - offset} payload bytes, "
            f"expected {expected}"
        )

    codes = np.frombuffer(
        payload, dtype=np.uint8, count=code_bytes, offset=offset
    ).reshape(boards, squares).copy()
    offset += code_bytes
    if codes.size and int(codes.max()) > _MAX_PIECE_CODE:
        raise ValueError("sample record contains an invalid piece code")

    l_coords = np.frombuffer(
        payload, dtype="<i2", count=boards, offset=offset
    ).astype(np.int64)
    offset += 2 * boards
    t_coords = np.frombuffer(
        payload, dtype="<u2", count=boards, offset=offset
    ).astype(np.int64)
    offset += 2 * boards

    actions = _unpack_actions(payload[offset : offset + action_bytes], candidates, packed)
    offset += action_bytes

    policy = np.frombuffer(
        payload, dtype="<f2", count=policy_len, offset=offset
    ).astype(np.float32)
    offset += 2 * policy_len
    if not np.isfinite(policy).all():
        raise ValueError("sample record contains non-finite policy values")
    np.maximum(policy, 0.0, out=policy)
    total = float(policy.sum())
    if total <= 0.0:
        raise ValueError("sample record policy is empty")
    policy /= total

    metadata = _decode_metadata(payload[offset : offset + metadata_len])

    return {
        "board_codes": codes,
        "l_coords": l_coords,
        "t_coords": t_coords,
        "urgency": float(urgency),
        "side_to_move": 1 if flags & _FLAG_SIDE else 0,
        "actions": actions,
        "policy": policy,
        "has_submit": has_submit,
        "value_target": float(value),
        "player": 1 if flags & _FLAG_PLAYER else 0,
        "move_count": int(move_count),
        "metadata": metadata,
    }


__all__ = [
    "CODEC_VERSION",
    "codes_to_planes",
    "decode_sample",
    "encode_sample",
    "peek_move_count",
    "planes_to_codes",
]
