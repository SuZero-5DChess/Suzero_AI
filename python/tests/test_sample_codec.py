"""Compact sample codec: round trips, compression and corrupt records."""

import zlib

import numpy as np
import pytest

from suzero_5dchess.protocol import PIECE_CHANNELS, STANDARD, VERY_SMALL
from suzero_5dchess.replay import ReplayBuffer
from suzero_5dchess.sample import PositionSample, collate_samples
from suzero_5dchess.sample_codec import (
    _HEADER,
    codes_to_planes,
    decode_sample,
    encode_sample,
    planes_to_codes,
)


def sample(squares: int = 64) -> PositionSample:
    codes = np.zeros((3, squares), dtype=np.uint8)
    codes[0, 0] = 1
    codes[1, 5] = PIECE_CHANNELS
    codes[2, squares - 1] = 7
    return PositionSample(
        board_codes=codes,
        l_coords=np.array([0, 2, 4], dtype=np.int64),
        t_coords=np.array([1, 3, 5], dtype=np.int64),
        urgency=17.5,
        side_to_move=1,
        actions=np.array(
            [[0, 0, 0, 0, 1, 1, 0, 0], [2, 3, 1, 0, 2, 3, 2, -1]],
            dtype=np.int64,
        ),
        policy=np.array([0.6, 0.3, 0.1], dtype=np.float32),
        has_submit=True,
        value_target=0.75,
        player=1,
        move_count=42,
        metadata={"weight": 1.5, "plies": 9.0},
    )


def round_trip(decoded: dict[str, object], original: PositionSample) -> None:
    np.testing.assert_array_equal(decoded["board_codes"], original.board_codes)
    np.testing.assert_array_equal(decoded["l_coords"], original.l_coords)
    np.testing.assert_array_equal(decoded["t_coords"], original.t_coords)
    np.testing.assert_array_equal(decoded["actions"], original.actions)
    np.testing.assert_allclose(decoded["policy"], original.policy, atol=1e-3)
    assert float(np.sum(decoded["policy"])) == pytest.approx(1.0, abs=1e-6)
    assert decoded["urgency"] == pytest.approx(original.urgency)
    assert decoded["value_target"] == pytest.approx(original.value_target, abs=1e-3)
    assert decoded["side_to_move"] == original.side_to_move
    assert decoded["has_submit"] is original.has_submit
    assert decoded["player"] == original.player
    assert decoded["move_count"] == original.move_count
    assert decoded["metadata"] == original.metadata


def test_round_trip_preserves_every_field():
    original = sample()
    round_trip(decode_sample(encode_sample(original)), original)


def test_round_trip_without_submit_or_metadata():
    original = sample()
    original.has_submit = False
    original.policy = np.array([0.7, 0.3], dtype=np.float32)
    original.metadata = {}
    round_trip(decode_sample(encode_sample(original)), original)


def test_planes_are_rebuilt_exactly_from_codes():
    original = sample()
    planes = codes_to_planes(original.board_codes)
    assert planes.dtype == np.float32
    assert planes.shape == (3, PIECE_CHANNELS, 64)
    occupancy = (original.board_codes > 0).astype(np.float32)
    np.testing.assert_array_equal(planes.sum(axis=1), occupancy)
    assert float(planes.sum()) == float(occupancy.sum()) == 3.0
    np.testing.assert_array_equal(planes_to_codes(planes), original.board_codes)


@pytest.mark.parametrize("mutation", ["non_binary", "two_pieces", "channels"])
def test_planes_to_codes_rejects_malformed_planes(mutation):
    planes = codes_to_planes(sample().board_codes)
    if mutation == "non_binary":
        planes[0, 0, 0] = 0.5
    elif mutation == "two_pieces":
        planes[0, 1, 0] = 1.0
    else:
        planes = planes[:, :11]
    with pytest.raises(ValueError):
        planes_to_codes(planes)


def test_collate_expands_padded_codes():
    batch = collate_samples([sample()], STANDARD)
    planes = batch["board_planes"]
    assert planes.shape == (1, 3, PIECE_CHANNELS, 64)
    np.testing.assert_array_equal(planes[0], codes_to_planes(sample().board_codes))
    assert planes.sum() == 3.0


def test_compression_shrinks_the_float32_layout():
    original = sample()
    blob = encode_sample(original)
    float32_bytes = original.board_codes.size * PIECE_CHANNELS * 4
    assert len(blob) < float32_bytes // 20


def test_fallback_for_coordinates_outside_packed_byte_range():
    original = sample(squares=256)  # 16x16 boards
    original.l_coords = np.array([-200, 2, 4], dtype=np.int64)
    original.t_coords = np.array([1, 300, 5], dtype=np.int64)
    round_trip(decode_sample(encode_sample(original)), original)


def test_corrupt_version_and_truncated_blob_rejected():
    payload = zlib.decompress(encode_sample(sample()))
    bad_version = bytes([2]) + payload[1:]
    with pytest.raises(ValueError, match="version"):
        decode_sample(zlib.compress(bad_version))
    with pytest.raises(ValueError, match="zlib"):
        decode_sample(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="truncated"):
        decode_sample(zlib.compress(b""))


def test_invalid_piece_code_rejected_on_decode():
    payload = bytearray(zlib.decompress(encode_sample(sample())))
    payload[_HEADER.size] = PIECE_CHANNELS + 1
    with pytest.raises(ValueError, match="piece code"):
        decode_sample(zlib.compress(bytes(payload)))


def test_legacy_plane_records_still_load():
    original = sample()
    legacy = original.as_dict()
    legacy["board_planes"] = codes_to_planes(original.board_codes)
    del legacy["board_codes"]
    buffer = ReplayBuffer(4, STANDARD)
    buffer.load_state_dict({"samples": [legacy]})
    saved = buffer.sample(1)[0]
    np.testing.assert_array_equal(saved.board_codes, original.board_codes)
    np.testing.assert_allclose(saved.policy, original.policy, atol=1e-3)


def test_very_small_variant_uses_sixteen_squares():
    original = sample(squares=VERY_SMALL.board_squares)
    original.actions = np.array([[0, 0, 0, 0, 1, 1, 0, 0]], dtype=np.int64)
    original.policy = np.array([0.5, 0.5], dtype=np.float32)
    buffer = ReplayBuffer(2, VERY_SMALL)
    buffer.push([original])
    saved = buffer.sample(1)[0]
    assert saved.board_codes.shape == (3, 16)
    fields = (
        "board_codes",
        "l_coords",
        "t_coords",
        "actions",
        "policy",
        "urgency",
        "value_target",
        "side_to_move",
        "has_submit",
        "player",
        "move_count",
        "metadata",
    )
    round_trip({name: getattr(saved, name) for name in fields}, original)
