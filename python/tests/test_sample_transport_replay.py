"""Ragged self-play transport and validated replay snapshot regressions."""

import numpy as np
import pytest

from suzero_5dchess.cpp_self_play import _samples_by_game
from suzero_5dchess.cpp_selfplay_processes import _extract_arrays
from suzero_5dchess.protocol import VERY_SMALL, STANDARD
from suzero_5dchess.replay import ReplayBuffer, ReplayDataSource
from suzero_5dchess.sample import PositionSample, collate_samples


def sample():
    return PositionSample(
        board_codes=np.zeros((2, 16), dtype=np.uint8),
        l_coords=np.array([5, 5], dtype=np.int64),
        t_coords=np.array([1, 2], dtype=np.int64),
        urgency=23.0,
        side_to_move=0,
        actions=np.array([[0, 0, 1, 0, 1, 0, 1, 0]], dtype=np.int64),
        policy=np.array([0.75, 0.25], dtype=np.float32),
        has_submit=True,
        metadata={"weight": 1.0},
    )


def test_replay_owns_immutable_validated_snapshot():
    original = sample()
    buffer = ReplayBuffer(3, VERY_SMALL)
    buffer.push([original])
    saved = buffer.sample(1)[0]
    original.board_codes[:] = 13
    original.actions[:] = 0
    original.policy[:] = 0
    original.assign_outcome(-1.0)
    original.metadata["weight"] = 7.0
    assert saved.value_target == 0.0
    assert saved.metadata["weight"] == 1.0
    assert saved.policy.tolist() == [0.75, 0.25]
    assert not saved.board_codes.any()
    for name in ("board_codes", "l_coords", "t_coords", "actions", "policy"):
        array = getattr(saved, name)
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 9
    with pytest.raises(AttributeError):
        saved.value_target = 1.0
    with pytest.raises(AttributeError):
        saved.policy = original.policy
    with pytest.raises(AttributeError):
        del saved.actions
    with pytest.raises(TypeError):
        saved.metadata["weight"] = 2.0
    collate_samples([saved], VERY_SMALL)


@pytest.mark.parametrize("field", ["board_codes", "actions", "policy"])
def test_invalid_content_rejected_at_push_and_load(field):
    invalid = sample()
    getattr(invalid, field)[:] = 0 if field != "board_codes" else 13
    buffer = ReplayBuffer(3, VERY_SMALL)
    buffer.push([sample()])
    with pytest.raises(ValueError):
        buffer.push([sample(), invalid])
    assert len(buffer) == 1
    with pytest.raises(ValueError):
        buffer.load_state_dict({"samples": [invalid.as_dict()]})
    assert len(buffer) == 1


def test_batch_only_skips_content_checks_for_replay(monkeypatch):
    import suzero_5dchess.sample as module

    original = sample()
    buffer = ReplayBuffer(3, VERY_SMALL)
    buffer.push([original])
    source = ReplayDataSource(buffer, 1, 2, VERY_SMALL)

    def fail(*args, **kwargs):
        raise AssertionError("content scan on training path")

    monkeypatch.setattr(module.np, "unique", fail)
    monkeypatch.setattr(module.np, "isin", fail)
    for _ in range(2):
        source.begin_epoch()
        assert source.next_batch().sample_count == 1
    with pytest.raises(AssertionError):
        collate_samples([original], VERY_SMALL)
    with pytest.raises(ValueError, match="shape"):
        collate_samples(buffer.sample(1), STANDARD)


def test_checkpoint_state_does_not_alias_replay():
    buffer = ReplayBuffer(3, VERY_SMALL)
    buffer.push([sample()])
    state = buffer.state_dict()
    assert state["format"] == "gn5d_replay_zlib_v1"
    restored = ReplayBuffer(3, VERY_SMALL)
    restored.load_state_dict(state)
    replacement = np.zeros_like(state["offsets"])
    replacement[:] = 0
    state["offsets"] = replacement
    for replay in (buffer, restored):
        saved = replay.sample(1)[0]
        assert not saved.board_codes.any()
        assert saved.metadata["weight"] == 1.0
        assert not saved.board_codes.flags.writeable
        assert not saved.actions.flags.writeable


def flat_payload():
    codes = np.zeros((4, 16), dtype=np.uint8)
    codes[0, 0] = 1
    codes[1, 3] = 2
    codes[2, 7] = 12
    codes[3, 15] = 5
    return {
        "board_codes": codes,
        "l_coords": np.arange(4, dtype=np.int64),
        "t_coords": np.arange(4, dtype=np.int64),
        "board_offsets": np.array([0, 1, 4], dtype=np.int64),
        "action_offsets": np.array([0, 2, 3], dtype=np.int64),
        "used_board_counts": np.array([1, 3], dtype=np.int64),
        "action_counts": np.array([2, 1], dtype=np.int64),
        "actions": np.arange(24, dtype=np.int64).reshape(3, 8),
        "policy": np.array([0.75, 0.25, 1.0], dtype=np.float32),
        "has_submit": np.array([True, False]),
        "urgency": np.array([24, 22], dtype=np.float32),
        "side_to_move": np.array([0, 1], dtype=np.int64),
        "player": np.array([0, 1], dtype=np.int64),
        "move_count": np.array([1, 2], dtype=np.int64),
        "game_index": np.array([0, 1], dtype=np.int64),
        "value": np.array([1, -1], dtype=np.float32),
    }


def test_flat_transport_round_trip_and_independent_storage(tmp_path):
    raw = flat_payload()
    arrays = _extract_arrays(raw)
    assert "board_offsets" in arrays and "action_offsets" in arrays
    path = tmp_path / "samples.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as data:
        grouped = _samples_by_game({key: data[key] for key in data.files})
    a, b = grouped[0][0], grouped[1][0]
    np.testing.assert_array_equal(a.board_codes, raw["board_codes"][:1])
    np.testing.assert_array_equal(b.board_codes, raw["board_codes"][1:])
    np.testing.assert_array_equal(a.actions, raw["actions"][:1])
    np.testing.assert_array_equal(b.actions, raw["actions"][2:])
    assert a.policy.tolist() == [0.75, 0.25]
    assert b.policy.tolist() == [1.0]
    assert a.board_codes.base is None
    assert b.actions.base is None


@pytest.mark.parametrize("offsets", [
    np.array([0, 1, 5], dtype=np.int64),
    np.array([1, 2, 4], dtype=np.int64),
    np.array([0, 3, 4], dtype=np.int64),
    np.array([0, -1, 4], dtype=np.int64),
    np.array([0, 1, 4], dtype=np.float32),
    np.array([0, 4], dtype=np.int64),
])
def test_flat_rejects_corrupt_offsets(offsets):
    raw = flat_payload()
    raw["board_offsets"] = offsets
    with pytest.raises(ValueError, match="offsets"):
        _samples_by_game(raw)


@pytest.mark.parametrize("keys", [("action_offsets",), ("board_offsets",), ("board_offsets", "action_offsets")])
def test_flat_requires_both_offsets(keys):
    raw = flat_payload()
    for key in keys:
        del raw[key]
    with pytest.raises(ValueError, match="offsets"):
        _samples_by_game(raw)


def test_worker_rejects_object_offsets():
    raw = flat_payload()
    raw["board_offsets"] = np.array([0, 1, 4], dtype=object)
    with pytest.raises(RuntimeError, match="int64"):
        _extract_arrays(raw)


def test_empty_flat_transport():
    raw = flat_payload()
    for key, array in list(raw.items()):
        raw[key] = array[:0] if not key.endswith("offsets") else np.array([0], dtype=np.int64)
    assert _samples_by_game(raw) == {}
