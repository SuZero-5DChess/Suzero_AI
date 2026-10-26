"""Late-game row weighting, duplication and weighted no-replacement draws."""

from __future__ import annotations

import numpy as np
import pytest

from suzero_5dchess.endgame_sampling import (
    EndgameSamplingConfig,
    endgame_weights,
    extra_copies,
    game_progress,
    repeat_by_weight,
    sampling_weights,
    weighted_without_replacement,
)


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="enabled"):
        EndgameSamplingConfig(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strength"):
        EndgameSamplingConfig(strength=-1.0)
    with pytest.raises(ValueError, match="exponent"):
        EndgameSamplingConfig(exponent=0.0)
    with pytest.raises(ValueError, match="finite"):
        EndgameSamplingConfig(strength=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        EndgameSamplingConfig(exponent=float("inf"))


def test_identity_tracks_enabled_settings():
    assert EndgameSamplingConfig().identity() is None
    assert EndgameSamplingConfig(enabled=True, strength=2, exponent=1).identity() == {
        "strength": 2.0,
        "exponent": 1.0,
    }


def test_progress_is_relative_to_the_game_end():
    progress = game_progress(np.array([0, 5, 10, 20], dtype=np.int64))
    assert progress.tolist() == [0.0, 0.25, 0.5, 1.0]
    # A row set with no measurable progress (all zero) is never boosted.
    assert game_progress(np.zeros(3, dtype=np.int64)).tolist() == [0.0, 0.0, 0.0]
    assert game_progress(np.empty(0, dtype=np.int64)).size == 0


def test_progress_rejects_invalid_counts():
    with pytest.raises(ValueError, match="flat"):
        game_progress(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="non-negative"):
        game_progress(np.array([1, -1], dtype=np.int64))


def test_weights_ramp_from_one_to_the_configured_strength():
    progress = np.array([0.0, 0.5, 1.0])
    config = EndgameSamplingConfig(enabled=True, strength=2.0, exponent=2.0)
    assert sampling_weights(progress, config).tolist() == [1.0, 1.5, 3.0]
    linear = EndgameSamplingConfig(enabled=True, strength=2.0, exponent=1.0)
    assert sampling_weights(progress, linear).tolist() == [1.0, 2.0, 3.0]
    # Disabled means uniform, independent of the shape of progress.
    assert sampling_weights(progress, EndgameSamplingConfig()).tolist() == [1.0] * 3


def test_extra_copies_expect_weight_minus_one():
    weights = np.array([1.0, 1.5, 3.0])
    rng = np.random.default_rng(11)
    draws = np.array([extra_copies(weights, rng) for _ in range(4000)])
    mean = draws.mean(axis=0)
    assert np.allclose(mean, weights - 1.0, atol=0.05)
    # Integer parts are guaranteed; the first row never gets a copy.
    assert (draws[:, 0] == 0).all()
    assert (draws[:, 2] >= 2).all()


def test_repeat_by_weight_keeps_every_row_once_before_extras():
    rows = np.array([4, 5, 6], dtype=np.int64)
    weights = np.array([1.0, 1.0, 3.0])
    repeated = repeat_by_weight(rows, weights, np.random.default_rng(3))
    assert repeated[:3].tolist() == [4, 5, 6]
    assert repeated.tolist().count(6) >= 3
    assert repeated.tolist().count(4) == 1


def test_weighted_without_replacement_is_distinct_and_weighted():
    rows = np.arange(5, dtype=np.int64)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    rng = np.random.default_rng(7)
    draws = [weighted_without_replacement(rows, weights, 2, rng) for _ in range(300)]
    for draw in draws:
        assert len(set(draw.tolist())) == 2
        assert 4 in draw.tolist()
    # The heavy row is always present; the other slot is mostly the next
    # heaviest four rows with equal weight.
    others = [value for draw in draws for value in draw if value != 4]
    assert len(others) == 300
    assert len(set(others)) > 1


def test_weighted_without_replacement_validates_arguments():
    rows = np.arange(3, dtype=np.int64)
    weights = np.ones(3)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="count"):
        weighted_without_replacement(rows, weights, 4, rng)
    with pytest.raises(ValueError, match="count"):
        weighted_without_replacement(rows, weights, 0, rng)
    with pytest.raises(ValueError, match="positive"):
        weighted_without_replacement(rows, np.array([1.0, 0.0, 1.0]), 1, rng)
    with pytest.raises(ValueError, match="equal"):
        weighted_without_replacement(rows, np.ones(2), 1, rng)


def test_endgame_weights_call_the_two_steps_in_order():
    counts = np.array([0, 1, 2, 4], dtype=np.int64)
    config = EndgameSamplingConfig(enabled=True, strength=1.0, exponent=2.0)
    assert np.allclose(
        endgame_weights(counts, config), [1.0, 1.0625, 1.25, 2.0]
    )
