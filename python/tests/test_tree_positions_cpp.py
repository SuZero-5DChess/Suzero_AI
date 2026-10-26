"""Real C++ ONNX tree-target and multi-process transport integration tests."""

import numpy as np
import pytest
import torch

from suzero_5dchess.config import NetworkConfig
from suzero_5dchess.cpp_self_play import _selfplay_results
from suzero_5dchess.network import build_network
from suzero_5dchess.protocol import VERY_SMALL
from suzero_5dchess.sample import validate_sample

native = pytest.importorskip("suzero_5dchess._5dchess")
pytestmark = pytest.mark.skipif(
    getattr(native, "TREE_POSITION_FORMAT_VERSION", 0) != 1,
    reason="requires rebuilt tree-position native extension",
)


@pytest.fixture(scope="module")
def binding_kwargs(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from suzero_5dchess.export_onnx import export_network

    network = build_network(NetworkConfig(), VERY_SMALL)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    path = export_network(network, VERY_SMALL, tmp_path_factory.mktemp("tree-native") / "model.onnx", verify=False)
    return dict(
        model_path=str(path), metadata_path=str(path) + ".json", provider="cpu",
        cuda_device_id=0, variant_pgn=VERY_SMALL.pgn, board_side=4,
        line_shift=VERY_SMALL.line_shift, num_games=2, num_threads=1,
        num_simulations=96, fast_simulations=2, playout_cap_randomization=False,
        full_search_probability=1.0, c_puct=2.0, dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25, max_depth=16, temperature=1.0,
        temperature_final=0.1, temperature_threshold=30, min_board_limit=4,
        max_board_limit=4, material_scale=2.0, max_batch_size=16, seed=41,
        record_games=True, record_tree_min_visits=1,
        record_tree_max_depth=5, record_tree_max_per_turn=4,
        record_tree_sample_probability=1.0,
    )


def assert_samples_equal(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        for field in ("board_codes", "l_coords", "t_coords", "actions", "policy"):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
        assert left.metadata == right.metadata
        assert left.value_target == right.value_target
        assert left.player == right.player
        assert left.move_count == right.move_count


def test_native_tree_targets_preserve_game_and_search_budget(binding_kwargs):
    # Omit record_tree_positions: the binding must enable it by default.
    off = native.run_selfplay(**{**binding_kwargs, "record_tree_positions": False})
    on = native.run_selfplay(**binding_kwargs)
    assert on["stats"]["rows"] == off["stats"]["rows"]
    assert on["stats"]["requests"] == off["stats"]["requests"]
    assert on["sample_source"].dtype == np.int64
    assert on["search_visits"].dtype == np.int64
    changed_value = False
    total_sides = 0
    for plain, enabled in zip(_selfplay_results(off), _selfplay_results(on)):
        assert plain.moves == enabled.moves
        assert plain.final_fen == enabled.final_fen
        assert plain.white_outcome == enabled.white_outcome
        main = [s for s in enabled.samples if not s.metadata.get("tree_position")]
        sides = [s for s in enabled.samples if s.metadata.get("tree_position")]
        assert_samples_equal(plain.samples, main)
        assert len(sides) == enabled.tree_samples
        total_sides += len(sides)
        for sample in enabled.samples:
            validate_sample(sample, VERY_SMALL)
        for sample in sides:
            assert sample.metadata["search_visits"] >= 1
            outcome = enabled.white_outcome if sample.player == 0 else -enabled.white_outcome
            changed_value |= abs(sample.value_target - outcome) > 1e-6
    assert total_sides > 0
    assert changed_value, "side search values must not be overwritten by real-game outcomes"


def test_native_probability_zero_and_fast_search_skip_tree_rows(binding_kwargs):
    for overrides in (
        {"record_tree_sample_probability": 0.0},
        {"playout_cap_randomization": True, "full_search_probability": 0.0},
    ):
        raw = native.run_selfplay(**{**binding_kwargs, **overrides})
        assert not raw["sample_source"].any()
        assert all(game["tree_samples"] == 0 for game in raw["games"])


def test_native_multiprocess_retains_tree_metadata(binding_kwargs):
    from suzero_5dchess.cpp_selfplay_processes import run_self_play_processes

    expected = _selfplay_results(native.run_selfplay(**binding_kwargs))
    actual = run_self_play_processes(
        {**binding_kwargs, "num_threads": 2}, processes=2, total_server_threads=2,
    )
    assert len(actual) == len(expected)
    assert sum(game.tree_samples for game in actual) > 0
    for left, right in zip(expected, actual):
        assert left.moves == right.moves
        assert left.final_fen == right.final_fen
        assert left.tree_samples == right.tree_samples
        assert_samples_equal(left.samples, right.samples)
