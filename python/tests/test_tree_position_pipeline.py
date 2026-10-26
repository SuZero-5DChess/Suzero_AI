"""Tree rows participate in training, generated counts and replay checkpoints."""

import json
import math

import pytest
import torch

from suzero_5dchess.config import MctsConfig, SelfPlayConfig, TrainConfig
from suzero_5dchess.protocol import VERY_SMALL

native = pytest.importorskip("suzero_5dchess._5dchess")


@pytest.mark.parametrize("backend", ["python", "cpp_onnx"])
def test_tree_rows_train_and_round_trip_checkpoint(tmp_path, monkeypatch, backend):
    from suzero_5dchess import pipeline
    from suzero_5dchess.network import build_network

    if backend == "cpp_onnx":
        if getattr(native, "TREE_POSITION_FORMAT_VERSION", 0) != 1:
            pytest.skip("requires rebuilt native tree-position extension")
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    def zero_network(config, variant):
        network = build_network(config, variant)
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
        return network

    monkeypatch.setattr(pipeline, "build_network", zero_network)
    config = TrainConfig(
        device="cpu", seed=41, num_iterations=1, epochs_per_iteration=1,
        steps_per_epoch=1, batch_size=2, min_replay_size=2,
        replay_buffer_size=200, save_interval=1, export_interval=0,
        tensorboard=False, gate_interval=0, checkpoint_dir=str(tmp_path / "run"),
        self_play=SelfPlayConfig(
            num_games=1, min_board_limit=4, max_board_limit=4,
            playout_cap_randomization=False, backend=backend,
            record_tree_positions=True, record_tree_min_visits=1,
            record_tree_sample_probability=1.0,
        ),
        mcts=MctsConfig(num_simulations=96, max_depth=16),
    )
    summary = pipeline.run_training(config)
    record = summary["iterations"][0]
    counts = record["self_play"]
    assert counts["tree_samples"] > 0
    assert counts["samples"] == counts["tree_samples"] + counts["played_samples"]
    assert summary["total_samples"] == counts["samples"] == summary["buffer_size"]
    assert math.isfinite(record["training"]["metrics"]["loss"])
    paths = pipeline.RunPaths.from_config(config)
    # The default backend persists one NPZ per game; read the rows back from it.
    store = pipeline.GameStore(paths.replay_store, VERY_SMALL, 200)
    store.scan()
    assert len(store) == counts["samples"]
    sides = [s for s in store.iter_samples() if s.metadata.get("tree_position")]
    assert len(sides) == counts["tree_samples"]
    assert all(s.metadata["search_visits"] >= 1 for s in sides)
    logged = json.loads((paths.log_dir / "training_log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert logged["self_play"]["tree_samples"] == len(sides)
    checkpoint = pipeline._read_checkpoint(paths.latest)
    assert checkpoint["train_config"]["self_play"]["record_tree_positions"] is True
