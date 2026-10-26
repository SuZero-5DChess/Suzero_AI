"""Gate routing, durable candidates and training progress without native matches."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from suzero_5dchess import pipeline
from suzero_5dchess.config import NetworkConfig, SelfPlayConfig, TrainConfig
from suzero_5dchess.gating import ModelGate
from suzero_5dchess.sample import PositionSample
from suzero_5dchess.self_play import SelfPlayResult


def config_for(path, **changes):
    config = TrainConfig(
        checkpoint_dir=str(path), device="cpu", tensorboard=False,
        num_iterations=2, epochs_per_iteration=1, steps_per_epoch=1,
        batch_size=2, min_replay_size=1, replay_buffer_size=20,
        save_interval=10, export_interval=0,
        network=NetworkConfig(d_model=8, n_heads=2, d_ff=16,
                              action_mlp_hidden=8, dropout=0.0),
        self_play=SelfPlayConfig(num_games=2), gate_interval=1,
        gate_games=2, gate_workers=1, gate_simulations=1,
        gate_min_finished=2, gate_provider="cpu", gate_max_boards=2,
    )
    return replace(config, **changes)


def weights(network):
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}


def equal_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def fake_games(network, *args, **kwargs):
    sample = PositionSample(
        board_codes=np.zeros((1, 16), dtype=np.uint8),
        l_coords=np.array([0]), t_coords=np.array([1]), urgency=24.0,
        side_to_move=0, actions=np.array([[0, 0, 1, 0, 1, 0, 1, 0]]),
        policy=np.array([1.0], dtype=np.float32), has_submit=False,
        value_target=1.0,
    )
    return [SelfPlayResult(
        samples=[sample], white_outcome=1.0, terminal_reason="material",
        semimoves=1, turns=1, board_limit=2, board_count=2, final_player=0,
    ) for _ in range(2)]


def install_gate(monkeypatch, runner):
    instances = []

    def factory(config, paths, spec, device):
        gate = ModelGate(config, paths, spec, device, run_match=runner)
        instances.append(gate)
        return gate

    monkeypatch.setattr(pipeline, "ModelGate", factory)
    return instances


def totals(wins=0, losses=0, draws=0, truncated=0):
    games = [
        {"terminal": True, "truncated": False,
         "participant_utilities": [utility, -utility]}
        for utility, count in ((1.0, wins), (-1.0, losses), (0.0, draws))
        for _ in range(count)
    ]
    games += [{"terminal": False, "truncated": True,
               "participant_utilities": [None, None]}
              for _ in range(truncated)]
    return {"totals": [{"wins": wins, "losses": losses, "draws": draws,
                        "truncated": truncated}], "games": games}


@pytest.mark.parametrize("backend", ["python", "cpp_onnx"])
def test_disabled_uses_training_network_and_original_log_keys(tmp_path, monkeypatch, backend):
    cfg = config_for(tmp_path, gate_interval=0)
    cfg.self_play.backend = backend
    networks = []
    original = pipeline._make_trainer

    def trainer(network, *args):
        networks.append(network)
        return original(network, *args)

    def generate(network, *args, **kwargs):
        assert network is networks[0]
        return fake_games(network)

    def unexpected(*args, **kwargs):
        raise AssertionError("disabled gating must not construct ModelGate")

    monkeypatch.setattr(pipeline, "_make_trainer", trainer)
    monkeypatch.setattr(pipeline, "ModelGate", unexpected)
    monkeypatch.setattr(pipeline, "generate_games", generate)
    monkeypatch.setattr(pipeline, "generate_games_cpp", generate)
    summary = pipeline.run_training(cfg)
    paths = pipeline.RunPaths.from_config(cfg)
    assert not paths.accepted_dir.exists()
    assert not paths.gate_log_dir.exists()
    for record in summary["iterations"]:
        assert "gate" not in record
        assert "selfplay_model_iteration" not in record
    checkpoint = torch.load(paths.latest, weights_only=False)
    assert "accepted" not in checkpoint["extra_state"]
    assert not paths.iteration_checkpoint(1).exists()


@pytest.mark.parametrize("backend", ["python", "cpp_onnx"])
@pytest.mark.parametrize("promote", [False, True])
def test_gate_routes_only_accepted_and_never_rolls_back_training(
    tmp_path, monkeypatch, backend, promote,
):
    cfg = config_for(tmp_path)
    cfg.self_play.backend = backend
    observed, candidates = [], []
    paths = pipeline.RunPaths.from_config(cfg)

    def generate(network, *args, **kwargs):
        assert not network.training
        observed.append(weights(network))
        if backend == "cpp_onnx":
            assert kwargs["model_path"] == paths.root / "selfplay" / f"model_iter_{len(observed):04d}.onnx"
        return fake_games(network)

    def runner(candidate, accepted, iteration):
        # Even with sparse saves, a candidate is resumable before gate publication.
        checkpoint = torch.load(paths.iteration_checkpoint(iteration), weights_only=False)
        assert checkpoint["extra_state"]["iteration"] == iteration
        assert "optimizer" in checkpoint
        assert equal_weights(checkpoint["model"], weights(candidate))
        assert candidate is not accepted
        candidates.append(weights(candidate))
        return totals(wins=2) if promote else totals(losses=2)

    install_gate(monkeypatch, runner)
    monkeypatch.setattr(pipeline, "generate_games", generate)
    monkeypatch.setattr(pipeline, "generate_games_cpp", generate)
    summary = pipeline.run_training(cfg)
    first, second = summary["iterations"]
    assert first["selfplay_model_iteration"] == 0
    assert second["selfplay_model_iteration"] == (1 if promote else 0)
    assert equal_weights(observed[1], candidates[0] if promote else observed[0])
    assert not equal_weights(candidates[0], observed[0])
    assert not equal_weights(candidates[1], candidates[0])
    assert first["gate"]["accepted_before"] == 0
    assert first["gate"]["accepted_after"] == (1 if promote else 0)
    assert second["gate"]["accepted_before"] == (1 if promote else 0)
    assert second["gate"]["accepted_after"] == (2 if promote else 0)
    assert first["gate"]["decision"] == ("accepted" if promote else "rejected")
    state = json.loads(paths.accepted_state.read_text())
    assert state["iteration"] == (2 if promote else 0)
    final = torch.load(paths.latest, weights_only=False)
    assert final["extra_state"]["iteration"] == 2
    assert final["extra_state"]["accepted"]["iteration"] == state["iteration"]
    assert equal_weights(final["model"], candidates[-1])
    assert paths.iteration_checkpoint(1).exists()
    for iteration in (1, 2):
        assert torch.load(paths.iteration_checkpoint(iteration), weights_only=False)["extra_state"]["iteration"] == iteration
    records = [json.loads(line) for line in (paths.log_dir / "training_log.jsonl").read_text().splitlines()]
    assert [r["selfplay_model_iteration"] for r in records] == [0, 1 if promote else 0]
    assert [r["gate"] for r in records] == [first["gate"], second["gate"]]
    assert len(list(paths.gate_log_dir.glob("*.json"))) == 2


def test_resume_restores_accepted_and_bootstrap_uses_resumed_weights(tmp_path, monkeypatch):
    cfg = config_for(tmp_path, num_iterations=1)
    monkeypatch.setattr(pipeline, "generate_games", fake_games)
    install_gate(monkeypatch, lambda *_: totals(wins=2))
    pipeline.run_training(cfg)
    paths = pipeline.RunPaths.from_config(cfg)
    first = torch.load(paths.latest, weights_only=False)
    observed = []

    def generate(network, *args, **kwargs):
        observed.append(weights(network))
        return fake_games(network)

    monkeypatch.setattr(pipeline, "generate_games", generate)
    install_gate(monkeypatch, lambda *_: totals(losses=2))
    resumed = pipeline.run_training(cfg, resume=paths.latest)
    record = resumed["iterations"][0]
    assert record["selfplay_model_iteration"] == 1
    assert record["gate"]["accepted_before"] == record["gate"]["accepted_after"] == 1
    assert equal_weights(observed[0], first["model"])
    assert json.loads(paths.accepted_state.read_text())["iteration"] == 1

    # Enabling gate on an existing ungated run bootstraps the restored network.
    ungated = config_for(tmp_path / "ungated", num_iterations=1, gate_interval=0)
    pipeline.run_training(ungated)
    old_paths = pipeline.RunPaths.from_config(ungated)
    restored = torch.load(old_paths.latest, weights_only=False)
    observed.clear()
    enabled = replace(ungated, gate_interval=2)
    pipeline.run_training(enabled, resume=old_paths.latest)
    assert equal_weights(observed[0], restored["model"])
    assert json.loads(old_paths.accepted_state.read_text())["iteration"] == 1


@pytest.mark.parametrize("no_training", [False, True])
def test_skipped_gate_records_reason_without_match(tmp_path, monkeypatch, no_training):
    cfg = config_for(tmp_path, gate_interval=3)
    if no_training:
        cfg.min_replay_size = 20
    monkeypatch.setattr(pipeline, "generate_games", fake_games)

    def unexpected(*args):
        raise AssertionError("skipped gate must not run a match")

    install_gate(monkeypatch, unexpected)
    result = pipeline.run_training(cfg)
    for row in result["iterations"]:
        assert row["selfplay_model_iteration"] == 0
        assert row["gate"]["skipped"]
        assert row["gate"]["reason"] == ("no_training" if no_training else "interval")
        assert row["gate"]["accepted_after"] == 0
    assert not pipeline.RunPaths.from_config(cfg).gate_log_dir.exists()


def test_match_error_rejects_and_training_continues(tmp_path, monkeypatch):
    cfg = config_for(tmp_path)
    monkeypatch.setattr(pipeline, "generate_games", fake_games)

    def fail(*args):
        raise RuntimeError("injected match failure")

    install_gate(monkeypatch, fail)
    result = pipeline.run_training(cfg)
    assert result["end_iteration"] == 2
    for row in result["iterations"]:
        assert row["gate"]["reason"] == "match_error"
        assert row["gate"]["accepted_before"] == row["gate"]["accepted_after"] == 0
        report = json.loads(open(row["gate"]["match_log"], encoding="utf-8").read())
        assert "injected match failure" in report["error"]


def test_accepted_ahead_of_resume_fails_before_selfplay(tmp_path, monkeypatch):
    cfg = config_for(tmp_path)
    monkeypatch.setattr(pipeline, "generate_games", fake_games)
    install_gate(monkeypatch, lambda *_: totals(wins=2))
    pipeline.run_training(cfg)
    paths = pipeline.RunPaths.from_config(cfg)

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid resume must fail before self-play")

    monkeypatch.setattr(pipeline, "generate_games", unexpected)
    with pytest.raises(ValueError, match="ahead"):
        pipeline.run_training(cfg, resume=paths.iteration_checkpoint(1))
