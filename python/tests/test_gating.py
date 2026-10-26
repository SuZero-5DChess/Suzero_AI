"""Gate persistence and decisions with injected runners: no native matches or ONNX."""

from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from suzero_5dchess import gating
from suzero_5dchess.checkpoint import CHECKPOINT_FORMAT
from suzero_5dchess.config import GateConfig, NetworkConfig, TrainConfig
from suzero_5dchess.gating import ModelGate, decide_gate
from suzero_5dchess.network import build_network
from suzero_5dchess.protocol import ACTION_ENCODING, VERY_SMALL


def config(**changes):
    return TrainConfig(
        network=NetworkConfig(d_model=16, n_heads=4, d_ff=32,
                              action_mlp_hidden=16, dropout=0.1),
        **{"gate_interval": 2, "gate_games": 4, "gate_min_finished": 2,
           "gate_provider": "cpu", **changes},
    )


def paths(root):
    return SimpleNamespace(accepted_dir=root / "accepted",
                           accepted_state=root / "accepted" / "state.json",
                           gate_log_dir=root / "logs" / "gates")


def raw_result(wins=3, losses=1, draws=0, truncated=0):
    games = [
        {"terminal": True, "truncated": False,
         "participant_utilities": [utility, -utility]}
        for utility, count in ((1.0, wins), (-1.0, losses), (0.0, draws))
        for _ in range(count)
    ]
    games += [{"terminal": False, "truncated": True,
               "participant_utilities": [None, None]}
              for _ in range(truncated)]
    return {"totals": [{"wins": wins, "losses": losses,
                        "draws": draws, "truncated": truncated}], "games": games}


def make_gate(root, *, cfg=None, runner=None):
    cfg = cfg or config()
    net = build_network(cfg.network, VERY_SMALL)
    gate = ModelGate(cfg, paths(root), VERY_SMALL, torch.device("cpu"),
                     run_match=runner if runner is not None else lambda *args: raw_result())
    return gate, net


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("counts,scores,threshold,minimum,decision,reason,average", [
    ((1, 1, 0, 0), [0.5, -0.25], 0.125, 2, "accepted", "score", 0.125),
    ((1, 1, 0, 0), [0.5, -0.25], 0.126, 2, "rejected", "score", 0.125),
    ((1, 0, 0, 0), [0.04], 0.05, 1, "rejected", "score", 0.04),
    ((1, 0, 0, 0), [1.0], 0.05, 2, "rejected", "insufficient_finished", 1.0),
    ((0, 0, 0, 0), [], 0.05, 0, "rejected", "insufficient_finished", None),
    ((0, 0, 2, 0), [0.0, 0.0], 0.0, 2, "accepted", "score", 0.0),
    ((0, 0, 0, 2), [], 0.05, 0, "rejected", "insufficient_finished", None),
])
def test_pure_decision(counts, scores, threshold, minimum, decision, reason, average):
    totals = dict(zip(("wins", "losses", "draws", "truncated"), counts))
    result = decide_gate(totals, scores, threshold=threshold, min_finished=minimum)
    assert result.decision == decision
    assert result.reason == reason
    if average is None:
        assert result.average_score is None
    else:
        assert result.average_score == pytest.approx(average)
    assert result.finished == sum(counts[:3])
    assert result.games == sum(counts)
    assert result.as_dict()["threshold"] == threshold
    assert gating.GateConfig is GateConfig


def test_finished_scores_keep_material_score_and_skip_truncation():
    result = {"games": [
        {"truncated": False, "participant_utilities": [0.42, -0.42]},
        {"truncated": True, "participant_utilities": [None, None]},
        {"truncated": False, "participant_utilities": [-1.0, 1.0]},
    ]}
    assert gating.finished_scores(result) == [0.42, -1.0]


@pytest.mark.parametrize("result", [
    {}, {"games": "x"}, {"games": [None]},
    {"games": [{"truncated": 0, "participant_utilities": [1.0, -1.0]}]},
    {"games": [{"truncated": False, "participant_utilities": [None, None]}]},
    {"games": [{"truncated": False, "participant_utilities": [float("nan"), 0.0]}]},
    {"games": [{"truncated": False, "participant_utilities": [1.0]}]},
])
def test_finished_scores_reject_malformed_matches(result):
    with pytest.raises(ValueError):
        gating.finished_scores(result)


def test_decision_rejects_score_count_mismatch():
    with pytest.raises(ValueError, match="cover"):
        decide_gate({"wins": 2, "losses": 0, "draws": 0, "truncated": 0},
                    [1.0], threshold=0.05, min_finished=1)


def test_disabled_never_touches_paths_or_binding(tmp_path, monkeypatch):
    def fail():
        pytest.fail("disabled gate requested the native binding")
    monkeypatch.setattr(gating, "_require_binding", fail)
    cfg = config(gate_interval=0)
    net = build_network(cfg.network, VERY_SMALL)
    # No path attributes at all: a disabled gate must not inspect the filesystem.
    gate = ModelGate(cfg, object(), VERY_SMALL, torch.device("cpu"))
    assert gate.resolve(net) is net
    assert gate.bootstrap_or_load(net, 7).iteration == 7
    assert gate.evaluate(net, 8, trained=True).enabled is False
    assert gate.evaluate(net, 8, trained=False).skipped
    assert list(tmp_path.iterdir()) == []


def test_binding_required_only_for_default_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(gating, "_require_binding", lambda: None)
    monkeypatch.setattr(gating, "_5dchess", SimpleNamespace(run_match=None))
    with pytest.raises(RuntimeError, match="rebuild"):
        ModelGate(config(), paths(tmp_path), VERY_SMALL, torch.device("cpu"))
    make_gate(tmp_path)  # Injected complete runners require neither ONNX nor binding.
    assert not paths(tmp_path).accepted_dir.exists()


def test_bootstrap_promote_and_fresh_reload(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="suzero_5dchess.gating")
    calls = []
    def runner(candidate, accepted, iteration):
        calls.append(iteration)
        assert accepted is not candidate
        assert not accepted.training
        assert not candidate.training
        assert next(accepted.parameters()).data_ptr() != next(candidate.parameters()).data_ptr()
        return raw_result()

    gate, net = make_gate(tmp_path, runner=runner)
    modes = [module.training for module in net.modules()]
    state = gate.bootstrap_or_load(net, 0)
    assert state.iteration == 0
    assert state.checkpoint == "agent_iter_0000.pt"
    assert state.updated_at.endswith("Z")
    assert datetime.fromisoformat(state.updated_at[:-1] + "+00:00").utcoffset().total_seconds() == 0
    assert read_json(gate.paths.accepted_state)["updated_at"] == state.updated_at
    initial = (gate.paths.accepted_dir / state.checkpoint).read_bytes()
    accepted = gate.resolve(net)
    before = next(accepted.parameters()).detach().clone()
    with torch.no_grad():
        next(net.parameters()).add_(1)
    assert torch.equal(next(accepted.parameters()), before)
    decision = gate.evaluate(net, 2, trained=True)
    assert calls == [2]
    assert (decision.decision, decision.reason) == ("accepted", "score")
    assert (decision.accepted_before, decision.accepted_after) == (0, 2)
    assert [module.training for module in net.modules()] == modes
    assert (gate.paths.accepted_dir / state.checkpoint).read_bytes() == initial
    full = read_json(Path(decision.match_log))
    assert full["summary"] == decision.as_dict()
    assert full["match"] == raw_result()
    snapshot = torch.load(gate.paths.accepted_dir / gate.state.checkpoint,
                          weights_only=False)
    assert snapshot["extra_state"]["format"] == CHECKPOINT_FORMAT
    assert snapshot["extra_state"]["action_encoding"] == ACTION_ENCODING
    assert snapshot["extra_state"]["network_config"] == asdict(net.config)
    assert gate.summary() == {"iteration": 2, "checkpoint": "agent_iter_0002.pt"}
    fresh, other_net = make_gate(tmp_path)
    assert fresh.bootstrap_or_load(other_net, 2).iteration == 2
    assert fresh.state.updated_at == gate.state.updated_at
    assert "bootstrapped accepted iteration 0" in caplog.text
    assert "loaded accepted iteration 2" in caplog.text
    assert "candidate iteration 2 decision=accepted reason=score accepted=0 -> 2" in caplog.text
    assert (
        "candidate result: 3 wins, 1 losses, 0 draws (4 finished); "
        "average score=+0.5000 (candidate perspective, forced material included), "
        "win rate=75.0% (wins/finished); threshold=+0.0500"
    ) in caplog.text
    loaded = fresh.resolve(other_net)
    assert all(torch.equal(value, loaded.state_dict()[key])
               for key, value in net.state_dict().items())
    assert loaded is not net and loaded is not other_net
    assert not loaded.training


def test_decision_log_reports_average_score_without_truncation_noise(caplog):
    from suzero_5dchess.gating import GateDecision

    caplog.set_level(logging.INFO, logger="suzero_5dchess.gating")
    ModelGate._log_decision(GateDecision(
        candidate_iteration=4, wins=3, losses=1, draws=0, finished=4,
        average_score=0.4, threshold=0.05, seconds=1.5,
    ))
    assert "average score=+0.4000" in caplog.text
    assert "threshold=+0.0500" in caplog.text
    assert "truncated" not in caplog.text
    caplog.clear()
    ModelGate._log_decision(GateDecision(
        candidate_iteration=5, wins=3, losses=1, draws=0, truncated=2,
        finished=4, average_score=0.4, threshold=0.05,
    ))
    assert "2 truncated" in caplog.text


def test_reload_uses_state_and_checkpoint_stat_cache(tmp_path, monkeypatch):
    gate, net = make_gate(tmp_path)
    gate.bootstrap_or_load(net, 0)
    original = gating.load_network_checkpoint
    loads = []
    def load(*args, **kwargs):
        loads.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(gating, "load_network_checkpoint", load)
    first = gate.resolve(net)
    assert gate.resolve(net) is first
    assert len(loads) == 1
    path = gate.paths.accepted_dir / gate.state.checkpoint
    snapshot = torch.load(path, weights_only=False)
    key = next(iter(snapshot["model"]))
    snapshot["model"][key].add_(1)
    stamp = path.stat().st_mtime_ns
    torch.save(snapshot, path)
    os.utime(path, ns=(stamp + 1_000_000, stamp + 1_000_000))
    changed = gate.resolve(net)
    assert changed is not first
    assert len(loads) == 2
    assert torch.equal(changed.state_dict()[key], snapshot["model"][key])
    # Re-read the pointer, not merely a constructor-time state object.
    state = read_json(gate.paths.accepted_state)
    state["variant"] = "standard"
    write_json(gate.paths.accepted_state, state)
    with pytest.raises(ValueError, match="variant"):
        gate.resolve(net)


@pytest.mark.parametrize("change,match", [
    ({"format": 2}, "format"), ({"format": True}, "format"),
    ({"iteration": -1}, "iteration"), ({"iteration": True}, "iteration"),
    ({"iteration": 1.0}, "iteration"), ({"iteration": 3}, "ahead"),
    ({"variant": "standard"}, "variant"), ({"network_config": {}}, "network_config"),
    ({"checkpoint": "../outside.pt"}, "basename"),
    ({"checkpoint": "..\\outside.pt"}, "basename"),
    ({"checkpoint": "C:outside.pt"}, "basename"),
    ({"checkpoint": "missing.pt"}, "does not exist"),
    ({"updated_at": 123}, "updated_at"),
    ({"updated_at": "not a timestampZ"}, "Invalid isoformat"),
    ({"updated_at": "2026-10-25T00:00:00+00:00"}, "ending in Z"),
    ({"updated_at": "2026-10-25Z"}, "UTC ISO timestamp"),
])
def test_invalid_state_is_fatal_without_fallback(tmp_path, change, match):
    gate, net = make_gate(tmp_path)
    gate.bootstrap_or_load(net, 0)
    state = read_json(gate.paths.accepted_state)
    state.update(change)
    write_json(gate.paths.accepted_state, state)
    contents = gate.paths.accepted_state.read_bytes()
    fresh, net = make_gate(tmp_path)
    with pytest.raises(ValueError, match=match):
        fresh.bootstrap_or_load(net, 2)
    assert gate.paths.accepted_state.read_bytes() == contents
    assert len(list(gate.paths.accepted_dir.glob("*.pt"))) == 1


def test_malformed_json_and_unloadable_checkpoint_are_fatal(tmp_path):
    gate, net = make_gate(tmp_path)
    gate.bootstrap_or_load(net, 0)
    state_bytes = gate.paths.accepted_state.read_bytes()
    gate.paths.accepted_state.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid gate state"):
        gate.bootstrap_or_load(net, 0)
    gate.paths.accepted_state.write_bytes(state_bytes)
    checkpoint = gate.paths.accepted_dir / gate.state.checkpoint
    checkpoint.write_bytes(b"broken checkpoint")
    with pytest.raises(ValueError, match="accepted checkpoint"):
        gate.bootstrap_or_load(net, 0)


@pytest.mark.parametrize("field,value", [
    ("format", "wrong"), ("action_encoding", "wrong"),
    ("variant", "standard"), ("network_config", {}), ("iteration", 1),
    ("iteration", False),
])
def test_checkpoint_metadata_must_match_state(tmp_path, field, value):
    gate, net = make_gate(tmp_path)
    gate.bootstrap_or_load(net, 0)
    checkpoint = gate.paths.accepted_dir / gate.state.checkpoint
    data = torch.load(checkpoint, weights_only=False)
    data["extra_state"][field] = value
    torch.save(data, checkpoint)
    with pytest.raises(ValueError, match="checkpoint"):
        gate.bootstrap_or_load(net, 0)


def test_checkpoint_weights_must_be_loadable(tmp_path):
    gate, net = make_gate(tmp_path)
    gate.bootstrap_or_load(net, 0)
    checkpoint = gate.paths.accepted_dir / gate.state.checkpoint
    data = torch.load(checkpoint, weights_only=False)
    data["model"].pop(next(iter(data["model"])))
    torch.save(data, checkpoint)
    with pytest.raises(ValueError, match="cannot load accepted checkpoint"):
        gate.bootstrap_or_load(net, 0)


def test_immutable_snapshot_reused_only_for_identical_weights(tmp_path):
    gate, net = make_gate(tmp_path)
    gate.bootstrap_or_load(net, 0)
    snapshot = gate.paths.accepted_dir / gate.state.checkpoint
    original = snapshot.read_bytes()
    gate.paths.accepted_state.unlink()
    gate.bootstrap_or_load(net, 0)
    assert snapshot.read_bytes() == original
    gate.paths.accepted_state.unlink()
    with torch.no_grad():
        next(net.parameters()).add_(1)
    with pytest.raises(ValueError, match="immutable"):
        gate.bootstrap_or_load(net, 0)
    assert not gate.paths.accepted_state.exists()
    assert snapshot.read_bytes() == original


def test_rng_modes_and_skip_priority(tmp_path):
    calls = []
    def runner(candidate, accepted, iteration):
        torch.rand(17)
        candidate.train()
        accepted.train()
        calls.append(iteration)
        return raw_result(wins=0, losses=4)
    gate, net = make_gate(tmp_path, runner=runner)
    net.train()
    net.blocks[0].eval()
    modes = [module.training for module in net.modules()]
    rng = torch.get_rng_state().clone()
    gate.bootstrap_or_load(net, 0)
    gate.resolve(net)
    assert torch.equal(torch.get_rng_state(), rng)
    no_training = gate.evaluate(net, 1, trained=False)
    assert no_training.reason == "no_training" and no_training.skipped
    interval = gate.evaluate(net, 1, trained=True)
    assert interval.reason == "interval" and interval.skipped
    assert calls == []
    assert not gate.paths.gate_log_dir.exists()
    decision = gate.evaluate(net, 2, trained=True)
    assert decision.decision == "rejected" and decision.reason == "score"
    assert gate.state.iteration == 0
    assert torch.equal(torch.get_rng_state(), rng)
    assert [module.training for module in net.modules()] == modes
    assert not gate.resolve(net).training
    assert len(list(gate.paths.gate_log_dir.glob("*.json"))) == 1


def test_runner_error_logged_and_preserves_rng_and_modes(tmp_path):
    def runner(candidate, accepted, iteration):
        torch.rand(7)
        candidate.eval()
        accepted.train()
        raise RuntimeError("fake runner failed")
    gate, net = make_gate(tmp_path, runner=runner)
    gate.bootstrap_or_load(net, 0)
    rng = torch.get_rng_state().clone()
    decision = gate.evaluate(net, 2, trained=True)
    assert decision.reason == "match_error" and not decision.skipped
    assert decision.accepted_after == 0
    assert net.training
    assert torch.equal(torch.get_rng_state(), rng)
    report = read_json(Path(decision.match_log))
    assert report["match"] is None
    assert report["error"] == "RuntimeError: fake runner failed"
    assert report["summary"] == decision.as_dict()


def test_promotion_io_error_is_fatal_not_match_error(tmp_path, monkeypatch):
    gate, net = make_gate(tmp_path)
    gate.bootstrap_or_load(net, 0)
    state = gate.paths.accepted_state.read_bytes()
    replace = gating.os.replace
    def fail_state(source, target):
        if Path(target) == gate.paths.accepted_state:
            raise OSError("state publication failed")
        return replace(source, target)
    monkeypatch.setattr(gating.os, "replace", fail_state)
    with pytest.raises(OSError, match="publication"):
        gate.evaluate(net, 2, trained=True)
    assert gate.paths.accepted_state.read_bytes() == state
    assert gate.state.iteration == 0
    assert (gate.paths.accepted_dir / "agent_iter_0002.pt").is_file()
    assert not gate.paths.gate_log_dir.exists()
    assert not list(gate.paths.accepted_dir.glob("*.tmp"))


def test_snapshot_failure_does_not_publish_state(tmp_path, monkeypatch):
    gate, net = make_gate(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("snapshot failed")
    monkeypatch.setattr(gating.torch, "save", fail)
    with pytest.raises(OSError, match="snapshot failed"):
        gate.bootstrap_or_load(net, 0)
    assert not gate.paths.accepted_state.exists()


def test_ahead_state_explains_safe_recovery(tmp_path):
    gate, net = make_gate(tmp_path)
    gate.bootstrap_or_load(net, 2)
    with pytest.raises(ValueError) as error:
        gate.bootstrap_or_load(net, 0)
    message = str(error.value)
    assert "checkpoint aligned" in message
    assert "accepted/state.json" in message
    assert "new run directory" in message


@pytest.mark.parametrize("result", [None, {}, {"totals": []},
    {"totals": [{"wins": -1, "losses": 0, "draws": 0, "truncated": 0}]}])
def test_malformed_runner_results_fail_closed(tmp_path, result):
    gate, net = make_gate(tmp_path, runner=lambda *args: result)
    gate.bootstrap_or_load(net, 0)
    decision = gate.evaluate(net, 2, trained=True)
    assert decision.reason == "match_error"
    assert decision.decision == "rejected"
    assert decision.accepted_after == 0
    report = read_json(Path(decision.match_log))
    assert report["match"] == result
    assert report["error"]


def test_default_runner_exports_copies_and_forwards_gate_settings(tmp_path, monkeypatch):
    cfg = config(gate_workers=3, gate_simulations=7, gate_max_boards=9,
                 gate_temperature=0.2)
    monkeypatch.setattr(gating, "_require_binding", lambda: None)
    monkeypatch.setattr(gating, "_5dchess", SimpleNamespace(run_match=lambda: None))
    exports, matches = [], []

    def export(network, spec, path):
        torch.rand(5)
        assert next(network.parameters()).device.type == "cpu"
        assert not network.training
        exports.append((network, spec, Path(path)))
        return Path(path)

    def match(a, b, **kwargs):
        matches.append((a, b, kwargs))
        return raw_result(wins=0, losses=4)

    monkeypatch.setattr(gating, "export_network", export)
    monkeypatch.setattr(gating, "play_model_match", match)
    net = build_network(cfg.network, VERY_SMALL)
    gate = ModelGate(cfg, paths(tmp_path), VERY_SMALL, torch.device("cpu"))
    gate.bootstrap_or_load(net, 0)
    accepted = gate.resolve(net)
    rng = torch.get_rng_state().clone()
    result = gate.evaluate(net, 2, trained=True)
    assert result.decision == "rejected"
    assert torch.equal(torch.get_rng_state(), rng)
    assert net.training
    assert len(exports) == 2 and len(matches) == 1
    assert exports[0][0] is not net and exports[1][0] is not accepted
    a, b, kwargs = matches[0]
    assert a.name == "candidate_iter_2" and b.name == "accepted_iter_0"
    assert a.model == exports[0][2] and b.model == exports[1][2]
    for player in (a, b):
        assert player.simulations == 7
        assert player.temperature == 0.2
        assert player.provider == "cpu"
    assert kwargs == {
        "variant": VERY_SMALL, "games": 4, "workers": 3, "max_moves": 0,
        "board_limit": 9, "material_scale": cfg.self_play.material_scale,
        "seed": cfg.seed + 2 * 1009,
    }
