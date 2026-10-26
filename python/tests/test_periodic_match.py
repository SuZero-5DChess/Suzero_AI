"""Periodic opponent selection, resume, legacy inference and ONNX coverage."""

from dataclasses import replace
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from suzero_5dchess.config import (
    NetworkConfig, SelfPlayConfig, TrainConfig, debug_train_config,
)
from suzero_5dchess.network import build_network
from suzero_5dchess.periodic_match import PeriodicModelMatches
from suzero_5dchess.protocol import VERY_SMALL


@pytest.mark.parametrize("changes", [
    {"model_match_interval": -1}, {"model_match_lookback": 0},
    {"model_match_games": 3}, {"model_match_simulations": 0},
    {"model_match_workers": 0}, {"model_match_workers": -1},
    {"model_match_workers": 2**31},
    {"model_match_max_boards": -1}, {"model_match_provider": "gpu"},
])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        TrainConfig(**changes)


def test_color_win_rates_exclude_draws_and_truncations():
    from suzero_5dchess.periodic_match import _win_rates_by_color

    rates = _win_rates_by_color([
        {
            "white": {"wins": 3, "losses": 1, "draws": 0, "truncated": 2},
            "black": {"wins": 1, "losses": 1, "draws": 2, "truncated": 1},
        },
        {
            "white": {"wins": 0, "losses": 2, "draws": 0, "truncated": 2},
            "black": {"wins": 1, "losses": 0, "draws": 1, "truncated": 2},
        },
    ])
    assert rates == {
        "current": {"white": 0.75, "black": 0.25},
        "opponent": {"white": 0.0, "black": 0.5},
    }


def test_launcher_does_not_configure_periodic_matches():
    text = (Path(__file__).resolve().parents[2] / "run_train.bat").read_text(encoding="utf-8")
    for knob in ("EVALUATION_INTERVAL", "MATCH_LOOKBACK", "MATCH_GAMES", "MATCH_WORKERS",
                 "MATCH_SIMS", "MATCH_MAX_BOARDS", "MATCH_PROVIDER", "MATCH_REFERENCE"):
        assert f"set \"{knob}=" not in text
    assert "--model-match-interval %" not in text
    assert 'if defined DEBUG_MODE set "DEFAULT_ARGS=--record-games %CONTINUOUS_ARG%"' in text


def test_disabled_matches_do_not_load_missing_reference(tmp_path):
    cfg = TrainConfig(model_match_reference=str(tmp_path / "missing.pt"))
    matches = PeriodicModelMatches(cfg, tmp_path, tmp_path)
    net = tiny_network()
    matches.snapshot(net, 0)
    assert matches.run(net, 1) is None
    assert not matches.directory.exists()


def test_cli_options():
    from suzero_5dchess.cli import build_parser, _build_config
    cfg = _build_config(build_parser().parse_args([
        "train", "--model-match-interval", "7", "--model-match-lookback", "13",
        "--model-match-games", "4", "--model-match-workers", "6", "--model-match-sims", "3",
        "--model-match-reference", "old.pt", "--model-match-max-boards", "9",
        "--model-match-provider", "cuda",
    ]))
    assert (cfg.model_match_interval, cfg.model_match_lookback) == (7, 13)
    assert (cfg.model_match_games, cfg.model_match_simulations) == (4, 3)
    assert cfg.model_match_workers == 6
    assert TrainConfig().model_match_workers == 20
    assert TrainConfig().model_match_max_boards == 25
    assert cfg.model_match_reference == "old.pt"
    assert cfg.model_match_provider == "cuda"
    assert cfg.model_match_max_boards == 9
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg


def tiny_network():
    return build_network(NetworkConfig(d_model=16, n_heads=4, d_ff=32,
                                      action_mlp_hidden=16, dropout=0.0))


def require_match():
    binding = pytest.importorskip("suzero_5dchess._5dchess")
    if not hasattr(binding, "run_match"):
        pytest.skip("rebuild the model match binding")
    pytest.importorskip("onnx")


def test_exact_history_and_fixed_real_matches(tmp_path, caplog):
    require_match()
    caplog.set_level(logging.INFO, logger="suzero_5dchess.periodic_match")
    network = tiny_network()
    fixed = tmp_path / "fixed.pt"
    cfg = TrainConfig(model_match_interval=2, model_match_lookback=3,
                      model_match_games=2, model_match_simulations=2,
                      model_match_max_boards=2, model_match_reference=str(fixed),
                      self_play=SelfPlayConfig(min_board_limit=2,
                                               max_board_limit=2))
    from dataclasses import asdict
    torch.save({"model": network.state_dict(), "extra_state": {
        "variant": "very_small", "network_config": asdict(network.config),
    }}, fixed)
    matches = PeriodicModelMatches(cfg, tmp_path, tmp_path / "logs")
    matches.snapshot(network, 0)
    assert matches.run(network, 1) is None
    result = matches.run(network, 2)
    assert result["previous"]["skipped"]
    total = result["fixed"]["totals"][0]
    assert total["truncated"] == 0
    assert total["wins"] + total["losses"] + total["draws"] == 2
    assert result["fixed"]["win_rate"] is not None
    assert result["fixed"]["material_games"] == 2
    assert result["fixed"]["material_average_score"] is not None
    assert "0 truncated games" in caplog.text
    assert "win rate over normally finished games" in caplog.text
    assert "score rate" not in caplog.text
    assert "average score over 2 material-ended games" in caplog.text
    assert "current model perspective; positive means ahead" in caplog.text
    matches.snapshot(network, 1)
    matches.snapshot(network, 3)
    before = torch.get_rng_state().clone()
    result = matches.run(network, 4)
    assert torch.equal(before, torch.get_rng_state())
    assert result["previous"]["opponent_iteration"] == 1
    assert result["previous"]["material_games"] == 2
    assert result["previous"]["material_average_score"] is not None
    from suzero_5dchess.game_record import load_record

    for key in ("previous", "fixed"):
        full = json.loads(Path(result[key]["output"]).read_text(encoding="utf-8"))
        assert len(full["match"]["games"]) == 2
        assert full["summary"]["board_limit"] == 2
        assert full["summary"]["workers"] == 2
        assert result[key]["workers"] == 2
        assert len(result[key]["gn5d_records"]) == 2
        assert full["summary"]["gn5d_records"] == result[key]["gn5d_records"]
        for game_id, record_path in enumerate(result[key]["gn5d_records"], start=1):
            record = load_record(record_path)
            assert record["meta"]["game_id"] == game_id
            assert record["meta"]["backend"] == f"model_match_{key}"
            assert record["result"]["terminal_reason"] == "material"
    # A missing exact target must not silently use the available iteration 3.
    result = matches.run(network, 8)
    assert result["previous"]["skipped"]
    assert result["fixed"]["games"] == 2


def test_pipeline_history_survives_resume_and_sparse_saves(tmp_path):
    require_match()
    from suzero_5dchess.pipeline import RunPaths, run_training
    cfg = replace(debug_train_config(), checkpoint_dir=str(tmp_path),
                  num_iterations=2, save_interval=10, export_interval=0,
                  model_match_interval=2, model_match_lookback=3,
                  model_match_games=2, model_match_simulations=1,
                  model_match_max_boards=2)
    cfg.network = tiny_network().config
    cfg.self_play.num_games = 2
    cfg.self_play.min_board_limit = 2
    cfg.self_play.max_board_limit = 2
    cfg.mcts.num_simulations = 1
    first = run_training(cfg)
    assert first["iterations"][1]["model_matches"]["previous"]["skipped"]
    assert not RunPaths.from_config(cfg).iteration_checkpoint(1).exists()
    resumed = run_training(cfg, resume=RunPaths.from_config(cfg).latest)
    history = resumed["iterations"][1]["model_matches"]["previous"]
    assert history["opponent_iteration"] == 1
    assert history["totals"][0]["truncated"] == 0
    assert history["material_games"] == 2


def legacy_state():
    """A small reference-format LSTM state, including the original key layout."""
    from suzero_5dchess.legacy_model import LegacyLstmNetwork
    cfg = tiny_network().config
    half = cfg.d_model // 2
    pos = torch.arange(-50, 50).float()[:, None]
    freq = torch.exp(torch.arange(0, half, 2).float() * (-np.log(10000.0) / half))
    table = torch.stack((torch.sin(pos * freq), torch.cos(pos * freq)), -1).flatten(-2)
    network = LegacyLstmNetwork(cfg, VERY_SMALL, table)
    network.pos_encoder.pe.copy_(table)
    weights = {}
    for key, value in network.state_dict().items():
        if key.startswith("blocks."):
            key = key.replace("blocks.", "transformer.layers.", 1)
            key = key.replace("qkv.weight", "self_attn.in_proj_weight")
            key = key.replace("qkv.bias", "self_attn.in_proj_bias")
            if ".out_proj." in key:
                key = key.replace(".out_proj.", ".self_attn.out_proj.")
        weights[key] = value
    from dataclasses import asdict
    return {"action_encoding": "smallstep_v1", "model_state_dict": weights,
            "config": {"variant_name": "very_small", "network": {
                **asdict(cfg), "architecture": "lstm", "board_side": 4,
                "board_squares": 16, "piece_channels": 12,
            }}}


def test_parallel_matches_against_legacy_checkpoint(tmp_path):
    require_match()
    from suzero_5dchess.export_onnx import export_network
    from suzero_5dchess.match import MatchPlayer, play_model_match

    current = export_network(tiny_network(), VERY_SMALL, tmp_path / "current.onnx")
    legacy = tmp_path / "legacy.pt"
    torch.save(legacy_state(), legacy)
    result = play_model_match(
        MatchPlayer(current, simulations=2), MatchPlayer(legacy, simulations=2),
        games=4, workers=3, max_moves=8, board_limit=1,
    )
    assert all(game["terminal"] for game in result["games"])
    assert [game["a_is_white"] for game in result["games"]] == [True, False] * 2
    for total, colors in zip(result["totals"], result["by_color"]):
        assert total["truncated"] == 0
        assert sum(total.values()) == 4
        assert sum(colors["white"].values()) == sum(colors["black"].values()) == 2


def test_legacy_markers_and_dynamic_onnx(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from suzero_5dchess.legacy_model import load_legacy_checkpoint
    from suzero_5dchess.export_onnx import export_network
    from suzero_5dchess.protocol import ONNX_INPUT_NAMES
    state = legacy_state()
    network = load_legacy_checkpoint(state, VERY_SMALL)
    actions = torch.tensor([[[1, 0, 2, -1, 3, 1, 3, 0],
                             [1, 0, 2, -1, 0, 0, 9, 7]]])
    ls, ts = torch.tensor([[4, 5, 8]]), torch.tensor([[2, 3, 5]])
    markers = network.action_markers(actions, ls, ts)
    expected = torch.zeros(1, 2, 3, 16)
    expected[0, :, 0, 1] = -1
    expected[0, 0, 1, 7] = 1
    assert torch.equal(markers, expected)
    path = export_network(network, VERY_SMALL, tmp_path / "legacy.onnx")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch, boards, q in ((1, 1, 1), (3, 5, 6)):
        inputs = (
            torch.randn(batch, boards, 12, 16),
            torch.arange(boards).expand(batch, boards) + 4,
            torch.arange(boards).expand(batch, boards) + 2,
            torch.tensor([boards, 1, boards - 1][:batch]),
            torch.ones(batch), torch.arange(batch) % 2,
            actions[:, :1].expand(batch, q, 8),
        )
        with torch.no_grad():
            expected = network(*inputs)
        actual = session.run(None, {name: value.numpy() for name, value in zip(ONNX_INPUT_NAMES, inputs)})
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a.numpy(), b, atol=1e-4, rtol=1e-4)
    state["action_encoding"] = "halfstep"
    with pytest.raises(ValueError, match="smallstep_v1"):
        load_legacy_checkpoint(state, VERY_SMALL)
