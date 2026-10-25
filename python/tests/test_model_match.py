"""Model-match integration uses separate encodings, weights and architectures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
import threading

import pytest
import torch

from suzero_5dchess.match import MatchPlayer, material_average_score, play_model_match
from suzero_5dchess.protocol import VERY_SMALL


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    binding = pytest.importorskip("suzero_5dchess._5dchess")
    if not hasattr(binding, "run_match"):
        pytest.skip("rebuild the C++ model-match binding")
    pytest.importorskip("onnx")
    from suzero_5dchess.config import NetworkConfig
    from suzero_5dchess.export_onnx import export_network
    from suzero_5dchess.network import build_network

    directory = tmp_path_factory.mktemp("match-models")
    paths = []
    with torch.random.fork_rng():
        for index, width in enumerate((16, 32)):
            torch.manual_seed(72 + index)
            config = NetworkConfig(d_model=width, n_heads=4, n_layers=1,
                                   d_ff=width * 2, action_mlp_hidden=width,
                                   dropout=0.0)
            network = build_network(config, VERY_SMALL)
            spec = replace(VERY_SMALL, line_shift=5 + index * 3)
            paths.append(export_network(network, spec, directory / f"{index}.onnx"))
            torch.save({"model": network.state_dict(), "extra_state": {
                "variant": VERY_SMALL.name, "network_config": asdict(config),
            }}, directory / f"{index}.pt")
    return paths[0], paths[1], directory / "1.pt"


def players(models):
    return (MatchPlayer(models[0], "small", simulations=2, max_depth=8),
            MatchPlayer(models[1], "large", simulations=4, max_depth=12,
                        reverse_move_order=True))


@pytest.mark.parametrize("workers", [1, 3, 20])
def test_real_heterogeneous_models_and_paired_statistics(models, workers):
    progress = []
    caller = threading.get_ident()
    def record(done, game):
        assert threading.get_ident() == caller
        progress.append((done, game))
    result = play_model_match(*players(models), games=4, max_moves=0, board_limit=2,
                              workers=workers, progress=record)
    assert result["participants"] == ("small", "large")
    assert [done for done, _ in progress] == [1, 2, 3, 4]
    assert len(result["games"]) == 4
    assert sorted(game["index"] for _, game in progress) == [0, 1, 2, 3]
    completed_games = {game["index"]: game for _, game in progress}
    for index, game in enumerate(result["games"]):
        assert game["a_is_white"] == (index % 2 == 0)
        assert game["terminal"] and not game["truncated"]
        assert game["terminal_reason"] == "material"
        assert len(game["moves"]) == 1
        assert "SUBMIT" not in game["moves"][0]["action"]
        assert game["moves"][0]["player"] == 0
        assert game["moves"][0]["participant_index"] == index % 2
        assert game["participant_utilities"][0] == -game["participant_utilities"][1]
        assert completed_games[index] == game
    for total, colors in zip(result["totals"], result["by_color"]):
        assert total["truncated"] == 0
        assert total["wins"] + total["losses"] + total["draws"] == 4
        for color in ("white", "black"):
            assert sum(colors[color].values()) == 2
    assert material_average_score(result) == pytest.approx(
        -material_average_score(result, 1)
    )


def test_parallel_cuda_matches(models):
    ort = pytest.importorskip("onnxruntime")
    if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDA device and ONNX Runtime CUDA provider required")
    a, b = (replace(player, provider="cuda") for player in players(models))
    result = play_model_match(a, b, games=4, workers=4, max_moves=8, board_limit=1)
    assert len(result["games"]) == 4
    assert all(game["terminal"] and not game["truncated"] for game in result["games"])
    for total, colors in zip(result["totals"], result["by_color"]):
        assert sum(total.values()) == 4
        assert sum(colors["white"].values()) == sum(colors["black"].values()) == 2


@pytest.mark.parametrize("workers", [1, 3])
def test_truncation_not_a_draw_and_repeatable(models, workers):
    kwargs = dict(games=2, max_moves=1, seed=998, workers=workers)
    result = play_model_match(*players(models), **kwargs)
    assert result == play_model_match(*players(models), **kwargs)
    for total in result["totals"]:
        assert total == {"wins": 0, "losses": 0, "draws": 0, "truncated": 2}
    for game in result["games"]:
        assert game["truncated"] and not game["terminal"]
        assert game["final_utility"] is None
        assert game["participant_utilities"] == [None, None]
        assert len(game["moves"]) == 1
    assert material_average_score(result) is None


def test_material_average_score_uses_only_material_terminals():
    result = {"games": [
        {"terminal": True, "terminal_reason": "material",
         "participant_utilities": [0.3, -0.3]},
        {"terminal": True, "terminal_reason": "capture_king",
         "participant_utilities": [1.0, -1.0]},
        {"terminal": False, "terminal_reason": "",
         "participant_utilities": [None, None]},
        {"terminal": True, "terminal_reason": "material",
         "participant_utilities": [-0.1, 0.1]},
    ]}
    assert material_average_score(result) == pytest.approx(0.1)
    assert material_average_score(result, 1) == pytest.approx(-0.1)


def test_checkpoint_conversion_does_not_modify_checkpoint(models):
    before = hashlib.sha256(models[2].read_bytes()).digest()
    a, b = players(models)
    result = play_model_match(a, replace(b, model=models[2]), games=2, max_moves=1)
    assert result["totals"][0]["truncated"] == 2
    assert hashlib.sha256(models[2].read_bytes()).digest() == before
    assert not Path(str(models[2]) + ".onnx").exists()


@pytest.mark.parametrize("kwargs", [
    {"games": 0}, {"games": 3}, {"games": 2**31}, {"max_moves": -1},
    {"workers": 0}, {"workers": -1}, {"workers": 2**31},
    {"seed": -1}, {"seed": 2**64}, {"board_limit": 0},
    {"material_scale": float("nan")}, {"material_scale": -1.0},
])
def test_invalid_series_rejected_before_model_loading(kwargs):
    with pytest.raises(ValueError):
        play_model_match(MatchPlayer("missing.onnx"), MatchPlayer("missing.onnx"), **kwargs)


@pytest.mark.parametrize("changes", [
    {"simulations": 0}, {"max_depth": 0}, {"c_puct": float("nan")},
    {"temperature": -0.5}, {"provider": "gpu"}, {"intra_op_threads": 0},
])
def test_invalid_bot_rejected(models, changes):
    a, b = players(models)
    with pytest.raises(ValueError):
        play_model_match(replace(a, **changes), b)


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_progress_exception_propagates(models, workers, error):
    seen = []
    def stop(done, game):
        seen.append(done)
        raise error("stop match")
    with pytest.raises(error, match="stop match"):
        play_model_match(*players(models), games=4, max_moves=1,
                         workers=workers, progress=stop)
    assert seen == [1]


def test_cli_independent_options_and_output(models, tmp_path, capsys):
    from suzero_5dchess.cli import build_parser

    output = tmp_path / "match.json"
    args = build_parser().parse_args([
        "match", "--model-a", str(models[0]), "--model-b", str(models[1]),
        "--sims-a", "2", "--sims-b", "3", "--max-moves", "1", "--workers", "2",
        "--c-puct-b", "1.7", "--reverse-moves-b", "--output", str(output),
    ])
    assert args.sims_a == 2 and args.sims_b == 3
    assert args.c_puct_b == 1.7 and args.reverse_moves_b
    assert args.func(args) == 0
    full = json.loads(output.read_text(encoding="utf-8"))
    assert len(full["games"]) == 2
    assert full["totals"][0]["truncated"] == 2
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert "games" not in summary and summary["games_played"] == 2
    assert summary["material_average_scores"] == [None, None]
    assert "match 2/2" in captured.err
    assert "material average score:" not in captured.err
    with pytest.raises(SystemExit, match="already exists"):
        args.func(args)
