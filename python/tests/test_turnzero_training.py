"""Standard - Turn Zero rules, CLI defaults and end-to-end training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from suzero_5dchess.cli import _build_config, _resolve_resume, build_parser
from suzero_5dchess.config import NetworkConfig, TrainConfig
from suzero_5dchess.network import build_network
from suzero_5dchess.pipeline import RunPaths, run_training
from suzero_5dchess.protocol import (
    ONNX_INPUT_NAMES,
    STANDARD,
    STANDARD_TURN_ZERO,
    VERY_SMALL,
    resolve_variant,
)


def test_turnzero_variant_and_parameter_budget() -> None:
    spec = resolve_variant("standard_turn_zero")
    assert spec is STANDARD_TURN_ZERO
    assert spec.pgn == '[Board "Standard - Turn Zero"]\n[Mode "5D"]\n'
    assert (spec.board_side, spec.board_squares, spec.piece_channels) == (8, 64, 12)
    assert spec.pgn != STANDARD.pgn
    assert TrainConfig(variant=spec.name).variant == spec.name

    parser = build_parser()
    config = _build_config(parser.parse_args(["train", "--variant", spec.name]))
    network = build_network(config.network, config.variant)
    assert config.network.max_params == 300_000
    assert network.config.d_model == 128
    assert network.parameter_count == 270_202

    small = _build_config(parser.parse_args(["train"]))
    assert small.variant == VERY_SMALL.name
    assert small.network.max_params == 200_000
    assert build_network(small.network).parameter_count == 192_058

    explicit = _build_config(parser.parse_args([
        "train", "--variant", spec.name, "--max-params", "200000",
    ]))
    with pytest.raises(ValueError, match="exceeding"):
        build_network(explicit.network, explicit.variant)


def test_turnzero_default_simulations() -> None:
    parser = build_parser()
    turnzero = _build_config(parser.parse_args(["train", "--variant", STANDARD_TURN_ZERO.name]))
    assert turnzero.mcts.num_simulations == 1000

    explicit = _build_config(parser.parse_args([
        "train", "--variant", STANDARD_TURN_ZERO.name, "--sims", "400",
    ]))
    assert explicit.mcts.num_simulations == 400

    # The debug preset stays tiny; only an explicit --sims overrides it.
    debug = _build_config(parser.parse_args([
        "train", "--variant", STANDARD_TURN_ZERO.name, "--debug",
    ]))
    assert debug.mcts.num_simulations == 2

    # Other variants keep the shared default.
    assert _build_config(parser.parse_args(["train"])).mcts.num_simulations == 800


def test_turnzero_cli_subcommands_and_paths() -> None:
    parser = build_parser()
    for command, flags in (
        ("export", ["--checkpoint", "latest.pt", "--output", "model.onnx"]),
        ("evaluate", ["--checkpoint-a", "a.pt", "--checkpoint-b", "b.pt"]),
        ("match", ["--model-a", "a.onnx", "--model-b", "b.onnx"]),
    ):
        args = parser.parse_args([command, "--variant", STANDARD_TURN_ZERO.name, *flags])
        assert args.variant == STANDARD_TURN_ZERO.name

    config = _build_config(parser.parse_args([
        "train", "--variant", STANDARD_TURN_ZERO.name,
    ]))
    paths = RunPaths.from_config(config)
    assert paths.latest == Path("runs/standard_turn_zero/cls_action_mlp/latest.pt")
    assert paths.replay == Path("runs/standard_turn_zero/cls_action_mlp/replay_buffer.pt")
    assert paths.model_path.name == "standard_turn_zero_latest_fp32.onnx"
    assert config.model_match_reference is None

    debug = _build_config(parser.parse_args([
        "train", "--variant", STANDARD_TURN_ZERO.name, "--debug",
    ]))
    assert RunPaths.from_config(debug).latest == Path("runs/debug/standard_turn_zero/latest.pt")
    assert debug.network.max_params == 300_000
    small_debug = _build_config(parser.parse_args(["train", "--debug"]))
    assert RunPaths.from_config(small_debug).latest == Path("runs/debug/latest.pt")

    custom = _build_config(parser.parse_args([
        "train", "--variant", STANDARD_TURN_ZERO.name, "--debug",
        "--checkpoint-dir", "custom-run",
    ]))
    assert RunPaths.from_config(custom).latest == Path("custom-run/latest.pt")


def test_turnzero_native_initial_history() -> None:
    pytest.importorskip("suzero_5dchess._5dchess")
    from suzero_5dchess.env import GameEnv

    env = GameEnv(STANDARD_TURN_ZERO)
    observation = env.encode()
    assert env.current_player == 0
    assert env.action_count == 20
    assert not env.has_submit
    assert env.board_count == 2
    assert observation.board_planes.shape == (2, 12, 64)
    assert observation.t_coords.tolist() == [0, 1]
    assert observation.l_coords.tolist() == [5, 5]
    snapshot = env.record_snapshot()
    assert [(board["t"], board["c"]) for board in snapshot["boards"]] == [
        (0, True), (1, False),
    ]
    assert GameEnv(STANDARD).board_count == 1
    clone = env.clone()
    assert clone.state_key == env.state_key
    env.apply_action(0)
    assert clone.board_count == 2
    assert env.forced_submit
    env.apply_action(0)
    assert env.current_player == 1
    assert 0 in env.encode().t_coords


def test_turnzero_dynamic_onnx_export(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from suzero_5dchess.export_onnx import export_network

    spec = STANDARD_TURN_ZERO
    network = build_network(NetworkConfig(max_params=300_000), spec).eval()
    path = export_network(network, spec, tmp_path / "turnzero.onnx")
    metadata = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    assert metadata["variant"] == spec.name
    assert metadata["variant_pgn"] == spec.pgn
    assert metadata["board_side"] == 8
    assert metadata["board_squares"] == 64
    assert metadata["parameter_count"] == 270_202

    rng = np.random.default_rng(7)
    batch, boards, candidates = 3, 5, 6
    inputs = (
        rng.integers(0, 2, (batch, boards, 12, 64)).astype(np.float32),
        rng.integers(-2, 8, (batch, boards)).astype(np.int64),
        rng.integers(0, 8, (batch, boards)).astype(np.int64),
        np.array([5, 2, 4], dtype=np.int64),
        rng.random(batch).astype(np.float32),
        np.array([0, 1, 0], dtype=np.int64),
        rng.integers(0, 8, (batch, candidates, 8)).astype(np.int64),
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, dict(zip(ONNX_INPUT_NAMES, inputs)))
    with torch.no_grad():
        expected = network(*[torch.from_numpy(value) for value in inputs])
    for reference, exported in zip(expected, actual):
        np.testing.assert_allclose(reference.numpy(), exported, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("backend,processes", [("python", 1), ("cpp_onnx", 1), ("cpp_onnx", 2)])
def test_turnzero_training_records_export_and_resume(
    tmp_path: Path, backend: str, processes: int,
) -> None:
    pytest.importorskip("suzero_5dchess._5dchess")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from suzero_5dchess.game_record import load_record, replay_record
    from suzero_5dchess.sample import validate_sample

    args = build_parser().parse_args([
        "train", "--variant", STANDARD_TURN_ZERO.name, "--debug", "--device", "cpu",
        "--checkpoint-dir", str(tmp_path / "run"),
        "--selfplay-backend", backend, "--cpp-onnx-provider", "cpu",
        "--selfplay-processes", str(processes), "--num-server-threads", "2",
        "--workers", "2", "--games", "2", "--sims", "2",
        "--min-board-limit", "4", "--max-board-limit", "4", "--batch-size", "1",
        "--min-replay-size", "1", "--steps-per-epoch", "1", "--record-games",
    ])
    config = _build_config(args)
    summary = run_training(config)
    assert summary["end_iteration"] == 1
    assert summary["total_games"] == 2
    training = summary["iterations"][0]["training"]
    assert training["skipped"] is False
    assert training["steps"] == 1
    assert np.isfinite(training["metrics"]["loss"])

    paths = RunPaths.from_config(config)
    checkpoint = torch.load(paths.latest, map_location="cpu", weights_only=False)
    assert checkpoint["extra_state"]["variant"] == STANDARD_TURN_ZERO.name
    assert "replay" not in checkpoint["extra_state"]
    assert checkpoint["train_state"]["trainer_global_step"] == 1
    from suzero_5dchess.pipeline import _load_replay_buffer
    from suzero_5dchess.replay import ReplayBuffer

    replay = ReplayBuffer(config.replay_buffer_size)
    _load_replay_buffer(replay, paths.replay)
    for sample in replay.sample(len(replay)):
        validate_sample(sample, STANDARD_TURN_ZERO)
        assert sample.board_planes.shape[1:] == (12, 64)

    metadata = json.loads(Path(str(paths.model_path) + ".json").read_text(encoding="utf-8"))
    assert metadata["variant"] == STANDARD_TURN_ZERO.name
    assert metadata["board_squares"] == 64
    archives = list((paths.log_dir / "games").rglob("*.gn5d"))
    assert len(archives) == 2
    for archive in archives:
        record = load_record(archive)
        assert record["variant"] == STANDARD_TURN_ZERO.name
        assert isinstance(record["initial"], str)
        initial_lines = record["initial"].strip().splitlines()
        assert len(initial_lines) == 2
        assert ":0:0:" in initial_lines[0]
        assert record["result"]["terminal_reason"] == "material"
        assert record["actions"][-1] != "submit"
        final = replay_record(record)
        assert final.is_terminal
        assert len(final.record_snapshot()["boards"]) >= 2

    assert _resolve_resume(config, "latest") == paths.latest
    resumed = run_training(config, resume=paths.latest)
    assert resumed["start_iteration"] == 1
    assert resumed["end_iteration"] == 2
    assert resumed["total_games"] == 4
    assert resumed["iterations"][0]["training"]["steps"] == 1
    assert paths.iteration_checkpoint(2).is_file()
    assert len(list((paths.log_dir / "games").rglob("*.gn5d"))) == 4
    saved = torch.load(paths.latest, map_location="cpu", weights_only=False)
    assert saved["train_state"]["trainer_global_step"] == 2

    for other_variant in (VERY_SMALL.name, STANDARD.name):
        incompatible = TrainConfig(variant=other_variant)
        with pytest.raises(SystemExit, match="checkpoint variant"):
            _resolve_resume(incompatible, str(paths.latest))
