"""Tests for the 5D chess training package."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from suzero_5dchess.config import (
    MctsConfig,
    NetworkConfig,
    SelfPlayConfig,
    TrainConfig,
    debug_train_config,
)
from suzero_5dchess.pipeline import RunPaths, run_training
from suzero_5dchess.protocol import VERY_SMALL

pytest.importorskip("suzero_5dchess._5dchess")


def test_replay_save_is_compressed_and_round_trips(tmp_path: Path) -> None:
    from dataclasses import fields

    from suzero_5dchess.pipeline import _load_replay_buffer, _save_replay_buffer
    from suzero_5dchess.replay import ReplayBuffer

    sample = _synthetic_sample(q=2, has_submit=True, policy=[0.2, 0.3, 0.5], player=1)
    sample.board_codes = np.zeros((80, 64), dtype=np.uint8)
    sample.board_codes[0, 0] = 1
    sample.board_codes[79, 63] = 12
    sample.l_coords = np.arange(80, dtype=np.int64)
    sample.t_coords = np.arange(80, dtype=np.int64)
    sample.value_target = -0.75
    sample.move_count = 42
    sample.metadata = {"weight": 0.5}
    buffer = ReplayBuffer(5)
    buffer.push([sample])

    state = buffer.state_dict()
    assert state["format"] == "gn5d_replay_zlib_v1"
    assert state["count"] == 1
    assert state["blobs"].dtype == np.uint8

    path = tmp_path / "replay_buffer.pt"
    _save_replay_buffer(buffer, path)
    old_plane_bytes = sample.board_codes.shape[0] * 12 * sample.board_codes.shape[1] * 4
    assert path.stat().st_size < old_plane_bytes // 4

    restored = ReplayBuffer(5)
    _load_replay_buffer(restored, path)
    actual = restored.sample(1)[0]
    for field in fields(sample):
        expected_value = getattr(sample, field.name)
        actual_value = getattr(actual, field.name)
        if isinstance(expected_value, np.ndarray):
            assert isinstance(actual_value, np.ndarray)
            if field.name == "policy":
                np.testing.assert_allclose(actual_value, expected_value, atol=1e-3)
            else:
                np.testing.assert_array_equal(actual_value, expected_value)
        elif field.name == "metadata":
            assert dict(actual_value) == expected_value
        else:
            assert actual_value == expected_value
    assert not Path(str(path) + ".tmp").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_replay_load_preserves_fifo_capacity(tmp_path: Path, legacy: bool) -> None:
    from suzero_5dchess.pipeline import _load_replay_buffer
    from suzero_5dchess.replay import ReplayBuffer

    samples = [
        _synthetic_sample(q=1, has_submit=False, policy=[1.0], player=0)
        for _ in range(3)
    ]
    for index, sample in enumerate(samples):
        sample.move_count = index
    buffer = ReplayBuffer(3)
    buffer.push(samples)
    state = {"capacity": 3, "samples": samples} if legacy else buffer.state_dict()
    path = tmp_path / "replay_buffer.pt"
    torch.save(state, path)
    restored = ReplayBuffer(2)
    _load_replay_buffer(restored, path)
    assert restored.capacity == 2
    assert len(restored) == 2
    assert sorted(sample.move_count for sample in restored.sample(2)) == [1, 2]
    assert all(isinstance(sample.policy, np.ndarray) for sample in restored.sample(2))


def test_replay_save_failure_preserves_previous_file(tmp_path: Path, monkeypatch) -> None:
    from suzero_5dchess.pipeline import _save_replay_buffer
    from suzero_5dchess.replay import ReplayBuffer

    path = tmp_path / "replay_buffer.pt"
    path.write_bytes(b"previous replay")

    def fail_save(state, target):
        Path(target).write_bytes(b"partial replay")
        raise MemoryError("simulated serialization failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(MemoryError):
        _save_replay_buffer(ReplayBuffer(2), path)
    assert path.read_bytes() == b"previous replay"


def test_parameter_budget_keeps_d_model() -> None:
    from suzero_5dchess.network import build_network

    network = build_network(NetworkConfig(), VERY_SMALL)
    assert network.config.d_model == 128
    assert network.parameter_count == 192_058
    assert network.parameter_count <= 200_000


def test_dynamic_onnx_export(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from suzero_5dchess.export_onnx import export_network
    from suzero_5dchess.network import build_network

    network = build_network(NetworkConfig(), VERY_SMALL)
    path = export_network(network, VERY_SMALL, tmp_path / "model.onnx", verify=True)
    sidecar = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    assert sidecar["io_protocol"] == "smallstep_movekey_v1"
    assert sidecar["board_side"] == 4
    assert sidecar["board_squares"] == 16
    assert sidecar["piece_channels"] == 12

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    batch, boards, q = 3, 5, 6
    names = [
        "board_planes",
        "l_coords",
        "t_coords",
        "used_board_counts",
        "urgency",
        "side_to_move",
        "actions",
    ]
    inputs = (
        rng.integers(0, 2, (batch, boards, 12, 16)).astype(np.float32),
        rng.integers(-2, 8, (batch, boards)).astype(np.int64),
        rng.integers(0, 8, (batch, boards)).astype(np.int64),
        np.array([5, 3, 4], dtype=np.int64),
        rng.random(batch).astype(np.float32),
        np.array([0, 1, 0], dtype=np.int64),
        rng.integers(0, 4, (batch, q, 8)).astype(np.int64),
    )
    actual = session.run(None, dict(zip(names, inputs)))
    network.eval()
    with torch.no_grad():
        expected = tuple(
            tensor.numpy()
            for tensor in network(*[torch.from_numpy(value) for value in inputs])
        )
    for expected_value, actual_value in zip(expected, actual):
        assert np.allclose(expected_value, actual_value, atol=1e-4, rtol=1e-4)


def _synthetic_sample(
    *, q: int, has_submit: bool, policy: list[float], player: int
) -> "PositionSample":
    from suzero_5dchess.sample import PositionSample

    actions = np.zeros((q, 8), dtype=np.int64)
    for row in range(q):
        actions[row] = (row % 4, row % 4, 1, 0, (row + 1) % 4, (row + 1) % 4, 1, 0)
    return PositionSample(
        board_codes=np.zeros((1, 16), dtype=np.uint8),
        l_coords=np.array([5], dtype=np.int64),
        t_coords=np.array([1], dtype=np.int64),
        urgency=24.0,
        side_to_move=player,
        actions=actions,
        policy=np.asarray(policy, dtype=np.float32),
        has_submit=has_submit,
        player=player,
    )


def test_collate_puts_submit_in_last_column() -> None:
    from suzero_5dchess.sample import collate_samples

    with_submit = _synthetic_sample(
        q=2, has_submit=True, policy=[0.2, 0.3, 0.5], player=0
    )
    without_submit = _synthetic_sample(q=1, has_submit=False, policy=[1.0], player=1)
    batch = collate_samples([with_submit, without_submit], VERY_SMALL)

    assert batch["actions"].shape == (2, 2, 8)
    assert batch["policy"].shape == (2, 3)
    assert batch["policy_mask"][0].tolist() == [True, True, True]
    assert batch["policy_mask"][1].tolist() == [True, False, False]
    assert batch["policy"][0, 2] == pytest.approx(0.5)
    assert batch["policy"][1, 1] == pytest.approx(0.0)
    assert batch["used_board_counts"].tolist() == [1, 1]


def test_select_action_flips_value_across_player_change() -> None:
    from suzero_5dchess.mcts import MCTS, Node

    parent = Node(
        state=None,  # type: ignore[arg-type]
        player=0,
        priors=np.array([0.5, 0.5], dtype=np.float32),
        expanded=True,
    )
    # Child 0 keeps the player and has the lower value from the parent's
    # perspective.  Child 1 submits (player 1), so its +0.1 must be read as
    # -0.1 by the parent and must lose.
    parent.children[0] = Node(
        state=None, player=0, visits=10, value_sum=0.5  # type: ignore[arg-type]
    )
    parent.children[1] = Node(
        state=None, player=1, visits=10, value_sum=1.0  # type: ignore[arg-type]
    )
    search = MCTS(None, MctsConfig(num_simulations=1, c_puct=0.0))  # type: ignore[arg-type]
    assert search._select_action(parent) == 0


def test_visit_policy_counts_simulations_only() -> None:
    from suzero_5dchess.env import GameEnv
    from suzero_5dchess.mcts import MCTS, TorchEvaluator
    from suzero_5dchess.network import build_network

    network = build_network(NetworkConfig(), VERY_SMALL)
    network.eval()
    env = GameEnv(VERY_SMALL)
    search = MCTS(TorchEvaluator(network), MctsConfig(num_simulations=8))
    root = search.run(env)

    assert root.visits == 8
    assert sum(child.visits for child in root.children.values()) == 8
    assert float(root.visit_policy().sum()) == pytest.approx(1.0)


@pytest.mark.parametrize("fen,sign", [
    ("[3k/4/4/KR2:0:1:w]", 1),
    ("[kr2/4/4/3K:0:1:b]", -1),
])
def test_self_play_produces_valid_labelled_samples(fen: str, sign: int) -> None:
    from suzero_5dchess.network import build_network
    from suzero_5dchess.sample import validate_sample
    from suzero_5dchess.self_play import play_game

    spec = replace(
        VERY_SMALL,
        pgn=f'[Board "Custom"]\n[Size "4x4"]\n{fen}\n',
    )
    network = build_network(NetworkConfig(), spec)
    network.eval()
    result = play_game(
        network,
        spec,
        MctsConfig(num_simulations=2),
        SelfPlayConfig(
            playout_cap_randomization=False,
            min_board_limit=2,
            max_board_limit=2,
            record_games=True,
        ),
        seed=11,
    )
    assert result.samples
    assert result.terminal_reason == "material"
    assert result.board_count >= result.board_limit
    assert result.white_outcome == pytest.approx(sign * np.tanh(2.0))
    assert result.moves[-1]["type"] == "move"
    for sample in result.samples:
        validate_sample(sample, spec)
        assert abs(float(sample.policy.sum()) - 1.0) < 1e-4
        expected = result.white_outcome if sample.player == 0 else -result.white_outcome
        assert sample.value_target == pytest.approx(expected)


@pytest.mark.parametrize("fen,sign", [
    ("[3k/4/4/KR2:0:1:w]", 1),
    ("[kr2/4/4/3K:0:1:b]", -1),
])
def test_cpp_self_play_uses_terminal_material_labels(tmp_path: Path, fen: str, sign: int) -> None:
    from suzero_5dchess.cpp_self_play import backend_available, generate_games
    from suzero_5dchess.network import build_network

    if not backend_available():
        pytest.skip("the built extension has no run_selfplay bridge")
    pytest.importorskip("onnx")
    spec = replace(
        VERY_SMALL,
        pgn=f'[Board "Custom"]\n[Size "4x4"]\n{fen}\n',
    )
    network = build_network(NetworkConfig(), spec).eval()
    result = generate_games(
        network,
        spec,
        MctsConfig(num_simulations=2),
        SelfPlayConfig(
            num_games=1,
            min_board_limit=2,
            max_board_limit=2,
            playout_cap_randomization=False,
            backend="cpp_onnx",
            record_games=True,
        ),
        model_path=tmp_path / "model.onnx",
    )[0]

    assert result.samples
    assert result.terminal_reason == "material"
    assert result.board_count >= result.board_limit
    assert result.white_outcome == pytest.approx(sign * np.tanh(2.0))
    assert result.moves[-1]["type"] == "move"
    for sample in result.samples:
        expected = result.white_outcome if sample.player == 0 else -result.white_outcome
        assert sample.value_target == pytest.approx(expected)


def test_parallel_self_play_matches_sequential() -> None:
    from suzero_5dchess.network import build_network
    from suzero_5dchess.self_play import generate_games

    network = build_network(NetworkConfig(), VERY_SMALL)
    base = SelfPlayConfig(
        num_games=2,
        playout_cap_randomization=False,
        min_board_limit=2,
        max_board_limit=2,
    )
    sequential = generate_games(
        network, VERY_SMALL, MctsConfig(num_simulations=2), base, seed=5
    )

    workers = SelfPlayConfig(
        num_games=2,
        playout_cap_randomization=False,
        min_board_limit=2,
        max_board_limit=2,
        workers=2,
    )
    parallel = generate_games(
        network, VERY_SMALL, MctsConfig(num_simulations=2), workers, seed=5
    )

    assert [game.white_outcome for game in parallel] == [
        game.white_outcome for game in sequential
    ]
    assert [len(game.samples) for game in parallel] == [
        len(game.samples) for game in sequential
    ]


def test_debug_pipeline_checkpoint_and_resume(tmp_path: Path) -> None:
    config = TrainConfig(
        num_iterations=1,
        epochs_per_iteration=1,
        steps_per_epoch=1,
        batch_size=1,
        min_replay_size=1,
        replay_buffer_size=200,
        save_interval=1,
        export_interval=1,
        tensorboard=False,
        checkpoint_dir=str(tmp_path / "run"),
        self_play=SelfPlayConfig(
            num_games=1,
            playout_cap_randomization=False,
            min_board_limit=2,
            max_board_limit=2,
            record_games=True,
        ),
        mcts=MctsConfig(num_simulations=2),
    )
    summary = run_training(config)
    assert summary["end_iteration"] == 1
    paths = RunPaths.from_config(config)
    assert paths.latest.exists()
    assert paths.final.exists()
    assert paths.model_path.exists()
    assert Path(str(paths.model_path) + ".json").exists()
    record = paths.log_dir / "games_iter_0001.json"
    assert record.exists()
    assert len(json.loads(record.read_text(encoding="utf-8"))) == 1

    resumed = run_training(config, resume=paths.latest)
    assert resumed["end_iteration"] == 2
    assert paths.iteration_checkpoint(2).exists()
    assert (paths.log_dir / "games_iter_0002.json").exists()


def test_iteration_numbers_are_sequential(tmp_path: Path) -> None:
    config = replace(
        debug_train_config(),
        num_iterations=2,
        checkpoint_dir=str(tmp_path / "run"),
    )
    summary = run_training(config)
    assert summary["end_iteration"] == 2
    assert [record["iteration"] for record in summary["iterations"]] == [1, 2]  # type: ignore[index]
    paths = RunPaths.from_config(config)
    assert paths.iteration_checkpoint(1).exists()
    assert paths.iteration_checkpoint(2).exists()

    resumed = run_training(config, resume=paths.latest)
    assert resumed["end_iteration"] == 4
    assert [record["iteration"] for record in resumed["iterations"]] == [3, 4]  # type: ignore[index]


def test_pipeline_writes_tensorboard(tmp_path: Path) -> None:
    pytest.importorskip("torch.utils.tensorboard")
    config = TrainConfig(
        num_iterations=1,
        epochs_per_iteration=1,
        steps_per_epoch=1,
        batch_size=1,
        min_replay_size=1,
        replay_buffer_size=200,
        save_interval=1,
        export_interval=0,
        tensorboard=True,
        checkpoint_dir=str(tmp_path / "run"),
        self_play=SelfPlayConfig(
            num_games=1,
            playout_cap_randomization=False,
            min_board_limit=2,
            max_board_limit=2,
        ),
        mcts=MctsConfig(num_simulations=2),
    )
    summary = run_training(config)
    assert summary["end_iteration"] == 1
    paths = RunPaths.from_config(config)
    events = list(paths.tensorboard_dir.glob("events.out.tfevents.*"))
    assert events


def test_evaluation_seeds_baseline_then_plays(tmp_path: Path) -> None:
    config = TrainConfig(
        num_iterations=2,
        epochs_per_iteration=1,
        steps_per_epoch=1,
        batch_size=1,
        min_replay_size=1,
        replay_buffer_size=200,
        save_interval=1,
        export_interval=0,
        evaluation=True,
        evaluation_interval=2,
        evaluation_games=2,
        evaluation_simulations=2,
        tensorboard=False,
        checkpoint_dir=str(tmp_path / "eval_run"),
        self_play=SelfPlayConfig(
            num_games=1,
            playout_cap_randomization=False,
            min_board_limit=2,
            max_board_limit=2,
        ),
        mcts=MctsConfig(num_simulations=2),
    )
    summary = run_training(config)
    paths = RunPaths.from_config(config)

    first, second = summary["iterations"]
    assert first["evaluation"] is None
    assert paths.marked.exists()
    assert second["evaluation"] is not None
    assert second["evaluation"]["games"] == 2
    assert (
        second["evaluation"]["a_wins"]
        + second["evaluation"]["b_wins"]
        + second["evaluation"]["draws"]
        == 2
    )


def test_selfplay_backend_interface_defaults() -> None:
    config = TrainConfig()
    assert config.self_play.backend == "python"
    assert config.self_play.cpp_onnx_provider == "cpu"

    with pytest.raises(ValueError):
        SelfPlayConfig(backend="onnx")
    with pytest.raises(ValueError):
        SelfPlayConfig(cpp_onnx_provider="gpu")


def test_cli_exposes_backend_and_board_limit_knobs() -> None:
    from suzero_5dchess.cli import _build_config, build_parser

    args = build_parser().parse_args(
        [
            "train",
            "--selfplay-backend",
            "cpp_onnx",
            "--cpp-onnx-provider",
            "cuda",
            "--min-board-limit",
            "5",
            "--max-board-limit",
            "9",
            "--evaluation-interval",
            "7",
        ]
    )
    config = _build_config(args)
    assert config.self_play.backend == "cpp_onnx"
    assert config.self_play.cpp_onnx_provider == "cuda"
    assert config.self_play.min_board_limit == 5
    assert config.self_play.max_board_limit == 9
    assert config.evaluation_interval == 7


def test_cpp_onnx_backend_runs_a_debug_iteration(tmp_path: Path) -> None:
    from suzero_5dchess.cpp_self_play import backend_available

    if not backend_available():
        pytest.skip("the built extension has no run_selfplay bridge")

    config = debug_train_config()
    config.checkpoint_dir = str(tmp_path / "run")
    config.self_play.backend = "cpp_onnx"
    config.self_play.cpp_onnx_provider = "cpu"
    config.self_play.workers = 2
    config.self_play.num_games = 2
    config.self_play.min_board_limit = 2
    config.self_play.max_board_limit = 2
    config.min_replay_size = 1
    summary = run_training(config)

    assert summary["end_iteration"] == 1
    assert summary["stopped_early"] is False
    assert summary["total_games"] == 2
    record = summary["iterations"][0]
    assert record["self_play"]["games"] == 2
    assert (tmp_path / "run" / "latest.pt").exists()


def test_cooperative_sigint_stops_after_current_iteration() -> None:
    import signal

    from suzero_5dchess import interrupts

    interrupts.install()
    try:
        assert not interrupts.stop_requested()
        # The first press only asks for a cooperative stop; it must not raise.
        signal.raise_signal(signal.SIGINT)
        assert interrupts.stop_requested()
    finally:
        interrupts.uninstall()
    assert not interrupts.stop_requested()


@pytest.mark.parametrize("iterations", [1, 2])
@pytest.mark.parametrize("stop_phase", ["selfplay", "training"])
def test_continuous_last_iteration_stop_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, iterations: int, stop_phase: str
) -> None:
    import signal

    from suzero_5dchess import cli, interrupts, pipeline
    from suzero_5dchess.self_play import SelfPlayResult

    config = debug_train_config()
    config.checkpoint_dir = str(tmp_path / "run")
    config.num_iterations = iterations
    config.export_interval = 0
    config.steps_per_epoch = 1
    config.batch_size = 2
    config.self_play.record_games = False
    config.device = "cpu"
    calls = 0
    original_training_phase = pipeline._training_phase

    def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert calls <= iterations, "continuous mode started another batch after Ctrl+C"
        results = [
            SelfPlayResult(
                samples=[_synthetic_sample(q=2, has_submit=False, policy=[0.5, 0.5], player=0)],
                white_outcome=0.0, terminal_reason="draw", semimoves=1,
                turns=1, board_limit=2, board_count=1, final_player=0,
            )
            for _ in range(2)
        ]
        if stop_phase == "selfplay" and calls == iterations:
            signal.raise_signal(signal.SIGINT)
        return results

    def train(*args, **kwargs):
        if stop_phase == "training" and calls == iterations:
            signal.raise_signal(signal.SIGINT)
        return original_training_phase(*args, **kwargs)

    monkeypatch.setattr(pipeline, "generate_games", generate)
    monkeypatch.setattr(pipeline, "_training_phase", train)
    paths = RunPaths.from_config(config)
    summary = cli._run_training_batches(config, None, paths, continuous=True)

    assert calls == iterations
    assert summary["stopped_early"] is True
    assert summary["end_iteration"] == iterations
    assert summary["total_games"] == iterations * 2
    assert not interrupts.stop_requested()
    assert paths.final.exists()
    extra = pipeline._read_checkpoint(paths.latest)
    assert extra["iteration"] == iterations


def test_cli_accepts_continuous_flag() -> None:
    from suzero_5dchess.cli import build_parser

    args = build_parser().parse_args(["train", "--continuous", "--iterations", "5"])
    assert args.continuous is True
    assert args.iterations == 5
    assert build_parser().parse_args(["train"]).continuous is False


def test_continuous_recording_reuses_one_archive_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from suzero_5dchess import cli

    config = replace(
        debug_train_config(),
        num_iterations=1,
        checkpoint_dir=str(tmp_path / "run"),
        self_play=replace(debug_train_config().self_play, record_games=True),
    )
    paths = RunPaths.from_config(config)
    lifecycle: list[str] = []
    received: list[object] = []

    class FakeArchiveWriter:
        def start(self) -> None:
            lifecycle.append("start")

        def close(self) -> None:
            lifecycle.append("close")

        def abort(self) -> None:
            lifecycle.append("abort")

    def fake_run_training(cfg, *, resume=None, archive_writer=None):
        received.append(archive_writer)
        paths.latest.parent.mkdir(parents=True, exist_ok=True)
        paths.latest.write_bytes(b"checkpoint")
        return {
            "start_iteration": len(received) - 1,
            "end_iteration": len(received),
            "total_games": len(received),
            "total_samples": len(received),
            "buffer_size": len(received),
            "iterations": [{"iteration": len(received)}],
            "stopped_early": len(received) == 2,
        }

    monkeypatch.setattr(cli, "ArchiveWriter", FakeArchiveWriter)
    monkeypatch.setattr(cli, "run_training", fake_run_training)

    summary = cli._run_training_batches(config, None, paths, continuous=True)

    assert summary["end_iteration"] == 2
    assert len(received) == 2
    assert received[0] is received[1]
    assert lifecycle == ["start", "close"]


def test_continuous_batches_resume_until_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from suzero_5dchess import cli

    config = replace(
        debug_train_config(),
        num_iterations=2,
        checkpoint_dir=str(tmp_path / "run"),
    )
    paths = RunPaths.from_config(config)
    calls: list[object] = []

    def fake_run_training(cfg, *, resume=None):
        calls.append(resume)
        paths.latest.parent.mkdir(parents=True, exist_ok=True)
        paths.latest.write_bytes(b"checkpoint")
        start = 2 * (len(calls) - 1)
        return {
            "start_iteration": start,
            "end_iteration": start + 2,
            "total_games": 2 * len(calls),
            "total_samples": 10 * len(calls),
            "buffer_size": 10 * len(calls),
            "iterations": [{"iteration": start + 1}, {"iteration": start + 2}],
            "stopped_early": len(calls) == 3,
        }

    monkeypatch.setattr(cli, "run_training", fake_run_training)
    summary = cli._run_training_batches(config, None, paths, continuous=True)

    # A fresh start, then two resumes; the third batch was interrupted.
    assert calls == [None, paths.latest, paths.latest]
    assert summary["end_iteration"] == 6
    assert summary["total_games"] == 6
    assert [record["iteration"] for record in summary["iterations"]] == [  # type: ignore[index]
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert summary["stopped_early"] is True

    # Without --continuous the caller gets exactly one batch.
    calls.clear()
    summary = cli._run_training_batches(config, None, paths, continuous=False)
    assert calls == [None]
    assert summary["end_iteration"] == 2


def test_continuous_requires_positive_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from suzero_5dchess import cli

    config = replace(
        debug_train_config(),
        num_iterations=0,
        checkpoint_dir=str(tmp_path / "run"),
    )
    paths = RunPaths.from_config(config)
    called = False

    def fake_run_training(cfg, *, resume=None):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(cli, "run_training", fake_run_training)
    with pytest.raises(SystemExit):
        cli._run_training_batches(config, None, paths, continuous=True)
    assert called is False
