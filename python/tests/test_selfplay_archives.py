"""Archive integration across Python and single/multi-process native self-play."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from suzero_5dchess.config import MctsConfig, NetworkConfig, SelfPlayConfig, TrainConfig
from suzero_5dchess.protocol import VERY_SMALL


@pytest.fixture
def native_records():
    binding = pytest.importorskip("suzero_5dchess._5dchess")
    if getattr(binding, "RECORD_FORMAT_VERSION", 0) != 1:
        pytest.skip("rebuild _5dchess for .gn5d support")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield binding
    torch.set_num_threads(previous)


@pytest.mark.parametrize("backend,processes", [("python", 1), ("cpp_onnx", 1), ("cpp_onnx", 2)])
def test_pipeline_archives_every_fast_game(
    tmp_path: Path, native_records, backend: str, processes: int
) -> None:
    from suzero_5dchess.game_record import load_record
    from suzero_5dchess.pipeline import RunPaths, run_training

    if backend == "cpp_onnx":
        pytest.importorskip("onnx")
    config = TrainConfig(
        device="cpu", checkpoint_dir=str(tmp_path / "run"),
        log_dir=str(tmp_path / "custom_logs"),
        num_iterations=1, export_interval=0, save_interval=1,
        tensorboard=False, min_replay_size=1, gate_interval=0,
        mcts=MctsConfig(num_simulations=2),
        self_play=SelfPlayConfig(
            num_games=2, workers=2, num_server_threads=processes,
            processes=processes, backend=backend,
            cpp_onnx_provider="cpu", record_games=True,
            min_board_limit=2, max_board_limit=2,
            full_search_probability=0.0,
            fast_simulations=2,
        ),
    )
    summary = run_training(config)
    assert summary["end_iteration"] == 1
    paths = RunPaths.from_config(config)
    archives = sorted((paths.log_dir / "games").rglob("*.gn5d"))
    assert len(archives) == 2
    records = [load_record(path) for path in archives]
    assert {record["meta"]["game_id"] for record in records} == {1, 2}
    for path, record in zip(archives, records):
        assert record["meta"]["backend"] == backend
        assert record["meta"]["iteration"] == 1
        assert path.parent.name == record["result"]["terminal_reason"]
        assert record["actions"]
        assert "replay" not in record
        assert all(action == "submit" or (isinstance(action, list) and len(action) == 8)
                   for action in record["actions"])
        assert sum(action != "submit" for action in record["actions"]) > 0
    # No full-search positions were sampled, but all played actions survived.
    iteration = summary["iterations"][0]
    assert iteration["self_play"]["samples"] == 0
    aggregate = json.loads((paths.log_dir / "games_iter_0001.json").read_text("utf-8"))
    assert all(game["moves"] and game["final_fen"] for game in aggregate)
    original_bytes = {path: path.read_bytes() for path in archives}

    resumed = run_training(config, resume=paths.latest)
    assert resumed["end_iteration"] == 2
    all_archives = sorted((paths.log_dir / "games").rglob("*.gn5d"))
    assert len(all_archives) == 4
    assert {load_record(path)["meta"]["game_id"] for path in all_archives} == {1, 2, 3, 4}
    assert all(path.read_bytes() == data for path, data in original_bytes.items())


def test_python_recording_does_not_change_game(native_records) -> None:
    from suzero_5dchess.game_record import build_record, replay_record
    from suzero_5dchess.network import build_network
    from suzero_5dchess.self_play import play_game

    torch.manual_seed(101)
    network = build_network(NetworkConfig(), VERY_SMALL).eval()
    config = SelfPlayConfig(
        min_board_limit=2, max_board_limit=2,
        full_search_probability=0.0, fast_simulations=2,
    )
    args = (network, VERY_SMALL, MctsConfig(num_simulations=2))
    plain = play_game(*args, config, seed=11)
    recorded = play_game(*args, replace(config, record_games=True), seed=11)
    assert not plain.moves and not plain.final_fen
    for key in ("white_outcome", "terminal_reason", "semimoves", "turns", "board_count", "final_player"):
        assert getattr(plain, key) == getattr(recorded, key)
    assert getattr(recorded, "terminal_reason") == "material"
    assert recorded.board_count >= recorded.board_limit
    assert recorded.moves[-1]["type"] == "move"
    record = build_record(recorded, VERY_SMALL, iteration=1, game_id=1, backend="python")
    assert replay_record(record).record_snapshot()["fen"] == recorded.final_fen


def test_old_native_extension_refuses_recording_before_export(monkeypatch, tmp_path: Path) -> None:
    from suzero_5dchess import cpp_self_play

    monkeypatch.setattr(cpp_self_play, "_require_binding", lambda: None)
    monkeypatch.setattr(cpp_self_play, "_5dchess", SimpleNamespace(run_selfplay=lambda **kwargs: None))

    def unexpected_export(*args):
        pytest.fail("old binding should fail before expensive export/self-play")

    monkeypatch.setattr(cpp_self_play, "_export_model", unexpected_export)
    with pytest.raises(RuntimeError, match="rebuilt _5dchess"):
        cpp_self_play.generate_games(
            object(), VERY_SMALL, MctsConfig(num_simulations=1),
            SelfPlayConfig(backend="cpp_onnx", record_games=True),
            model_path=tmp_path / "unused.onnx",
        )


def test_disabled_recording_writes_no_game_archives(tmp_path: Path, native_records) -> None:
    from suzero_5dchess.pipeline import RunPaths, run_training

    config = TrainConfig(
        device="cpu", checkpoint_dir=str(tmp_path), num_iterations=1,
        export_interval=0, tensorboard=False, gate_interval=0,
        mcts=MctsConfig(num_simulations=1),
        self_play=SelfPlayConfig(
            num_games=1,
            min_board_limit=1,
            max_board_limit=1,
            record_games=False,
        ),
    )
    run_training(config)
    paths = RunPaths.from_config(config)
    assert not list(paths.log_dir.rglob("*.gn5d"))
    assert not list(paths.log_dir.glob("games_iter_*.json"))
