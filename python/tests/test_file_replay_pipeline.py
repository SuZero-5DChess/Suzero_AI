"""Training integration for the per-game file replay backend (the default)."""

import json

import numpy as np
import pytest
import torch

from suzero_5dchess import pipeline
from suzero_5dchess.config import NetworkConfig, SelfPlayConfig, TrainConfig
from suzero_5dchess.game_store import GAME_STORE_FORMAT, GameStore
from suzero_5dchess.protocol import VERY_SMALL
from suzero_5dchess.replay import REPLAY_FORMAT
from suzero_5dchess.replay_window import WindowConfig
from suzero_5dchess.sample import PositionSample
from suzero_5dchess.self_play import SelfPlayResult


def sample(index):
    return PositionSample(
        board_codes=np.zeros((1, 16), dtype=np.uint8),
        l_coords=np.array([0], dtype=np.int64),
        t_coords=np.array([1], dtype=np.int64),
        urgency=24.0,
        side_to_move=0,
        actions=np.array([[0, 0, 1, 0, 1, 0, 1, 0]], dtype=np.int64),
        policy=np.array([1.0], dtype=np.float32),
        has_submit=False,
        move_count=index,
    )


def game(start, count=2):
    return SelfPlayResult(
        samples=[sample(index) for index in range(start, start + count)],
        white_outcome=0.0,
        terminal_reason="draw",
        semimoves=count,
        turns=2,
        board_limit=2,
        board_count=2,
        final_player=0,
    )


def config_for(path, **kwargs):
    fields = {
        "checkpoint_dir": str(path), "device": "cpu", "tensorboard": False,
        "num_iterations": 1, "epochs_per_iteration": 1, "steps_per_epoch": 1,
        "batch_size": 2, "min_replay_size": 2, "replay_buffer_size": 8,
        "export_interval": 0, "save_interval": 1, "gate_interval": 0,
        "network": NetworkConfig(d_model=8, n_heads=2, d_ff=16, action_mlp_hidden=8),
        "self_play": SelfPlayConfig(num_games=2),
        "replay_backend": "file",
    }
    fields.update(kwargs)
    return TrainConfig(**fields)


@pytest.fixture
def streaming_backend(monkeypatch):
    """A backend that streams each game exactly like the C++/Python paths."""

    calls = []

    def generate(*args, game_store=None, game_id_start=0, iteration=-1, **kwargs):
        calls.append({
            "store": game_store,
            "game_id_start": game_id_start,
            "iteration": iteration,
        })
        results = []
        for index in range(2):
            result = game(index * 2)
            if game_store is not None:
                game_store.append_result(
                    game_id_start + index, result, iteration=iteration
                )
            results.append(result)
        return results

    monkeypatch.setattr(pipeline, "generate_games", generate)
    return calls


def test_each_finished_game_is_on_disk_before_training(
    tmp_path, streaming_backend, monkeypatch
):
    config = config_for(tmp_path / "run")
    observed = []
    original = pipeline._training_phase

    def training(trainer, replay, cfg, **kwargs):
        # Training must already see the round on disk, not in RAM.
        observed.append((type(replay).__name__, len(replay), sorted(
            path.name for path in replay.directory.iterdir()
        )))
        return original(trainer, replay, cfg, **kwargs)

    monkeypatch.setattr(pipeline, "_training_phase", training)
    summary = pipeline.run_training(config)

    assert streaming_backend[0]["store"] is not None
    assert streaming_backend[0]["game_id_start"] == 0
    assert streaming_backend[0]["iteration"] == 1
    assert observed == [("GameStore", 4, ["replay.lock", "replay.log"])]

    iteration = summary["iterations"][0]
    assert summary["total_samples"] == 4
    assert summary["buffer_size"] == 4
    assert iteration["training"]["samples"] == 2  # one batch of two
    assert iteration["training"]["shuffle"]["backend"] == "file"
    counts = iteration["training"]["train_counts"]
    assert counts["samples"] == 4
    assert counts["trained"] == 2
    assert counts["untrained"] == 2
    assert counts["max"] == 1
    assert "replay_window" not in iteration  # window stays opt-in

    paths = pipeline.RunPaths.from_config(config)
    state = torch.load(paths.replay, weights_only=False)
    assert state["format"] == GAME_STORE_FORMAT
    assert state["frames"] == 2
    assert "blobs" not in state  # only counters are saved, not the frame index

    # Row counters use row identities, so a fresh process can merge them back
    # onto the log it scans from disk.
    assert paths.train_counts.is_file()
    restored = GameStore(paths.replay_store, VERY_SMALL, 64)
    restored.scan()
    assert restored.load_train_counts(paths.train_counts) == 4
    assert sorted(restored.train_counts) == [0, 0, 1, 1]

    checkpoint = torch.load(paths.latest, weights_only=False)
    assert checkpoint["extra_state"]["total_samples"] == 4
    assert not (paths.root / "replay_shuffle").exists()

    logged = json.loads(
        (paths.log_dir / "training_log.jsonl").read_text(encoding="utf-8")
    )
    assert logged["training"]["shuffle"]["games"] == 2


def test_results_the_backend_could_not_stream_are_appended(tmp_path, monkeypatch):
    def generate(*args, **kwargs):
        return [game(0), game(2)]

    monkeypatch.setattr(pipeline, "generate_games", generate)
    config = config_for(tmp_path / "run")
    summary = pipeline.run_training(config)

    assert summary["total_samples"] == 4
    assert summary["buffer_size"] == 4
    store = GameStore(
        pipeline.RunPaths.from_config(config).replay_store, VERY_SMALL, 8
    )
    store.scan()
    assert len(store) == 4
    assert [entry.game_id for entry in store.games] == [0, 1]


def test_resume_keeps_appending_to_the_same_window(
    tmp_path, streaming_backend
):
    config = config_for(tmp_path / "run")
    first = pipeline.run_training(config)
    paths = pipeline.RunPaths.from_config(config)
    first_state = torch.load(paths.replay, weights_only=False)

    second = pipeline.run_training(config, resume=paths.latest)
    assert second["start_iteration"] == 1
    assert second["end_iteration"] == 2
    assert second["total_samples"] == 8
    assert second["buffer_size"] == 8
    assert streaming_backend[1]["game_id_start"] == 2
    assert streaming_backend[1]["iteration"] == 2

    store = GameStore(paths.replay_store, VERY_SMALL, 8)
    store.scan()
    assert len(store) == 8
    assert [entry.game_id for entry in store.games] == [0, 1, 2, 3]
    state = torch.load(paths.replay, weights_only=False)
    assert state["total_generated"] == 8
    assert state["next_game_id"] == 4
    assert state["frames"] == first_state["frames"] + 2
    assert first["iterations"][0]["training"]["shuffle"]["generation"] != \
        second["iterations"][0]["training"]["shuffle"]["generation"]


def test_window_trims_whole_games_in_file_mode(tmp_path, streaming_backend):
    config = config_for(
        tmp_path / "run",
        num_iterations=2,
        epochs_per_iteration=1,
        steps_per_epoch=2,
        replay_buffer_size=2,
        window=WindowConfig(enabled=True, expand_per_row=0, max_samples=4),
    )
    summary = pipeline.run_training(config)
    first, second = summary["iterations"]
    # The first window keeps both games: nothing was served yet, so the target
    # of two rows may not evict the four freshly generated rows.
    assert first["replay_window"]["total_generated"] == 4
    assert first["replay_window"]["window_size"] == 2
    assert first["replay_window"]["buffered"] == 4
    assert first["replay_window"]["evicted"] == 0
    # Both games had a training pass, so the next round evicts the first
    # iteration's games and keeps the two new, not yet served ones.
    window = second["replay_window"]
    assert window["backend"] == "file"
    assert window["total_generated"] == 8
    assert window["window_size"] == 2
    assert window["buffered"] == 4
    assert window["evicted"] == 4
    assert summary["total_samples"] == 8
    assert summary["buffer_size"] == 4

    store = GameStore(pipeline.RunPaths.from_config(config).replay_store, VERY_SMALL, 4)
    store.scan()
    assert [entry.game_id for entry in store.games] == [2, 3]


def test_file_backend_rejects_uniform_window_sampling(tmp_path):
    with pytest.raises(ValueError, match="shuffled file-sequential"):
        config_for(
            tmp_path,
            window=WindowConfig(
                enabled=True, sampling="uniform", max_samples=8
            ),
        )


def test_ctrl_c_during_self_play_saves_samples_without_training(
    tmp_path, monkeypatch
):
    stopped = [False]

    def generate(*args, **kwargs):
        stopped[0] = True  # the signal was noticed while the batch ran
        return [game(0), game(2)]

    def training(*args, **kwargs):
        raise AssertionError("training must not start after Ctrl+C during self-play")

    monkeypatch.setattr(pipeline, "generate_games", generate)
    monkeypatch.setattr(pipeline, "_training_phase", training)
    monkeypatch.setattr(
        pipeline.interrupts, "stop_requested", lambda: stopped[0]
    )

    config = config_for(tmp_path / "run")
    summary = pipeline.run_training(config)

    assert summary["stopped_early"] is True
    assert summary["end_iteration"] == 0
    assert summary["iterations"] == []
    assert summary["total_games"] == 2
    assert summary["total_samples"] == 4
    assert summary["buffer_size"] == 4

    paths = pipeline.RunPaths.from_config(config)
    store = GameStore(paths.replay_store, VERY_SMALL, 8)
    store.scan()
    assert len(store) == 4
    assert [entry.game_id for entry in store.games] == [0, 1]

    checkpoint = torch.load(paths.latest, weights_only=False)
    assert checkpoint["extra_state"]["iteration"] == 0
    assert checkpoint["extra_state"]["total_games"] == 2
    assert checkpoint["extra_state"]["total_samples"] == 4
    # The batch stopped mid-iteration, so the record names the next round.
    assert checkpoint["extra_state"]["selfplay_progress"] == {
        "iteration": 1,
        "games": 2,
    }


def test_resume_after_ctrl_c_does_not_replay_saved_games(tmp_path, monkeypatch):
    stopped = [False]
    calls = []

    def generate(*args, game_store=None, game_id_start=0, iteration=-1, **kwargs):
        calls.append((game_id_start, iteration))
        results = []
        for index in range(2):
            result = game(index * 2)
            if game_store is not None:
                game_store.append_result(
                    game_id_start + index, result, iteration=iteration
                )
            results.append(result)
        if len(calls) == 1:
            stopped[0] = True
        return results

    monkeypatch.setattr(pipeline, "generate_games", generate)
    monkeypatch.setattr(
        pipeline.interrupts, "stop_requested", lambda: stopped[0]
    )

    config = config_for(tmp_path / "run")
    first = pipeline.run_training(config)
    assert first["stopped_early"] is True
    assert first["total_samples"] == 4

    # The interrupted iteration already played every game this config asks
    # for, so the resume trains on those saved games instead of replaying
    # them.  Iteration 1 never trained, so it is still entered and completed.
    stopped[0] = False
    paths = pipeline.RunPaths.from_config(config)
    second = pipeline.run_training(config, resume=paths.latest)

    assert calls == [(0, 1)]
    assert second["start_iteration"] == 0
    assert second["end_iteration"] == 1
    assert second["total_games"] == 2
    assert second["total_samples"] == 4

    store = GameStore(paths.replay_store, VERY_SMALL, 8)
    store.scan()
    assert [entry.game_id for entry in store.games] == [0, 1]
    state = torch.load(paths.replay, weights_only=False)
    assert state["next_game_id"] == 2
    assert state["total_generated"] == 4


def test_resume_tops_up_the_interrupted_iteration(tmp_path, monkeypatch):
    stopped = [False]
    calls = []
    advances = []

    from contextlib import contextmanager

    @contextmanager
    def progress(desc, total):
        seen = []
        advances.append(seen)
        yield seen.append

    def generate(*args, num_games=None, game_store=None, game_id_start=0,
                 iteration=-1, progress=None, **kwargs):
        calls.append({
            "num_games": num_games,
            "game_id_start": game_id_start,
            "iteration": iteration,
        })
        # A batch that is cut short after two games, as Ctrl+C does.
        count = 2 if num_games is None else min(num_games, 2)
        results = []
        for index in range(count):
            result = game(index * 2)
            if game_store is not None:
                game_store.append_result(
                    game_id_start + index, result, iteration=iteration
                )
            results.append(result)
            if progress is not None:
                progress(index + 1, count, result)
        if len(calls) == 1:
            stopped[0] = True
        return results

    monkeypatch.setattr(pipeline, "generate_games", generate)
    monkeypatch.setattr(pipeline, "self_play_progress", progress)
    monkeypatch.setattr(
        pipeline.interrupts, "stop_requested", lambda: stopped[0]
    )

    config = config_for(
        tmp_path / "run", self_play=SelfPlayConfig(num_games=4)
    )
    first = pipeline.run_training(config)
    assert first["stopped_early"] is True
    assert first["total_games"] == 2

    stopped[0] = False
    paths = pipeline.RunPaths.from_config(config)
    second = pipeline.run_training(config, resume=paths.latest)

    # Only the missing two games are played, behind the saved ids, and the
    # progress line counts them from the two already on disk.
    assert calls == [
        {"num_games": 4, "game_id_start": 0, "iteration": 1},
        {"num_games": 2, "game_id_start": 2, "iteration": 1},
    ]
    assert advances == [[0, 1, 2], [2, 3, 4]]
    assert second["start_iteration"] == 0
    assert second["end_iteration"] == 1
    assert second["total_games"] == 4
    assert second["total_samples"] == 8

    store = GameStore(paths.replay_store, VERY_SMALL, 8)
    store.scan()
    assert [entry.game_id for entry in store.games] == [0, 1, 2, 3]
    state = torch.load(paths.replay, weights_only=False)
    assert state["next_game_id"] == 4
    assert state["total_generated"] == 8

    checkpoint = torch.load(paths.latest, weights_only=False)
    assert checkpoint["extra_state"]["selfplay_progress"] == {
        "iteration": 1,
        "games": 4,
    }


def test_memory_backend_still_uses_the_replay_buffer(tmp_path, monkeypatch):
    def generate(*args, **kwargs):
        return [game(0), game(2)]

    monkeypatch.setattr(pipeline, "generate_games", generate)
    config = config_for(tmp_path / "run", replay_backend="memory")
    summary = pipeline.run_training(config)
    assert summary["buffer_size"] == 4
    paths = pipeline.RunPaths.from_config(config)
    state = torch.load(paths.replay, weights_only=False)
    assert state["format"] == REPLAY_FORMAT
    assert "blobs" in state
    assert not paths.replay_store.exists()


def test_cpp_backend_receives_the_store_and_game_id_range(tmp_path, monkeypatch):
    from suzero_5dchess.config import SelfPlayConfig

    captured = {}

    def generate(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(pipeline, "generate_games_cpp", generate)
    config = config_for(
        tmp_path / "run",
        min_replay_size=10,
        replay_buffer_size=10,
        self_play=SelfPlayConfig(num_games=2, backend="cpp_onnx"),
    )
    pipeline.run_training(config)
    assert isinstance(captured["game_store"], GameStore)
    assert captured["game_id_start"] == 0
    assert captured["iteration"] == 1


def test_pipeline_threads_the_repeat_rows_flag_into_the_round(
    tmp_path, streaming_backend
):
    config = config_for(tmp_path / "run", no_repeat_rows=False)
    summary = pipeline.run_training(config)
    shuffle = summary["iterations"][0]["training"]["shuffle"]
    assert shuffle["backend"] == "file"
    assert shuffle["no_repeat_rows"] is False
    assert shuffle["round_samples"] == 4


def test_endgame_sampling_is_wired_into_file_rounds(tmp_path, streaming_backend):
    from suzero_5dchess.endgame_sampling import EndgameSamplingConfig

    endgame = EndgameSamplingConfig(enabled=True, strength=2.0, exponent=1.0)
    config = config_for(tmp_path / "run", endgame=endgame)
    summary = pipeline.run_training(config)

    iteration = summary["iterations"][0]
    assert iteration["training"]["endgame"] == {
        "strength": 2.0, "exponent": 1.0,
    }
    shuffle = iteration["training"]["shuffle"]
    assert shuffle["endgame"] == {"strength": 2.0, "exponent": 1.0}

    paths = pipeline.RunPaths.from_config(config)
    store = GameStore(paths.replay_store, VERY_SMALL, 8)
    store.scan()
    round_state = store.read_last_round()
    assert round_state["endgame"] == {"strength": 2.0, "exponent": 1.0}
    # The final row of each two-row game gets two extra copies.
    assert round_state["count"] > round_state["pool_samples"]
    # selected entries are (game_id, local row); local row 1 is the game's end.
    rows = [row[1] for row in round_state["selected"]]
    assert rows.count(1) > rows.count(0)
    assert rows.count(1) >= 3


def test_endgame_sampling_is_wired_into_the_memory_backend(tmp_path, monkeypatch):
    from suzero_5dchess.endgame_sampling import EndgameSamplingConfig

    def generate(*args, **kwargs):
        return [game(0), game(2)]

    monkeypatch.setattr(pipeline, "generate_games", generate)
    endgame = EndgameSamplingConfig(enabled=True, strength=1.0, exponent=2.0)
    config = config_for(
        tmp_path / "run", replay_backend="memory", endgame=endgame,
    )
    summary = pipeline.run_training(config)
    assert summary["iterations"][0]["training"]["endgame"] == {
        "strength": 1.0, "exponent": 2.0,
    }
