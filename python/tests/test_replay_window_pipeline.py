"""Lossy replay integration, counter reconciliation and disabled compatibility."""

from dataclasses import replace
import json
import logging

import numpy as np
import pytest
import torch

from suzero_5dchess import pipeline
from suzero_5dchess.config import NetworkConfig, TrainConfig, SelfPlayConfig
from suzero_5dchess.protocol import VERY_SMALL
from suzero_5dchess.replay import ReplayBuffer
from suzero_5dchess.replay_window import WindowConfig, compute_window_size
from suzero_5dchess.sample import PositionSample
from suzero_5dchess.self_play import SelfPlayResult
from suzero_5dchess.train_counts import evictable_prefix


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


def config_for(path, **kwargs):
    # This module exercises the in-memory window and its published shuffle
    # generations; the file-per-game backend has its own tests.
    kwargs.setdefault("replay_backend", "memory")
    return TrainConfig(
        checkpoint_dir=str(path), device="cpu", tensorboard=False,
        num_iterations=2, epochs_per_iteration=2, steps_per_epoch=2,
        batch_size=2, min_replay_size=1, replay_buffer_size=4,
        export_interval=0, save_interval=1, gate_interval=0,
        network=NetworkConfig(d_model=8, n_heads=2, d_ff=16, action_mlp_hidden=8),
        self_play=SelfPlayConfig(num_games=2),
        window=WindowConfig(enabled=True, taper_exponent=1, expand_per_row=0.5,
                            max_samples=12),
        **kwargs,
    )


@pytest.fixture
def fake_games(monkeypatch):
    def generate(*args, **kwargs):
        return [
            SelfPlayResult(
                samples=[sample(index) for index in range(game * 4, game * 4 + 4)],
                white_outcome=0.0, terminal_reason="draw", semimoves=4,
                turns=2, board_limit=2, board_count=2, final_player=0,
            )
            for game in range(2)
        ]

    monkeypatch.setattr(pipeline, "generate_games", generate)


def test_round_trim_before_training_and_resume(tmp_path, monkeypatch, fake_games):
    config = config_for(tmp_path)
    observed = []
    trims = []
    original_training = pipeline._training_phase
    original_trim = ReplayBuffer.trim_to

    def training(trainer, buffer, cfg, **kwargs):
        n = buffer.total_generated
        observed.append((n, len(buffer), buffer.capacity))
        # The window is a soft cap: untrained front rows block eviction, so the
        # live size may exceed the target while that backlog waits for training.
        over = max(0, len(buffer) - compute_window_size(
            n, cfg.replay_buffer_size, cfg.window))
        assert evictable_prefix(buffer.train_counts, over) == 0
        return original_training(trainer, buffer, cfg, **kwargs)

    def trim(buffer, size):
        trims.append(size)
        return original_trim(buffer, size)

    monkeypatch.setattr(pipeline, "_training_phase", training)
    monkeypatch.setattr(ReplayBuffer, "trim_to", trim)
    summary = pipeline.run_training(config)
    # A fresh window is never trimmed before training: nothing was served yet.
    assert observed == [(8, 8, 12), (16, 10, 12)]
    assert trims == [4, 6, 10]  # one startup trim, then one per round, not per game
    first, second = summary["iterations"]
    # (window size, buffered, evicted, trained samples, round size) per round.
    for record, window_size, buffered, evicted, trained, round_samples in (
        (first, 6, 8, 0, 8, 8),
        # The second round excludes the previously trained rows, which left
        # exactly the eight new, never-served rows to draw from.
        (second, 10, 10, 6, 8, 8),
    ):
        window = record["replay_window"]
        assert window == {
            "total_generated": buffered + evicted,
            "window_size": window_size, "buffered": buffered,
            "evicted": evicted, "sampling": "shuffle", "keep_samples": None,
            "estimated_bytes": window["estimated_bytes"],
            "coverage": trained / buffered,
            "backend": "memory",
        }
        assert window["estimated_bytes"] > 0
        assert record["training"]["samples"] == trained
        # Every live row was served exactly once by the end of the iteration:
        # the round covers the whole window and the epoch budget happens to
        # reach every row.
        counts = record["training"]["train_counts"]
        assert counts["samples"] == buffered
        assert counts["trained"] == buffered
        assert counts["untrained"] == 0
        assert counts["max"] == 1
        shuffle = record["training"]["shuffle"]
        assert shuffle["pool_samples"] == buffered
        assert shuffle["round_samples"] == round_samples
        assert shuffle["remaining_samples"] == round_samples - trained
        assert shuffle["no_repeat_rows"] is True
        # The published round was fully consumed; another round would be
        # published lazily by the next epoch that needs rows.
        assert shuffle["waiting_for_shuffle"] is (trained == round_samples)
    assert first["training"]["shuffle"]["generation"] != second["training"]["shuffle"]["generation"]
    paths = pipeline.RunPaths.from_config(config)
    logged = [json.loads(line) for line in (paths.log_dir / "training_log.jsonl").read_text().splitlines()]
    assert [row["replay_window"] for row in logged] == [first["replay_window"], second["replay_window"]]
    state = torch.load(paths.replay, weights_only=False)
    assert state["total_generated"] == 16
    assert state["count"] == 10
    assert state["train_counts"].tolist() == [1] * 10
    shuffle = second["training"]["shuffle"]
    assert state["shuffle_state"] == {key: shuffle[key] for key in state["shuffle_state"]}
    checkpoint = torch.load(paths.latest, weights_only=False)
    assert checkpoint["extra_state"]["replay_shuffle"] == state["shuffle_state"]
    directory = paths.root / "replay_shuffle"
    assert json.loads((directory / "current.json").read_text())["generation"] == shuffle["generation"]
    assert not (directory / first["training"]["shuffle"]["generation"]).exists()

    resumed = pipeline.run_training(replace(config, num_iterations=1), resume=paths.latest)
    assert resumed["total_samples"] == 24
    assert resumed["buffer_size"] == 12
    window = resumed["iterations"][0]["replay_window"]
    assert window["window_size"] == 12
    assert window["evicted"] == 6  # includes push's hard-cap protection
    assert window["coverage"] == pytest.approx(8 / 12)
    assert observed[-1] == (24, 12, 12)
    assert trims == [4, 6, 10, 10, 12]


def test_shuffle_rounds_never_repeat_the_previous_round(tmp_path, fake_games, monkeypatch):
    published = []
    original = pipeline.publish_shuffle

    def publish(buffer, directory, **kwargs):
        manifest = original(buffer, directory, **kwargs)
        published.append({
            "exclude": sorted(kwargs.get("exclude", ())),
            "manifest": manifest,
            "pool": len(buffer),
            "generated": buffer.total_generated,
        })
        return manifest

    monkeypatch.setattr(pipeline, "publish_shuffle", publish)
    config = replace(
        config_for(tmp_path),
        window=WindowConfig(
            enabled=True, taper_exponent=1, expand_per_row=0.5,
            max_samples=12, keep_samples=3,
        ),
    )
    pipeline.run_training(config)
    manifests = [item["manifest"] for item in published if item["manifest"] is not None]
    assert len(manifests) >= 4

    for previous, current in zip(published, published[1:]):
        before, after = previous["manifest"], current["manifest"]
        if before is None or after is None:
            continue
        if previous["generated"] == current["generated"]:
            # Same phase: a mid-epoch round excludes exactly the round it
            # replaced, so within one epoch shuffles are row-disjoint.
            assert current["exclude"] == sorted(before["rows"])
            assert set(before["rows"]).isdisjoint(after["rows"])
        else:
            # Across phases the window was trimmed from the front, so the
            # previous round's indices shift by the eviction count.
            shift = previous["pool"] + (
                current["generated"] - previous["generated"]
            ) - current["pool"]
            expected = sorted(
                row - shift for row in before["rows"] if row >= shift
            )
            assert current["exclude"] == expected
            assert set(expected).isdisjoint(after["rows"])

    # The first phase needed more than one round to fill its epoch budget,
    # i.e. there are multiple shuffles within an epoch, not one per epoch.
    first_phase = [item["manifest"] for item in published
                   if item["manifest"] is not None and item["generated"] == 8]
    assert len(first_phase) >= 2


def test_disabled_sampling_logs_and_training_unchanged(tmp_path, fake_games, caplog, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("disabled replay published shuffle files")

    monkeypatch.setattr(pipeline, "publish_shuffle", fail)
    base = replace(config_for(tmp_path / "default"), window=WindowConfig())
    noisy_disabled = replace(
        base, checkpoint_dir=str(tmp_path / "disabled_knobs"),
        window=WindowConfig(enabled=False, max_samples=1, keep_samples=1,
                            sampling="shuffle", taper_scale=1),
    )
    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        first = pipeline.run_training(base)
        second = pipeline.run_training(noisy_disabled)
    assert "replay_window:" not in caplog.text
    assert "lossy replay window:" not in caplog.text
    for a, b in zip(first["iterations"], second["iterations"]):
        assert "replay_window" not in a and "replay_window" not in b
        assert a["training"] == b["training"]
        # Window disabled: max_samples=1 must not size the buffer, and the
        # capacity is soft until the trainer has served the front rows.
        assert a["buffer_size"] == b["buffer_size"] >= 4
        assert a.keys() == b.keys()
    states = [torch.load(pipeline.RunPaths.from_config(c).latest, weights_only=False)
              for c in (base, noisy_disabled)]
    for name, tensor in states[0]["model"].items():
        assert torch.equal(tensor, states[1]["model"][name])


@pytest.mark.parametrize("extra_n,replay_n,expected", [(30, 40, 40), (40, 30, 40), (0, 0, 8)])
def test_startup_max_rule_and_trim(tmp_path, caplog, extra_n, replay_n, expected):
    buffer = ReplayBuffer(12, VERY_SMALL)
    buffer.push([sample(i) for i in range(8)])
    # Only rows the trainer already served may be evicted at startup.
    buffer.record_trained(range(8))
    buffer.total_generated = replay_n
    config = replace(config_for(tmp_path), window=WindowConfig(enabled=True, max_samples=12,
                                                            expand_per_row=0))
    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        n = pipeline._initialize_replay_window(buffer, config, extra_n, replay_loaded=True)
    assert n == buffer.total_generated == expected
    assert len(buffer) == 4
    assert [x.move_count for x in buffer.sample_at(range(4))] == [4, 5, 6, 7]
    assert "replay counters differ" in caplog.text
    assert "startup_evicted=4" in caplog.text
    assert "estimated_window_bytes=" in caplog.text


@pytest.mark.parametrize("legacy,missing,smaller", [(True, False, False), (False, True, False), (False, False, True)])
def test_resume_old_missing_and_smaller_replay(tmp_path, fake_games, caplog, legacy, missing, smaller):
    config = config_for(tmp_path)
    pipeline.run_training(config)
    paths = pipeline.RunPaths.from_config(config)
    if legacy:
        state = torch.load(paths.replay, weights_only=False)
        del state["total_generated"]
        torch.save(state, paths.replay)
    if missing:
        paths.replay.unlink()
    if smaller:
        config = replace(config, window=replace(config.window, max_samples=5))
    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        resumed = pipeline.run_training(replace(config, num_iterations=1), resume=paths.latest)
    assert resumed["start_iteration"] == 2
    assert resumed["total_samples"] == 24
    if smaller:
        # Loading keeps only trained rows within the reduced cap; the new rows
        # that follow are untrained, so the buffer may stay above max_samples.
        assert "new_max_samples=5" in caplog.text
        assert "may discard up to 5 oldest trained samples on load" in caplog.text
        assert resumed["buffer_size"] == 8
        assert resumed["iterations"][0]["replay_window"]["buffered"] == 8
    else:
        assert resumed["buffer_size"] <= config.window.max_samples
    if legacy:
        assert "legacy replay has no total_generated" in caplog.text
    if missing:
        assert "historical replay samples are permanently lost" in caplog.text
        assert "historical N=16" in caplog.text


def test_skipped_training_publishes_no_shuffle_round(tmp_path, fake_games, monkeypatch):
    published = []
    original = pipeline.publish_shuffle

    def publish(*args, **kwargs):
        manifest = original(*args, **kwargs)
        published.append(manifest)
        return manifest

    monkeypatch.setattr(pipeline, "publish_shuffle", publish)
    config = replace(config_for(tmp_path), min_replay_size=20, replay_buffer_size=20,
                     window=WindowConfig(enabled=True, max_samples=20))
    records = pipeline.run_training(config)["iterations"]
    assert all(record["training"]["skipped"] is True for record in records)
    assert all(record["replay_window"]["coverage"] == 0 for record in records)
    # Publishing a round that will not be consumed would force the next phase
    # to exclude rows that were never trained on.
    assert published == []
    paths = pipeline.RunPaths.from_config(config)
    assert not (paths.root / "replay_shuffle").exists()


def test_enabled_uniform_ignores_keep_and_does_not_publish_files(tmp_path, fake_games, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("uniform replay published shuffle files")

    monkeypatch.setattr(pipeline, "publish_shuffle", fail)
    base = replace(config_for(tmp_path / "uniform"), window=WindowConfig(enabled=True, sampling="uniform", max_samples=12))
    subset = replace(base, checkpoint_dir=str(tmp_path / "uniform_keep"), window=replace(base.window, keep_samples=1))
    first = pipeline.run_training(base)
    second = pipeline.run_training(subset)
    for a, b in zip(first["iterations"], second["iterations"]):
        assert a["training"] == b["training"]
        assert a["training"]["samples"] == 8
        assert "shuffle" not in a["training"]
    for config in (base, subset):
        assert not (pipeline.RunPaths.from_config(config).root / "replay_shuffle").exists()


def test_window_tensorboard_tags():
    calls = {}

    class Writer:
        def add_scalar(self, tag, value, step):
            calls[tag] = value

    pipeline._log_tensorboard(Writer(), {"replay_window": {
        "window_size": 10, "buffered": 8, "coverage": 0.5,
    }}, eval_state={}, done=1)
    assert calls["Replay/window_size"] == 10
    assert calls["Replay/buffered"] == 8
    assert calls["Replay/coverage"] == 0.5


def test_file_backend_rounds_never_repeat_the_previous_round(
    tmp_path, fake_games, monkeypatch
):
    from suzero_5dchess.game_store import GameStore

    rounds = []
    original = GameStore.write_last_round

    def record(self, state):
        rounds.append({**state, "games": [list(item) for item in state["games"]]})
        return original(self, state)

    monkeypatch.setattr(GameStore, "write_last_round", record)
    config = replace(
        config_for(tmp_path, replay_backend="file"),
        window=WindowConfig(
            enabled=True, taper_exponent=1, expand_per_row=0.5,
            max_samples=12, keep_samples=3,
        ),
    )
    summary = pipeline.run_training(config)
    assert len(rounds) >= 4
    for before, after in zip(rounds, rounds[1:]):
        assert set(map(tuple, before["selected"])).isdisjoint(
            map(tuple, after["selected"])
        )
    for iteration in summary["iterations"]:
        shuffle = iteration["training"]["shuffle"]
        assert shuffle["no_repeat_rows"] is True
        assert shuffle["rounds"] >= 2
    # A single epoch draws more than one round, so shuffles happen inside an
    # epoch rather than once per phase.
    first_phase = [state for state in rounds if state["iteration"] == 0]
    assert len(first_phase) >= 2

