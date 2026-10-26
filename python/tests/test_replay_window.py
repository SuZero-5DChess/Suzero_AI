"""Dynamic replay-window schedules, compact storage and sampling regressions."""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest
import torch

from suzero_5dchess.config import TrainConfig
from suzero_5dchess.protocol import VERY_SMALL
from suzero_5dchess.replay import ReplayBuffer, ReplayDataSource
from suzero_5dchess.replay_window import WindowConfig, compute_window_size
from suzero_5dchess.sample import PositionSample, collate_samples
from suzero_5dchess.sample_codec import encode_sample


def sample(index: int = 0) -> PositionSample:
    """Small but distinct, ragged records exercise every collated tensor."""

    boards = 1 + index % 3
    codes = np.zeros((boards, VERY_SMALL.board_squares), dtype=np.uint8)
    codes[:, index % VERY_SMALL.board_squares] = 1 + index % 12
    actions = np.array(
        [[0, 0, 1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 2, 1, 1, 0]],
        dtype=np.int64,
    )[: 1 + index % 2]
    submit = index % 2 == 0
    return PositionSample(
        board_codes=codes,
        l_coords=np.arange(boards, dtype=np.int64) - 1,
        t_coords=np.arange(boards, dtype=np.int64) + index,
        urgency=float(index + 1),
        side_to_move=index % 2,
        actions=actions,
        policy=np.array([0.75, 0.25], dtype=np.float32),
        has_submit=submit,
        value_target=(index % 3 - 1) * 0.5,
        player=index % 2,
        move_count=index,
        metadata={"weight": float(index + 1)},
    )


def make_buffer(count: int, capacity: int | None = None) -> ReplayBuffer:
    buffer = ReplayBuffer(capacity or max(1, count), VERY_SMALL)
    buffer.push([sample(index) for index in range(count)])
    return buffer


def assert_batch_matches(batch, buffer: ReplayBuffer, indices) -> None:
    expected = collate_samples(buffer.sample_at(indices), VERY_SMALL)
    assert batch is not None
    assert batch.sample_count == len(indices)
    assert set(batch.data) == set(expected)
    for key, value in expected.items():
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        assert batch.data[key].dtype == tensor.dtype, key
        assert batch.data[key].shape == tensor.shape, key
        assert torch.equal(batch.data[key], tensor), key


def take_epoch(source: ReplayDataSource):
    source.begin_epoch()
    batches = []
    while not source.is_exhausted():
        batch = source.next_batch()
        assert batch is not None
        batches.append(batch)
    assert source.next_batch() is None
    return batches


def batch_indices(batches) -> list[int]:
    return [int(value) - 1 for batch in batches for value in batch.data["urgency"]]


def test_window_defaults_are_opt_in():
    assert WindowConfig() == WindowConfig(
        enabled=False,
        expand_per_row=0.4,
        taper_exponent=0.65,
        taper_scale=None,
        max_samples=500_000,
        keep_samples=None,
        sampling="shuffle",
    )
    assert TrainConfig().window == WindowConfig()


@pytest.mark.parametrize("total", [0, 1, 50, 99, 100])
def test_window_at_or_below_default_scale_is_minimum(total):
    assert compute_window_size(total, 100, WindowConfig()) == 100


@pytest.mark.parametrize(
    ("total", "expected"), [(100, 100), (400, 180), (440, 187), (900, 260)]
)
def test_window_matches_hand_calculated_square_root_curve(total, expected):
    # e=.5 and scale=100 reduce the unclamped expression to 20 + 8*sqrt(N).
    window = WindowConfig(taper_exponent=0.5)
    assert compute_window_size(total, 100, window) == expected


def test_window_linear_exponent_floors_instead_of_rounding():
    window = WindowConfig(taper_exponent=1.0)
    assert compute_window_size(104, 100, window) == 101  # floor(101.6)
    assert compute_window_size(200, 100, window) == 140


@pytest.mark.parametrize("scale", [25, 400])
@pytest.mark.parametrize("total", [0, 10, 25, 100, 400, 900])
def test_window_custom_scale_uses_scale_not_minimum(scale, total):
    window = WindowConfig(taper_exponent=0.5, taper_scale=scale)
    raw = 100 + 0.4 * ((math.sqrt(total) - math.sqrt(scale)) / (0.5 / math.sqrt(scale)))
    expected = math.floor(max(100, min(500_000, raw)))
    assert compute_window_size(total, 100, window) == expected


def test_window_is_monotone_and_capped():
    window = WindowConfig(max_samples=700)
    totals = [0, 1, 99, 100, 101, 200, 1_000, 10_000, 100_000, 10**9]
    sizes = [compute_window_size(total, 100, window) for total in totals]
    assert sizes == sorted(sizes)
    assert all(100 <= size <= 700 for size in sizes)
    assert sizes[-1] == 700
    assert compute_window_size(10**9, 100, WindowConfig(expand_per_row=0)) == 100
    assert compute_window_size(10**9, 100, WindowConfig(max_samples=100)) == 100


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expand_per_row", -0.1),
        ("expand_per_row", float("nan")),
        ("expand_per_row", float("inf")),
        ("taper_exponent", 0),
        ("taper_exponent", -0.1),
        ("taper_exponent", 1.1),
        ("taper_exponent", float("nan")),
        ("taper_exponent", float("inf")),
        ("taper_scale", 0),
        ("taper_scale", -1),
        ("max_samples", 0),
        ("max_samples", -1),
        ("keep_samples", 0),
        ("keep_samples", -1),
        ("sampling", "replacement"),
    ],
)
def test_window_rejects_invalid_configuration(field, value):
    with pytest.raises(ValueError):
        WindowConfig(**{field: value})


@pytest.mark.parametrize(("total", "minimum"), [(-1, 100), (1, 0), (1, -1)])
def test_window_rejects_invalid_counts(total, minimum):
    with pytest.raises(ValueError):
        compute_window_size(total, minimum, WindowConfig())


def test_train_config_only_checks_window_capacity_when_enabled():
    small = WindowConfig(max_samples=100)
    assert TrainConfig(window=small).window.max_samples == 100
    with pytest.raises(ValueError, match="max_samples"):
        TrainConfig(window=WindowConfig(enabled=True, max_samples=100))
    default_size = TrainConfig().replay_buffer_size
    equal = WindowConfig(enabled=True, max_samples=default_size)
    assert TrainConfig(window=equal).window.max_samples == default_size


def test_train_config_revalidates_mutated_window():
    window = WindowConfig()
    window.taper_exponent = 0
    with pytest.raises(ValueError):
        TrainConfig(window=window)


def test_window_config_json_round_trip_and_old_dict_compatibility():
    window = WindowConfig(
        enabled=True,
        expand_per_row=0.7,
        taper_exponent=0.5,
        taper_scale=20_000,
        max_samples=80_000,
        keep_samples=123,
        sampling="uniform",
    )
    config = TrainConfig(
        window=window, replay_backend="memory", replay_buffer_size=50_000
    )
    data = json.loads(json.dumps(config.to_dict()))
    assert data["window"] == window.__dict__
    assert TrainConfig.from_dict(data).to_dict() == config.to_dict()
    del data["window"]
    assert TrainConfig.from_dict(data).window == WindowConfig()
    data["window"] = {"enabled": True}
    assert TrainConfig.from_dict(data).window == WindowConfig(enabled=True)
    assert TrainConfig.from_dict({}).window == WindowConfig()


def test_cli_maps_all_window_flags_and_preserves_defaults():
    from suzero_5dchess.cli import _build_config, build_parser

    parser = build_parser()
    assert _build_config(parser.parse_args(["train"])).window == WindowConfig()
    args = parser.parse_args(
        [
            "train", "--window",
            "--replay-size", "50000",
            "--window-expand-per-row", "0.7",
            "--window-taper-exponent", "0.5",
            "--window-taper-scale", "20000",
            "--window-max-samples", "80000",
            "--window-keep-samples", "123",
            "--window-sampling", "uniform",
            "--replay-backend", "memory",
        ]
    )
    assert _build_config(args).window == WindowConfig(
        enabled=True,
        expand_per_row=0.7,
        taper_exponent=0.5,
        taper_scale=20_000,
        max_samples=80_000,
        keep_samples=123,
        sampling="uniform",
    )
    # Supplying a tuning flag alone must not silently enable window mode.
    args = parser.parse_args(["train", "--window-max-samples", "100"])
    assert _build_config(args).window == WindowConfig(max_samples=100)


@pytest.mark.parametrize(
    "flags",
    [
        ["--window", "--window-max-samples", "100"],
        ["--window-expand-per-row", "-0.1"],
        ["--window-taper-exponent", "0"],
        ["--window-keep-samples", "0"],
    ],
)
def test_cli_runs_nested_window_validation(flags):
    from suzero_5dchess.cli import _build_config, build_parser

    args = build_parser().parse_args(["train", *flags])
    with pytest.raises(ValueError):
        _build_config(args)


def test_fifo_trim_and_generated_counter_are_independent():
    buffer = make_buffer(5, capacity=3)
    assert buffer.total_generated == 5
    # Capacity alone cannot evict rows the trainer never served.
    assert len(buffer) == 5
    assert [record.move_count for record in buffer.sample_at([0, 1, 2])] == [0, 1, 2]
    buffer.record_trained(range(5))
    assert buffer.trim_to(2) == 3
    assert [record.move_count for record in buffer.sample_at([0, 1])] == [3, 4]
    assert buffer.trim_to(20) == 0
    assert buffer.capacity == 3
    assert buffer.total_generated == 5
    with pytest.raises(ValueError):
        buffer.trim_to(-1)
    assert len(buffer) == 2
    assert buffer.trim_to(0) == 2
    assert buffer.trim_to(0) == 0
    assert len(buffer) == 0
    buffer.push([])
    assert buffer.total_generated == 5
    buffer.push([sample(5)])
    assert buffer.total_generated == 6
    assert buffer.sample_at([0])[0].move_count == 5


def test_sample_at_preserves_order_duplicates_and_sample_decode_semantics():
    buffer = make_buffer(5)
    indices = [3, 1, 3, 0]
    records = buffer.sample_at(indices)
    assert [record.move_count for record in records] == indices
    assert buffer.sample_at([]) == []
    for record in records:
        assert not record.board_codes.flags.writeable
        assert not record.policy.flags.writeable
        with pytest.raises(AttributeError):
            record.value_target = 1.0
    seed = 13
    chosen = np.random.default_rng(seed).choice(5, size=4, replace=False)
    sampled = buffer.sample(4, np.random.default_rng(seed))
    expected = collate_samples(buffer.sample_at(chosen), VERY_SMALL)
    actual = collate_samples(sampled, VERY_SMALL)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("index", [-1, 5])
def test_sample_at_rejects_out_of_range_indices(index):
    with pytest.raises(IndexError):
        make_buffer(5).sample_at([index])


def test_estimated_bytes_includes_blob_objects_and_list_storage():
    buffer = make_buffer(0, capacity=10)
    empty_bytes = buffer.estimated_bytes()
    assert empty_bytes == sys.getsizeof(buffer._blobs)
    buffer.push([sample(index) for index in range(8)])
    expected = sys.getsizeof(buffer._blobs) + sum(sys.getsizeof(blob) for blob in buffer._blobs)
    assert buffer.estimated_bytes() == expected
    assert expected > sum(len(blob) for blob in buffer._blobs)
    assert expected > empty_bytes
    buffer.record_trained(range(8))
    buffer.trim_to(0)
    assert buffer.estimated_bytes() == sys.getsizeof(buffer._blobs)
    assert buffer.estimated_bytes() < expected


def test_sixty_thousand_real_blobs_push_save_load_without_old_capacity_limit(monkeypatch, tmp_path):
    import suzero_5dchess.replay as module

    originals = [sample(index) for index in range(8)]
    # Generate genuine compressed records before replacing repeated CPU work.
    blobs = [encode_sample(record) for record in originals]
    buffer = ReplayBuffer(60_000, VERY_SMALL)
    with monkeypatch.context() as patch:
        patch.setattr(module, "validate_sample", lambda *args: None)
        patch.setattr(module, "encode_sample", lambda record: blobs[record.move_count])
        buffer.push(originals * 7_500)
        buffer.record_trained(range(60_000))
        buffer.push([originals[0]])
    assert len(buffer) == 60_000
    assert buffer.total_generated == 60_001
    state = buffer.state_dict()
    assert state["count"] == 60_000
    assert state["total_generated"] == 60_001
    assert state["offsets"].shape == (60_001,)
    assert state["offsets"][-1] == len(state["blobs"])
    path = tmp_path / "large-replay.pt"
    torch.save(state, path)
    restored = ReplayBuffer(60_000, VERY_SMALL)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    assert len(restored) == 60_000
    assert restored.total_generated == 60_001
    indices = [0, 7, 50_000, 59_998, 59_999]
    assert [record.move_count for record in restored.sample_at(indices)] == [
        (index + 1) % 8 for index in indices
    ]
    expected = collate_samples(buffer.sample_at(indices), VERY_SMALL)
    actual = collate_samples(restored.sample_at(indices), VERY_SMALL)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    assert restored.trim_to(3) == 59_997
    assert restored.total_generated == 60_001
    restored.push([sample(8)])
    assert restored.total_generated == 60_002


@pytest.mark.parametrize("format_name", ["compact", "legacy_dict", "legacy_sample", "legacy_tensor", "legacy_planes"])
@pytest.mark.parametrize("counter", [None, 91])
def test_load_old_and_new_formats_with_capacity_and_counter_compatibility(format_name, counter):
    originals = [sample(index) for index in range(4)]
    if format_name == "compact":
        state = make_buffer(4).state_dict()
        state.pop("total_generated")
    else:
        records = []
        for original in originals:
            if format_name == "legacy_sample":
                records.append(original)
                continue
            record = original.as_dict()
            if format_name == "legacy_planes":
                record["board_planes"] = original.board_planes
                del record["board_codes"]
            if format_name == "legacy_tensor":
                record = {
                    key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                    for key, value in record.items()
                }
            records.append(record)
        state = {"capacity": 100, "samples": records}
    if counter is not None:
        state["total_generated"] = counter
    restored = make_buffer(1, capacity=2)
    restored.load_state_dict(state)
    assert restored.capacity == 2
    # Nothing in these snapshots proves a row was trained, so the smaller
    # capacity cannot discard any of them on load.
    assert len(restored) == 4
    assert restored.total_generated == (0 if counter is None else counter)
    assert [record.move_count for record in restored.sample_at([0, 1])] == [0, 1]
    actual = collate_samples(restored.sample_at([0, 1]), VERY_SMALL)
    expected = collate_samples(originals[:2], VERY_SMALL)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    restored.push([sample(4)])
    assert restored.total_generated == (0 if counter is None else counter) + 1


@pytest.mark.parametrize("keep", [None, 12, 20])
def test_shuffle_epochs_match_seeded_permutations_without_repeats(keep):
    buffer = make_buffer(12)
    source = ReplayDataSource(buffer, 3, 99, VERY_SMALL, seed=17, sampling="shuffle", keep_samples=keep)
    twin = ReplayDataSource(buffer, 3, 99, VERY_SMALL, seed=17, sampling="shuffle", keep_samples=keep)
    orders = []
    for epoch in (1, 2, 3):
        batches = take_epoch(source)
        assert source.steps_per_epoch == 4
        assert all(batch.sample_count == 3 for batch in batches)
        order = batch_indices(batches)
        expected = np.random.default_rng(17 + epoch).permutation(np.arange(12))
        assert order == expected.tolist()
        assert sorted(order) == list(range(12))
        assert batch_indices(take_epoch(twin)) == order
        for step, batch in enumerate(batches):
            assert_batch_matches(batch, buffer, expected[step * 3 : (step + 1) * 3])
        orders.append(order)
    assert orders[0] != orders[1]
    assert not source.has_validation()
    assert list(source.validation_batches()) == []


def test_shuffle_subset_is_selected_once_and_fixed_across_epochs():
    buffer = make_buffer(20, capacity=30)
    seed = 37
    pool = np.random.default_rng(seed).choice(20, size=8, replace=False)
    source = ReplayDataSource(buffer, 2, 99, VERY_SMALL, seed=seed, sampling="shuffle", keep_samples=8)
    # Later appends must not rebuild the source's construction-time pool.
    buffer.push([sample(index) for index in range(20, 24)])
    orders = []
    for epoch in (1, 2, 3):
        batches = take_epoch(source)
        assert source.steps_per_epoch == 4
        order = batch_indices(batches)
        expected = np.random.default_rng(seed + epoch).permutation(pool)
        assert order == expected.tolist()
        assert set(order) == set(pool)
        assert len(order) == len(set(order)) == 8
        orders.append(order)
    assert orders[0] != orders[1]


@pytest.mark.parametrize(
    ("count", "batch_size", "requested", "keep", "sizes"),
    [
        (13, 4, 99, None, [4, 4, 4]),
        (12, 3, 2, None, [3, 3]),
        (3, 8, 99, None, [3]),
        (20, 8, 99, 3, [3]),
        (0, 4, 99, None, []),
    ],
)
def test_shuffle_step_cap_partial_batch_and_empty_pool(count, batch_size, requested, keep, sizes):
    source = ReplayDataSource(
        make_buffer(count), batch_size, requested, VERY_SMALL,
        sampling="shuffle", keep_samples=keep,
    )
    for _ in range(2):
        batches = take_epoch(source)
        assert source.steps_per_epoch == len(sizes)
        assert [batch.sample_count for batch in batches] == sizes
        indices = batch_indices(batches)
        assert len(indices) == len(set(indices))


@pytest.mark.parametrize("count", [0, 2, 13])
@pytest.mark.parametrize("explicit_uniform", [False, True])
def test_uniform_matches_original_choice_path_tensor_for_tensor(count, explicit_uniform):
    buffer = make_buffer(count)
    options = {"sampling": "uniform", "keep_samples": 1} if explicit_uniform else {}
    source = ReplayDataSource(buffer, 4, 99, VERY_SMALL, seed=31, **options)
    for epoch in (1, 2, 3):
        batches = take_epoch(source)
        expected_steps = min(99, max(1, count // 4)) if count else 0
        assert source.steps_per_epoch == expected_steps
        assert len(batches) == expected_steps
        rng = np.random.default_rng(31 + epoch)
        for batch in batches:
            # This is the original uniform implementation's RNG call, not a
            # call through ReplayBuffer.sample or ReplayDataSource internals.
            indices = rng.choice(count, size=min(4, count), replace=False)
            assert_batch_matches(batch, buffer, indices)


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"steps_per_epoch": 0},
        {"sampling": "replacement"},
        {"sampling": "shuffle", "keep_samples": 0},
        {"sampling": "shuffle", "keep_samples": -1},
    ],
)
def test_data_source_rejects_invalid_sampling_options(options):
    values = {"batch_size": 2, "steps_per_epoch": 3}
    values.update(options)
    with pytest.raises(ValueError):
        ReplayDataSource(make_buffer(4), spec=VERY_SMALL, **values)


def test_launcher_exposes_the_no_repeat_rows_switch():
    from pathlib import Path

    launcher = Path(__file__).resolve().parents[2] / "run_train.bat"
    text = launcher.read_text(encoding="utf-8")
    assert 'if not defined NO_REPEAT_ROWS set "NO_REPEAT_ROWS=0"' in text
    assert (
        'if "%NO_REPEAT_ROWS%"=="0" set "DEFAULT_ARGS=%DEFAULT_ARGS% --repeat-rows"'
        in text
    )
    assert (
        'if not "%NO_REPEAT_ROWS%"=="0" set '
        '"DEFAULT_ARGS=%DEFAULT_ARGS% --no-repeat-rows"' in text
    )


def test_buffer_progress_is_relative_to_each_pushed_game_and_persists():
    from suzero_5dchess.endgame_sampling import EndgameSamplingConfig

    buffer = ReplayBuffer(8, VERY_SMALL)
    buffer.push([sample(index) for index in range(4)])  # move_count 0..3
    buffer.push([sample(index) for index in range(2)])  # a second game
    assert np.allclose(buffer.progress, [0.0, 1 / 3, 2 / 3, 1.0, 0.0, 1.0])

    restored = ReplayBuffer(8, VERY_SMALL)
    restored.load_state_dict(buffer.state_dict())
    assert np.allclose(restored.progress, buffer.progress)

    # Old snapshots without progress fall back to the stored move counts and
    # the window-wide maximum, because the game boundaries are gone.
    state = buffer.state_dict()
    state.pop("progress")
    legacy = ReplayBuffer(8, VERY_SMALL)
    legacy.load_state_dict(state)
    assert np.allclose(legacy.progress, [0.0, 1 / 3, 2 / 3, 1.0, 0.0, 1 / 3])
    shard = ReplayBuffer(8, VERY_SMALL)
    shard.load_state_dict(state, derive_progress=False)
    assert shard.progress == [0.0] * 6


def test_replay_data_source_spends_draws_on_late_rows():
    from suzero_5dchess.endgame_sampling import EndgameSamplingConfig

    buffer = ReplayBuffer(8, VERY_SMALL)
    buffer.push([sample(index) for index in range(8)])  # move_count 0..7
    seen = []
    original = buffer.sample_at

    def record(indices):
        seen.extend(int(index) for index in indices)
        return original(indices)

    buffer.sample_at = record
    source = ReplayDataSource(
        buffer, 1, 1, VERY_SMALL, seed=4,
        endgame=EndgameSamplingConfig(enabled=True, strength=100.0, exponent=2.0),
    )
    for _ in range(40):
        source.begin_epoch()
        assert source.next_batch() is not None
    # Late indices carry most of the weight (7 ≈ 101, 6 ≈ 74, 5 ≈ 51, 0 = 1).
    assert seen.count(7) > 3 * seen.count(0)
    assert np.mean(seen) > 4.5
