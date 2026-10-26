"""Per-sample training-count tracking across the replay backends."""

from __future__ import annotations

import numpy as np
import pytest

from suzero_5dchess import train_counts
from suzero_5dchess.file_replay import FileReplayDataSource, publish_shuffle
from suzero_5dchess.game_store import GameStore
from suzero_5dchess.protocol import VERY_SMALL
from suzero_5dchess.replay import ReplayBuffer, ReplayDataSource
from suzero_5dchess.sample import PositionSample


def sample(index: int = 0) -> PositionSample:
    codes = np.zeros((1, VERY_SMALL.board_squares), dtype=np.uint8)
    codes[0, index % VERY_SMALL.board_squares] = 1 + index % 12
    return PositionSample(
        board_codes=codes,
        l_coords=np.array([0], dtype=np.int64),
        t_coords=np.array([index + 1], dtype=np.int64),
        urgency=float(index + 1),
        side_to_move=index % 2,
        actions=np.array([[0, 0, 1, 0, 1, 0, 1, 0]], dtype=np.int64),
        policy=np.array([1.0], dtype=np.float32),
        has_submit=False,
        value_target=(index % 3 - 1) * 0.5,
        player=index % 2,
        move_count=index,
    )


def make_buffer(count: int, capacity: int | None = None) -> ReplayBuffer:
    buffer = ReplayBuffer(capacity or max(1, count), VERY_SMALL)
    buffer.push([sample(index) for index in range(count)])
    return buffer


def drain(source, epochs: int = 4) -> int:
    """Run every step the source is willing to serve; return sample count."""

    served = 0
    for _ in range(epochs):
        source.begin_epoch()
        while not source.is_exhausted():
            batch = source.next_batch()
            if batch is None:
                break
            served += batch.sample_count
    return served


def test_summarize_reports_untrained_rows_and_top_bucket():
    summary = train_counts.summarize([0, 0, 1, 2, 2, 3, 11])

    assert summary["samples"] == 7
    assert summary["trained"] == 5
    assert summary["untrained"] == 2
    assert summary["total_trains"] == 19
    assert summary["mean"] == round(19 / 7, 4)
    assert summary["median"] == 2
    assert summary["p90"] == 11
    assert summary["max"] == 11
    assert list(summary["histogram"]) == train_counts.HISTOGRAM_KEYS
    assert summary["histogram"]["0"] == 2
    assert summary["histogram"]["2"] == 2
    assert summary["histogram"]["10+"] == 1


def test_summarize_empty_window_is_zeroed():
    summary = train_counts.summarize([])

    assert summary["samples"] == 0
    assert summary["trained"] == summary["untrained"] == 0
    assert summary["histogram"] == {key: 0 for key in train_counts.HISTOGRAM_KEYS}


def test_record_ignores_rows_outside_the_window():
    counts = [0, 0]

    assert train_counts.record(counts, [0, 1, 5, -1, 0]) == 3
    assert counts == [2, 1]


@pytest.mark.parametrize("sampling", ["uniform", "shuffle"])
def test_replay_data_source_counts_every_served_row(sampling):
    buffer = make_buffer(6, capacity=8)
    source = ReplayDataSource(buffer, 2, 3, VERY_SMALL, seed=5, sampling=sampling)

    served = drain(source, epochs=2)

    assert served == 12
    assert sum(buffer.train_counts) == served
    if sampling == "shuffle":
        # Each epoch is one full permutation, so every row is served twice.
        assert buffer.train_counts == [2] * 6


def test_replay_buffer_state_round_trips_counters():
    buffer = make_buffer(5)
    buffer.record_trained([1, 1, 3])
    assert buffer.train_counts == [0, 2, 0, 1, 0]

    state = buffer.state_dict()
    assert np.asarray(state["train_counts"]).tolist() == [0, 2, 0, 1, 0]

    restored = ReplayBuffer(5, VERY_SMALL)
    restored.load_state_dict(state)
    assert restored.train_counts == [0, 2, 0, 1, 0]


def test_replay_trim_drops_counters_with_their_rows():
    buffer = make_buffer(4)
    buffer.record_trained([0, 1, 1])

    assert buffer.trim_to(2) == 2
    assert buffer.train_counts == [0, 0]

    buffer.push([sample(9)])
    assert buffer.train_counts == [0, 0, 0]


def test_evictable_prefix_stops_at_the_first_untrained_row():
    counts = [2, 1, 0, 3]

    assert train_counts.evictable_prefix(counts, 0) == 0
    assert train_counts.evictable_prefix(counts, 1) == 1
    assert train_counts.evictable_prefix(counts, 2) == 2
    # The untrained row at index 2 protects itself and everything behind it.
    assert train_counts.evictable_prefix(counts, 4) == 2
    assert train_counts.evictable_prefix([], 3) == 0


def test_untrained_first_is_a_stable_partition():
    order = np.array([4, 0, 3, 1, 2])

    assert train_counts.untrained_first(order, [1, 0, 1, 0, 1]).tolist() == [
        3, 1, 4, 0, 2
    ]
    assert train_counts.untrained_first(order, [1] * 5).tolist() == order.tolist()
    assert train_counts.untrained_first(order, [0] * 5).tolist() == order.tolist()
    assert train_counts.untrained_first(
        np.empty(0, dtype=np.int64), [0]
    ).size == 0


def test_replay_buffer_keeps_untrained_rows_above_capacity():
    buffer = ReplayBuffer(3, VERY_SMALL)
    buffer.push([sample(index) for index in range(5)])
    # Fresh rows were never served, so the capacity cannot evict any of them.
    assert len(buffer) == 5
    assert buffer.train_counts == [0] * 5

    buffer.record_trained([0, 1])
    buffer.push([sample(5)])
    # Only the two served rows leave; untrained row 2 protects rows 3..5.
    assert len(buffer) == 4
    assert buffer.train_counts == [0, 0, 0, 0]
    assert [item.move_count for item in buffer.sample_at(range(4))] == [2, 3, 4, 5]

    assert buffer.trim_to(1) == 0
    assert len(buffer) == 4

    buffer.record_trained(range(4))
    assert buffer.trim_to(1) == 3
    assert [item.move_count for item in buffer.sample_at([0])] == [5]


def test_replay_buffer_load_keeps_untrained_rows_above_capacity():
    buffer = make_buffer(4)
    # The first row has a pass, the rest do not: the front trim must stop at
    # row 1 instead of discarding three never-served samples.
    buffer.record_trained([0])

    restored = ReplayBuffer(2, VERY_SMALL)
    restored.load_state_dict(buffer.state_dict())

    assert len(restored) == 3
    assert restored.train_counts == [0, 0, 0]
    assert [item.move_count for item in restored.sample_at(range(3))] == [1, 2, 3]


def test_shuffle_epoch_serves_untrained_rows_first():
    buffer = make_buffer(6)
    buffer.record_trained([0, 1, 2])
    source = ReplayDataSource(buffer, 2, 2, VERY_SMALL, seed=1, sampling="shuffle")

    source.begin_epoch()
    while not source.is_exhausted():
        assert source.next_batch() is not None

    # Four served rows: the three new rows plus one repeat of a trained row.
    assert buffer.train_counts[3:] == [1, 1, 1]
    assert sum(buffer.train_counts[:3]) == 4


def game_samples(start: int, count: int) -> list[PositionSample]:
    return [sample(index) for index in range(start, start + count)]


def test_game_store_counts_follow_served_rounds(tmp_path):
    store = GameStore(tmp_path / "store", VERY_SMALL, 16)
    store.append(game_samples(0, 4), game_id=1)
    store.append(game_samples(4, 4), game_id=2)

    source = store.training_source(2, 2, VERY_SMALL, seed=3)
    served = drain(source)

    assert served == 8
    assert sorted(store.train_counts) == [1] * 8


def test_game_store_counter_sidecar_maps_rows_after_partial_trim(tmp_path):
    store = GameStore(tmp_path / "store", VERY_SMALL, 16)
    store.append(game_samples(0, 8), game_id=7)
    for row in range(8):
        store.record_trained([row] * (row + 1))
    assert store.train_counts == list(range(1, 9))

    path = tmp_path / "train_counts.npz"
    store.save_train_counts(path)
    store.trim_to(4)
    assert store.train_counts == [5, 6, 7, 8]

    restored = GameStore(tmp_path / "store", VERY_SMALL, 16)
    restored.scan()
    assert restored.load_train_counts(path) == 4
    assert restored.train_counts == [5, 6, 7, 8]


def test_game_store_counter_sidecar_ignores_rows_that_left_the_window(tmp_path):
    store = GameStore(tmp_path / "store", VERY_SMALL, 16)
    store.append(game_samples(0, 4), game_id=1)
    store.append(game_samples(4, 4), game_id=2)
    store.record_trained([0, 1, 2, 3])

    path = tmp_path / "train_counts.npz"
    store.save_train_counts(path)
    store.trim_to(4)  # drops game 1 entirely
    assert store.train_counts == [0, 0, 0, 0]

    store.record_trained([0, 1])
    # Game 2 is still present, so its (zero) counters merge without erasing
    # the two rows recorded after the save.
    assert store.load_train_counts(path) == 4
    assert store.train_counts == [1, 1, 0, 0]


def test_file_replay_counts_rows_split_across_shards(tmp_path):
    buffer = make_buffer(8, capacity=10)
    publish_shuffle(buffer, tmp_path, iteration=1, seed=7, rows_per_file=3)
    source = FileReplayDataSource(
        tmp_path, 4, 99, VERY_SMALL, train_counts=buffer.train_counts
    )

    served = drain(source)

    assert served == 8
    assert buffer.train_counts == [1] * 8
