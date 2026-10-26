"""Row-log replay store: shared appends, row-granular trimming, shuffle rounds."""

import numpy as np
import pytest

from suzero_5dchess.game_store import GAME_FILE_FORMAT, GAME_STORE_FORMAT, GameStore
from suzero_5dchess.protocol import VERY_SMALL
from suzero_5dchess.row_log import HEADER, LOG_NAME, MAGIC, VERSION
from suzero_5dchess.sample import PositionSample
from suzero_5dchess.sample_codec import encode_sample
from suzero_5dchess.self_play import SelfPlayResult


def sample(index):
    # A unique single piece per sample makes rows identifiable after collation
    # without relying on fields the training batch does not carry.
    codes = np.zeros((1, 16), dtype=np.uint8)
    codes[0, index % 16] = 1
    return PositionSample(
        board_codes=codes,
        l_coords=np.array([0], dtype=np.int64),
        t_coords=np.array([1], dtype=np.int64),
        urgency=24.0,
        side_to_move=0,
        actions=np.array([[0, 0, 1, 0, 1, 0, 1, 0]], dtype=np.int64),
        policy=np.array([1.0], dtype=np.float32),
        has_submit=False,
        move_count=index,
    )


def game(start, count=4):
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


def store_for(tmp_path, capacity=64):
    return GameStore(tmp_path / "replay_store", VERY_SMALL, capacity)


def test_append_scan_and_state_round_trip(tmp_path):
    store = store_for(tmp_path)
    assert store.append(game(0).samples, game_id=3, iteration=1) == 4
    assert store.append(game(4).samples, game_id=4, iteration=1) == 4
    assert len(store) == 8
    assert store.total_generated == 8
    assert store.next_game_id == 5
    assert [entry.game_id for entry in store.games] == [3, 4]

    files = sorted(path.name for path in (tmp_path / "replay_store").iterdir())
    assert files == ["replay.lock", "replay.log"]

    # A fresh process rebuilds the window from the frame headers alone.
    scanned = GameStore(tmp_path / "replay_store", VERY_SMALL, 64)
    scanned.scan()
    assert len(scanned) == 8
    assert scanned.total_generated == 8
    assert scanned.next_game_id == 5
    assert [entry.iteration for entry in scanned.games] == [1, 1]

    state = store.state_dict()
    assert state["format"] == GAME_STORE_FORMAT
    assert state["count"] == 8
    assert state["frames"] == 2
    # The window's blobs never live in the checkpoint.
    assert "blobs" not in state

    restored = GameStore(tmp_path / "replay_store", VERY_SMALL, 64)
    restored.scan()
    restored.load_state_dict({**state, "total_generated": 40, "next_game_id": 9})
    assert restored.total_generated == 40
    assert restored.next_game_id == 9


def test_legacy_store_state_loads_without_its_file_list(tmp_path):
    store = store_for(tmp_path)
    store.append(game(0).samples, game_id=0)
    store.load_state_dict({
        "format": "suzero_game_store_v1",
        "capacity": 64,
        "count": 4,
        "total_generated": 1234,
        "next_game_id": 77,
        "files": [{
            "name": "g000000000000-00000004.npz",
            "game_id": 0,
            "count": 4,
            "iteration": -1,
        }],
        "shuffle_state": {"rounds": 2},
    })
    assert store.total_generated == 1234
    assert store.next_game_id == 77
    assert store.shuffle_state == {"rounds": 2}
    assert len(store) == 4
    with pytest.raises(ValueError, match="not a file game store"):
        store.load_state_dict({"format": "something-else"})


def test_append_never_overwrites_a_game(tmp_path):
    store = store_for(tmp_path)
    store.append(game(0).samples, game_id=7)
    with pytest.raises(FileExistsError):
        store.append(game(4).samples, game_id=7)
    assert len(store) == 4
    assert len(store.games) == 1


def test_empty_game_writes_no_file(tmp_path):
    store = store_for(tmp_path)
    result = game(0)
    result.samples = []
    result.sample_count = 0
    assert store.append_result(0, result) == 0
    assert len(store) == 0
    assert list((tmp_path / "replay_store").iterdir()) == []
    assert result.persisted is True


def test_append_result_releases_rows_but_keeps_the_count(tmp_path):
    store = store_for(tmp_path)
    result = game(0)
    assert store.append_result(0, result, iteration=2) == 4
    assert result.persisted is True
    assert result.samples == []
    assert result.sample_count == 4
    assert len(store) == 4
    # Releasing is opt-out for callers that still need the rows.
    kept = game(4)
    store.append_result(1, kept, release=False)
    assert len(kept.samples) == 4


def test_trim_to_drops_the_oldest_rows_and_may_cut_a_game(tmp_path):
    store = store_for(tmp_path)
    for index in range(3):
        store.append(game(index * 4).samples, game_id=index)
    assert len(store) == 12
    store.record_trained(range(12))

    # The unit is a row, not a game: game 0 goes and game 1 keeps its last two
    # rows, which keep their original row numbers.
    assert store.trim_to(6) == 6
    assert len(store) == 6
    assert [entry.game_id for entry in store.games] == [1, 2]
    assert [entry.count for entry in store.games] == [2, 4]
    assert [entry.skip for entry in store.games] == [2, 0]
    assert [sample.move_count for sample in store.iter_samples()] == [
        6, 7, 8, 9, 10, 11
    ]
    assert store.total_generated == 12

    # The rewritten log holds only what survived, and a fresh index agrees.
    log = tmp_path / "replay_store" / LOG_NAME
    scanned = GameStore(tmp_path / "replay_store", VERY_SMALL, 64)
    scanned.scan()
    assert len(scanned) == 6
    assert [(entry.game_id, entry.skip) for entry in scanned.games] == [(1, 2), (2, 0)]
    assert log.stat().st_size == scanned.estimated_bytes()


def test_trim_keeps_a_buffer_and_drops_a_batch(tmp_path):
    store = store_for(tmp_path)
    for index in range(10):
        store.append(game(index * 4).samples, game_id=index)
    assert len(store) == 40
    store.record_trained(range(40))

    # Five rows over the target is inside the seven-row buffer: nothing moves.
    assert store.trim_to(35) == 0
    assert len(store) == 40

    # Ten rows over the target is outside it: one batch goes at once.
    assert store.trim_to(30) == 10
    assert len(store) == 30


def test_appends_drop_a_batch_instead_of_trimming_every_game(tmp_path):
    store = store_for(tmp_path, capacity=8)
    for index in range(2):
        store.append(game(index * 4).samples, game_id=index)
    assert len(store) == 8
    store.record_trained(range(8))

    # 12 rows against a capacity of 8 with a one-row buffer: the oldest five
    # rows go, leaving room for the next games before the log is rewritten.
    store.append(game(8).samples, game_id=2)
    assert len(store) == 7
    assert [(entry.game_id, entry.skip, entry.count) for entry in store.games] == [
        (1, 1, 3),
        (2, 0, 4),
    ]


def test_trim_keeps_untrained_rows_above_the_target(tmp_path):
    store = store_for(tmp_path)
    store.append(game(0).samples, game_id=0)
    store.append(game(4).samples, game_id=1)
    assert len(store) == 8

    # Nothing was served yet: no row is evictable, so the whole window stays.
    assert store.trim_to(2) == 0
    assert len(store) == 8
    assert store.train_counts == [0] * 8

    store.record_trained(range(4))
    # Only the served game leaves; the unserved front row of game 1 would have
    # blocked the rest anyway.
    assert store.trim_to(2) == 4
    assert [sample.move_count for sample in store.iter_samples()] == [4, 5, 6, 7]
    assert store.train_counts == [0] * 4


def test_append_capacity_keeps_untrained_rows(tmp_path):
    store = store_for(tmp_path, capacity=4)
    store.append(game(0).samples, game_id=0)
    store.append(game(4).samples, game_id=1)
    # Neither game was served, so exceeding the capacity evicts nothing.
    assert len(store) == 8

    store.record_trained(range(4))
    store.append(game(8).samples, game_id=2)
    # Now the served game can leave; the unserved rows stay above capacity.
    assert [entry.game_id for entry in store.games] == [1, 2]
    assert len(store) == 8
    assert store.train_counts == [0] * 8


def test_next_game_id_follows_the_highest_id_not_the_last_append(tmp_path):
    store = store_for(tmp_path)
    # Parallel self-play workers append interleaved id ranges.
    store.append(game(0).samples, game_id=5)
    store.append(game(4).samples, game_id=2)
    assert store.next_game_id == 6
    assert store.append(game(8).samples) == 4
    assert store.games[-1].game_id == 6


def test_refresh_picks_up_another_writers_files(tmp_path):
    parent = store_for(tmp_path)
    parent.append(game(0).samples, game_id=0)

    # Simulate a self-play child appending directly to the shared directory.
    child = GameStore(parent.directory, VERY_SMALL, 1, enforce_capacity=False)
    child.append(game(4).samples, game_id=1)
    child.append(game(8).samples, game_id=2)

    assert parent.refresh() == 8
    assert parent.refresh() == 0
    assert len(parent) == 12
    assert parent.total_generated == 12
    assert parent.next_game_id == 3


def batch_ids(batch):
    planes = batch.data["board_planes"].numpy()
    return [int(np.argmax(row[0, 0])) for row in planes]


def test_training_source_visits_every_row_once(tmp_path):
    store = store_for(tmp_path, capacity=64)
    expected = []
    for index in range(3):
        produced = game(index * 2, count=2)
        store.append(produced.samples, game_id=index)
        expected.extend(index * 2 + offset for offset in range(2))

    source = store.training_source(
        2, 3, VERY_SMALL, seed=11,
    )
    source.begin_epoch()
    seen = []
    while not source.is_exhausted():
        batch = source.next_batch()
        assert batch is not None
        seen.extend(batch_ids(batch))
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == 6
    assert source.summary()["backend"] == "file"
    assert source.summary()["games"] == 3
    assert source.summary()["generation"].startswith("generation-")

    # The same seed reproduces the same generation and row order.
    other = store.training_source(2, 3, VERY_SMALL, seed=11)
    assert other.generation == source.generation
    other.begin_epoch()
    assert batch_ids(other.next_batch()) == seen[:2]


def test_training_source_partial_rounds_move_across_games(tmp_path):
    """A phase that reads part of its round must not always take the oldest games.

    Training consumes ``epochs * steps * batch`` rows, which can be much less
    than a full round.  If the round is grouped by game index the served prefix
    is the same (oldest) games every phase, so those rows are retrained while
    newer rows are trimmed before they are ever served.
    """

    store = store_for(tmp_path, capacity=4096)
    for index in range(200):
        store.append(game(index * 4).samples, game_id=index)

    def drain(source):
        source.begin_epoch()
        while not source.is_exhausted():
            if source.next_batch() is None:
                break

    first = store.training_source(
        8, 5, VERY_SMALL, iteration=0, seed=3, no_repeat_rows=False,
    )
    drain(first)
    served = {row for row, count in enumerate(store.train_counts) if count}
    assert len(served) == 40

    second = store.training_source(
        8, 5, VERY_SMALL, iteration=1, seed=3 + 1009, no_repeat_rows=False,
    )
    drain(second)
    fresh = {
        row for row, count in enumerate(store.train_counts)
        if count and row not in served
    }
    assert len(fresh) >= 20


def test_training_source_keep_samples_is_a_bounded_pool(tmp_path):
    store = store_for(tmp_path, capacity=64)
    for index in range(4):
        store.append(game(index * 4).samples, game_id=index)

    rounds = []
    original = store.write_last_round

    def record(state):
        rounds.append(state)
        return original(state)

    store.write_last_round = record
    source = store.training_source(
        4, 4, VERY_SMALL, seed=3, keep_samples=5,
    )
    source.begin_epoch()
    seen = []
    while not source.is_exhausted():
        batch = source.next_batch()
        if batch is None:
            break
        seen.extend(batch_ids(batch))
    # One epoch runs its full four-step budget; rounds of five rows are drawn
    # as needed, and each round excludes the rows drawn by the one before it.
    assert len(seen) == 16
    assert source.pool_samples == 16
    assert len(rounds) >= 2
    assert all(state["count"] == 5 for state in rounds)
    for before, after in zip(rounds, rounds[1:]):
        assert set(map(tuple, before["selected"])).isdisjoint(
            map(tuple, after["selected"])
        )
    assert source.summary()["no_repeat_rows"] is True
    assert source.summary()["rounds"] >= 1


def test_training_source_rounds_never_repeat_the_previous_round(tmp_path):
    store = store_for(tmp_path, capacity=64)
    for index in range(4):
        store.append(game(index * 4).samples, game_id=index)

    rounds = []
    original = store.write_last_round

    def record(state):
        rounds.append(state)
        return original(state)

    store.write_last_round = record
    source = store.training_source(
        4, 3, VERY_SMALL, iteration=3, seed=17, keep_samples=5,
    )
    for epoch in range(2):
        source.begin_epoch()
        while not source.is_exhausted():
            if source.next_batch() is None:
                break
    assert len(rounds) >= 3
    for before, after in zip(rounds, rounds[1:]):
        assert set(map(tuple, before["selected"])).isdisjoint(
            map(tuple, after["selected"])
        )
    assert all(
        state["iteration"] == 3 and state["phase_seed"] == 17 for state in rounds
    )


def test_next_phase_excludes_the_previous_phase_round(tmp_path):
    store = store_for(tmp_path, capacity=64)
    for index in range(4):
        store.append(game(index * 4).samples, game_id=index)

    first = store.training_source(
        4, 2, VERY_SMALL, iteration=0, seed=5, keep_samples=6,
    )
    first.begin_epoch()
    while not first.is_exhausted():
        assert first.next_batch() is not None
    previous = store.read_last_round()
    assert previous is not None

    store.training_source(
        4, 2, VERY_SMALL, iteration=1, seed=5 + 1009, keep_samples=6,
    )
    current = store.read_last_round()
    assert current is not None
    assert current["round"] == 0 and current["iteration"] == 1
    # Only rows the previous phase actually served are excluded.  The last
    # round it published was not consumed whole, so its unread rows stay
    # eligible instead of being skipped a second time.
    row_of: dict[tuple[int, int], int] = {}
    row = 0
    for entry in store.games:
        for local in range(entry.skip, entry.skip + entry.count):
            row_of[(entry.game_id, local)] = row
            row += 1
    trained_previous = {
        tuple(item)
        for item in previous["selected"]
        if store.train_counts[row_of[tuple(item)]] > 0
    }
    assert trained_previous
    assert set(map(tuple, current["selected"])).isdisjoint(trained_previous)


def test_next_phase_reoffers_untrained_rows_from_the_previous_round(tmp_path):
    store = store_for(tmp_path, capacity=64)
    for index in range(2):
        store.append(game(index * 4).samples, game_id=index)

    first = store.training_source(
        2, 1, VERY_SMALL, iteration=0, seed=5, keep_samples=4,
    )
    first.begin_epoch()
    while not first.is_exhausted():
        assert first.next_batch() is not None
    previous = store.read_last_round()
    assert previous is not None
    row_of: dict[tuple[int, int], int] = {}
    row = 0
    for entry in store.games:
        for local in range(entry.skip, entry.skip + entry.count):
            row_of[(entry.game_id, local)] = row
            row += 1
    selected = {tuple(item) for item in previous["selected"]}
    served = {item for item in selected if store.train_counts[row_of[item]] > 0}
    unread = selected - served
    assert served and unread

    store.training_source(
        2, 1, VERY_SMALL, iteration=1, seed=5 + 1009, keep_samples=6,
    )
    current = store.read_last_round()
    assert current is not None
    chosen = {tuple(item) for item in current["selected"]}
    # Only rows the previous phase actually served are excluded; its unread
    # rows stay eligible instead of being skipped a second time.
    assert unread <= chosen
    assert served.isdisjoint(chosen)


def test_retry_reuses_the_phase_first_round(tmp_path):
    store = store_for(tmp_path, capacity=64)
    store.append(game(0).samples, game_id=0)
    state = tmp_path / "replay_store" / "last_round.json"
    first = store.training_source(
        2, 2, VERY_SMALL, iteration=5, seed=41, keep_samples=3,
    )
    recorded = state.read_bytes()
    retry = store.training_source(
        2, 2, VERY_SMALL, iteration=5, seed=41, keep_samples=3,
    )
    assert retry.generation == first.generation
    assert retry.summary()["round_samples"] == first.summary()["round_samples"]
    assert state.read_bytes() == recorded


def test_evicted_games_drop_out_of_the_previous_round(tmp_path):
    store = store_for(tmp_path, capacity=64)
    store.append(game(0).samples, game_id=0)
    first = store.training_source(4, 1, VERY_SMALL, iteration=0, seed=2)
    first.begin_epoch()
    assert first.next_batch() is not None
    previous = store.read_last_round()
    assert {row[0] for row in previous["selected"]} == {0}

    store.append(game(4).samples, game_id=1)
    assert store.trim_to(4) == 4
    store.training_source(4, 1, VERY_SMALL, iteration=1, seed=2 + 1009)
    current = store.read_last_round()
    assert {row[0] for row in current["selected"]} == {1}


@pytest.mark.parametrize("damage", ["format", "count", "row"])
def test_invalid_round_state_is_rejected(tmp_path, damage):
    import json

    store = store_for(tmp_path, capacity=64)
    store.append(game(0).samples, game_id=0)
    store.training_source(2, 1, VERY_SMALL, seed=1)
    path = tmp_path / "replay_store" / "last_round.json"
    data = json.loads(path.read_text())
    if damage == "format":
        data["format"] = "not-the-round-format"
    elif damage == "count":
        data["count"] += 1
    else:
        data["selected"][0][0] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        store.read_last_round()
    with pytest.raises(ValueError):
        store.training_source(2, 1, VERY_SMALL, seed=1)


def test_round_sidecar_is_not_an_archive(tmp_path):
    store = store_for(tmp_path, capacity=64)
    store.append(game(0).samples, game_id=0)
    store.training_source(2, 1, VERY_SMALL, seed=1)
    assert (tmp_path / "replay_store" / "last_round.json").is_file()

    scanned = store_for(tmp_path, capacity=64)
    scanned.scan()
    assert len(scanned) == 4
    assert [entry.game_id for entry in scanned.games] == [0]


def test_all_rows_selected_waits_for_new_samples(tmp_path):
    store = store_for(tmp_path, capacity=64)
    for index in range(2):
        store.append(game(index * 4).samples, game_id=index)
    source = store.training_source(4, 1, VERY_SMALL, iteration=0, seed=1)
    for _ in range(2):
        source.begin_epoch()
        assert source.next_batch().sample_count == 4
    source.begin_epoch()
    assert source.steps_per_epoch == 0
    assert source.next_batch() is None
    assert source.summary()["waiting_for_shuffle"] is True


def test_state_and_generation_stay_json_safe(tmp_path):
    import json

    store = store_for(tmp_path)
    store.append(game(0).samples, game_id=0)
    source = store.training_source(2, 2, VERY_SMALL, seed=1)
    store.shuffle_state = source.state_dict()
    json.dumps(store.state_dict())
    assert store.state_dict()["shuffle_state"]["generation"] == source.generation


def test_child_transport_points_at_the_canonical_directory(tmp_path):
    store = store_for(tmp_path)
    transport = store.child_transport(5)
    assert transport == {
        "directory": str(store.directory.resolve()),
        "variant": VERY_SMALL.name,
        "iteration": 5,
    }


def test_log_frame_header_names_the_format(tmp_path):
    store = store_for(tmp_path)
    store.append(game(0).samples, game_id=0)
    raw = (tmp_path / "replay_store" / LOG_NAME).read_bytes()
    magic, length, checksum, version, rows, skipped, game_id, iteration = (
        HEADER.unpack_from(raw, 0)
    )
    assert magic == MAGIC and version == VERSION
    assert (rows, skipped, game_id) == (4, 0, 0)
    assert length == len(raw) - HEADER.size
    assert checksum != 0


def legacy_archive(directory, game_id, start, count=2):
    """Write one archive in the retired per-game NPZ layout."""

    encoded = [
        encode_sample(sample(index)) for index in range(start, start + count)
    ]
    offsets = np.zeros(count + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(blob) for blob in encoded])
    np.savez(
        directory / f"g{game_id:012d}-{count:08d}.npz",
        format=np.array(GAME_FILE_FORMAT),
        game_id=np.array(game_id, dtype=np.int64),
        iteration=np.array(-1, dtype=np.int64),
        count=np.array(count, dtype=np.int64),
        total_generated=np.array(start + count, dtype=np.int64),
        offsets=offsets,
        blobs=np.frombuffer(b"".join(encoded), dtype=np.uint8),
    )


def test_legacy_game_archives_are_imported_and_parked(tmp_path):
    directory = tmp_path / "replay_store"
    directory.mkdir()
    legacy_archive(directory, 0, 0)
    legacy_archive(directory, 1, 2)

    store = GameStore(directory, VERY_SMALL, 64)
    store.scan()
    assert [entry.game_id for entry in store.games] == [0, 1]
    assert [entry.count for entry in store.games] == [2, 2]
    assert [item.move_count for item in store.iter_samples()] == [0, 1, 2, 3]
    assert store.next_game_id == 2
    assert list(directory.glob("g*.npz")) == []
    parked = sorted(path.name for path in (directory / "legacy").iterdir())
    assert parked == [
        "g000000000000-00000002.npz", "g000000000001-00000002.npz"
    ]

    # Appending continues behind the imported ids.
    assert store.append(game(4, count=2).samples) == 2
    assert [entry.game_id for entry in store.games] == [0, 1, 2]

    # An archive left behind by an interrupted import is not imported twice.
    legacy_archive(directory, 0, 0)
    again = GameStore(directory, VERY_SMALL, 64)
    again.scan()
    assert [entry.game_id for entry in again.games] == [0, 1, 2]
    assert len(again) == 6


def test_rows_trimmed_at_the_front_drop_out_of_the_previous_round(tmp_path):
    store = store_for(tmp_path, capacity=64)
    for game_id in range(2):
        store.append(game(game_id * 4).samples, game_id=game_id)
    first = store.training_source(8, 1, VERY_SMALL, iteration=0, seed=2)
    first.begin_epoch()
    assert first.next_batch() is not None
    previous = store.read_last_round()
    assert {row[0] for row in previous["selected"]} == {0, 1}

    # The two oldest rows go; game 0 keeps rows 2 and 3 under their numbers.
    assert store.trim_to(6) == 2
    assert [(entry.game_id, entry.skip, entry.count) for entry in store.games] == [
        (0, 2, 2),
        (1, 0, 4),
    ]
    store.append(game(8).samples, game_id=2)
    store.training_source(8, 1, VERY_SMALL, iteration=1, seed=2 + 1009)
    current = store.read_last_round()
    assert current["pool_samples"] == 10
    # The six rows that survived the cut were already used, so the new round
    # draws exactly the game that arrived after it.
    assert {tuple(row) for row in current["selected"]} == {
        (2, 0), (2, 1), (2, 2), (2, 3)
    }


def test_training_source_no_repeat_rows_can_be_disabled(tmp_path):
    store = store_for(tmp_path, capacity=64)
    # One game of four rows is exactly one full round.
    store.append(game(0).samples, game_id=0)

    def consume(source):
        source.begin_epoch()
        seen = []
        while not source.is_exhausted():
            batch = source.next_batch()
            if batch is None:
                break
            seen.extend(batch_ids(batch))
        return seen

    disjoint = store.training_source(2, 2, VERY_SMALL, iteration=0, seed=9)
    assert len(consume(disjoint)) == 4
    assert consume(disjoint) == []
    # The previous round used every row, so a disjoint rule publishes nothing.
    assert disjoint.summary()["rounds"] == 0
    assert disjoint.summary()["no_repeat_rows"] is True

    repeated = store.training_source(
        2, 2, VERY_SMALL, iteration=1, seed=10, no_repeat_rows=False,
    )
    rounds = []
    original = store.write_last_round

    def record(state):
        rounds.append(state)
        return original(state)

    store.write_last_round = record
    assert len(consume(repeated)) == 4
    # Every round is drawn from the whole pool, so each new epoch publishes
    # another full round instead of running dry.
    assert len(consume(repeated)) == 4
    assert len(consume(repeated)) == 4
    assert repeated.summary()["rounds"] == 2
    assert repeated.summary()["no_repeat_rows"] is False
    assert rounds and rounds[-1]["count"] == 4

    with pytest.raises(ValueError, match="no_repeat_rows"):
        store.training_source(2, 2, VERY_SMALL, seed=1, no_repeat_rows=1)


def test_row_move_counts_are_cached_and_sliced_after_trim(tmp_path):
    store = store_for(tmp_path)
    store.append(game(0).samples, game_id=0)
    assert store.row_move_counts(store.games[0]).tolist() == [0, 1, 2, 3]
    store.record_trained(range(4))
    assert store.trim_to(2) == 2
    entry = store.games[0]
    assert (entry.skip, entry.count) == (2, 2)
    # The cached array follows the surviving row identity.
    assert store.row_move_counts(entry).tolist() == [2, 3]


def test_endgame_sampling_repeats_late_rows_in_the_round(tmp_path):
    from suzero_5dchess.endgame_sampling import EndgameSamplingConfig

    store = store_for(tmp_path)
    store.append(game(0).samples, game_id=0)  # move_count 0..3
    source = store.training_source(
        8, 1, VERY_SMALL, iteration=0, seed=5,
        endgame=EndgameSamplingConfig(enabled=True, strength=4.0, exponent=1.0),
    )
    state = store.read_last_round()
    assert state is not None
    rows = [row[1] for row in state["selected"]]
    assert sorted(set(rows)) == [0, 1, 2, 3]  # every row still shown once
    assert rows.count(0) == 1
    assert rows.count(3) >= 5  # final row weight 5 → four extra copies
    assert state["count"] == len(rows)
    assert state["endgame"] == {"strength": 4.0, "exponent": 1.0}
    assert source.pool_samples == 4
    assert source.summary()["endgame"] == {
        "strength": 4.0, "exponent": 1.0,
    }


def test_round_sidecar_only_allows_duplicates_with_endgame(tmp_path):
    store = store_for(tmp_path)
    store.append(game(0).samples, game_id=0)
    round_state = {
        "iteration": 0, "phase_seed": 0, "round": 0, "seed": 0,
        "keep_samples": None, "pool_samples": 4, "games": [[0, 4]],
        "count": 5,
        "selected": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 3]],
    }
    with pytest.raises(ValueError, match="duplicate game round row"):
        store.write_last_round(dict(round_state))
    store.write_last_round({
        **round_state, "endgame": {"strength": 1.0, "exponent": 2.0},
    })
    assert store.read_last_round()["selected"][-1] == [0, 3]
    with pytest.raises(ValueError, match="endgame"):
        store.write_last_round({
            **round_state, "endgame": {"strength": -1.0, "exponent": 2.0},
        })


def test_endgame_settings_do_not_reuse_an_older_round(tmp_path):
    from suzero_5dchess.endgame_sampling import EndgameSamplingConfig

    store = store_for(tmp_path)
    store.append(game(0).samples, game_id=0)
    store.training_source(
        2, 1, VERY_SMALL, iteration=0, seed=3, no_repeat_rows=False,
    )
    plain = store.read_last_round()
    assert plain["endgame"] is None

    store.training_source(
        2, 1, VERY_SMALL, iteration=0, seed=3, no_repeat_rows=False,
        endgame=EndgameSamplingConfig(enabled=True, strength=2.0, exponent=1.0),
    )
    weighted = store.read_last_round()
    assert weighted["endgame"] == {"strength": 2.0, "exponent": 1.0}
    # The stale round is not reused: duplicates would not fit its count.
    assert weighted["count"] > plain["count"]
