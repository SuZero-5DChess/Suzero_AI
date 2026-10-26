"""Row log engine: frame codec, shared appends, torn tails, batch trimming."""

from pathlib import Path

import pytest

from suzero_5dchess.row_log import HEADER, LOG_NAME, RowLog


def rows(count: int, tag: int = 0) -> list[bytes]:
    """``count`` one-byte rows, all carrying ``tag``."""

    return [bytes([tag])] * count


def test_append_and_load_round_trip(tmp_path):
    log = RowLog(tmp_path)
    frame = log.append(game_id=4, iteration=7, blobs=[b"aa", b"bbb"])
    assert (
        frame.offset, frame.rows, frame.skipped, frame.game_id, frame.iteration
    ) == (0, 2, 0, 4, 7)
    assert log.read(frame) == [b"aa", b"bbb"]

    reopened = RowLog(tmp_path)
    assert reopened.load() == [frame]
    assert reopened.rows == 2
    assert reopened.end == reopened.path.stat().st_size == frame.length

    magic, length, _crc, _version, count, skipped, game_id, iteration = (
        HEADER.unpack_from((tmp_path / LOG_NAME).read_bytes(), 0)
    )
    assert magic == b"RWL1" and length == (reopened.end - HEADER.size)
    assert (count, skipped, game_id, iteration) == (2, 0, 4, 7)


def test_two_writers_append_to_the_same_log(tmp_path):
    first, second = RowLog(tmp_path), RowLog(tmp_path)
    first.append(game_id=0, iteration=1, blobs=rows(1))
    second.append(game_id=1, iteration=1, blobs=rows(2, tag=1))

    # Each writer indexes what the other appended before its own frame.
    assert [frame.game_id for frame in first.frames] == [0]
    assert first.refresh() == 2
    assert [frame.game_id for frame in first.frames] == [0, 1]
    # The second writer saw the first frame before appending its own.
    assert [frame.game_id for frame in second.frames] == [0, 1]
    assert second.refresh() == 0

    third = first.append(game_id=2, iteration=1, blobs=rows(3, tag=2))
    assert second.refresh() == 3
    assert [frame.game_id for frame in second.frames] == [0, 1, 2]
    assert second.read(third) == [b"\x02", b"\x02", b"\x02"]

    fresh = RowLog(tmp_path)
    assert [frame.game_id for frame in fresh.load()] == [0, 1, 2]
    assert fresh.rows == 6


def test_an_empty_append_writes_nothing(tmp_path):
    log = RowLog(tmp_path)
    with pytest.raises(ValueError, match="at least one row"):
        log.append(game_id=0, iteration=0, blobs=[])
    assert not log.path.exists()
    assert log.append_many([]) == []


def test_a_reader_ignores_a_torn_tail_and_a_writer_repairs_it(tmp_path):
    log = RowLog(tmp_path)
    frame = log.append(game_id=0, iteration=1, blobs=rows(32))
    raw = log.path.read_bytes()
    cut = frame.length - 5
    log.path.write_bytes(raw[:cut])

    # Reading stops at the tear without touching the file.
    reader = RowLog(tmp_path)
    assert reader.load() == []
    assert reader.end == 0
    assert log.path.stat().st_size == cut

    # The next writer cuts the partial frame before appending behind it.
    writer = RowLog(tmp_path)
    repaired = writer.append(game_id=1, iteration=1, blobs=rows(4, tag=1))
    assert repaired.offset == 0
    assert [item.game_id for item in writer.frames] == [1]
    assert writer.end == writer.path.stat().st_size

    reopened = RowLog(tmp_path)
    assert [item.game_id for item in reopened.load()] == [1]


def test_unique_appends_refuse_an_id_the_log_already_holds(tmp_path):
    first = RowLog(tmp_path)
    first.append(game_id=3, iteration=1, blobs=rows(1), unique=True)

    second = RowLog(tmp_path)
    second.load()
    with pytest.raises(FileExistsError):
        second.append(game_id=3, iteration=1, blobs=rows(1), unique=True)

    # A writer that never indexed the log syncs before checking.
    stale = RowLog(tmp_path)
    with pytest.raises(FileExistsError):
        stale.append(game_id=3, iteration=1, blobs=rows(1), unique=True)
    assert [frame.game_id for frame in RowLog(tmp_path).load()] == [3]


def test_drop_cuts_the_oldest_rows_and_keeps_row_numbers(tmp_path):
    log = RowLog(tmp_path)
    for game_id in range(3):
        log.append(game_id=game_id, iteration=1, blobs=rows(3, tag=game_id))
    assert log.rows == 9
    before = log.path.stat().st_size

    assert log.drop(4) == 4
    assert [(frame.game_id, frame.skipped, frame.live_rows) for frame in log.frames] == [
        (1, 1, 2),
        (2, 0, 3),
    ]
    assert log.read(log.frames[0]) == [b"\x01", b"\x01"]
    assert log.end == log.path.stat().st_size < before

    assert log.drop(5) == 5
    assert log.rows == 0 and log.frames == [] and log.end == 0
    assert log.path.stat().st_size == 0

    # Appending after a full drop starts a fresh file at offset zero.
    frame = log.append(game_id=3, iteration=2, blobs=rows(1, tag=3))
    assert frame.offset == 0
    reopened = RowLog(tmp_path)
    assert [(item.game_id, item.iteration) for item in reopened.load()] == [(3, 2)]


def test_a_stale_frame_is_rejected(tmp_path):
    log = RowLog(tmp_path)
    stale = log.append(game_id=0, iteration=1, blobs=rows(2))
    log.append(game_id=1, iteration=1, blobs=rows(2, tag=1))

    log.drop(stale.live_rows)
    with pytest.raises(ValueError, match="moved|stale"):
        log.read(stale)


def test_drop_more_than_the_log_holds_empties_it(tmp_path):
    log = RowLog(tmp_path)
    log.append(game_id=0, iteration=1, blobs=rows(2))
    assert log.drop(99) == 2
    assert log.rows == 0
    assert log.path.is_file()


def test_drop_of_zero_rows_does_not_rewrite(tmp_path):
    log = RowLog(tmp_path)
    log.append(game_id=0, iteration=1, blobs=rows(2))
    before = log.path.stat().st_mtime_ns
    size = log.path.stat().st_size
    assert log.drop(0) == 0
    assert log.path.stat().st_size == size
    assert log.path.stat().st_mtime_ns == before


def test_the_log_keeps_many_games_in_one_file(tmp_path):
    log = RowLog(tmp_path)
    log.append_many(
        (game_id, 3, rows(2, tag=game_id % 251), 0) for game_id in range(50)
    )
    assert len(log.frames) == 50
    assert log.rows == 100
    assert sorted(path.name for path in Path(tmp_path).iterdir()) == [
        "replay.lock", "replay.log"
    ]
    assert log.path.stat().st_size == log.end
