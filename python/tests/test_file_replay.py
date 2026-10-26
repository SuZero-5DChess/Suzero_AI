"""Published shuffle rounds exclude the previous round's rows at row level."""

from __future__ import annotations

import json
from pathlib import Path
import uuid

import numpy as np
import pytest
import torch

from suzero_5dchess import file_replay
from suzero_5dchess.endgame_sampling import EndgameSamplingConfig
from suzero_5dchess.file_replay import (
    FileReplayDataSource,
    clear_shuffle,
    previous_round_exclusion,
    publish_shuffle,
)
from suzero_5dchess.protocol import VERY_SMALL
from suzero_5dchess.replay import ReplayBuffer
from suzero_5dchess.sample import PositionSample, collate_samples


def sample(index):
    boards = 1 + index % 2
    codes = np.zeros((boards, 16), dtype=np.uint8)
    codes[:, index % 16] = 1 + index % 12
    return PositionSample(
        board_codes=codes,
        l_coords=np.arange(boards, dtype=np.int64) - 1,
        t_coords=np.arange(boards, dtype=np.int64) + index,
        urgency=float(index + 1),
        side_to_move=index % 2,
        actions=np.array([[0, 0, 1, 0, 1, 0, 1, 0]], dtype=np.int64),
        policy=np.array([0.75, 0.25], dtype=np.float32) if index % 2 else np.array([1.0], dtype=np.float32),
        has_submit=bool(index % 2),
        value_target=(index % 3 - 1) * 0.5,
        player=index % 2,
        move_count=index,
        metadata={"weight": float(index + 1)},
    )


def make_buffer(count=11):
    buffer = ReplayBuffer(max(count, 1), VERY_SMALL)
    buffer.push([sample(index) for index in range(count)])
    return buffer


def published_blobs(directory, manifest):
    blobs = []
    for item in manifest["files"]:
        path = directory / manifest["generation"] / item["name"]
        with np.load(path, allow_pickle=False) as archive:
            state = {key: archive[key] for key in archive.files}
        for key in ("format", "capacity", "count", "total_generated"):
            state[key] = state[key].item()
        assert state["count"] == item["count"]
        assert state["blobs"].dtype == np.uint8
        assert state["offsets"].dtype == np.int64
        assert all(not np.asarray(value).dtype.hasobject for value in state.values())
        replay = ReplayBuffer(item["count"], VERY_SMALL)
        replay.load_state_dict(state)
        blobs.extend(replay.encoded_at(range(len(replay))))
    return blobs


def take_epoch(source):
    source.begin_epoch()
    batches = []
    while not source.is_exhausted():
        batch = source.next_batch()
        if batch is None:
            break
        batches.append(batch)
    assert source.next_batch() is None
    return batches


def assert_batches_equal(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.sample_count == b.sample_count
        assert a.data.keys() == b.data.keys()
        for key in a.data:
            assert a.data[key].dtype == b.data[key].dtype, key
            assert torch.equal(a.data[key], b.data[key]), key


def assert_batch_matches(batch, buffer, indices):
    expected = collate_samples(buffer.sample_at(indices), VERY_SMALL)
    assert batch.sample_count == len(indices)
    assert batch.data.keys() == expected.keys()
    for key, value in expected.items():
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        assert batch.data[key].dtype == tensor.dtype, key
        assert torch.equal(batch.data[key], tensor), key


@pytest.mark.parametrize("count", [0, 1, 10])
def test_publish_manifest_and_lossless_blob_shards(tmp_path, count):
    buffer = make_buffer(count)
    manifest = publish_shuffle(buffer, tmp_path, iteration=7, seed=19, rows_per_file=3)
    current = json.loads((tmp_path / "current.json").read_text())
    stored = json.loads((tmp_path / manifest["generation"] / "manifest.json").read_text())
    assert current == stored == manifest
    assert manifest["format"] == file_replay.SHUFFLE_FORMAT
    assert manifest["count"] == count
    assert len(manifest["files"]) == (count + 2) // 3
    assert sum(item["count"] for item in manifest["files"]) == count
    assert all(0 < item["count"] <= 3 for item in manifest["files"])
    assert manifest["producer"]["iteration"] == 7
    assert manifest["producer"]["total_generated"] == count
    assert manifest["producer"]["seed"] == 19
    assert manifest["producer"]["keep_samples"] is None
    assert manifest["producer"]["rows_per_file"] == 3
    assert manifest["producer"]["digest"]
    assert manifest["pool_samples"] == count
    assert manifest["rows"] == np.random.default_rng(19).permutation(np.arange(count)).tolist()
    order = np.random.default_rng(19).permutation(np.arange(count))
    assert published_blobs(tmp_path, manifest) == buffer.encoded_at(order)
    assert buffer.total_generated == len(buffer) == count


@pytest.mark.parametrize("keep", [None, 3, 10, 99])
def test_shuffle_subset_is_deterministic_without_replacement(tmp_path, keep):
    buffer = make_buffer(10)
    first = publish_shuffle(buffer, tmp_path / "a", iteration=0, seed=41, keep_samples=keep, rows_per_file=2)
    second = publish_shuffle(buffer, tmp_path / "b", iteration=0, seed=41, keep_samples=keep, rows_per_file=2)
    rng = np.random.default_rng(41)
    pool = rng.choice(10, size=keep, replace=False) if keep is not None and keep < 10 else np.arange(10)
    order = rng.permutation(pool)
    blobs = published_blobs(tmp_path / "a", first)
    expected_count = 10 if keep is None else min(keep, 10)
    assert first["count"] == expected_count
    assert blobs == published_blobs(tmp_path / "b", second) == buffer.encoded_at(order)
    assert len(blobs) == len(set(blobs))


def test_round_serves_untrained_rows_before_trained_ones(tmp_path):
    buffer = make_buffer(6)
    buffer.record_trained([0, 2, 4])

    manifest = publish_shuffle(buffer, tmp_path, iteration=1, seed=13)

    # The round still contains every row, but the three never-served rows come
    # first so the epoch budget reaches them before repeating old samples.
    assert sorted(manifest["rows"]) == list(range(6))
    assert set(manifest["rows"][:3]) == {1, 3, 5}
    assert set(manifest["rows"][3:]) == {0, 2, 4}


def test_excluded_rows_are_never_selected_and_retries_reuse(tmp_path):
    buffer = make_buffer(10)
    exclude = [0, 3, 7]
    first = publish_shuffle(
        buffer, tmp_path, iteration=0, seed=5, exclude=exclude, rows_per_file=3
    )
    assert not set(first["rows"]) & set(exclude)
    assert sorted(first["rows"]) == [1, 2, 4, 5, 6, 8, 9]
    assert first["pool_samples"] == 10
    assert first["producer"]["exclude_digest"]
    # A retry of the same phase reuses the round even though the exclusion is
    # derived again from the round that this publication produced.
    assert publish_shuffle(
        buffer, tmp_path, iteration=0, seed=5, exclude=exclude,
        rows_per_file=3,
    ) == first


def test_all_rows_excluded_publishes_no_round(tmp_path):
    buffer = make_buffer(5)
    assert publish_shuffle(
        buffer, tmp_path, iteration=0, seed=1, exclude=range(5)
    ) is None
    assert not (tmp_path / "current.json").exists()


def test_publisher_advances_rounds_without_repeating_previous_rows(tmp_path):
    buffer = make_buffer(10)
    manifests = []

    def publish(exclude):
        manifest = publish_shuffle(
            buffer, tmp_path, iteration=0,
            seed=100 + len(manifests), keep_samples=4,
            exclude=exclude, rows_per_file=2,
        )
        manifests.append(manifest)
        return manifest

    publish([])
    source = FileReplayDataSource(tmp_path, 4, 2, VERY_SMALL, publisher=publish)
    batches = take_epoch(source)
    # Two steps of four rows fit in the epoch budget; the first round holds
    # only four rows, so the source publishes a second round mid-epoch.
    assert [batch.sample_count for batch in batches] == [4, 4]
    assert len(manifests) == 2
    assert source.summary()["rounds"] == 1
    assert set(manifests[0]["rows"]).isdisjoint(manifests[1]["rows"])
    assert_batch_matches(batches[0], buffer, manifests[0]["rows"])
    assert_batch_matches(batches[1], buffer, manifests[1]["rows"])

    # The next epoch continues the rule: only the immediately preceding round
    # is excluded, so the first round's rows become eligible again.
    take_epoch(source)
    assert len(manifests) == 4
    assert set(manifests[2]["rows"]).isdisjoint(manifests[1]["rows"])
    assert set(manifests[3]["rows"]).isdisjoint(manifests[2]["rows"])
    assert all(row in range(10) for row in manifests[3]["rows"])


def test_previous_round_exclusion_follows_window_evictions(tmp_path):
    buffer = ReplayBuffer(20, VERY_SMALL)
    buffer.push([sample(index) for index in range(6)])
    manifest = publish_shuffle(buffer, tmp_path, iteration=0, seed=1)
    # Only rows the trainer already served may leave the window.
    buffer.record_trained(range(6))
    buffer.push([sample(index) for index in range(6, 14)])
    buffer.trim_to(10)
    # 6 published + 8 appended - 10 retained = 4 rows evicted from the front.
    rows = previous_round_exclusion(
        tmp_path, pool_samples=len(buffer), total_generated=buffer.total_generated
    )
    assert sorted(rows) == [0, 1]
    assert set(rows) <= set(range(len(buffer)))
    buffer.trim_to(0)
    assert previous_round_exclusion(
        tmp_path, pool_samples=0, total_generated=buffer.total_generated
    ) == []
    assert manifest["rows"]


def test_epochs_continue_consuming_one_published_round(tmp_path):
    buffer = make_buffer(11)
    manifest = publish_shuffle(buffer, tmp_path, iteration=0, seed=9, rows_per_file=4)
    source = FileReplayDataSource(tmp_path, 3, 2, VERY_SMALL)
    order = np.random.default_rng(9).permutation(11)
    initial_files = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    seen = 0
    for expected_sizes, expected_steps in (([3, 3], 2), ([3, 2], 2), ([], 0)):
        batches = take_epoch(source)
        assert source.steps_per_epoch == expected_steps
        assert [batch.sample_count for batch in batches] == expected_sizes
        for batch in batches:
            assert_batch_matches(batch, buffer, order[seen:seen + batch.sample_count])
            seen += batch.sample_count
        summary = source.summary()
        assert summary["remaining_samples"] == 11 - seen
        assert summary["pool_samples"] == summary["round_samples"] == 11
        assert summary["files"] == 3
        assert summary["no_repeat_rows"] is True
        assert summary["waiting_for_shuffle"] is (seen == 11)
    assert seen == 11
    state = source.state_dict()
    assert state["generation"] == manifest["generation"]
    assert state["file_index"] == 3 and state["row_offset"] == 0
    assert state["files_used"] == [item["name"] for item in manifest["files"]]
    assert source.next_batch() is None
    assert not source.has_validation()
    assert list(source.validation_batches()) == []
    # Consumption is checkpointed by the caller, not a sidecar write per batch.
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == initial_files


@pytest.mark.parametrize(("count", "batch", "steps"), [(0, 3, 99), (2, 8, 99), (12, 3, 1)])
def test_empty_partial_and_requested_epoch_budget(tmp_path, count, batch, steps):
    publish_shuffle(make_buffer(count), tmp_path, iteration=0, seed=0, rows_per_file=2)
    source = FileReplayDataSource(tmp_path, batch, steps, VERY_SMALL)
    batches = take_epoch(source)
    assert source.steps_per_epoch == (steps if count else 0)
    assert len(batches) == (min(steps, -(-count // batch)) if count else 0)
    consumed = sum(item.sample_count for item in batches)
    assert consumed == (min(count, batch * steps) if count else 0)
    assert source.remaining_samples == count - consumed


def test_cursor_checkpoint_restores_mid_shard_without_repeat(tmp_path):
    buffer = make_buffer(11)
    directory = tmp_path / "shuffle"
    publish_shuffle(buffer, directory, iteration=0, seed=3, rows_per_file=4)
    uninterrupted = FileReplayDataSource(directory, 3, 1, VERY_SMALL)
    assert len(take_epoch(uninterrupted)) == 1
    state = uninterrupted.state_dict()
    assert state["file_index"] == 0 and state["row_offset"] == 3
    assert state["files_used"] == []
    buffer.shuffle_state = state
    checkpoint = buffer.state_dict()
    path = tmp_path / "replay.pt"
    torch.save(checkpoint, path)
    restored_buffer = ReplayBuffer(11, VERY_SMALL)
    restored_buffer.load_state_dict(torch.load(path, weights_only=False))
    assert restored_buffer.shuffle_state == state
    resumed = FileReplayDataSource(directory, 3, 1, VERY_SMALL, state=restored_buffer.shuffle_state)
    assert resumed.summary() == uninterrupted.summary()
    for _ in range(4):
        assert_batches_equal(take_epoch(resumed), take_epoch(uninterrupted))
        assert resumed.state_dict() == uninterrupted.state_dict()
    assert resumed.waiting_for_shuffle
    exhausted = FileReplayDataSource(directory, 3, 1, VERY_SMALL, state=resumed.state_dict())
    assert take_epoch(exhausted) == []
    assert exhausted.waiting_for_shuffle
    # Returned replay snapshots own their cursor data, just like their blobs.
    checkpoint["shuffle_state"]["files_used"].append("not-a-real-file")
    assert buffer.shuffle_state == state
    assert restored_buffer.shuffle_state == state


def test_indexed_replay_export_omits_cursor_and_preserves_record_order(tmp_path):
    buffer = make_buffer(5)
    buffer.shuffle_state = {"generation": "old", "file_index": 1, "row_offset": 2, "files_used": ["000000.npz"]}
    for indices in ([4, 1, 4], []):
        state = buffer.state_dict(indices)
        assert "shuffle_state" not in state
        restored = ReplayBuffer(5, VERY_SMALL)
        restored.load_state_dict(state)
        assert restored.encoded_at(range(len(restored))) == buffer.encoded_at(indices)
        assert restored.total_generated == 5
    legacy = buffer.state_dict()
    legacy.pop("shuffle_state")
    buffer.load_state_dict(legacy)
    assert buffer.shuffle_state is None


def test_retry_publish_reuses_generation_and_saved_cursor(tmp_path, monkeypatch):
    buffer = make_buffer(8)
    first = publish_shuffle(buffer, tmp_path, iteration=4, seed=5, rows_per_file=3)
    source = FileReplayDataSource(tmp_path, 2, 1, VERY_SMALL)
    take_epoch(source)
    state = source.state_dict()
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    def fail(*args, **kwargs):
        raise AssertionError("idempotent retry rewrote a shard")

    monkeypatch.setattr(file_replay.np, "savez", fail)
    assert publish_shuffle(buffer, tmp_path, iteration=4, seed=5, rows_per_file=3) == first
    resumed = FileReplayDataSource(tmp_path, 2, 1, VERY_SMALL, state=state)
    assert resumed.state_dict() == state
    assert_batches_equal(take_epoch(resumed), take_epoch(source))
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("changed", ["iteration", "seed", "keep", "shard_size", "counter", "content"])
def test_changed_producer_publishes_new_generation_and_ignores_old_cursor(tmp_path, changed):
    buffer = make_buffer(8)
    options = {"iteration": 0, "seed": 5, "rows_per_file": 3}
    first = publish_shuffle(buffer, tmp_path, **options)
    source = FileReplayDataSource(tmp_path, 2, 1, VERY_SMALL)
    take_epoch(source)
    old_state = source.state_dict()
    if changed == "iteration":
        options["iteration"] = 1
    elif changed == "seed":
        options["seed"] = 6
    elif changed == "keep":
        options["keep_samples"] = 4
    elif changed == "shard_size":
        options["rows_per_file"] = 2
    elif changed == "counter":
        buffer.total_generated += 1
    else:
        # Same number of rows and N, different content must invalidate reuse.
        buffer = ReplayBuffer(8, VERY_SMALL)
        buffer.push([sample(index + 1) for index in range(8)])
    second = publish_shuffle(buffer, tmp_path, **options)
    assert second["generation"] != first["generation"]
    assert not (tmp_path / first["generation"]).exists()
    resumed = FileReplayDataSource(tmp_path, 2, 1, VERY_SMALL, state=old_state)
    fresh = FileReplayDataSource(tmp_path, 2, 1, VERY_SMALL)
    assert resumed.remaining_samples == second["count"]
    assert resumed.state_dict()["file_index"] == resumed.state_dict()["row_offset"] == 0
    assert_batches_equal(take_epoch(resumed), take_epoch(fresh))


@pytest.mark.parametrize("stage", ["shard", "pointer"])
def test_failed_publication_preserves_previous_complete_generation(tmp_path, monkeypatch, stage):
    buffer = make_buffer(8)
    first = publish_shuffle(buffer, tmp_path, iteration=0, seed=5, rows_per_file=3)
    pointer = (tmp_path / "current.json").read_bytes()
    old_blobs = published_blobs(tmp_path, first)
    if stage == "shard":
        original = file_replay.np.savez
        calls = 0

        def fail(path, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected shard failure")
            return original(path, **kwargs)

        target, name = file_replay.np, "savez"
    else:
        original = file_replay.os.replace

        def fail(source, destination):
            if Path(destination) == tmp_path / "current.json":
                raise OSError("injected pointer failure")
            return original(source, destination)

        target, name = file_replay.os, "replace"
    with monkeypatch.context() as patch:
        patch.setattr(target, name, fail)
        with pytest.raises(OSError, match="injected"):
            publish_shuffle(buffer, tmp_path, iteration=1, seed=6, rows_per_file=3)
    assert (tmp_path / "current.json").read_bytes() == pointer
    assert published_blobs(tmp_path, first) == old_blobs
    assert sum(batch.sample_count for batch in take_epoch(FileReplayDataSource(tmp_path, 2, 99, VERY_SMALL))) == 8
    # Recovery may leave an unpublished owned generation, but never a broken
    # current pointer; the next successful publication can replace it safely.
    latest = publish_shuffle(buffer, tmp_path, iteration=2, seed=7, rows_per_file=3)
    assert latest["generation"] != first["generation"]
    assert not (tmp_path / first["generation"]).exists()
    assert len(published_blobs(tmp_path, latest)) == 8


def make_staging(directory):
    generation = "generation-" + uuid.uuid4().hex
    staging = directory / (generation + ".tmp")
    staging.mkdir()
    (staging / "owner.json").write_text(json.dumps({
        "format": file_replay.SHUFFLE_FORMAT, "generation": generation,
    }))
    (staging / "000000.npz").write_bytes(b"interrupted shard write")
    (staging / "manifest.json.tmp").write_bytes(b"{interrupted manifest")
    return staging


@pytest.mark.parametrize("action", ["clear", "publish", "retry"])
def test_owned_crash_staging_is_removed_by_cleanup_or_publication(tmp_path, action):
    buffer = make_buffer(5)
    current = publish_shuffle(buffer, tmp_path, iteration=0, seed=1)
    staging = make_staging(tmp_path)
    if action == "clear":
        clear_shuffle(tmp_path)
        assert not (tmp_path / "current.json").exists()
    else:
        manifest = publish_shuffle(buffer, tmp_path, iteration=0 if action == "retry" else 1, seed=1)
        assert manifest["count"] == 5
        assert len(published_blobs(tmp_path, manifest)) == 5
        if action == "retry":
            assert manifest == current
        else:
            assert manifest["generation"] != current["generation"]
    assert not staging.exists()


@pytest.mark.parametrize("action", ["clear", "publish"])
def test_owned_staging_with_unknown_contents_is_not_deleted(tmp_path, action):
    staging = make_staging(tmp_path)
    note = staging / "user-notes.txt"
    note.write_text("not ours")
    before = {path.name: path.read_bytes() for path in staging.iterdir()}
    with pytest.raises(ValueError):
        if action == "clear":
            clear_shuffle(tmp_path)
        else:
            publish_shuffle(make_buffer(5), tmp_path, iteration=0, seed=1)
    assert {path.name: path.read_bytes() for path in staging.iterdir()} == before


@pytest.mark.parametrize("field", ["format", "generation"])
def test_staging_requires_matching_ownership_marker(tmp_path, field):
    staging = make_staging(tmp_path)
    marker = staging / "owner.json"
    owner = json.loads(marker.read_text())
    owner[field] = "not-the-owned-generation"
    marker.write_text(json.dumps(owner))
    before = {path.name: path.read_bytes() for path in staging.iterdir()}
    with pytest.raises(ValueError):
        clear_shuffle(tmp_path)
    assert {path.name: path.read_bytes() for path in staging.iterdir()} == before


def test_clear_removes_only_owned_generations_and_is_idempotent(tmp_path):
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me")
    unknown = tmp_path / "generation-user-backups"
    unknown.mkdir()
    (unknown / "backup.txt").write_text("also keep me")
    first = publish_shuffle(make_buffer(5), tmp_path, iteration=0, seed=1)
    clear_shuffle(tmp_path)
    clear_shuffle(tmp_path)
    clear_shuffle(tmp_path / "not-created")
    assert not (tmp_path / "current.json").exists()
    assert not (tmp_path / first["generation"]).exists()
    assert unrelated.read_text() == "keep me"
    assert (unknown / "backup.txt").read_text() == "also keep me"


def test_cleanup_refuses_unknown_contents_in_owned_generation(tmp_path):
    manifest = publish_shuffle(make_buffer(5), tmp_path, iteration=0, seed=1)
    note = tmp_path / manifest["generation"] / "user-notes.txt"
    note.write_text("must not delete")
    with pytest.raises(ValueError):
        clear_shuffle(tmp_path)
    assert note.read_text() == "must not delete"
    assert all((note.parent / item["name"]).exists() for item in manifest["files"])


@pytest.mark.parametrize("field", ["generation", "shard"])
@pytest.mark.parametrize("unsafe", ["../outside.npz", "/outside.npz", "nested/000000.npz"])
def test_manifest_rejects_path_escape_without_touching_other_files(tmp_path, field, unsafe):
    directory = tmp_path / "shuffle"
    manifest = publish_shuffle(make_buffer(5), directory, iteration=0, seed=1)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"unrelated contents")
    if field == "generation":
        manifest["generation"] = unsafe
    else:
        manifest["files"][0]["name"] = unsafe
    (directory / "current.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        FileReplayDataSource(directory, 2, 2, VERY_SMALL)
    with pytest.raises(ValueError):
        clear_shuffle(directory)
    assert outside.read_bytes() == b"unrelated contents"


@pytest.mark.parametrize("damage", ["missing", "invalid_npz", "wrong_count", "invalid_blob"])
def test_missing_or_corrupt_shard_fails_instead_of_silent_repeat(tmp_path, damage):
    manifest = publish_shuffle(make_buffer(5), tmp_path, iteration=0, seed=1, rows_per_file=3)
    shard = tmp_path / manifest["generation"] / manifest["files"][0]["name"]
    if damage == "missing":
        shard.unlink()
    elif damage == "invalid_npz":
        shard.write_bytes(b"not an npz archive")
    else:
        with np.load(shard, allow_pickle=False) as archive:
            state = {key: archive[key] for key in archive.files}
        if damage == "wrong_count":
            state["count"] = np.array(99)
        else:
            state["blobs"] = np.zeros_like(state["blobs"])
        np.savez(shard, **state)
    with pytest.raises((OSError, ValueError)):
        source = FileReplayDataSource(tmp_path, 2, 2, VERY_SMALL)
        source.begin_epoch()
        source.next_batch()
    assert json.loads((tmp_path / "current.json").read_text())["generation"] == manifest["generation"]


@pytest.mark.parametrize(
    "cursor", [{"file_index": -1}, {"file_index": 99}, {"row_offset": -1}, {"row_offset": 4}]
)
def test_same_generation_rejects_invalid_cursor(tmp_path, cursor):
    publish_shuffle(make_buffer(5), tmp_path, iteration=0, seed=1, rows_per_file=4)
    state = FileReplayDataSource(tmp_path, 2, 1, VERY_SMALL).state_dict()
    state.update(cursor)
    with pytest.raises(ValueError):
        FileReplayDataSource(tmp_path, 2, 1, VERY_SMALL, state=state)


def test_file_source_without_repeat_rows_draws_every_round_from_the_pool(tmp_path):
    buffer = make_buffer(10)
    manifests = []
    exclusions = []

    def publish(exclude):
        exclusions.append(list(exclude))
        manifest = publish_shuffle(
            buffer, tmp_path, iteration=0,
            seed=300 + len(manifests), keep_samples=4,
            exclude=exclude, rows_per_file=2, no_repeat_rows=False,
        )
        manifests.append(manifest)
        return manifest

    publish([])
    source = FileReplayDataSource(
        tmp_path, 4, 2, VERY_SMALL, publisher=publish, no_repeat_rows=False,
    )
    take_epoch(source)
    # The mid-epoch round asks for no exclusion, so it may repeat rows.
    assert exclusions == [[], []]
    assert source.summary()["no_repeat_rows"] is False
    assert len(manifests) == 2
    assert manifests[0]["count"] == manifests[1]["count"] == 4
    # Every row of the pool is eligible for the second round.
    assert len(set(manifests[1]["rows"])) == 4


def test_publish_shuffle_repeats_late_rows_inside_the_round(tmp_path):
    buffer = make_buffer(4)  # move_count 0..3 → progress 0, 1/3, 2/3, 1
    endgame = EndgameSamplingConfig(enabled=True, strength=4.0, exponent=1.0)
    manifest = publish_shuffle(
        buffer, tmp_path, iteration=0, seed=2,
        no_repeat_rows=False, endgame=endgame,
    )
    assert manifest is not None
    rows = manifest["rows"]
    assert sorted(set(rows)) == [0, 1, 2, 3]  # coverage is unchanged
    assert rows.count(0) == 1
    assert rows.count(3) >= 5  # weight 5: four guaranteed extra copies
    assert manifest["producer"]["endgame"] == {
        "strength": 4.0, "exponent": 1.0,
    }
    assert manifest["count"] == len(rows)


def test_endgame_settings_are_part_of_the_round_identity(tmp_path):
    buffer = make_buffer(4)
    first = publish_shuffle(
        buffer, tmp_path, iteration=0, seed=2, no_repeat_rows=False,
        endgame=EndgameSamplingConfig(enabled=True, strength=1.0, exponent=2.0),
    )
    same = publish_shuffle(
        buffer, tmp_path, iteration=0, seed=2, no_repeat_rows=False,
        endgame=EndgameSamplingConfig(enabled=True, strength=1.0, exponent=2.0),
    )
    assert first is not None and same is not None
    assert same["generation"] == first["generation"]

    changed = publish_shuffle(
        buffer, tmp_path, iteration=0, seed=2, no_repeat_rows=False,
        endgame=EndgameSamplingConfig(enabled=True, strength=3.0, exponent=2.0),
    )
    assert changed is not None
    assert changed["generation"] != first["generation"]


def test_disabled_endgame_sampling_keeps_one_row_each(tmp_path):
    buffer = make_buffer(4)
    manifest = publish_shuffle(
        buffer, tmp_path, iteration=0, seed=2, no_repeat_rows=False,
    )
    assert manifest is not None
    assert sorted(manifest["rows"]) == [0, 1, 2, 3]
    assert manifest["producer"]["endgame"] is None


def _rewrite_round(directory, manifest):
    for path in (
        directory / "current.json",
        directory / str(manifest["generation"]) / "manifest.json",
    ):
        path.write_text(json.dumps(manifest))


def test_manifest_rejects_duplicate_rows_unless_endgame_is_declared(tmp_path):
    manifest = publish_shuffle(
        make_buffer(4), tmp_path, iteration=0, seed=2, no_repeat_rows=False,
    )
    manifest["rows"][1] = manifest["rows"][0]
    _rewrite_round(tmp_path, manifest)
    with pytest.raises(ValueError, match="duplicate shuffle row index"):
        FileReplayDataSource(tmp_path, 2, 2, VERY_SMALL)

    manifest["producer"]["endgame"] = {"strength": 1.0, "exponent": 2.0}
    _rewrite_round(tmp_path, manifest)
    source = FileReplayDataSource(tmp_path, 2, 2, VERY_SMALL)
    source.begin_epoch()
    assert source.next_batch().sample_count == 2


@pytest.mark.parametrize(
    "endgame", ["weighted", {"strength": -1.0}, {"strength": 1.0, "exponent": 0}],
)
def test_manifest_rejects_invalid_endgame_settings(tmp_path, endgame):
    manifest = publish_shuffle(
        make_buffer(4), tmp_path, iteration=0, seed=2, no_repeat_rows=False,
    )
    manifest["producer"]["endgame"] = endgame
    _rewrite_round(tmp_path, manifest)
    with pytest.raises(ValueError, match="shuffle endgame settings"):
        FileReplayDataSource(tmp_path, 2, 2, VERY_SMALL)
