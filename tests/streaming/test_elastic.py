# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import pytest
import torch

from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.elastic import (
    _round_down_drop_first,
    canonical_chunk_order,
    canonical_item_stream,
    lockstep_stream_from_worker_seqs,
    restripe_items,
    sample_in_epoch_from_state,
    topology_changed,
    worker_plan_to_chunks,
)
from litdata.streaming.item_loader import TokensLoader
from litdata.streaming.posix_fast import PosixFastProfile
from litdata.utilities.env import _DistributedEnv
from tests.streaming.test_item_loader import _write_int_dataset


def _flatten_plan(plans):
    items = []
    for visits in plans:
        for chunk_index, ids in visits:
            items.extend((chunk_index, i) for i in ids)
    return items


def test_topology_changed_and_sample_in_epoch():
    state = {"world_size": 8, "num_workers": 2, "batch_size": 4, "num_samples_yielded": 10}
    assert topology_changed(state, world_size=8, num_workers=2, batch_size=4) is False
    assert topology_changed(state, world_size=2, num_workers=2, batch_size=4) is True
    assert topology_changed(state, world_size=8, num_workers=8, batch_size=4) is True
    assert topology_changed(state, world_size=8, num_workers=2, batch_size=8) is True
    assert (
        topology_changed(
            {"num_workers": 0, "world_size": 1, "batch_size": 1}, world_size=1, num_workers=1, batch_size=1
        )
        is False
    )
    assert (
        topology_changed(
            {"num_workers": 0, "world_size": 1, "batch_size": 1}, world_size=1, num_workers=0, batch_size=1
        )
        is False
    )
    assert sample_in_epoch_from_state(state) == 80
    assert sample_in_epoch_from_state({"sample_in_epoch": 12, "num_samples_yielded": 3, "world_size": 8}) == 12


def test_lockstep_stream_keeps_unequal_worker_tails():
    short = [(0, i) for i in range(10)]
    long = [(1, i) for i in range(50)]
    stream = lockstep_stream_from_worker_seqs([short, long], world_size=1, num_workers=2, batch_size=1)
    assert len(stream) == 60
    assert len(set(stream)) == 60
    assert set(stream) == set(short) | set(long)


def test_restripe_item_no_duplicates_and_drop_prefix():
    intervals = [[0, 0, 8, 8] for _ in range(8)]
    stream = canonical_item_stream(intervals, seed=42, epoch=1, shuffle=True, num_canonical_nodes=2)
    assert len(stream) == 64
    assert len(set(stream)) == 64

    drop_first = 16
    plans = restripe_items(stream, world_size=2, num_workers=2, batch_size=4, drop_first=drop_first, drop_last=False)
    remaining = _flatten_plan(plans)
    assert len(remaining) == len(set(remaining))
    prefix = set(stream[:drop_first])
    assert prefix.isdisjoint(set(remaining))
    assert set(remaining) == set(stream[drop_first:])


def test_restripe_workers_2_to_8_same_remaining_set():
    intervals = [[0, 0, 4, 4] for _ in range(16)]
    stream = canonical_item_stream(intervals, seed=7, epoch=3, shuffle=True, num_canonical_nodes=1)
    drop_first = 24
    a = set(_flatten_plan(restripe_items(stream, world_size=1, num_workers=2, batch_size=4, drop_first=drop_first)))
    b = set(_flatten_plan(restripe_items(stream, world_size=1, num_workers=8, batch_size=4, drop_first=drop_first)))
    assert a == b
    assert len(a) == len(stream) - drop_first


def test_restripe_world_size_2_to_1():
    intervals = [[0, 0, 5, 5] for _ in range(10)]
    stream = canonical_item_stream(intervals, seed=1, epoch=1, shuffle=False, num_canonical_nodes=2)
    drop_first = 10
    plans = restripe_items(stream, world_size=1, num_workers=1, batch_size=5, drop_first=drop_first)
    remaining = _flatten_plan(plans)
    assert remaining == stream[drop_first:]


def test_restripe_chunk_granularity_keeps_whole_chunks():
    intervals = [[0, 0, 4, 4] for _ in range(6)]
    stream = canonical_item_stream(intervals, seed=0, epoch=1, shuffle=False, num_canonical_nodes=1)
    plans = restripe_items(
        stream, world_size=1, num_workers=2, batch_size=4, drop_first=6, drop_last=False, granularity="chunk"
    )
    remaining = _flatten_plan(plans)
    chunks = [c for c, _ in remaining]
    assert remaining
    assert all(chunks.count(c) == 4 for c in set(chunks))
    # The chunk that was mid-read at drop_first=6 (item 2 of chunk 1) is skipped entirely.
    first_remaining_chunk = remaining[0][0]
    assert stream[6][0] != first_remaining_chunk or stream[5][0] != stream[6][0]


def test_worker_plan_to_chunks_stop_length():
    visits = [(3, [1, 2, 3]), (5, [0, 1])]
    chunks, intervals, item_lists = worker_plan_to_chunks(visits)
    assert chunks == [3, 5]
    assert item_lists == [[1, 2, 3], [0, 1]]
    assert sum(iv[2] - iv[1] for iv in intervals) == 5


def _all_ids_from_loader(loader, max_batches=None):
    ids = []
    for i, batch in enumerate(loader):
        ids.extend(torch.as_tensor(batch).reshape(-1).tolist())
        if max_batches is not None and i + 1 >= max_batches:
            break
    return ids


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_dataloader_elastic_workers_2_to_8(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=128, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=2, batch_size=4)
    _all_ids_from_loader(loader, max_batches=6)
    state = loader.state_dict()
    assert "sample_in_epoch" in state["dataset"]

    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=8, batch_size=4)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert len(rest) == len(set(rest))
    assert rest, "elastic resume should still yield remaining samples"


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_dataloader_strict_resume_same_workers_not_elastic(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=64, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=0, batch_size=4)
    _all_ids_from_loader(loader, max_batches=5)
    state = loader.state_dict()
    assert state["dataset"].get("state_version") == 2
    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=0, batch_size=4)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert rest
    assert len(rest) == len(set(rest))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_simulated_world_size_2_to_1_canonical(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=80, chunk_size=8)

    def make_ds(world, rank, state=None):
        ds = StreamingDataset(data_dir, shuffle=True, seed=42, drop_last=True)
        ds.distributed_env = _DistributedEnv(world, rank, 1)
        ds.batch_size = 4
        ds.num_workers = 1
        if state is not None:
            ds.load_state_dict(state)
        return ds

    proto = make_ds(2, 0)
    base_state = proto.state_dict(0, 1, 4)
    base_state["resume_mode"] = "elastic"
    base_state["sample_in_epoch"] = 0
    base_state["world_size"] = 2
    base_state["num_workers"] = 1
    base_state["batch_size"] = 4

    consumed = []
    local_n = 8
    for rank in (0, 1):
        ds = make_ds(2, rank, dict(base_state))
        it = iter(ds)
        for _ in range(local_n):
            consumed.append(int(next(it)))
    assert len(consumed) == len(set(consumed))

    resume_state = dict(base_state)
    resume_state["sample_in_epoch"] = local_n * 2
    resume_state["num_samples_yielded"] = local_n
    resume_state["world_size"] = 2

    rest = []
    ds = make_ds(1, 0, resume_state)
    for item in ds:
        rest.append(int(item))
    assert len(rest) == len(set(rest))
    assert set(consumed).isdisjoint(set(rest))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_window_shuffle_chunk_granularity_resume(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=96, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    dataset.posix_fast = PosixFastProfile(kind="nfs")
    loader = StreamingDataLoader(dataset, num_workers=2, batch_size=4)
    _all_ids_from_loader(loader, max_batches=4)
    state = loader.state_dict()

    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    dataset_b.posix_fast = PosixFastProfile(kind="nfs")
    loader_b = StreamingDataLoader(dataset_b, num_workers=4, batch_size=4)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert len(rest) == len(set(rest))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_tokens_loader_elastic_workers(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    from litdata.streaming import Cache

    data_dir = os.path.join(tmpdir, "tok")
    os.makedirs(data_dir)
    cache = Cache(input_dir=str(data_dir), chunk_size=40, item_loader=TokensLoader(20))
    counter = 0
    for i in range(40):
        cache[i] = torch.arange(counter, counter + 20).to(torch.int)
        counter += 20
    cache.done()
    cache.merge()

    dataset = StreamingDataset(data_dir, item_loader=TokensLoader(20), shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=2, batch_size=2)
    _all_ids_from_loader(loader, max_batches=5)
    state = loader.state_dict()
    dataset_b = StreamingDataset(data_dir, item_loader=TokensLoader(20), shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=4, batch_size=2)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert rest


def test_round_down_drop_first_aligns_to_global_batch():
    assert _round_down_drop_first(0, 8, 4) == 0
    assert _round_down_drop_first(32, 8, 4) == 32
    assert _round_down_drop_first(33, 8, 4) == 32
    assert _round_down_drop_first(63, 8, 4) == 32
    assert _round_down_drop_first(-5, 2, 4) == 0


def test_canonical_stream_independent_of_physical_world_size():
    intervals = [[0, 0, 4, 4] for _ in range(12)]
    a = canonical_item_stream(intervals, seed=9, epoch=2, shuffle=True, num_canonical_nodes=4)
    b = canonical_item_stream(intervals, seed=9, epoch=2, shuffle=True, num_canonical_nodes=4)
    c = canonical_item_stream(intervals, seed=9, epoch=2, shuffle=True, num_canonical_nodes=2)
    assert a == b
    assert len(set(a)) == len(a)
    assert set(a) == set(c)
    # Without a window, flattening NCN buckets keeps shuffled chunk order. Window shuffle
    # inside buckets is what makes NCN change the 1D stream.
    w4 = canonical_chunk_order(12, seed=9, epoch=2, shuffle=True, num_canonical_nodes=4, window=3)
    w2 = canonical_chunk_order(12, seed=9, epoch=2, shuffle=True, num_canonical_nodes=2, window=3)
    assert sorted(w4) == sorted(w2) == list(range(12))
    assert w4 != w2


def test_canonical_chunk_order_empty_and_window():
    assert canonical_chunk_order(0, seed=1, epoch=1, shuffle=True, num_canonical_nodes=4) == []
    order = canonical_chunk_order(16, seed=3, epoch=1, shuffle=True, num_canonical_nodes=4, window=4)
    assert sorted(order) == list(range(16))


def test_restripe_unaligned_drop_first_skips_remainder():
    stream = [(i // 4, i % 4) for i in range(40)]
    plans = restripe_items(stream, world_size=2, num_workers=1, batch_size=4, drop_first=13)
    remaining = _flatten_plan(plans)
    assert len(remaining) == len(set(remaining))
    assert set(remaining) == set(stream[8:])


def test_restripe_drop_last_equal_per_worker():
    stream = [(i // 3, i % 3) for i in range(50)]
    plans = restripe_items(stream, world_size=2, num_workers=2, batch_size=4, drop_first=0, drop_last=True)
    counts = [len(_flatten_plan([p])) for p in plans]
    assert len(set(counts)) == 1
    assert counts[0] % 4 == 0
    remaining = _flatten_plan(plans)
    assert len(remaining) == len(set(remaining))
    assert len(remaining) <= 48


def test_restripe_constant_global_batch_size_same_set():
    intervals = [[0, 0, 8, 8] for _ in range(8)]
    stream = canonical_item_stream(intervals, seed=42, epoch=1, shuffle=True, num_canonical_nodes=4)
    drop_first = 16
    a = set(_flatten_plan(restripe_items(stream, world_size=8, num_workers=1, batch_size=4, drop_first=drop_first)))
    b = set(_flatten_plan(restripe_items(stream, world_size=4, num_workers=1, batch_size=8, drop_first=drop_first)))
    assert a == b


def test_restripe_8_to_2_to_8_same_remaining_set():
    intervals = [[0, 0, 4, 4] for _ in range(16)]
    stream = canonical_item_stream(intervals, seed=11, epoch=1, shuffle=True, num_canonical_nodes=8)
    drop_first = 32
    a = set(_flatten_plan(restripe_items(stream, world_size=8, num_workers=1, batch_size=4, drop_first=drop_first)))
    b = set(_flatten_plan(restripe_items(stream, world_size=2, num_workers=1, batch_size=4, drop_first=drop_first)))
    c = set(_flatten_plan(restripe_items(stream, world_size=8, num_workers=1, batch_size=4, drop_first=drop_first)))
    assert a == b == c


def test_restripe_uneven_chunk_sizes_no_duplicates():
    intervals = [[0, 0, 3, 3], [0, 0, 11, 11], [0, 0, 1, 1], [0, 0, 7, 7], [0, 0, 5, 5]]
    stream = canonical_item_stream(intervals, seed=5, epoch=4, shuffle=True, num_canonical_nodes=2)
    assert len(stream) == 27
    plans = restripe_items(stream, world_size=2, num_workers=3, batch_size=2, drop_first=5)
    remaining = _flatten_plan(plans)
    assert len(remaining) == len(set(remaining))
    assert set(remaining) == set(stream[4:])


def test_restripe_drop_past_end_is_empty():
    stream = [(0, i) for i in range(10)]
    plans = restripe_items(stream, world_size=2, num_workers=2, batch_size=2, drop_first=10_000)
    assert _flatten_plan(plans) == []


def test_v1_checkpoint_infers_sample_in_epoch():
    state = {"num_samples_yielded": 7, "world_size": 4}
    assert sample_in_epoch_from_state(state) == 28
    assert topology_changed(state, world_size=4, num_workers=2, batch_size=8) is False
    assert topology_changed({"num_workers": 0}, world_size=1, num_workers=1, batch_size=1) is False
    assert topology_changed({"num_workers": 0}, world_size=1, num_workers=0, batch_size=1) is False


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_dataloader_workers_8_to_2(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=128, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=8, batch_size=4)
    _all_ids_from_loader(loader, max_batches=5)
    state = loader.state_dict()
    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=2, batch_size=4)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert rest
    assert len(rest) == len(set(rest))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_elastic_second_checkpoint_advances_cursor(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=96, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=2, batch_size=4)
    _all_ids_from_loader(loader, max_batches=6)
    state = loader.state_dict()
    cursor0 = state["dataset"]["sample_in_epoch"]

    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=0, batch_size=4)
    loader_b.load_state_dict(state)
    rest1 = _all_ids_from_loader(loader_b, max_batches=3)
    state2 = loader_b.state_dict()
    assert state2["dataset"].get("resume_mode") == "elastic"
    assert state2["dataset"]["sample_in_epoch"] == cursor0 + 12

    dataset_c = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_c = StreamingDataLoader(dataset_c, num_workers=0, batch_size=4)
    loader_c.load_state_dict(state2)
    rest2 = _all_ids_from_loader(loader_c)
    assert len(rest1) == len(set(rest1))
    assert len(rest2) == len(set(rest2))
    assert set(rest1).isdisjoint(set(rest2))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_unaligned_sample_in_epoch_is_rounded_before_save(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=64, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=0, batch_size=4)
    _all_ids_from_loader(loader, max_batches=2)
    state = loader.state_dict()
    state["dataset"]["sample_in_epoch"] = 13
    state["dataset"]["num_workers"] = 2

    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=0, batch_size=4)
    loader_b.load_state_dict(state)
    # world_size=1, batch_size=4 → stride 4, 13 rounds down to 12 even before the first batch.
    assert dataset_b._elastic_drop_first == 12
    _all_ids_from_loader(loader_b, max_batches=1)
    assert loader_b.state_dict()["dataset"]["sample_in_epoch"] == 16


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_elastic_shuffle_false_and_drop_last(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=80, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=False, seed=42, drop_last=True)
    loader = StreamingDataLoader(dataset, num_workers=2, batch_size=4, drop_last=True)
    _all_ids_from_loader(loader, max_batches=4)
    state = loader.state_dict()
    dataset_b = StreamingDataset(data_dir, shuffle=False, seed=42, drop_last=True)
    loader_b = StreamingDataLoader(dataset_b, num_workers=4, batch_size=4, drop_last=True)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert len(rest) == len(set(rest))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_elastic_completes_then_next_epoch_is_not_stuck(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=48, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=0, batch_size=4)
    _all_ids_from_loader(loader, max_batches=3)
    state = loader.state_dict()

    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=2, batch_size=4)
    loader_b.load_state_dict(state)
    first_epoch_rest = _all_ids_from_loader(loader_b)
    second_epoch = _all_ids_from_loader(loader_b)
    assert first_epoch_rest
    assert second_epoch
    assert len(second_epoch) == len(set(second_epoch))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_v1_checkpoint_without_sample_in_epoch_still_restripes(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=64, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=0, batch_size=4)
    _all_ids_from_loader(loader, max_batches=4)
    state = loader.state_dict()
    ds_state = dict(state["dataset"])
    yielded = ds_state["num_samples_yielded"]
    world = ds_state["world_size"]
    ds_state.pop("sample_in_epoch", None)
    ds_state.pop("state_version", None)
    ds_state.pop("resume_mode", None)
    ds_state["num_workers"] = 2
    state["dataset"] = ds_state
    assert sample_in_epoch_from_state(ds_state) == yielded * world

    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=2, batch_size=4)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert rest
    assert len(rest) == len(set(rest))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
@pytest.mark.parametrize("shuffle", [True, False])
def test_same_topology_pause_resume_equals_full_epoch(tmpdir, monkeypatch, shuffle):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=64, chunk_size=8)
    full = _all_ids_from_loader(
        StreamingDataLoader(StreamingDataset(data_dir, shuffle=shuffle, seed=42), num_workers=0, batch_size=4)
    )
    dataset = StreamingDataset(data_dir, shuffle=shuffle, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=0, batch_size=4)
    first = _all_ids_from_loader(loader, max_batches=5)
    dataset_b = StreamingDataset(data_dir, shuffle=shuffle, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=0, batch_size=4)
    loader_b.load_state_dict(loader.state_dict())
    rest = _all_ids_from_loader(loader_b)
    assert first + rest == full
    assert len(full) == len(set(full))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_worker_change_does_not_repeat_consumed_samples(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=96, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=0, batch_size=4)
    first = _all_ids_from_loader(loader, max_batches=7)
    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=6, batch_size=4)
    loader_b.load_state_dict(loader.state_dict())
    rest = _all_ids_from_loader(loader_b)
    assert set(first).isdisjoint(set(rest))
    assert len(rest) == len(set(rest))
    assert set(first) | set(rest) <= set(range(96))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_world_size_2_to_1_from_fresh_canonical_epoch(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=80, chunk_size=8)

    def make_ds(world, rank, state=None):
        ds = StreamingDataset(data_dir, shuffle=True, seed=42, drop_last=True)
        ds.distributed_env = _DistributedEnv(world, rank, 1)
        ds.batch_size = 4
        ds.num_workers = 1
        if state is not None:
            ds.load_state_dict(state)
        return ds

    consumed = []
    local_n = 8
    rank0_state = None
    for rank in (0, 1):
        ds = make_ds(2, rank)
        it = iter(ds)
        for _ in range(local_n):
            consumed.append(int(next(it)))
        if rank == 0:
            rank0_state = ds.state_dict(local_n, 1, 4)
    assert len(consumed) == len(set(consumed))
    assert rank0_state is not None
    rank0_state["sample_in_epoch"] = local_n * 2
    rank0_state["world_size"] = 2

    rest = [int(x) for x in make_ds(1, 0, rank0_state)]
    assert len(rest) == len(set(rest))
    assert set(consumed).isdisjoint(set(rest))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_force_override_still_restripes_worker_mismatch(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=64, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=0, batch_size=4)
    first = _all_ids_from_loader(loader, max_batches=4)
    state = loader.state_dict()
    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42, force_override_state_dict=True)
    loader_b = StreamingDataLoader(dataset_b, num_workers=4, batch_size=4)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert set(first).isdisjoint(set(rest))
    assert rest


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_shuffled_epochs_are_different(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=48, chunk_size=8)
    loader = StreamingDataLoader(StreamingDataset(data_dir, shuffle=True, seed=42), num_workers=0, batch_size=4)
    epoch1 = _all_ids_from_loader(loader)
    epoch2 = _all_ids_from_loader(loader)
    assert epoch1 != epoch2
    assert sorted(epoch1) == sorted(epoch2)


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_num_canonical_nodes_frozen_across_resume(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=64, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42, num_canonical_nodes=4)
    loader = StreamingDataLoader(dataset, num_workers=0, batch_size=4)
    _all_ids_from_loader(loader, max_batches=3)
    state = loader.state_dict()
    assert state["dataset"]["num_canonical_nodes"] == 4
    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42, num_canonical_nodes=4)
    loader_b = StreamingDataLoader(dataset_b, num_workers=2, batch_size=4)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert rest
    assert loader_b.state_dict()["dataset"]["num_canonical_nodes"] == 4


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_num_canonical_nodes_defaults_to_initial_world_size(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=64, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=0, batch_size=4)
    _all_ids_from_loader(loader, max_batches=3)
    state = loader.state_dict()
    assert state["dataset"]["num_canonical_nodes"] == 1
    assert state["dataset"]["initial_world_size"] == 1
    assert state["dataset"]["initial_num_nodes"] == 1
    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=4, batch_size=4)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert rest
    saved = loader_b.state_dict()["dataset"]
    assert saved["num_canonical_nodes"] == 1
    assert saved["initial_world_size"] == 1
    assert saved["initial_num_nodes"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_v1_same_topology_prefix_replay_still_runs(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=48, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=0, batch_size=4)
    _all_ids_from_loader(loader, max_batches=3)
    state = loader.state_dict()
    ds_state = dict(state["dataset"])
    ds_state.pop("state_version", None)
    ds_state.pop("resume_mode", None)
    ds_state.pop("sample_in_epoch", None)
    ds_state["num_workers"] = 1
    state["dataset"] = ds_state
    dataset_b = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader_b = StreamingDataLoader(dataset_b, num_workers=0, batch_size=4)
    loader_b.load_state_dict(state)
    rest = _all_ids_from_loader(loader_b)
    assert rest


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_drop_last_keeps_ranks_equal_length(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=64, chunk_size=8)

    def drain(rank):
        ds = StreamingDataset(data_dir, shuffle=True, seed=7, drop_last=True)
        ds.distributed_env = _DistributedEnv(2, rank, 1)
        ds.batch_size = 4
        ds.num_workers = 1
        return [int(x) for x in ds]

    a, b = drain(0), drain(1)
    assert len(a) == len(b)
    assert len(a) == len(set(a))
    assert set(a).isdisjoint(set(b))
