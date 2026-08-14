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
    canonical_item_stream,
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
    assert sample_in_epoch_from_state(state) == 80
    assert sample_in_epoch_from_state({"sample_in_epoch": 12, "num_samples_yielded": 3, "world_size": 8}) == 12


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
        if max_batches is not None and i >= max_batches:
            break
        ids.extend(torch.as_tensor(batch).reshape(-1).tolist())
    return ids


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_dataloader_elastic_workers_2_to_8(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(os.path.join(tmpdir, "data"), num_items=128, chunk_size=8)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    loader = StreamingDataLoader(dataset, num_workers=2, batch_size=4)
    first = _all_ids_from_loader(loader, max_batches=6)
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
    assert state["dataset"].get("resume_mode") != "elastic"
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
