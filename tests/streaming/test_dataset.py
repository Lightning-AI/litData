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

import json
import logging
import os
import random
import shutil
import sys
from time import sleep
from typing import Any, Dict, Optional
from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from litdata import optimize, train_test_split
from litdata.constants import _ZSTD_AVAILABLE
from litdata.processing import functions
from litdata.streaming import Cache
from litdata.streaming import dataset as dataset_module
from litdata.streaming import resolver as resolver_module
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import (
    _INDEX_FILENAME,
    Dir,
    StreamingDataset,
    _replay_chunks_sampling,
    _replay_sampling,
)
from litdata.streaming.item_loader import TokensLoader
from litdata.streaming.reader import BinaryReader
from litdata.streaming.shuffle import FullShuffle, NoShuffle
from litdata.utilities import dataset_utilities as dataset_utilities_module
from litdata.utilities.dataset_utilities import load_index_file
from litdata.utilities.env import _DistributedEnv, _WorkerEnv
from litdata.utilities.shuffle import _associate_chunks_and_intervals_to_workers
from tests.streaming.utils import filter_lock_files


def seed_everything(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
@pytest.mark.timeout(30)
def test_streaming_dataset(tmpdir, monkeypatch, compression):
    seed_everything(42)

    with pytest.raises(FileNotFoundError, match="The provided dataset path"):
        dataset = StreamingDataset(input_dir=str(tmpdir.join("tmpfolder")))

    with pytest.raises(ValueError, match="The provided dataset"):
        dataset = StreamingDataset(input_dir=str(tmpdir))

    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(60):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir))

    assert len(dataset) == 60
    for i in range(60):
        assert dataset[i] == i

    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 60
    for i in range(60):
        assert next(dataset_iter) == i

    dataloader = StreamingDataLoader(dataset, num_workers=0, batch_size=1)
    assert len(dataloader) == 60
    dataloader = DataLoader(dataset, num_workers=2, batch_size=1)
    assert len(dataloader) == 60
    dataloader = DataLoader(dataset, num_workers=2, batch_size=2)
    assert len(dataloader) == 30


def _simple_optimize_fn(index):
    return index


@pytest.mark.parametrize(
    ("chunk_bytes", "chunk_size"),
    [
        ("64MB", None),
        (None, 5),  # at max 5 items in a chunk
        (None, 75),  # at max 75 items in a chunk
        (None, 1200),  # at max 1200 items in a chunk
    ],
)
@pytest.mark.parametrize("keep_data_ordered", [True, False])
def test_optimize_dataset(keep_data_ordered, chunk_bytes, chunk_size, tmpdir, monkeypatch):
    data_dir = str(tmpdir / "optimized")

    optimize(
        fn=_simple_optimize_fn,
        inputs=list(range(1000)),
        output_dir=data_dir,
        num_workers=4,
        chunk_bytes=chunk_bytes,
        chunk_size=chunk_size,
        keep_data_ordered=keep_data_ordered,
    )

    sleep(2)  # wait for the cache to be created

    ds = StreamingDataset(input_dir=data_dir)

    expected_dataset = list(range(1000))
    actual_dataset = ds[:]

    assert len(actual_dataset) == len(expected_dataset)

    if not keep_data_ordered:
        # in shared queue, the order of the chunks is not guaranteed
        assert sorted(actual_dataset) == expected_dataset
    else:
        assert actual_dataset == expected_dataset


@pytest.mark.timeout(30)
def test_streaming_dataset_max_pre_download(tmpdir):
    seed_everything(42)

    cache = Cache(str(tmpdir), chunk_size=10)
    for i in range(60):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert len(dataset) == 60
    for i in range(60):
        assert dataset[i] == i
    assert dataset.cache._reader._max_pre_download == 2

    dataset = StreamingDataset(input_dir=str(tmpdir), max_pre_download=10)
    assert len(dataset) == 60
    for i in range(60):
        assert dataset[i] == i
    assert dataset.cache._reader._max_pre_download == 10


@pytest.mark.timeout(30)
def test_streaming_dataset_max_cache_dir(tmpdir, caplog):
    seed_everything(42)

    cache = Cache(str(tmpdir), chunk_size=10)
    for i in range(60):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert len(dataset) == 60
    for i in range(60):
        assert dataset[i] == i

    with caplog.at_level(logging.WARNING):
        StreamingDataset(input_dir=str(tmpdir), max_cache_size="25GB")
        StreamingDataset(input_dir=str(tmpdir), max_cache_size="30GB")
        StreamingDataset(input_dir=str(tmpdir), max_cache_size="50GB")
        StreamingDataset(input_dir=str(tmpdir), max_cache_size="100GB")
    assert len(caplog.messages) == 0

    with caplog.at_level(logging.WARNING):
        StreamingDataset(input_dir=str(tmpdir), max_cache_size="500MB")
        StreamingDataset(input_dir=str(tmpdir), max_cache_size="1GB")
        StreamingDataset(input_dir=str(tmpdir), max_cache_size="10GB")
        StreamingDataset(input_dir=str(tmpdir), max_cache_size="20GB")
    assert len(caplog.messages) == 4
    assert all("The provided `max_cache_size` is less than 25GB." in record.message for record in caplog.records), (
        "Expected warning about the `max_cache_size` being less than 25GB was not logged"
    )


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
@pytest.mark.timeout(30)
def test_streaming_dataset_distributed_no_shuffle(drop_last, tmpdir, compression):
    seed_everything(42)

    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(101):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=False, drop_last=drop_last)
    assert not dataset.shuffle
    _ = dataset[0]  # init shuffler
    assert isinstance(dataset.shuffler, NoShuffle)

    for i in range(101):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(1, 0, 1)
    assert len(dataset) == 101

    dataset.distributed_env = _DistributedEnv(2, 0, 1)
    assert len(dataset) == 50

    dataset.distributed_env = _DistributedEnv(2, 1, 1)
    assert len(dataset) == 50 + int(not drop_last)

    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50 + int(not drop_last)

    dataset.distributed_env = _DistributedEnv(2, 0, 1)

    process_1_1 = list(dataset_iter)

    assert len(process_1_1) == 50
    assert process_1_1[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset_iter = iter(dataset)

    assert len(dataset_iter) == 50
    process_1_2 = list(dataset_iter)
    assert process_1_2[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    assert len(process_1_2) == 50

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=False, drop_last=drop_last)
    dataset.distributed_env = _DistributedEnv(2, 1, 1)

    assert len(dataset) == 50 + int(not drop_last)
    dataset_iter = iter(dataset)

    process_2_1 = list(dataset_iter)
    assert process_2_1[:10] == [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

    assert len(process_2_1) == 50 + int(not drop_last)
    dataset_iter = iter(dataset)

    assert len(dataset_iter) == 50 + int(not drop_last)
    process_2_2 = list(dataset_iter)

    assert process_2_2[:10] == [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

    assert len(process_2_2) == 50 + int(not drop_last)

    _, workers_intervals = dataset.shuffler.get_chunks_and_intervals_per_workers(
        dataset.distributed_env, 1, 1, dataset.current_epoch
    )

    assert process_1_1 == process_1_2

    found_list = []
    for i in process_1_1:
        found = False
        for interval in workers_intervals[0]:
            if interval[1] <= i <= interval[2]:
                found = True
                break
        found_list.append(found)

    assert all(found_list) is True

    found_list = []
    for i in process_2_1:
        found = False
        for interval in workers_intervals[1]:
            if interval[1] <= i <= interval[2]:
                found = True
                break
        found_list.append(found)

    assert all(found_list) is True

    assert len([i for i in process_1_1 if i in process_2_1]) == 0
    assert len([i for i in process_1_2 if i in process_2_2]) == 0


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
@pytest.mark.timeout(60)
def test_streaming_dataset_distributed_full_shuffle_odd(drop_last, tmpdir, compression):
    seed_everything(42)

    cache = Cache(input_dir=str(tmpdir), chunk_size=10, compression=compression)
    for i in range(1097):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    assert dataset.shuffle
    _ = dataset[0]
    assert isinstance(dataset.shuffler, FullShuffle)

    for i in range(1097):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(2, 0, 1)
    assert len(dataset) == 548
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 548
    process_1_1 = list(dataset_iter)
    assert process_1_1[:10] == [531, 536, 538, 530, 535, 537, 534, 539, 533, 532]
    assert len(process_1_1) == 548

    dataset_2 = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    iter(dataset_2)
    assert isinstance(dataset_2.shuffler, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(2, 1, 1)
    assert len(dataset_2) == 548 + int(not drop_last)
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 548 + int(not drop_last)
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [248, 249, 884, 887, 888, 883, 886, 882, 889, 880]
    assert len(process_2_1) == 548 + int(not drop_last)
    assert len([i for i in process_1_1 if i in process_2_1]) == 0


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param(
            "zstd",
            marks=pytest.mark.skipif(
                condition=not _ZSTD_AVAILABLE or sys.platform == "darwin", reason="Requires: ['zstd']"
            ),
        ),
    ],
)
@pytest.mark.timeout(60)
def test_streaming_dataset_distributed_full_shuffle_even(drop_last, tmpdir, compression):
    seed_everything(42)

    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(1222):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    assert dataset.shuffle
    _ = dataset[0]
    assert isinstance(dataset.shuffler, FullShuffle)

    for i in range(1222):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(2, 0, 1)
    assert len(dataset) == 611
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 611
    process_1_1 = list(dataset_iter)
    assert process_1_1[:10] == [278, 272, 270, 273, 276, 275, 274, 271, 277, 279]
    assert len(process_1_1) == 611

    dataset_2 = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    iter(dataset_2)
    assert isinstance(dataset_2.shuffler, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(2, 1, 1)
    assert len(dataset_2) == 611
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 611
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [999, 993, 991, 994, 997, 996, 995, 992, 998, 527]
    assert len(process_2_1) == 611
    assert len([i for i in process_1_1 if i in process_2_1]) == 0


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
@pytest.mark.timeout(60)
def test_streaming_dataset_distributed_full_shuffle_even_multi_nodes(drop_last, tmpdir, compression):
    seed_everything(42)

    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(1222):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    assert dataset.shuffle
    _ = dataset[0]
    assert isinstance(dataset.shuffler, FullShuffle)

    for i in range(1222):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(4, 0, 2)
    assert len(dataset) == 305
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 305
    process_1_1 = list(dataset_iter)
    assert process_1_1[:10] == [271, 273, 276, 272, 279, 270, 274, 275, 278, 277]
    assert len(process_1_1) == 305

    dataset_2 = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    iter(dataset_2)
    assert isinstance(dataset_2.shuffler, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(4, 1, 2)
    assert len(dataset_2) == 305
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 305
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [418, 417, 419, 416, 415, 348, 341, 343, 347, 346]
    assert len(process_2_1) == 305
    assert len([i for i in process_1_1 if i in process_2_1]) == 0

    dataset_2 = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    iter(dataset_2)
    assert isinstance(dataset_2.shuffler, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(4, 1, 2)
    dataset_2.current_epoch = 2
    assert len(dataset_2) == 310
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 310
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [231, 236, 232, 235, 234, 238, 239, 237, 230, 233]
    assert len(process_2_1) == 310
    assert len([i for i in process_1_1 if i in process_2_1]) != 0


def test_streaming_dataset_deepcopy(tmpdir):
    seed_everything(42)

    remote_dir = os.path.join(tmpdir, "remote_dir")

    os.makedirs(remote_dir, exist_ok=True)

    cache = Cache(remote_dir, chunk_size=10)
    for i in range(10):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=remote_dir, shuffle=True)
    assert dataset.cache is None
    iter(dataset)
    assert dataset.cache is not None
    assert dataset.cache._reader._prepare_thread is None
    dataset.cache._reader._prepare_thread = True
    dataloader = DataLoader(dataset, num_workers=1)

    batches = []
    for batch in dataloader:
        batches.append(batch)

    assert len(batches) == 10


def test_dataset_cache_recreation(tmpdir):
    """Test that we recreate the cache and other objects only when appropriate."""
    cache = Cache(str(tmpdir), chunk_size=10)
    for i in range(10):
        cache[i] = i
    cache.done()
    cache.merge()

    # repated `len()` calls
    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert not dataset.cache
    assert not dataset.shuffler
    len(dataset)
    assert not dataset.cache
    shuffler = dataset.shuffler
    assert isinstance(shuffler, NoShuffle)
    len(dataset)
    assert dataset.shuffler is shuffler

    # repeated `iter()` calls
    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert not dataset.cache
    assert not dataset.shuffler
    iter(dataset)
    cache = dataset.cache
    shuffler = dataset.shuffler
    assert isinstance(cache, Cache)
    assert isinstance(shuffler, NoShuffle)
    iter(dataset)
    assert isinstance(dataset.cache, Cache)
    assert isinstance(dataset.shuffler, NoShuffle)
    assert dataset.cache is not cache  # cache gets recreated
    assert dataset.shuffler is not shuffler  # shuffler gets recreated

    # repeated `getitem()` calls
    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert not dataset.cache
    assert not dataset.shuffler
    _ = dataset[0]
    cache = dataset.cache
    shuffler = dataset.shuffler
    assert isinstance(cache, Cache)
    assert isinstance(shuffler, NoShuffle)
    _ = dataset[1]
    assert dataset.cache is cache  # cache gets reused
    assert dataset.shuffler is shuffler  # shuffler gets reused


def test_dataset_for_text_tokens(tmpdir):
    seed_everything(42)

    block_size = 1024 + 1
    cache = Cache(input_dir=str(tmpdir), chunk_size=block_size * 11, item_loader=TokensLoader(block_size))
    text_idxs_list = []

    counter = 0
    while True:
        text_ids = torch.randint(0, 1000, (np.random.randint(0, 1000),)).to(torch.int)
        text_idxs_list.append(text_ids)
        chunk_filepath = cache._add_item(counter, text_ids)
        if chunk_filepath:
            break
        counter += 1

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size))

    assert len(dataset) == 10

    cache_0 = dataset[0]
    cache_1 = dataset[1]
    cache_2 = dataset[2]
    cache_3 = dataset[3]
    assert len(cache_0) == block_size
    assert len(cache_1) == block_size
    assert not torch.equal(cache_0, cache[1])
    indices = torch.cat(text_idxs_list, dim=0)
    assert torch.equal(cache_0, indices[: len(cache_0)])
    assert torch.equal(cache_1, indices[len(cache_0) : len(cache_0) + len(cache_1)])

    dataloader = DataLoader(StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size)), batch_size=2)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            assert torch.equal(torch.stack([cache_0, cache_1]), batch)
        elif batch_idx == 1:
            assert torch.equal(torch.stack([cache_2, cache_3]), batch)
        else:
            break


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_dataset_for_text_tokens_with_large_num_chunks(tmpdir):
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (512, 512))

    block_size = 1024
    cache = Cache(input_dir=str(tmpdir), chunk_bytes="10KB", item_loader=TokensLoader(block_size))

    for i in range(1500):
        text_ids = torch.randint(0, 10001, (torch.randint(100, 1001, (1,)).item(),)).numpy()
        cache._add_item(i, text_ids)

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=True)
    for _ in dataset:
        pass


def test_dataset_with_1d_array(tmpdir):
    seed_everything(42)

    cache = Cache(input_dir=str(tmpdir), chunk_size=100)
    text_idxs_list = []

    for i in range(100):
        text_ids = torch.randint(0, 1000, (np.random.randint(0, 1000),)).to(torch.int)
        text_idxs_list.append(text_ids)

        chunk_filepath = cache._add_item(i, text_ids)
        if chunk_filepath:
            print(i)
            break

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=False)

    assert len(dataset) == 100

    for i in range(100):
        generated = dataset[i]
        expected = text_idxs_list[i]
        assert torch.equal(expected, generated)


def test_dataset_for_text_tokens_multiple_workers(tmpdir):
    seed_everything(42)

    block_size = 10
    cache = Cache(input_dir=str(tmpdir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(10):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    for i in range(20):
        sequence = cache[i]
        assert sequence[0].item() == i * block_size
        assert sequence[-1].item() == (i + 1) * block_size - 1

    assert len(os.listdir(tmpdir)) == 6

    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)

    assert len(dataset) == 20

    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)

    assert len(dataloader) == 10

    expected = [
        [0, 10],
        [100, 110],
        [20, 30],
        [120, 130],
        [40, 50],
        [140, 150],
        [60, 70],
        [160, 170],
        [80, 90],
        [180, 190],
    ]

    result = []
    for batch in dataloader:
        result.append(batch[:, 0].tolist())
    assert result == expected


@pytest.mark.timeout(60)
def test_dataset_for_text_tokens_with_large_block_size_multiple_workers(tmpdir):
    # test to reproduce ERROR: Unexpected segmentation fault encountered in worker
    seed_everything(42)

    block_size = 2048 + 1
    cache = Cache(input_dir=str(tmpdir), chunk_bytes="64MB", item_loader=TokensLoader(block_size))

    for i in range(5000):
        text_ids = torch.randint(low=0, high=127, size=(8192,))
        cache[i] = text_ids

    cache.done()
    cache.merge()

    dataset = StreamingDataset(
        input_dir=str(tmpdir),
        item_loader=TokensLoader(block_size=2049),
        shuffle=True,
        drop_last=True,
    )
    dataloader = StreamingDataLoader(dataset, batch_size=8, num_workers=4, shuffle=True, drop_last=True)

    for _ in dataloader:
        pass


def test_dataset_for_text_tokens_distributed_num_workers(tmpdir):
    seed_everything(42)

    block_size = 10
    cache = Cache(input_dir=str(tmpdir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(10):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    for i in range(20):
        sequence = cache[i]
        assert sequence[0].item() == i * block_size
        assert sequence[-1].item() == (i + 1) * block_size - 1

    assert len([f for f in os.listdir(tmpdir) if f.endswith(".bin")]) == 5

    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)

    assert len(dataset) == 20

    dataset.distributed_env = _DistributedEnv(2, 0, 1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    assert len(dataloader) == 5

    expected = [[0, 10], [20, 30], [40, 50], [60, 70], [80, 90]]

    for batch_idx, batch in enumerate(dataloader):
        assert [batch[0][0].item(), batch[1][0].item()] == expected[batch_idx]

    dataset.distributed_env = _DistributedEnv(2, 1, 1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    assert len(dataloader) == 5

    expected = [[100, 110], [120, 130], [140, 150], [160, 170], [180, 190]]

    for batch_idx, batch in enumerate(dataloader):
        assert [batch[0][0].item(), batch[1][0].item()] == expected[batch_idx]

    for batch_idx, batch in enumerate(dataloader):
        assert [batch[0][0].item(), batch[1][0].item()] == expected[batch_idx]


def optimize_fn(item):
    return torch.arange(item[0], item[0] + 20).to(torch.int)


@pytest.mark.flaky(reruns=3)
def test_dataset_for_text_tokens_distributed_num_workers_end_to_end(tmpdir, monkeypatch):
    monkeypatch.setattr(functions, "_get_input_dir", lambda x: str(tmpdir))

    seed_everything(42)

    with open(tmpdir / "a.txt", "w") as f:
        f.write("hello")

    inputs = [(v, str(tmpdir / "a.txt")) for v in range(0, 200, 20)]

    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    functions.optimize(
        optimize_fn,
        inputs,
        output_dir=str(tmpdir),
        num_workers=2,
        chunk_size=2,
        reorder_files=False,
        num_downloaders=1,
        item_loader=TokensLoader(),
    )

    assert len([f for f in os.listdir(tmpdir) if f.endswith(".bin")]) == 10

    block_size = 10
    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)

    L = len(dataset)
    assert L == 20

    for i in range(L):
        sequence = dataset[i]
        assert sequence[0].item() == i * block_size
        assert sequence[-1].item() == (i + 1) * block_size - 1

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("GLOBAL_RANK", "0")
    monkeypatch.setenv("NNODES", "1")
    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)
    dataloader = StreamingDataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    assert dataset.drop_last  # in distributed setting, this is forced automatically

    # L = 20, world size 2, num workers 2
    # L / (2 * 2) = 5 items per worker
    #
    # `utilities::shuffle::_associate_chunks_and_intervals_to_workers`
    #       -> will associate 4 items to one worker and 6 items to other worker
    #
    # drop last -> no effect as each worker has complete batches (though one will produce 1 extra batch)
    # one worker will yield 2 batches, other will yield 3 batches => len(dataloader) = 5
    assert len(dataloader) == 5

    expected = [[0, 10], [60, 70], [20, 30], [80, 90], [40, 50]]
    returned = []
    for batch in dataloader:
        returned.append(batch[:, 0].tolist())
    assert returned == expected

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("GLOBAL_RANK", "1")
    monkeypatch.setenv("NNODES", "1")
    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)
    dataloader = StreamingDataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    assert dataset.drop_last  # in distributed setting, this is forced automatically

    assert len(dataloader) == 5

    expected = [[100, 110], [160, 170], [120, 130], [180, 190], [140, 150]]
    returned = []
    for batch in dataloader:
        returned.append(batch[:, 0].tolist())
    assert returned == expected


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
def test_s3_streaming_dataset(monkeypatch):
    downloader = mock.MagicMock()

    def fn(remote_chunkpath: str, local_chunkpath: str):
        with open(local_chunkpath, "w") as f:
            json.dump({"chunks": [{"chunk_size": 2, "filename": "0.bin"}]}, f)

    downloader.download_file = fn

    monkeypatch.setattr(dataset_utilities_module, "get_downloader", mock.MagicMock(return_value=downloader))

    dataset = StreamingDataset(input_dir="s3://pl-flash-data/optimized_tiny_imagenet")
    assert dataset.input_dir.url == "s3://pl-flash-data/optimized_tiny_imagenet"
    assert dataset.input_dir.path.endswith(
        "chunks/597d6184e3ba942b36c8b6357a890033/597d6184e3ba942b36c8b6357a890033"
    )  # it won't be None, and a cache dir will be created


class EmulateS3StreamingDataset(StreamingDataset):
    def _create_cache(self, worker_env: _WorkerEnv) -> Cache:
        cache_dir = os.path.join(self.input_dir.path)
        os.makedirs(cache_dir, exist_ok=True)

        cache = Cache(
            input_dir=Dir(cache_dir, self.input_dir.url),
            item_loader=self.item_loader,
            chunk_bytes=1,
            serializers=self.serializers,
        )
        cache._reader._try_load_config()

        if not cache.filled:
            raise ValueError(
                f"The provided dataset `{self.input_dir}` doesn't contain any {_INDEX_FILENAME} file."
                " HINT: Did you successfully optimize a dataset to the provided `input_dir`?"
            )

        return cache


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
def test_dataset_reshuffling_every_epoch(tmpdir):
    seed_everything(42)

    data_dir = os.path.join(tmpdir, "data")
    cache_dir = os.path.join(tmpdir, "cache_dir")

    os.makedirs(data_dir)
    os.makedirs(cache_dir)

    block_size = 20
    cache = Cache(input_dir=str(data_dir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(100):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    assert len([f for f in os.listdir(data_dir) if f.endswith(".bin")]) == 50

    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir), item_loader=TokensLoader(block_size), shuffle=True
    )

    dataset.current_epoch = 1
    dataloader = StreamingDataLoader(dataset, num_workers=2, batch_size=2, prefetch_factor=1)

    dataloader_iter = iter(dataloader)

    _ = next(dataloader_iter)
    state_dict_0 = dataloader.state_dict()

    assert state_dict_0["dataset"]["num_samples_yielded"] == 2
    assert state_dict_0["dataset"]["num_workers"] == 2
    assert state_dict_0["dataset"]["batch_size"] == 2

    _ = next(dataloader_iter)
    state_dict_1 = dataloader.state_dict()
    assert state_dict_1["dataset"]["num_samples_yielded"] == 4
    assert state_dict_1["dataset"]["num_workers"] == 2
    assert state_dict_1["dataset"]["batch_size"] == 2

    batch_2 = next(dataloader_iter)
    state_dict_2 = dataloader.state_dict()
    assert state_dict_2["dataset"]["num_samples_yielded"] == 6
    assert state_dict_2["dataset"]["num_workers"] == 2
    assert state_dict_2["dataset"]["batch_size"] == 2

    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir),
        item_loader=TokensLoader(block_size),
        shuffle=True,
    )

    dataloader = StreamingDataLoader(dataset, num_workers=2, batch_size=2, prefetch_factor=1)
    dataloader.load_state_dict(state_dict_1)

    dataloader_iter = iter(dataloader)
    batch_0_restart = next(dataloader_iter)

    state_dict_2 = dataloader.state_dict()["dataset"]

    assert state_dict_2["num_samples_yielded"] == 6
    assert state_dict_2["num_workers"] == 2
    assert state_dict_2["batch_size"] == 2

    assert torch.equal(batch_2, batch_0_restart)

    assert len(os.listdir(cache_dir)) >= 5


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
def test_resumable_dataset_two_workers_2_epochs(tmpdir):
    seed_everything(42)

    data_dir = os.path.join(tmpdir, "data")
    cache_dir = os.path.join(tmpdir, "cache_dir")

    os.makedirs(data_dir)
    os.makedirs(cache_dir)

    block_size = 20
    cache = Cache(input_dir=str(data_dir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(100):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    assert len([f for f in os.listdir(data_dir) if f.endswith(".bin")]) == 50

    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir), item_loader=TokensLoader(block_size), shuffle=True
    )

    dataloader = StreamingDataLoader(dataset, num_workers=2, batch_size=2, prefetch_factor=1, persistent_workers=True)

    batches_epoch_1 = []
    for batch in dataloader:
        batches_epoch_1.append(batch)

    assert len(filter_lock_files(os.listdir(cache_dir))) == 51

    batches_epoch_2 = []
    for batch in dataloader:
        batches_epoch_2.append(batch)

    assert len(filter_lock_files(os.listdir(cache_dir))) == 51
    assert not all(torch.equal(b1, b2) for b1, b2 in zip(batches_epoch_1, batches_epoch_2))


def _simple_preprocess(_):
    for _ in range(10):
        yield torch.randint(0, 100, size=(10,), dtype=torch.int64)


def _get_simulated_s3_dataloader(cache_dir, data_dir, shuffle=False):
    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir),
        item_loader=TokensLoader(block_size=10),
        shuffle=shuffle,
    )
    return StreamingDataLoader(dataset, batch_size=2, num_workers=2)


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
@mock.patch.dict(os.environ, {}, clear=True)
@pytest.mark.timeout(60)
@pytest.mark.parametrize("shuffle", [True, False])
def test_dataset_resume_on_future_chunks(shuffle, tmpdir, monkeypatch):
    """Tests resuming from a chunk past the first chunk, when subsequent chunks don't have the same size."""
    s3_cache_dir = str(tmpdir / "s3cache")
    optimize_data_cache_dir = str(tmpdir / "optimize_data_cache")
    optimize_cache_dir = str(tmpdir / "optimize_cache")
    data_dir = str(tmpdir / "optimized")
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", optimize_data_cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", optimize_cache_dir)

    optimize(
        fn=_simple_preprocess,
        inputs=list(range(8)),
        output_dir=data_dir,
        chunk_size=190,
        num_workers=4,
        num_uploaders=1,
        item_loader=TokensLoader(block_size=10),
    )
    assert set(os.listdir(data_dir)) == {
        "chunk-0-0.bin",
        "chunk-0-1.bin",
        "chunk-1-0.bin",
        "chunk-1-1.bin",
        "chunk-2-0.bin",
        "chunk-2-1.bin",
        "chunk-3-0.bin",
        "chunk-3-1.bin",
        "index.json",
    }

    os.mkdir(s3_cache_dir)
    train_dataloader = _get_simulated_s3_dataloader(s3_cache_dir, data_dir, shuffle=shuffle)
    batches_to_fetch = 16
    batch_to_resume_from = None
    dataloader_state = None

    for i, batch in enumerate(train_dataloader):
        if i == batches_to_fetch:
            dataloader_state = train_dataloader.state_dict()
        if i == batches_to_fetch + 1:
            batch_to_resume_from = batch
            break

    shutil.rmtree(s3_cache_dir)
    os.mkdir(s3_cache_dir)
    train_dataloader = _get_simulated_s3_dataloader(s3_cache_dir, data_dir, shuffle=shuffle)
    assert dataloader_state is not None
    assert batch_to_resume_from is not None
    train_dataloader.load_state_dict(dataloader_state)
    # The next batch after resuming must match what we should have gotten next in the initial loop
    assert torch.equal(next(iter(train_dataloader)), batch_to_resume_from)


@pytest.mark.timeout(60)
@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
def test_dataset_valid_state(tmpdir, monkeypatch):
    seed_everything(42)

    index_json_content: Optional[Dict[str, Any]] = None

    def mock_resolve_dataset(dir_path: str) -> Dir:
        return Dir(
            path=dir_path,
            url=os.path.join(
                "s3://dummy_bucket/projects/project_id/datasets/",
                *dir_path.split("/")[3:],
            ),
        )

    downloader = mock.MagicMock()

    def fn(remote_chunkpath: str, local_chunkpath: str):
        assert index_json_content is not None
        with open(local_chunkpath, "w") as f:
            json.dump(index_json_content, f)

    downloader.download_file = fn

    monkeypatch.setattr(resolver_module, "_resolve_datasets", mock_resolve_dataset)
    monkeypatch.setattr(dataset_utilities_module, "get_downloader", mock.MagicMock(return_value=downloader))

    data_dir = os.path.join(tmpdir, "data")
    cache_dir = os.path.join(tmpdir, "cache_dir")

    os.makedirs(data_dir)
    os.makedirs(cache_dir)

    block_size = 20
    cache = Cache(input_dir=str(data_dir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(100):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    index_json_content = load_index_file(data_dir)

    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir),
        item_loader=TokensLoader(block_size),
        shuffle=False,
        drop_last=False,
    )
    dataloader = DataLoader(dataset, num_workers=1, batch_size=2)
    dataloader_iter = iter(dataloader)
    next(dataloader_iter)

    sleep(1)

    state_dict = dataset.state_dict(0, 1, 2)

    dataset.load_state_dict(state_dict)
    dataset.worker_env = _WorkerEnv(world_size=1, rank=0)
    dataset.cache = cache

    dataset._validate_state_dict()

    state_dict["drop_last"] = True
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match="The provided `drop_last` state doesn't match the current one. Found `False` instead of `True`.",
    ):
        dataset._validate_state_dict()

    state_dict["item_loader"] = {}
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match="The provided `item_loader` state doesn't match the current one."
        " Found `{'block_size': 20}` instead of `{}`.",
    ):
        dataset._validate_state_dict()

    state_dict["seed"] = 12
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match="The provided `seed` state doesn't match the current one. Found `42` instead of `12`.",
    ):
        dataset._validate_state_dict()

    state_dict["input_dir_url"] = "toto"
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match=f"The provided `input_dir` URL state doesn't match the current one."
        f" Found `{data_dir}` instead of `toto`.",
    ):
        dataset._validate_state_dict()

    state_dict["input_dir_path"] = "toto"
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match=f"The provided `input_dir` path state doesn't match the current one."
        f" Found `{cache_dir}` instead of `toto`.",
    ):
        dataset._validate_state_dict()

    state_dict["input_dir_path"] = "/teamspace/datasets/coco"
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match=f"The provided `input_dir` path state doesn't match the current one. Found `{cache_dir}` instead of ",
    ):
        dataset._validate_state_dict()

    state_dict["num_workers"] = "8"
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match="The provided `num_workers` state doesn't match the current one. Found `1` instead of `8`.",
    ):
        dataset._validate_state_dict()

    state_dict["shuffle"] = True
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match="The provided `shuffle` state doesn't match the current one. Found `False` instead of `True`.",
    ):
        dataset._validate_state_dict()


@pytest.mark.timeout(60)
@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
def test_dataset_valid_state_override(tmpdir, monkeypatch):
    seed_everything(42)

    index_json_content: Optional[Dict[str, Any]] = None

    def mock_resolve_dataset(dir_path: str) -> Dir:
        return Dir(
            path=dir_path,
            url=os.path.join(
                "s3://dummy_bucket/projects/project_id/datasets/",
                *dir_path.split("/")[3:],
            ),
        )

    downloader = mock.MagicMock()

    def fn(remote_chunkpath: str, local_chunkpath: str):
        assert index_json_content is not None
        with open(local_chunkpath, "w") as f:
            json.dump(index_json_content, f)

    downloader.download_file = fn

    monkeypatch.setattr(resolver_module, "_resolve_datasets", mock_resolve_dataset)
    monkeypatch.setattr(dataset_utilities_module, "get_downloader", mock.MagicMock(return_value=downloader))

    data_dir = os.path.join(tmpdir, "data")
    cache_dir = os.path.join(tmpdir, "cache_dir")

    os.makedirs(data_dir)
    os.makedirs(cache_dir)

    block_size = 20
    cache = Cache(input_dir=str(data_dir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(100):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    index_json_content = load_index_file(data_dir)

    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir),
        item_loader=TokensLoader(block_size),
        shuffle=False,
        drop_last=False,
        force_override_state_dict=True,
    )
    dataloader = DataLoader(dataset, num_workers=1, batch_size=2)
    dataloader_iter = iter(dataloader)
    next(dataloader_iter)

    sleep(1)

    state_dict = dataset.state_dict(0, 1, 2)

    dataset.load_state_dict(state_dict)
    dataset.worker_env = _WorkerEnv(world_size=1, rank=0)
    dataset.cache = cache

    dataset._validate_state_dict()

    state_dict["drop_last"] = True
    dataset.load_state_dict(state_dict)
    dataset._validate_state_dict()
    assert state_dict["drop_last"] is False, "drop_last not overridden"

    state_dict["item_loader"] = {}
    dataset.load_state_dict(state_dict)
    dataset._validate_state_dict()
    assert state_dict["item_loader"] == {"block_size": 20}, "item_loader not overridden"

    state_dict["seed"] = 12
    dataset.load_state_dict(state_dict)
    dataset._validate_state_dict()
    assert state_dict["seed"] == 42, "seed not overridden"

    state_dict["input_dir_url"] = "toto"
    dataset.load_state_dict(state_dict)
    dataset._validate_state_dict()
    assert state_dict["input_dir_url"] == data_dir, "input_dir_url not overridden"

    state_dict["input_dir_path"] = "toto"
    dataset.load_state_dict(state_dict)
    dataset._validate_state_dict()
    assert state_dict["input_dir_path"] == cache_dir, "input_dir_path not overridden"

    state_dict["num_workers"] = "8"
    dataset.load_state_dict(state_dict)
    dataset._validate_state_dict()
    assert state_dict["num_workers"] == 1, "num_workers not overridden"

    state_dict["shuffle"] = True
    dataset.load_state_dict(state_dict)
    dataset._validate_state_dict()
    assert state_dict["shuffle"] is False, "shuffle not overridden"


def test_replay_sampling():
    assert _replay_sampling(27, 8, 2) == {0: 16, 1: 11}  # {0: 8 + 8, 1: 8 + 3}
    assert _replay_sampling(27, 7, 2) == {0: 14, 1: 13}  # {0: 7 + 7, 1: 7 + 6}
    assert _replay_sampling(27, 6, 2) == {0: 15, 1: 12}  # {0: 6 + 6 + 3, 1: 6 + 6}
    assert _replay_sampling(27, 5, 2) == {0: 15, 1: 12}  # {0: 5 + 5 + 5, 1: 5 + 5 + 2}
    assert _replay_sampling(27, 4, 2) == {0: 15, 1: 12}  # {0: 4 + 4 + 4 + 3, 1: 4 + 4 + 4}
    assert _replay_sampling(27, 8, 3) == {0: 11, 1: 8, 2: 8}  # {0: 8 + 3, 1: 8, 2: 8}
    assert _replay_sampling(27, 4, 3) == {0: 11, 1: 8, 2: 8}  # {0: 4 + 4 + 3, 1: 4 + 4, 2: 4 + 4}


def test_replay_chunks_sampling():
    chunks_replica = range(10)
    intervals_replica = [(i, i, i + 5, i + 5) for i in range(0, 50, 5)]
    workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
        _DistributedEnv(2, 0, 1), chunks_replica, intervals_replica
    )
    assert workers_chunks == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert workers_intervals == [
        [[0, 0, 5, 5], [5, 5, 10, 10], [10, 10, 15, 15], [15, 15, 20, 20], [20, 20, 25, 25]],
        [[25, 25, 30, 30], [30, 30, 35, 35], [35, 35, 40, 40], [40, 40, 45, 45], [45, 45, 50, 50]],
    ]
    workers_intervals = {i: workers_intervals[i] for i in range(len(workers_intervals))}
    assert _replay_chunks_sampling(workers_intervals, {0: 16, 1: 11}) == ({0: 3, 1: 2}, {0: 1, 1: 1})
    assert _replay_chunks_sampling(workers_intervals, {0: 14, 1: 13}) == ({0: 2, 1: 2}, {0: 4, 1: 3})
    assert _replay_chunks_sampling(workers_intervals, {0: 15, 1: 12}) == ({0: 3, 1: 2}, {0: 0, 1: 2})

    # Test that replay stops at the right chunk
    workers_intervals = {0: [(0, 0, 10, 10), (10, 10, 20, 20), (20, 20, 21, 21), (21, 21, 30, 30)]}
    indexes = {0: 15}
    # Replay should stop at chunk index 1, because 15 - 10 = 5, which fits into with chunk idx 1
    chunk_indexes, indexes = _replay_chunks_sampling(workers_intervals, indexes)
    assert chunk_indexes == {0: 1}
    assert indexes == {0: 5}


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
def test_dataset_distributed_drop_last(tmpdir, monkeypatch, compression):
    class _DistributedEnvMock:
        def detect(cls):
            return _DistributedEnv(2, 0, 1)

    logger_mock = mock.MagicMock()

    monkeypatch.setattr(dataset_module, "_DistributedEnv", _DistributedEnvMock())
    monkeypatch.setattr(dataset_module, "logger", logger_mock)

    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(60):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir), drop_last=None)
    assert dataset.drop_last

    dataset = StreamingDataset(str(tmpdir), drop_last=False)
    assert not dataset.drop_last

    warn_msg = logger_mock.warning._mock_mock_calls[0].args[0]
    expected_warn_msg = (
        "You're operating within a distributed environment and have disabled the `drop_last` option."
        " Please note that this configuration may lead to training interruptions"
        " if your system depends on distributed collectives."
    )
    assert expected_warn_msg == warn_msg


@pytest.mark.flaky(reruns=3)
def test_subsample_streaming_dataset_with_token_loader(tmpdir, monkeypatch):
    monkeypatch.setattr(functions, "_get_input_dir", lambda x: str(tmpdir))

    seed_everything(42)

    with open(tmpdir / "a.txt", "w") as f:
        f.write("hello")

    inputs = [(v, str(tmpdir / "a.txt")) for v in range(0, 200, 20)]

    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    functions.optimize(
        optimize_fn,
        inputs,
        output_dir=str(tmpdir),
        num_workers=2,
        chunk_size=2,
        reorder_files=False,
        num_downloaders=1,
        item_loader=TokensLoader(),
    )

    assert len([f for f in os.listdir(tmpdir) if f.endswith(".bin")]) == 10

    block_size = 10
    dataset1 = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)
    dataset2 = StreamingDataset(
        input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False, subsample=0.4
    )

    assert len(dataset2) == int(len(dataset1) * 0.4)

    dataset3 = StreamingDataset(
        input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False, subsample=2.5
    )
    assert len(dataset3) == int(len(dataset1) * 2.5)


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_dataset_with_mosaic_mds_data(tmpdir):
    from PIL import Image
    from streaming import MDSWriter
    # example taken from: https://github.com/mosaicml/streaming

    # A dictionary mapping input fields to their data types
    columns = {"image": "jpeg", "class": "int"}
    # Shard compression, if any
    compression = "zstd"
    # Save the samples as shards using MDSWriter
    with MDSWriter(out=str(tmpdir), columns=columns, compression=compression) as out:
        for i in range(10):
            sample = {
                "image": Image.fromarray(np.random.randint(0, 256, (32, 32, 3), np.uint8)),
                "class": i,
            }
            out.write(sample)

    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert len(dataset) == 10
    for i in range(10):
        sample = dataset[i]
        assert sample["class"] == i

    assert [sample["class"] for sample in dataset[:]] == list(range(10))  # test slicing

    # -------------- train_test_split --------------

    train_ds, test_ds, val_ds = train_test_split(dataset, splits=[0.7, 0.2, 0.1])

    assert len(train_ds) == 7
    assert len(test_ds) == 2
    assert len(val_ds) == 1

    # -------------- subsample --------------

    dataset = StreamingDataset(input_dir=str(tmpdir), subsample=0.4)
    assert len(dataset) == 4
    assert [sample["class"] for sample in dataset[:]] == [0, 1, 2, 3]

    # -------------- and supersample ---------------

    dataset = StreamingDataset(input_dir=str(tmpdir), subsample=1.5)
    assert len(dataset) == 15
    assert [sample["class"] for sample in dataset[:]] == [x % 10 for x in range(15)]

    # -------------- works with dataloader --------------

    dataset = StreamingDataset(input_dir=str(tmpdir))
    dataloader = DataLoader(dataset, batch_size=4, drop_last=True)
    i = 0
    for batch in dataloader:
        assert len(batch["class"]) == 4
        assert len(batch["image"]) == 4
        assert list(batch["class"]) == [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        i += 1

    dataloader = DataLoader(dataset, batch_size=4, drop_last=False)
    i = 0
    for batch in dataloader:
        if i == 2:
            # last batch is smaller than batch_size
            assert len(batch["class"]) == 2
            assert len(batch["image"]) == 2
            assert list(batch["class"]) == [4 * i, 4 * i + 1]
            break
        assert len(batch["class"]) == 4
        assert len(batch["image"]) == 4
        assert list(batch["class"]) == [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        i += 1


@pytest.mark.parametrize("shuffle", [True, False])
def test_is_last_index_for_chunked_index_with_dataset(tmpdir, shuffle):
    # Create a dataset with 50 items, 10 items per chunk
    cache = Cache(str(tmpdir), chunk_size=10)
    for i in range(50):
        cache[i] = i
    cache.done()
    cache.merge()

    # List to store all ChunkedIndex objects passed to BinaryReader.read
    chunked_indexes = []

    # Patch the BinaryReader.read method to track the indices
    original_read = BinaryReader.read

    # Create a mock function that will capture the indices but still call the original
    def mock_read(self, index):
        chunked_indexes.append(index)
        return original_read(self, index)  # Call the original read method

    # Patch the read method directly in the BinaryReader class
    with patch("litdata.streaming.reader.BinaryReader.read", mock_read):
        dataset = StreamingDataset(str(tmpdir), shuffle=shuffle)
        assert len(dataset) == 50

        # Iterate through the dataset to trigger BinaryReader.read
        for _ in dataset:
            pass

    # Assertions
    # Ensure BinaryReader.read was called 50 times (once for each item)
    assert len(chunked_indexes) == 50, "Expected 50 calls to BinaryReader.read"

    # first chunked index has the chunk_indexes from dataset worker
    worker_chunks = chunked_indexes[0].chunk_indexes
    assert worker_chunks == dataset.worker_chunks, "Expected chunk_indexes to match dataset.worker_chunks"

    # Verify that exactly one index has is_last_index=True
    indexes = [idx for idx in chunked_indexes if idx.is_last_index]
    assert len(indexes) == 1, "Expected exactly one index with is_last_index=True"
    assert indexes[0].is_last_index, "Expected is_last_index=True for the last item"
    assert indexes[0].chunk_index == worker_chunks[-1], "Expected to match the last chunk"


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_dataset_as_iterator_and_non_iterator(tmpdir, local, shuffle):
    """Test that _chunks_queued_for_download flag is correctly set and reset in reader.

    This test verifies that:
    1. When iterating, _chunks_queued_for_download is enabled during iteration but reset when done
    2. When accessing by index, _chunks_queued_for_download is never enabled
    """
    # Create directories
    cache_dir = os.path.join(tmpdir, "cache_dir")
    data_dir = os.path.join(tmpdir, "data_dir")
    os.makedirs(cache_dir)
    os.makedirs(data_dir)

    # Create a dataset with 50 items, 10 items per chunk
    cache = Cache(str(data_dir), chunk_size=10)
    for i in range(50):
        cache[i] = i
    cache.done()
    cache.merge()

    # Create dataset with appropriate configuration
    input_dir = f"local:{data_dir}" if local else str(data_dir)
    dataset = StreamingDataset(input_dir, cache_dir=str(cache_dir) if local else None, shuffle=shuffle)
    dataset_length = len(dataset)
    assert dataset_length == 50

    # ACT & ASSERT - Test iterator mode
    for i, data in enumerate(dataset):
        assert data is not None
        if local and i < dataset_length - 1:
            # In iterator mode with local or remote data, _chunks_queued_for_download should be enabled
            assert dataset.cache._reader._chunks_queued_for_download is True, (
                "_chunks_queued_for_download should be enabled during iteration"
            )
        else:
            assert dataset.cache._reader._chunks_queued_for_download is False, (
                "_chunks_queued_for_download should be disabled when used as local dir without `local:` prefix"
                " or when iteration is done"
            )
    # After iteration, _chunks_queued_for_download should be reset
    assert dataset.cache._reader._chunks_queued_for_download is False

    # ACT & ASSERT - Test indexed access mode
    for i in range(dataset_length):
        data = dataset[i]
        assert data is not None
        # In indexed access mode, _chunks_queued_for_download should never be enabled
        assert dataset.cache._reader._chunks_queued_for_download is False

    assert dataset.cache._reader._chunks_queued_for_download is False


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
@pytest.mark.parametrize("shuffle", [True, False])
def test_cache_get_clear_after_dataset_stream(tmpdir, shuffle):
    """Test that the cache directory is cleaned up after streaming through the dataset.

    This test verifies that:
    1. When streaming through a dataset with a limited cache size,
       all temporary chunk files are properly removed after completion
    2. This behavior is consistent regardless of shuffle settings
    """
    # ARRANGE
    # Create directories for cache and data
    cache_dir = os.path.join(tmpdir, "cache_dir")
    data_dir = os.path.join(tmpdir, "data_dir")
    os.makedirs(cache_dir)
    os.makedirs(data_dir)

    # Create a dataset with 500 items, 100 items per chunk
    cache = Cache(str(data_dir), chunk_size=100)
    for i in range(500):
        cache[i] = i
    cache.done()
    cache.merge()

    # Configure input directory based on local parameter
    input_dir = f"local:{data_dir}"

    # Create dataset with appropriate configuration and limited cache size
    dataset = StreamingDataset(input_dir, cache_dir=str(cache_dir), shuffle=shuffle, max_cache_size=2000)
    dataset_length = len(dataset)
    assert dataset_length == 500

    # ACT
    # Stream through the entire dataset to trigger caching and cleanup
    for data in dataset:
        assert data is not None

    # ASSERT
    # Verify that the cache directory is empty after streaming is complete
    cache_contents = os.listdir(dataset.cache.cache_dir)
    chunks = [f for f in cache_contents if ".bin" in f]

    assert len(chunks) == 0, (
        f"All of the chunks should have been cleared from the cache directory after streaming."
        f"Found {len(chunks)} chunk files: {chunks}"
    )


@pytest.mark.parametrize("shuffle", [True, False])
def test_dataset_transform(tmpdir, shuffle):
    """Test if the dataset transform is applied correctly."""
    # Create a simple dataset
    # Create directories for cache and data
    cache_dir = os.path.join(tmpdir, "cache_dir")
    data_dir = os.path.join(tmpdir, "data_dir")
    os.makedirs(cache_dir)
    os.makedirs(data_dir)

    # Create a dataset with 100 items, 20 items per chunk
    cache = Cache(str(data_dir), chunk_size=20)
    for i in range(100):
        cache[i] = i
    cache.done()
    cache.merge()

    # Define a simple transform function
    def transform_fn(x, *args, **kwargs):
        """A simple transform function that doubles the input."""
        return x * 2

    dataset = StreamingDataset(data_dir, cache_dir=str(cache_dir), shuffle=shuffle, transform=transform_fn)
    dataset_length = len(dataset)
    assert dataset_length == 100

    # ACT
    # Stream through the entire dataset and store the results
    complete_data = []
    for data in dataset:
        assert data is not None
        complete_data.append(data)

    if shuffle:
        complete_data.sort()

    # ASSERT
    # Verify that the transform is applied correctly
    for i, item in enumerate(complete_data):
        assert item == i * 2, f"Expected {i * 2}, got {item}"


@pytest.mark.parametrize("shuffle", [True, False])
def test_dataset_transform_inheritance(tmpdir, shuffle):
    """Test if the dataset transform is applied correctly."""
    # Create a simple dataset
    # Create directories for cache and data
    cache_dir = os.path.join(tmpdir, "cache_dir")
    data_dir = os.path.join(tmpdir, "data_dir")
    os.makedirs(cache_dir)
    os.makedirs(data_dir)

    # Create a dataset with 100 items, 20 items per chunk
    cache = Cache(str(data_dir), chunk_size=20)
    for i in range(100):
        cache[i] = i
    cache.done()
    cache.merge()

    class StreamingDatasetWithTransform(StreamingDataset):
        """A custom dataset class that inherits from StreamingDataset and applies a transform."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        # Define a simple transform function
        def transform(self, x, *args, **kwargs):
            """A simple transform function that doubles the input."""
            return x * 2

    dataset = StreamingDatasetWithTransform(data_dir, cache_dir=str(cache_dir), shuffle=shuffle)
    dataset_length = len(dataset)
    assert dataset_length == 100

    # ACT
    # Stream through the entire dataset and store the results
    complete_data = []
    for data in dataset:
        assert data is not None
        complete_data.append(data)

    if shuffle:
        complete_data.sort()

    # ASSERT
    # Verify that the transform is applied correctly
    for i, item in enumerate(complete_data):
        assert item == i * 2, f"Expected {i * 2}, got {item}"
