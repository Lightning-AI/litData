import json
import os
import platform
import sys

import pytest

from litdata import StreamingDataLoader, StreamingDataset, train_test_split
from litdata.constants import _INDEX_FILENAME, _ZSTD_AVAILABLE
from litdata.streaming.cache import Cache

IS_WINDOWS = sys.platform.startswith("win") or platform.system() == "Windows"


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
def test_train_test_split(tmpdir, compression):
    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(100):
        cache[i] = i
    cache.done()
    cache.merge()

    my_streaming_dataset = StreamingDataset(input_dir=str(tmpdir))
    train_dataset, test_dataset = train_test_split(my_streaming_dataset, splits=[0.75, 0.25])

    assert len(train_dataset) == 75
    assert len(test_dataset) == 25


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
def test_split_a_subsampled_dataset(tmpdir, compression):
    cache = Cache(str(tmpdir), chunk_size=15, compression=compression)
    for i in range(1000):
        cache[i] = i
    cache.done()
    cache.merge()

    _sub_sampled_streaming_dataset = StreamingDataset(input_dir=str(tmpdir), subsample=0.3)

    assert len(_sub_sampled_streaming_dataset) == 300  # 1000 * 0.3

    _split_fraction = [0.2, 0.3, 0.4, 0.1]

    split_datasets = train_test_split(_sub_sampled_streaming_dataset, _split_fraction)

    assert all(len(split_datasets[i]) == int(300 * split) for i, split in enumerate(_split_fraction))

    # ------------- splits with 0 fraction of samples -------------

    _split_fraction = [0.0, 0.0, 1.0]

    split_datasets = train_test_split(_sub_sampled_streaming_dataset, _split_fraction)

    assert all(len(split_datasets[i]) == int(300 * split) for i, split in enumerate(_split_fraction))

    # ------------- test if some splits get 0 samples -------------

    _sub_sampled_streaming_dataset = StreamingDataset(input_dir=str(tmpdir), subsample=0.05)

    assert len(_sub_sampled_streaming_dataset) == 50  # 1000 * 0.05

    _split_fraction = [0.01, 0.01, 0.98]

    split_datasets = train_test_split(_sub_sampled_streaming_dataset, _split_fraction)

    assert all(len(split_datasets[i]) == int(50 * split) for i, split in enumerate(_split_fraction))


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
def test_train_test_split_with_streaming_dataloader(tmpdir, compression):
    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(200):
        cache[i] = i
    cache.done()
    cache.merge()

    my_streaming_dataset = StreamingDataset(input_dir=str(tmpdir))

    splits = [0.1, 0.2, 0.7, 0.0]

    ds = train_test_split(my_streaming_dataset, splits=splits)

    assert [len(ds[i]) for i in range(len(splits))] == [int(200 * split) for split in splits]

    # check that the indices are unique for each dataset (iterating over the datasets)
    visited_indices = set()
    for _ds in ds:
        for idx in range(len(_ds)):
            assert _ds[idx] not in visited_indices
            visited_indices.add(_ds[idx])

    # check that the indices are unique for each dataloader (iterating over the dataloader)
    visited_indices = set()
    for _ds in ds:
        dl = StreamingDataLoader(_ds, batch_size=10)
        for _dl in dl:
            for curr_idx in _dl:
                assert curr_idx not in visited_indices
                visited_indices.add(curr_idx)


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None, marks=pytest.mark.skipif(IS_WINDOWS, reason="slow on windows")),
        pytest.param(
            "zstd",
            marks=pytest.mark.skipif(not _ZSTD_AVAILABLE or IS_WINDOWS, reason="Requires: ['zstd']"),
        ),
    ],
)
def test_train_test_split_with_shuffle_parameter(tmpdir, compression):
    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(100):
        cache[i] = i

    cache.done()
    cache.merge()

    my_streaming_dataset = StreamingDataset(input_dir=str(tmpdir))

    train_shuffled, test_shuffled = train_test_split(my_streaming_dataset, splits=[0.8, 0.2], shuffle=True)
    train_no_shuffle, test_no_shuffle = train_test_split(my_streaming_dataset, splits=[0.8, 0.2], shuffle=False)

    assert len(train_shuffled) == 80
    assert len(train_no_shuffle) == 80
    assert len(test_shuffled) == 20
    assert len(test_no_shuffle) == 20

    shuffled_combined = train_shuffled.subsampled_files + test_shuffled.subsampled_files
    no_shuffle_combined = train_no_shuffle.subsampled_files + test_no_shuffle.subsampled_files
    assert shuffled_combined != no_shuffle_combined

    assert no_shuffle_combined == my_streaming_dataset.subsampled_files


def test_train_test_split_natural_sort_ordering(tmpdir):
    """chunk-0-10 must not appear before chunk-0-2 in the split when shuffle=False.

    When a dataset is written with 10+ workers, the merge step sorts per-worker
    index files lexicographically. This places chunk-0-10.bin between chunk-0-1.bin
    and chunk-0-2.bin in index.json, causing the first half of a 50/50 split to
    include chunk-0-10 while chunk-0-2 through chunk-0-9 end up in the second half.

    This test reproduces that scenario by rewriting index.json in lexicographic order
    after the cache is built, then asserts that train_test_split still returns chunks
    in natural order.
    """
    cache = Cache(str(tmpdir), chunk_size=2)
    for i in range(22):
        cache[i] = i
    cache.done()
    cache.merge()

    # Rewrite index.json in lexicographic order to reproduce the multi-worker bug.
    # Lexicographic sort places "chunk-0-10.bin" between "chunk-0-1.bin" and
    # "chunk-0-2.bin", which is the exact ordering produced when 10+ workers merge.
    index_path = os.path.join(str(tmpdir), _INDEX_FILENAME)
    with open(index_path) as f:
        data = json.load(f)
    data["chunks"].sort(key=lambda c: c["filename"])  # lexicographic, not natural
    with open(index_path, "w") as f:
        json.dump(data, f)

    dataset = StreamingDataset(input_dir=str(tmpdir))
    train_ds, test_ds = train_test_split(dataset, splits=[0.5, 0.5], shuffle=False)

    # chunk-0-10 must be in the test split (last), not in train.
    # Without natural-sort in train_test_split it lands in train because lexicographic
    # order puts it at index 1 (right after chunk-0-0).
    assert train_ds.subsampled_files == [
        "chunk-0-0.bin",
        "chunk-0-1.bin",
        "chunk-0-2.bin",
        "chunk-0-3.bin",
        "chunk-0-4.bin",
        "chunk-0-5.bin",
    ]
    assert test_ds.subsampled_files == [
        "chunk-0-5.bin",
        "chunk-0-6.bin",
        "chunk-0-7.bin",
        "chunk-0-8.bin",
        "chunk-0-9.bin",
        "chunk-0-10.bin",
    ]
