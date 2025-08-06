import pytest

from litdata import StreamingDataLoader, StreamingDataset, train_test_split
from litdata.constants import _ZSTD_AVAILABLE
from litdata.streaming.cache import Cache


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
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
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
