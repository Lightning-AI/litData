import os
import sys

import pytest
import torch
from torch import tensor

from litdata.constants import _VIZ_TRACKER_AVAILABLE
from litdata.streaming import (
    Cache,
    CombinedStreamingDataset,
    ParallelStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
)
from litdata.streaming import dataloader as streaming_dataloader_module


class TestStatefulDataset:
    def __init__(self, size, step):
        self.size = size
        self.step = step
        self.counter = 0
        self.shuffle = None
        self.drop_last = None

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle

    def __len__(self):
        return self.size

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter == self.size:
            raise StopIteration
        value = self.step * self.counter
        self.counter += 1
        return value

    def state_dict(self, *args, **kwargs):
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]

    def set_epoch(self, current_epoch):
        pass

    def set_drop_last(self, drop_last):
        self.drop_last = drop_last

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers


class TestCombinedStreamingDataset(CombinedStreamingDataset):
    def _check_datasets(self, datasets) -> None:
        pass

    def reset_state_dict(self):
        pass


def test_streaming_dataloader():
    dataset = TestCombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)],
        42,
        weights=(0.5, 0.5),
        iterate_over_all=False,
    )
    dataloader = StreamingDataLoader(dataset, batch_size=2)
    dataloader_iter = iter(dataloader)
    batches = []
    for batch in dataloader_iter:
        batches.append(batch)

    expected = [
        tensor([0, 0]),
        tensor([1, 2]),
        tensor([-1, -2]),
        tensor([-3, 3]),
        tensor([4, 5]),
        tensor([6, -4]),
        tensor([7, 8]),
        tensor([-5, -6]),
        tensor([9, -7]),
        tensor([-8]),
    ]

    for exp, gen in zip(expected, batches):
        assert torch.equal(exp, gen)

    assert dataloader.state_dict() == {
        "dataset": {"0": {"counter": 10}, "1": {"counter": 9}},
        "current_epoch": 1,
        "latest_worker_idx": 0,
        "num_samples_yielded": {0: [10, 9]},
    }


@pytest.mark.skip(reason="Profiling patches torch which leads to undesired test interactions")
@pytest.mark.skipif(not _VIZ_TRACKER_AVAILABLE, reason="viz tracker required")
@pytest.mark.parametrize("profile", [2, True])
def test_dataloader_profiling(profile, tmpdir, monkeypatch):
    monkeypatch.setattr(streaming_dataloader_module, "_VIZ_TRACKER_AVAILABLE", True)

    dataset = TestCombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)],
        42,
        weights=(0.5, 0.5),
        iterate_over_all=False,
    )
    dataloader = StreamingDataLoader(
        dataset, batch_size=2, profile_batches=profile, profile_dir=str(tmpdir), num_workers=1
    )
    dataloader_iter = iter(dataloader)
    batches = []
    for batch in dataloader_iter:
        batches.append(batch)

    assert os.path.exists(os.path.join(tmpdir, "result.json"))


def test_dataloader_shuffle():
    dataset = TestCombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)], 42, weights=(0.5, 0.5), iterate_over_all=False
    )
    assert dataset._datasets[0].shuffle is None
    assert dataset._datasets[1].shuffle is None
    StreamingDataLoader(dataset, batch_size=2, num_workers=1, shuffle=True)
    assert dataset._datasets[0].shuffle
    assert dataset._datasets[1].shuffle


class TestStatefulDatasetDict(TestStatefulDataset):
    def __next__(self):
        return {"value": super().__next__()}


def custom_collate_fn(samples):
    assert len(samples) == 2
    assert "value" in samples[0]
    return "received"


def test_custom_collate():
    dataset = TestCombinedStreamingDataset(
        [TestStatefulDatasetDict(10, 1), TestStatefulDatasetDict(10, -1)],
        42,
        weights=(0.5, 0.5),
        iterate_over_all=False,
    )
    assert dataset._datasets[0].shuffle is None
    assert dataset._datasets[1].shuffle is None
    dataloader = StreamingDataLoader(dataset, batch_size=2, num_workers=0, shuffle=True, collate_fn=custom_collate_fn)
    assert dataset._datasets[0].shuffle
    assert dataset._datasets[1].shuffle
    dataloader_iter = iter(dataloader)
    assert next(dataloader_iter) == "received"
    assert dataloader._num_samples_yielded_wrapper[0] == [dataset._datasets[0].counter, dataset._datasets[1].counter]


def test_custom_collate_multiworker():
    dataset = TestCombinedStreamingDataset(
        [TestStatefulDatasetDict(10, 1), TestStatefulDatasetDict(10, -1)],
        42,
        weights=(0.5, 0.5),
        iterate_over_all=False,
    )
    assert dataset._datasets[0].shuffle is None
    assert dataset._datasets[1].shuffle is None
    dataloader = StreamingDataLoader(dataset, batch_size=2, num_workers=3, shuffle=True, collate_fn=custom_collate_fn)
    assert dataset._datasets[0].shuffle
    assert dataset._datasets[1].shuffle
    dataloader_iter = iter(dataloader)
    assert next(dataloader_iter) == "received"
    assert dataloader._num_samples_yielded_wrapper[0] == [1, 1]
    assert next(dataloader_iter) == "received"
    assert dataloader._num_samples_yielded_wrapper[1] == [1, 1]
    assert next(dataloader_iter) == "received"
    assert dataloader._num_samples_yielded_wrapper[2] == [1, 1]
    assert next(dataloader_iter) == "received"
    assert dataloader._num_samples_yielded_wrapper[0] == [3, 1]

    # Iterate through the remaining samples
    try:
        while next(dataloader_iter) == "received":
            continue
    except AssertionError:
        assert dataloader._num_samples_yielded_wrapper == {0: [10, 8], 1: [10, 8], 2: [10, 8]}

    # Try calling the state_dict. No error should follow
    _state_dict = dataloader.state_dict()


def test_dataloader_no_workers(tmpdir):
    cache = Cache(input_dir=str(tmpdir), chunk_bytes="64MB")
    for i in range(1000):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir), shuffle=True)
    dataloader = StreamingDataLoader(dataset)
    assert len(dataset) == 1000
    assert len(dataloader) == 1000
    assert len(dataset) == 1000


@pytest.mark.timeout(120)
def test_dataloader_with_loading_states(tmpdir):
    cache = Cache(input_dir=str(tmpdir), chunk_bytes="64MB")
    for i in range(100):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir), shuffle=True)

    # Test dataloader without explicit num workers
    dataloader = StreamingDataLoader(dataset, batch_size=4)
    dataloader.load_state_dict(dataloader.state_dict())
    batch = next(iter(dataloader))
    assert len(batch) == 4, "Batch size should be 4"
    assert len(dataloader) == 25, "Dataloader length should be 25 (100 items / batch size 4)"

    # Test dataloader with num workers
    dataloader = StreamingDataLoader(dataset, batch_size=4, num_workers=2)
    assert len(dataloader) == 25, "Dataloader length should be 25 (100 items / batch size 4)"

    # Verify dataloader state after partial iteration
    for batch_idx, batch in enumerate(dataloader):
        assert dataloader.current_epoch == 1, "Current epoch should be 1"
        if batch_idx == 10:
            break
    dataloader.load_state_dict(dataloader.state_dict())
    assert dataloader.restore
    # Verify remaining batches in the first epoch
    count = 0
    for _ in dataloader:
        assert dataloader.current_epoch == 1, "Current epoch should be 1"
        count += 1
    # we consumed 11 batches (batch_idx==10) before.
    assert count == 14, "There should be at least 14 batches remaining in the first epoch"
    assert not dataloader.restore

    # Verify batches in the second epoch
    count = 0
    for _ in dataloader:
        assert dataloader.current_epoch == 2, "Current epoch should be 2"
        count += 1
    assert count >= 25, "There should be at least 25 batches in the second epoch"

    # Verify that the datalaoder can resume after complete last epoch
    dataloader.load_state_dict(dataloader.state_dict())
    assert not dataloader.restore
    count = 0
    for _ in dataloader:
        assert dataloader.current_epoch == 3, "Current epoch should be 3"
        count += 1
    assert count >= 25, "There should be at least 25 batches in the third epoch"


@pytest.mark.timeout(120)
def test_dataloader_states_with_persistent_workers(tmpdir):
    cache = Cache(input_dir=str(tmpdir), chunk_bytes="64MB")
    for i in range(100):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir), shuffle=True)

    dataloader = StreamingDataLoader(dataset, batch_size=4, num_workers=2)
    assert len(dataloader) == 25, "Dataloader length should be 25 (100 items / batch size 4)"

    # Verify dataloader state after partial iteration
    for batch_idx, batch in enumerate(dataloader):
        assert dataloader.current_epoch == 1, "Current epoch should be 1"
        if batch_idx == 10:
            break

    prev_dataloader_state = dataloader.state_dict()
    dataloader = StreamingDataLoader(dataset, batch_size=4, num_workers=2, persistent_workers=True)
    dataloader.load_state_dict(prev_dataloader_state)
    assert dataloader.restore

    # Verify remaining batches in the first epoch
    count = 0
    for _ in dataloader:
        assert dataloader.current_epoch == 1, "Current epoch should be 1"
        count += 1
    # batch_idx==10 means we consumed 11 batches before.
    assert count == 14, "There should be at least 14 batches remaining in the first epoch"
    assert not dataloader.restore

    # Verify batches in the second epoch
    count = 0
    for _ in dataloader:
        assert dataloader.current_epoch == 2, "Current epoch should be 2"
        count += 1
    assert count >= 25, "There should be at least 25 batches in the second epoch"

    # Verify that the datalaoder can resume after complete last epoch
    dataloader.load_state_dict(dataloader.state_dict())
    assert not dataloader.restore
    count = 0
    for _ in dataloader:
        assert dataloader.current_epoch == 3, "Current epoch should be 3"
        count += 1
    assert count >= 25, "There should be at least 25 batches in the third epoch"


@pytest.mark.timeout(60)
def test_resume_dataloader_with_new_dataset(tmpdir):
    dataset_1_path = tmpdir.join("dataset_1")
    dataset_2_path = tmpdir.join("dataset_2")
    for dataset in [dataset_1_path, dataset_2_path]:
        cache = Cache(input_dir=str(dataset), chunk_bytes="64MB")
        for i in range(50):
            cache[i] = i
        cache.done()
        cache.merge()
    dataset = StreamingDataset(str(dataset_1_path), shuffle=True)
    dataloader = StreamingDataLoader(dataset, batch_size=4, num_workers=2)
    for _ in dataloader:
        assert dataloader.current_epoch == 1, "Current epoch should be 1"

    dataloader_state = dataloader.state_dict()
    dataset = StreamingDataset(str(dataset_2_path), shuffle=True)
    dataloader = StreamingDataLoader(dataset, batch_size=4, num_workers=2)
    dataloader.load_state_dict(dataloader_state)
    for _ in dataloader:
        assert dataloader.current_epoch == 2, "Current epoch should be 2"


def test_resume_dataloader_after_some_workers_are_done(tmpdir):
    # see https://github.com/Lightning-AI/litData/issues/563
    dset_path = tmpdir.join("dataset")
    cache = Cache(input_dir=str(dset_path), chunk_size=1)
    for i in range(3):
        cache[i] = i
    cache.done()
    cache.merge()
    dset = StreamingDataset(str(dset_path), shuffle=False)
    dloader = StreamingDataLoader(dset, batch_size=1, num_workers=2, shuffle=False)
    # worker 0 is assigned with samples 0 and 1, worker 1 is assigned with sample 2
    # the workers alternate, so the expected sequence is [0, 2, 1] and not [0, 1, 2]
    expected_sequence = [0, 2, 1]
    for i, x in enumerate(dloader):
        assert x == expected_sequence[i]
        if i == 1:
            break
    dloader.load_state_dict(dloader.state_dict())
    for x in dloader:
        assert x == expected_sequence[2]


def simple_transform(samples):
    x, y = samples
    return x + y


def rng_transform(samples, rng):
    x, y = samples
    return rng["random"].random() * x, rng["numpy"].random() * y, torch.rand(1, generator=rng["torch"])


@pytest.mark.timeout(120)
@pytest.mark.parametrize("length", [None, 7])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("transform", [None, simple_transform, rng_transform])
@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="too slow in CI")
def test_resume_parallel_dataset(tmp_path, length, num_workers, transform):
    dset_paths = [str(tmp_path / f"dataset_{i}") for i in range(2)]
    for dset_path in dset_paths:
        cache = Cache(input_dir=dset_path, chunk_size=1)
        for i in range(10):
            cache[i] = i
        cache.done()
        cache.merge()
    dloader = StreamingDataLoader(
        ParallelStreamingDataset(
            [StreamingDataset(dset_path) for dset_path in dset_paths],
            length=length,
            transform=transform,
        ),
        num_workers=num_workers,
    )
    for _ in dloader:
        pass
    state = dloader.state_dict()
    data = []
    for x in dloader:
        data.append(x)
    dloader.load_state_dict(state)
    for i, x in enumerate(dloader):
        assert x == data[i]


# Define a simple transform function
def transform_fn(x, *args, **kwargs):
    """A simple transform function that doubles the input."""
    return x * 2


@pytest.mark.parametrize("shuffle", [True, False])
def test_dataloader_dataset_transform(tmpdir, shuffle):
    """Test if the dataset's transform is applied correctly with dataloader."""
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

    dataset = StreamingDataset(data_dir, cache_dir=str(cache_dir), shuffle=shuffle, transform=transform_fn)
    dataset_length = len(dataset)
    assert dataset_length == 100

    # ACT
    dl = StreamingDataLoader(dataset, batch_size=10, num_workers=2, shuffle=shuffle)

    complete_data = []
    for batch in dl:
        complete_data.extend(batch)

    complete_data.sort()
    print(f"Complete data: {complete_data}")

    # ASSERT
    # Verify that the transform is applied correctly
    for i, item in enumerate(complete_data):
        assert item == i * 2, f"Expected {i * 2}, got {item}"


class StreamingDatasetWithTransform(StreamingDataset):
    """A custom dataset class that inherits from StreamingDataset and applies a transform."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Define a simple transform function
    def transform(self, x, *args, **kwargs):
        """A simple transform function that doubles the input."""
        return x * 2


@pytest.mark.parametrize("shuffle", [True, False])
def test_dataloader_dataset_transform_inheritance(tmpdir, shuffle):
    """Test if the dataset's transform is applied correctly with dataloader."""
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

    dataset = StreamingDatasetWithTransform(data_dir, cache_dir=str(cache_dir), shuffle=shuffle)
    dataset_length = len(dataset)
    assert dataset_length == 100

    # ACT
    dl = StreamingDataLoader(dataset, batch_size=10, num_workers=2, shuffle=shuffle)

    complete_data = []
    for batch in dl:
        complete_data.extend(batch)

    complete_data.sort()
    print(f"Complete data: {complete_data}")

    # ASSERT
    # Verify that the transform is applied correctly
    for i, item in enumerate(complete_data):
        assert item == i * 2, f"Expected {i * 2}, got {item}"
