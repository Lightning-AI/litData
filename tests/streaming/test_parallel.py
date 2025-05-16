import functools
import os
import sys
from copy import deepcopy
from unittest.mock import ANY, MagicMock

import pytest
import torch
from torch.utils.data import IterableDataset

from litdata.streaming.cache import Cache
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import Dir, StreamingDataset
from litdata.streaming.parallel import ParallelStreamingDataset


class TestParallelStreamingDataset(ParallelStreamingDataset):
    def _check_datasets(self, datasets) -> None:
        pass

    def reset_state_dict(self):
        pass


@pytest.mark.parametrize(
    ("dset_1", "dset_2", "length", "expected", "outputs"),
    [
        (range(5), range(0, -5, -1), None, 5, [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]),
        (range(10), range(0, -5, -1), None, 5, [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]),
        (range(5), range(0, -5, -1), 5, 5, [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]),
        (range(10), range(0, -5, -1), 5, 5, [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]),
        (range(5), range(0, -5, -1), 8, 8, [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4), (0, 0), (1, -1), (2, -2)]),
        (range(10), range(0, -5, -1), 8, 8, [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4), (5, 0), (6, -1), (7, -2)]),
        (range(10), range(0), None, 0, []),
    ],
)
def test_parallel_dataset(dset_1, dset_2, length, expected, outputs):
    dset = TestParallelStreamingDataset([dset_1, dset_2], length=length)
    assert len(dset) == expected
    assert list(dset) == outputs


def test_parallel_dataset_errors():
    with pytest.raises(RuntimeError, match="The provided datasets should be instances of"):
        ParallelStreamingDataset([range(5), range(5)])
    with pytest.raises(ValueError, match=r"`length` must be `None`, an integer, or `float\('inf'\)`"):
        TestParallelStreamingDataset([range(5), range(5)], length="foo")
    dset = TestParallelStreamingDataset([range(5), range(5)])
    dset._length = "foo"
    with pytest.raises(ValueError, match="Invalid ParallelStreamingDataset _length attribute"):
        len(dset)
    dset = TestParallelStreamingDataset([range(0), range(5)], length=1)
    assert list(dset) == []
    dset = TestParallelStreamingDataset([range(0), range(5)], length=-1)
    assert list(iter(dset)) == []  # negative length is actually supported


def test_parallel_transform_errors():
    with pytest.raises(ValueError, match="transform function must take 1 or 2 arguments"):
        TestParallelStreamingDataset([range(5), range(5)], transform=lambda: None)

    with pytest.raises(ValueError, match="transform function must take 1 or 2 arguments"):
        TestParallelStreamingDataset([range(5), range(5)], transform=lambda x, y, z: None)

    TestParallelStreamingDataset([range(5), range(5)], transform=functools.partial(lambda x, y, z: None, z=None))

    TestParallelStreamingDataset(
        [range(5), range(5)], transform=functools.partial(lambda x, y, z: None, y=None, z=None)
    )

    with pytest.raises(ValueError, match="transform function must take 1 or 2 arguments"):
        TestParallelStreamingDataset(
            [range(5), range(5)], transform=functools.partial(lambda x, y, z: 0, x=None, y=None, z=None)
        )

    def transform(x, y, z=None): ...

    with pytest.raises(ValueError, match="transform function must take 1 or 2 arguments"):
        TestParallelStreamingDataset([range(5), range(5)], transform=transform)

    TestParallelStreamingDataset([range(5), range(5)], transform=functools.partial(transform, z=None))

    def transform(x, y=None): ...

    TestParallelStreamingDataset([range(5), range(5)], transform=transform)

    def transform(x=None, y=None): ...

    TestParallelStreamingDataset([range(5), range(5)], transform=transform)


@pytest.mark.parametrize(
    ("dset_1", "dset_2", "length", "expected"),
    [
        (range(5), range(5), None, [5, 5]),
        (range(5), range(5), 5, [5, 5]),
        (range(5), range(5), 8, [3, 3]),
        (range(5), range(3), None, [3, 3]),
        (range(5), range(3), 5, [5, 2]),
        (range(5), range(3), 8, [3, 2]),
    ],
)
def test_parallel_dataset_num_samples_yielded(dset_1, dset_2, length, expected):
    dset = TestParallelStreamingDataset([dset_1, dset_2], length=length)
    assert dset._num_samples_yielded is None
    dset_iter = iter(dset)
    assert dset_iter._num_samples_yielded == [0, 0]
    next(dset_iter)
    assert dset_iter._num_samples_yielded == [1, 1]
    dset_iter = iter(dset)
    assert dset_iter._num_samples_yielded == [0, 0]
    list(dset_iter)
    assert dset_iter._num_samples_yielded == expected


def test_drop_last_and_shuffle():
    dset_mock_1 = MagicMock()
    dset_mock_2 = MagicMock()
    dset = TestParallelStreamingDataset([dset_mock_1, dset_mock_2])
    StreamingDataLoader(dset, shuffle=True, drop_last=True)
    dset_mock_1.set_shuffle.assert_called()
    dset_mock_2.set_shuffle.assert_called()
    dset_mock_1.set_drop_last.assert_called()
    dset_mock_2.set_drop_last.assert_called()
    dset_mock_1.set_num_workers.assert_called()
    dset_mock_2.set_num_workers.assert_called()
    dset_mock_1.set_batch_size.assert_called()
    dset_mock_2.set_batch_size.assert_called()


class DummyStatefulDataset:
    def __init__(self, length, step):
        self.length = length
        self.step = step
        self.counter = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter == self.length:
            raise StopIteration
        value = self.step * self.counter
        self.counter += 1
        return value

    def state_dict(self, *args, **kwargs):
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]


def test_parallel_dataset_state_dict():
    dataset = TestParallelStreamingDataset([DummyStatefulDataset(10, 1), DummyStatefulDataset(7, -1)], length=10)
    assert dataset.state_dict(0, 1) == {}
    dataset_iter = iter(dataset)
    assert dataset.state_dict(0, 1) == {"0": {"counter": 0}, "1": {"counter": 0}}

    data = []
    states = []
    for i, value in enumerate(dataset_iter):
        state = dataset.state_dict(i, 1)
        data.append(value)
        states.append(state)

    assert data == [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, 0), (8, -1), (9, -2)]
    assert states == [
        {"0": {"counter": 1}, "1": {"counter": 1}},
        {"0": {"counter": 2}, "1": {"counter": 2}},
        {"0": {"counter": 3}, "1": {"counter": 3}},
        {"0": {"counter": 4}, "1": {"counter": 4}},
        {"0": {"counter": 5}, "1": {"counter": 5}},
        {"0": {"counter": 6}, "1": {"counter": 6}},
        {"0": {"counter": 7}, "1": {"counter": 7}},
        {"0": {"counter": 8}, "1": {"counter": 1}},
        {"0": {"counter": 9}, "1": {"counter": 2}},
        {"0": {"counter": 10}, "1": {"counter": 3}},
    ]

    dataset_2 = TestParallelStreamingDataset([DummyStatefulDataset(10, 1), DummyStatefulDataset(7, -1)], length=10)
    assert dataset_2.state_dict(0, 1) == {}
    dataset2_iter = iter(dataset_2)

    data_2 = []
    for state in states[:-1]:
        dataset_2.load_state_dict({"dataset": state})
        data_2.append(next(dataset2_iter))

    assert data[1:] == data_2


class DummyIterableDataset(IterableDataset):
    def __init__(self, end, step):
        super().__init__()
        self.end = end
        self.step = step
        self.current_epoch = 0

    def __iter__(self):
        return iter(range(0, self.end, self.step))

    def __len__(self):
        return len(range(0, self.end, self.step))

    def state_dict(self, **kwargs):
        return kwargs

    def load_state_dict(self, state_dict) -> None:
        if state_dict:
            self._state_dict = state_dict

    def set_epoch(self, current_epoch):
        self.current_epoch = current_epoch

    def set_shuffle(self, _):
        pass

    def set_drop_last(self, _):
        pass

    def set_batch_size(self, _):
        pass

    def set_num_workers(self, _):
        pass


@pytest.mark.parametrize(
    ("batch_size", "length", "expected", "num_samples_yielded", "num_cycles"),
    [
        (
            1,
            None,
            [
                [torch.Tensor([0]), torch.Tensor([0])],
                [torch.Tensor([1]), torch.Tensor([-1])],
                [torch.Tensor([2]), torch.Tensor([-2])],
                [torch.Tensor([3]), torch.Tensor([-3])],
                [torch.Tensor([4]), torch.Tensor([-4])],
                [torch.Tensor([5]), torch.Tensor([-5])],
                [torch.Tensor([6]), torch.Tensor([-6])],
            ],
            [7, 7],
            [0, 0],
        ),
        (
            2,
            None,
            [
                [torch.Tensor([0, 1]), torch.Tensor([0, -1])],
                [torch.Tensor([2, 3]), torch.Tensor([-2, -3])],
                [torch.Tensor([4, 5]), torch.Tensor([-4, -5])],
                [torch.Tensor([6]), torch.Tensor([-6])],
            ],
            [7, 7],
            [0, 0],
        ),
        (
            1,
            13,
            [
                [torch.Tensor([0]), torch.Tensor([0])],
                [torch.Tensor([1]), torch.Tensor([-1])],
                [torch.Tensor([2]), torch.Tensor([-2])],
                [torch.Tensor([3]), torch.Tensor([-3])],
                [torch.Tensor([4]), torch.Tensor([-4])],
                [torch.Tensor([5]), torch.Tensor([-5])],
                [torch.Tensor([6]), torch.Tensor([-6])],
                [torch.Tensor([7]), torch.Tensor([0])],
                [torch.Tensor([8]), torch.Tensor([-1])],
                [torch.Tensor([9]), torch.Tensor([-2])],
                [torch.Tensor([0]), torch.Tensor([-3])],
                [torch.Tensor([1]), torch.Tensor([-4])],
                [torch.Tensor([2]), torch.Tensor([-5])],
            ],
            [3, 6],
            [1, 1],
        ),
        (
            2,
            13,
            [
                [torch.Tensor([0, 1]), torch.Tensor([0, -1])],
                [torch.Tensor([2, 3]), torch.Tensor([-2, -3])],
                [torch.Tensor([4, 5]), torch.Tensor([-4, -5])],
                [torch.Tensor([6, 7]), torch.Tensor([-6, 0])],
                [torch.Tensor([8, 9]), torch.Tensor([-1, -2])],
                [torch.Tensor([0, 1]), torch.Tensor([-3, -4])],
                [torch.Tensor([2]), torch.Tensor([-5])],
            ],
            [3, 6],
            [1, 1],
        ),
    ],
)
def test_parallel_dataset_with_dataloader_and_one_worker(batch_size, length, expected, num_samples_yielded, num_cycles):
    dset_1 = DummyIterableDataset(10, 1)
    dset_2 = DummyIterableDataset(-7, -1)
    dset = TestParallelStreamingDataset([dset_1, dset_2], length=length)
    dloader = StreamingDataLoader(dset, num_workers=1, batch_size=batch_size, prefetch_factor=1)

    outputs = list(dloader)
    assert len(outputs) == len(expected)
    for output, expected_output in zip(outputs, expected):
        assert len(output) == len(expected_output)
        for o, e in zip(output, expected_output):
            assert torch.equal(o, e)

    assert dloader.state_dict() == {
        "dataset": {
            "0": {"num_samples_yielded": num_samples_yielded[0], "num_workers": 1, "batch_size": batch_size},
            "1": {"num_samples_yielded": num_samples_yielded[1], "num_workers": 1, "batch_size": batch_size},
        },
        "current_epoch": 1,
        "latest_worker_idx": 0 if length is None else 1,
        "num_samples_yielded": {0: num_samples_yielded},
        "num_cycles": {0: num_cycles},
    }


def rng_transform(_, rngs, which):
    if which == "torch":
        return torch.rand(1, generator=rngs[which])
    return rngs[which].random()


@pytest.mark.parametrize("length", [None, 7])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("which", ["random", "numpy", "torch"])
@pytest.mark.parametrize("reset_rngs", [False, True])
@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="too slow in CI")
def test_parallel_dataset_rng(length, num_workers, which, reset_rngs):
    transform = functools.partial(rng_transform, which=which)

    dloader = StreamingDataLoader(
        TestParallelStreamingDataset(
            [DummyIterableDataset(10, 1)],
            length=length,
            transform=transform,
            seed=42,
            reset_rngs=reset_rngs,
        ),
        num_workers=num_workers,
    )
    epoch_1 = []
    for x in dloader:
        assert x not in epoch_1
        epoch_1.append(x)
    epoch_2 = []
    for x in dloader:
        assert x not in epoch_2
        epoch_2.append(x)
    for x1, x2 in zip(epoch_1, epoch_2):
        if reset_rngs and length is None:
            assert x1 == x2
        else:
            assert x1 != x2

    dloader = StreamingDataLoader(
        TestParallelStreamingDataset(
            [DummyIterableDataset(10, 1)],
            length=length,
            transform=transform,
            seed=42,
            reset_rngs=reset_rngs,
        ),
        num_workers=num_workers,
    )
    for x, old_x in zip(dloader, epoch_1):
        assert x == old_x

    dloader = StreamingDataLoader(
        TestParallelStreamingDataset(
            [DummyIterableDataset(10, 1)],
            length=length,
            transform=transform,
            seed=1337,
            reset_rngs=reset_rngs,
        ),
        num_workers=num_workers,
    )
    for x, old_x in zip(dloader, epoch_1):
        assert x != old_x


@pytest.mark.parametrize("parallel_dataset", [None, 3, float("inf")], indirect=True)
def test_parallel_dataset_dataloader_states_without_any_iterations(parallel_dataset):
    parallel_dataset, _ = parallel_dataset
    dataloader = StreamingDataLoader(parallel_dataset, batch_size=4)
    assert not dataloader.restore
    dataloader.load_state_dict(dataloader.state_dict())
    assert not dataloader.restore


@pytest.mark.timeout(120)
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("parallel_dataset", [None, 24], indirect=True)
@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="too slow in CI")
def test_parallel_dataset_dataloader_states_complete_iterations(parallel_dataset, num_workers):
    print(f"Testing with num_workers={num_workers}")

    parallel_dataset, length = parallel_dataset
    batch_size = 2

    dataloader = StreamingDataLoader(parallel_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    assert len(dataloader) == -(-len(parallel_dataset) // batch_size)

    # Verify dataloader state after complete last iteration
    epoch_1_data = []
    for data in dataloader:
        assert dataloader.current_epoch == 1, "Current epoch should be 1"
        epoch_1_data.append(data)

    dataloader.load_state_dict(dataloader.state_dict())
    assert not dataloader.restore

    epoch_1_data = [set(torch.cat(x).tolist()) for x in zip(*epoch_1_data)]
    assert all(len(x) == len(parallel_dataset) for x in epoch_1_data)

    epoch_2_data = []
    for data in dataloader:
        assert dataloader.current_epoch == 2, "Current epoch should be 2"
        epoch_2_data.append(data)

    assert not dataloader.restore

    epoch_2_data = [set(torch.cat(x).tolist()) for x in zip(*epoch_2_data)]
    assert all(len(x) == len(parallel_dataset) for x in epoch_2_data)

    if length is not None:
        # dataset length option is 24 and number of items on disk is 48 so the epochs should not overlap
        assert all(not x & y for x, y in zip(epoch_1_data, epoch_2_data)), "Epoch 1 and 2 data should not overlap"

    epoch_3_data = []
    for data in dataloader:
        assert dataloader.current_epoch == 3, "Current epoch should be 3"
        epoch_3_data.append(data)

    epoch_3_data = [set(torch.cat(x).tolist()) for x in zip(*epoch_3_data)]
    if length is None:
        assert all(len(x) == len(parallel_dataset) for x in epoch_3_data)
    else:
        # the datasets have cycled and shuffled so check new data overlaps with the previous epochs
        assert len(epoch_1_data[0] & epoch_3_data[0]) > 0
        assert len(epoch_2_data[0] & epoch_3_data[0]) > 0
        assert len(epoch_1_data[1] & epoch_3_data[1]) > 0
        assert len(epoch_2_data[1] & epoch_3_data[1]) > 0
        # dataset 1 length on disk is 48 so epoch 3 should have no dupes
        assert len(epoch_3_data[0]) == len(parallel_dataset)
        # dataset 2 length on disk is 56 so epoch 3 can have dupes since we cycled within epoch 3
        assert len(epoch_3_data[1]) <= len(parallel_dataset)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("break_at", [3, 7])
@pytest.mark.parametrize("parallel_dataset", [None, 20, 48], indirect=True)
def test_parallel_dataset_dataloader_states_partial_iterations(parallel_dataset, num_workers, break_at):
    print(f"Testing with num_workers={num_workers}, break_at={break_at}")

    parallel_dataset, _ = parallel_dataset
    batch_size = 2

    # Verify dataloader state after partial last iteration
    dataloader = StreamingDataLoader(parallel_dataset, batch_size=batch_size, num_workers=num_workers)

    total_batches = len(dataloader)
    assert total_batches == -(-len(parallel_dataset) // batch_size)

    assert not dataloader.restore, "Dataloader should not be in restore state initially."

    # Partial iteration up to 'break_at'
    for batch_idx, batch in enumerate(dataloader):
        assert dataloader.current_epoch == 1, "Current epoch should be 1 during first iteration"
        if batch_idx == break_at:
            break

    assert not dataloader.restore, (
        "Dataloader should not be in restore state after partial iteration, before loading state."
    )
    dataloader.load_state_dict(dataloader.state_dict())
    assert dataloader.restore, "Dataloader should be in restore state after loading the state from a partial iteration."

    # Verify remaining batches in the first epoch
    count = 0
    for _ in dataloader:
        assert dataloader.current_epoch == 1, "Current epoch should be 1 during restore"
        count += 1
    expected_batches = total_batches - break_at - 1
    assert count == expected_batches, f"There should be {expected_batches} remaining batches in the first epoch."
    assert not dataloader.restore, "Dataloader should not be in restore state after completing first epoch."

    # Verify batches in the second epoch
    samples_yielded = 0
    for batch in dataloader:
        assert dataloader.current_epoch == 2, "Current epoch should be 2 in the second iteration"
        assert all(len(b) == len(batch[0]) for b in batch), "All batches should have the same length."
        samples_yielded += len(batch[0])
    assert samples_yielded == len(parallel_dataset), "All samples should be yielded in the second epoch."


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="too slow in CI")
def test_parallel_dataset_with_dataloader_2_epochs_none_length(tmp_path):
    data_dir_1 = str(tmp_path / "data_1")
    data_dir_2 = str(tmp_path / "data_2")
    cache_dir_1 = str(tmp_path / "cache_dir_1")
    cache_dir_2 = str(tmp_path / "cache_dir_2")

    os.makedirs(data_dir_1)
    os.makedirs(data_dir_2)
    os.makedirs(cache_dir_1)
    os.makedirs(cache_dir_2)

    cache = Cache(input_dir=data_dir_1, chunk_size=2)

    for i in range(12):
        cache[i] = i

    cache.done()
    cache.merge()

    cache = Cache(input_dir=data_dir_2, chunk_size=2)

    for i in range(14):
        cache[i] = -i

    cache.done()
    cache.merge()

    dataset1 = StreamingDataset(input_dir=Dir(cache_dir_1, data_dir_1), shuffle=True)
    dataset2 = StreamingDataset(input_dir=Dir(cache_dir_2, data_dir_2), shuffle=True)
    dataset = ParallelStreamingDataset(datasets=[dataset1, dataset2], length=None)
    dataloader = StreamingDataLoader(dataset, num_workers=3, batch_size=2)

    assert dataset1.current_epoch == 1
    assert dataset2.current_epoch == 1

    expected_dataset_state = {
        "dataset": {
            "0": {
                "num_samples_yielded": 0,
                "num_workers": 3,
                "batch_size": 2,
                "current_epoch": 1,
                "input_dir_path": ANY,
                "input_dir_url": ANY,
                "cache_dir_path": None,
                "item_loader": None,
                "drop_last": False,
                "seed": 42,
                "world_size": 1,
                "shuffle": True,
                "subsampled_files": ANY,
                "region_of_interest": ANY,
            },
            "1": {
                "num_samples_yielded": 0,
                "num_workers": 3,
                "batch_size": 2,
                "current_epoch": 1,
                "input_dir_path": ANY,
                "input_dir_url": ANY,
                "cache_dir_path": None,
                "item_loader": None,
                "drop_last": False,
                "seed": 42,
                "world_size": 1,
                "shuffle": True,
                "subsampled_files": ANY,
                "region_of_interest": ANY,
            },
        },
        "current_epoch": 1,
        "latest_worker_idx": 0,
        "num_samples_yielded": {},
        "num_cycles": {},
    }
    expected_num_samples_yielded = [
        {0: [2, 2]},
        {0: [2, 2], 1: [2, 2]},
        {0: [2, 2], 1: [2, 2], 2: [2, 2]},
        {0: [4, 4], 1: [2, 2], 2: [2, 2]},
        {0: [4, 4], 1: [4, 4], 2: [2, 2]},
        {0: [4, 4], 1: [4, 4], 2: [4, 4]},
    ]
    expected_num_cycles = [
        {0: [0, 0]},
        {0: [0, 0], 1: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
    ]
    expected_current_epoch = [1, 1, 1, 1, 1, 1]
    dataset_1_current_epoch = [1, 1, 1, 1, 1, 1]
    dataset_2_current_epoch = [1, 1, 1, 1, 1, 1]
    expected_latest_worker_idx = [0, 1, 2, 0, 1, 2]
    expected_dataset0_samples_yielded = [2, 4, 6, 8, 10, 12]
    expected_dataset1_samples_yielded = [2, 4, 6, 8, 10, 12]

    batches_1 = []

    for idx, batch in enumerate(dataloader):
        batches_1.append(batch)
        curr_state_dict = dataloader.state_dict()

        expected_dataset_state["num_samples_yielded"] = expected_num_samples_yielded[idx]
        expected_dataset_state["num_cycles"] = expected_num_cycles[idx]
        expected_dataset_state["current_epoch"] = expected_current_epoch[idx]
        expected_dataset_state["latest_worker_idx"] = expected_latest_worker_idx[idx]
        expected_dataset_state["dataset"]["0"]["num_samples_yielded"] = expected_dataset0_samples_yielded[idx]
        expected_dataset_state["dataset"]["1"]["num_samples_yielded"] = expected_dataset1_samples_yielded[idx]
        expected_dataset_state["dataset"]["0"]["current_epoch"] = dataset_1_current_epoch[idx]
        expected_dataset_state["dataset"]["1"]["current_epoch"] = dataset_2_current_epoch[idx]

        assert curr_state_dict == expected_dataset_state

    assert dataset1.current_epoch == 1
    assert dataset2.current_epoch == 1

    saved_dataloader_state_dict = None

    batches_2 = []

    expected_num_samples_yielded = [
        {0: [2, 2]},
        {0: [2, 2], 1: [2, 2]},
        {0: [2, 2], 1: [2, 2], 2: [2, 2]},
        {0: [4, 4], 1: [2, 2], 2: [2, 2]},
        {0: [4, 4], 1: [4, 4], 2: [2, 2]},
        {0: [4, 4], 1: [4, 4], 2: [4, 4]},
    ]
    expected_num_cycles = [
        {0: [0, 0]},
        {0: [0, 0], 1: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
    ]
    expected_current_epoch = [2, 2, 2, 2, 2, 2]
    dataset_1_current_epoch = [2, 2, 2, 2, 2, 2]
    dataset_2_current_epoch = [2, 2, 2, 2, 2, 2]
    expected_latest_worker_idx = [0, 1, 2, 0, 1, 2]
    expected_dataset0_samples_yielded = [2, 4, 6, 8, 10, 12]
    expected_dataset1_samples_yielded = [2, 4, 6, 8, 10, 12]
    for idx, batch in enumerate(dataloader):
        batches_2.append(batch)
        curr_state_dict = dataloader.state_dict()

        expected_dataset_state["num_samples_yielded"] = expected_num_samples_yielded[idx]
        expected_dataset_state["num_cycles"] = expected_num_cycles[idx]
        expected_dataset_state["current_epoch"] = expected_current_epoch[idx]
        expected_dataset_state["latest_worker_idx"] = expected_latest_worker_idx[idx]
        expected_dataset_state["dataset"]["0"]["num_samples_yielded"] = expected_dataset0_samples_yielded[idx]
        expected_dataset_state["dataset"]["1"]["num_samples_yielded"] = expected_dataset1_samples_yielded[idx]
        expected_dataset_state["dataset"]["0"]["current_epoch"] = dataset_1_current_epoch[idx]
        expected_dataset_state["dataset"]["1"]["current_epoch"] = dataset_2_current_epoch[idx]

        assert curr_state_dict == expected_dataset_state

        if idx == 2:
            saved_dataloader_state_dict = deepcopy(curr_state_dict)

    assert dataset1.current_epoch == 2
    assert dataset2.current_epoch == 2

    assert len(batches_1) == len(batches_2)
    assert any(not torch.equal(x1, x2) for b1, b2 in zip(batches_1, batches_2) for x1, x2 in zip(b1, b2))

    assert saved_dataloader_state_dict is not None
    dataloader.load_state_dict(saved_dataloader_state_dict)

    assert dataloader.restore

    batches_23 = []
    states_23 = []
    for batch in dataloader:
        batches_23.append(batch)
        states_23.append(dataloader.state_dict())

    assert len(batches_2[3:]) == len(batches_23)
    assert all(torch.equal(x1, x2) for b1, b2 in zip(batches_2[3:], batches_23) for x1, x2 in zip(b1, b2))
    assert states_23[0]["current_epoch"] == 2

    assert not dataloader.restore


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="too slow in CI")
def test_parallel_dataset_with_dataloader_2_epochs_int_length(tmp_path):
    data_dir_1 = str(tmp_path / "data_1")
    data_dir_2 = str(tmp_path / "data_2")
    cache_dir_1 = str(tmp_path / "cache_dir_1")
    cache_dir_2 = str(tmp_path / "cache_dir_2")

    os.makedirs(data_dir_1)
    os.makedirs(data_dir_2)
    os.makedirs(cache_dir_1)
    os.makedirs(cache_dir_2)

    cache = Cache(input_dir=data_dir_1, chunk_size=2)

    for i in range(18):
        cache[i] = i

    cache.done()
    cache.merge()

    cache = Cache(input_dir=data_dir_2, chunk_size=2)

    for i in range(20):
        cache[i] = -i

    cache.done()
    cache.merge()

    dataset1 = StreamingDataset(input_dir=Dir(cache_dir_1, data_dir_1), shuffle=True)
    dataset2 = StreamingDataset(input_dir=Dir(cache_dir_2, data_dir_2), shuffle=True)
    dataset = ParallelStreamingDataset(datasets=[dataset1, dataset2], length=12)
    dataloader = StreamingDataLoader(dataset, num_workers=3, batch_size=2)

    assert dataset1.current_epoch == 1
    assert dataset2.current_epoch == 1

    expected_dataset_state = {
        "dataset": {
            "0": {
                "num_samples_yielded": 0,
                "num_workers": 3,
                "batch_size": 2,
                "current_epoch": 1,
                "input_dir_path": ANY,
                "input_dir_url": ANY,
                "cache_dir_path": None,
                "item_loader": None,
                "drop_last": False,
                "seed": 42,
                "world_size": 1,
                "shuffle": True,
                "subsampled_files": ANY,
                "region_of_interest": ANY,
            },
            "1": {
                "num_samples_yielded": 0,
                "num_workers": 3,
                "batch_size": 2,
                "current_epoch": 1,
                "input_dir_path": ANY,
                "input_dir_url": ANY,
                "cache_dir_path": None,
                "item_loader": None,
                "drop_last": False,
                "seed": 42,
                "world_size": 1,
                "shuffle": True,
                "subsampled_files": ANY,
                "region_of_interest": ANY,
            },
        },
        "current_epoch": 1,
        "latest_worker_idx": 0,
        "num_samples_yielded": {},
        "num_cycles": {},
    }
    expected_num_samples_yielded = [
        {0: [2, 2]},
        {0: [2, 2], 1: [2, 2]},
        {0: [2, 2], 1: [2, 2], 2: [2, 2]},
        {0: [4, 4], 1: [2, 2], 2: [2, 2]},
        {0: [4, 4], 1: [4, 4], 2: [2, 2]},
        {0: [4, 4], 1: [4, 4], 2: [4, 4]},
    ]
    expected_num_cycles = [
        {0: [0, 0]},
        {0: [0, 0], 1: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
    ]
    expected_current_epoch = [1, 1, 1, 1, 1, 1]
    dataset_1_current_epoch = [1, 1, 1, 1, 1, 1]
    dataset_2_current_epoch = [1, 1, 1, 1, 1, 1]
    expected_latest_worker_idx = [0, 1, 2, 0, 1, 2]
    expected_dataset0_samples_yielded = [2, 4, 6, 8, 10, 12]
    expected_dataset1_samples_yielded = [2, 4, 6, 8, 10, 12]

    batches_1 = []

    for idx, batch in enumerate(dataloader):
        batches_1.append(batch)
        curr_state_dict = dataloader.state_dict()

        expected_dataset_state["num_samples_yielded"] = expected_num_samples_yielded[idx]
        expected_dataset_state["num_cycles"] = expected_num_cycles[idx]
        expected_dataset_state["current_epoch"] = expected_current_epoch[idx]
        expected_dataset_state["latest_worker_idx"] = expected_latest_worker_idx[idx]
        expected_dataset_state["dataset"]["0"]["num_samples_yielded"] = expected_dataset0_samples_yielded[idx]
        expected_dataset_state["dataset"]["1"]["num_samples_yielded"] = expected_dataset1_samples_yielded[idx]
        expected_dataset_state["dataset"]["0"]["current_epoch"] = dataset_1_current_epoch[idx]
        expected_dataset_state["dataset"]["1"]["current_epoch"] = dataset_2_current_epoch[idx]

        assert curr_state_dict == expected_dataset_state

    assert dataset1.current_epoch == 1
    assert dataset2.current_epoch == 1

    saved_dataloader_state_dict = None

    batches_2 = []

    expected_num_samples_yielded = [
        {0: [6, 6], 1: [4, 4], 2: [4, 4]},
        {0: [6, 6], 1: [6, 6], 2: [4, 4]},
        {0: [6, 6], 1: [6, 6], 2: [6, 6]},
        {0: [2, 8], 1: [6, 6], 2: [6, 6]},
        {0: [2, 8], 1: [2, 2], 2: [6, 6]},
        {0: [2, 8], 1: [2, 2], 2: [2, 2]},
    ]
    expected_num_cycles = [
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [1, 0], 1: [0, 0], 2: [0, 0]},
        {0: [1, 0], 1: [1, 1], 2: [0, 0]},
        {0: [1, 0], 1: [1, 1], 2: [1, 1]},
    ]
    expected_current_epoch = [2, 2, 2, 2, 2, 2]
    dataset_1_current_epoch = [1, 1, 1, 2, 2, 2]
    dataset_2_current_epoch = [1, 1, 1, 1, 2, 2]
    expected_latest_worker_idx = [0, 1, 2, 0, 1, 2, 0]
    expected_dataset0_samples_yielded = [14, 16, 18, 2, 4, 6]
    expected_dataset1_samples_yielded = [14, 16, 18, 20, 2, 4]
    for idx, batch in enumerate(dataloader):
        batches_2.append(batch)
        curr_state_dict = dataloader.state_dict()

        expected_dataset_state["num_samples_yielded"] = expected_num_samples_yielded[idx]
        expected_dataset_state["num_cycles"] = expected_num_cycles[idx]
        expected_dataset_state["current_epoch"] = expected_current_epoch[idx]
        expected_dataset_state["latest_worker_idx"] = expected_latest_worker_idx[idx]
        expected_dataset_state["dataset"]["0"]["num_samples_yielded"] = expected_dataset0_samples_yielded[idx]
        expected_dataset_state["dataset"]["1"]["num_samples_yielded"] = expected_dataset1_samples_yielded[idx]
        expected_dataset_state["dataset"]["0"]["current_epoch"] = dataset_1_current_epoch[idx]
        expected_dataset_state["dataset"]["1"]["current_epoch"] = dataset_2_current_epoch[idx]

        assert curr_state_dict == expected_dataset_state

        if idx == 2:
            saved_dataloader_state_dict = deepcopy(curr_state_dict)

    assert dataset1.current_epoch == 2
    assert dataset2.current_epoch == 2

    assert len(batches_1) == len(batches_2)
    assert any(not torch.equal(x1, x2) for b1, b2 in zip(batches_1, batches_2) for x1, x2 in zip(b1, b2))

    assert saved_dataloader_state_dict is not None
    dataloader.load_state_dict(saved_dataloader_state_dict)

    assert dataloader.restore

    batches_23 = []
    states_23 = []
    for batch in dataloader:
        batches_23.append(batch)
        states_23.append(dataloader.state_dict())

    assert len(batches_2[3:]) == len(batches_23)
    assert all(torch.equal(x1, x2) for b1, b2 in zip(batches_2[3:], batches_23) for x1, x2 in zip(b1, b2))
    assert states_23[0]["current_epoch"] == 2

    assert not dataloader.restore
