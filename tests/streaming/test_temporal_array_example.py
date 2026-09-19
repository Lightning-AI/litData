"""Built-in window reads use ordinary optimize/PyTree chunks."""

import asyncio
import pickle
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from litdata import StreamingDataset, TemporalArrayLoader, optimize, train_test_split
from litdata.streaming import Cache
from litdata.streaming.downloader import Downloader


def make_record(n):
    return {
        "embedding": np.arange(n * 6, dtype=np.float32).reshape(n, 2, 3),
        "clock": torch.arange(n, dtype=torch.int64),
        "tensor": torch.arange(n * 2, dtype=torch.float64).reshape(n, 2),
        "valid": np.arange(n) % 2 == 0,
    }


@pytest.fixture(autouse=True)
def shutdown_runner():
    yield
    from litdata.raw import dataset as raw_dataset

    runner = raw_dataset._RUNNER
    raw_dataset._shutdown_runner_before_fork()
    if runner is not None:
        runner._executor.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def arrays(tmp_path):
    root = tmp_path / "arrays"
    cache = Cache(str(root), chunk_size=2)
    records = [make_record(n) for n in (30, 17, 41, 23)]
    for i, record in enumerate(records):
        cache[i] = record
    cache.done()
    cache.merge()
    return root, records


def assert_values(actual, record, start, frames, fields, owned=True):
    assert list(actual) == fields
    for name, value in actual.items():
        expected = record[name][start : start + frames]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(value, expected)
        else:
            np.testing.assert_array_equal(value, expected)
            assert not owned or (value.flags.owndata and value.flags.writeable)
        assert value.dtype == expected.dtype


def test_optimize_public_workflow(tmp_path):
    root = str(tmp_path / "optimized")
    optimize(make_record, [12, 20, 31], output_dir=root, chunk_size=2, num_workers=1)
    dataset = StreamingDataset(root)
    assert_values(dataset.read_window(1, 3, 7), make_record(20), 3, 7, list(make_record(20)))


def test_windows_and_full_records(arrays):
    root, records = arrays
    dataset = StreamingDataset(str(root), transform=lambda value: {"transformed": value})
    for index, start, frames, fields in [
        (0, 0, 1, ["valid"]),
        (0, 7, 17, ["embedding", "clock"]),
        (1, 0, 17, ["clock", "valid"]),
        (2, 40, 1, ["tensor", "embedding"]),
        (3, 0, 23, list(records[3])),
    ]:
        actual = dataset.read_window(index, start, frames, fields)
        assert_values(actual, records[index], start, frames, fields)
        for value in actual.values():
            value[...] = 0
        assert_values(dataset.read_window(index, start, frames, fields), records[index], start, frames, fields)
    assert "transformed" in dataset[0]
    plain = StreamingDataset(str(root))
    assert_values(plain[0], records[0], 0, 30, list(records[0]), owned=False)
    # Neither async ranges nor the sync loop runner are serialized with the dataset.
    resumed = pickle.loads(pickle.dumps(StreamingDataset(str(root))))  # noqa: S301
    assert_values(resumed.read_window(2, 3, 5), records[2], 3, 5, list(records[2]))


def attach_remote(dataset, root, monkeypatch):
    dataset.read_window(0, 0, 1, [])  # initialize metadata, without opening a payload
    config = dataset.cache._reader.config

    class FakeDownloader(Downloader):
        def __init__(self):
            super().__init__("s3://bucket/data", str(root), [])
            self.reads = []
            self.active = 0
            self.peak = 0
            self.fail = False
            self.block = False
            self.started = asyncio.Event()

        async def adownload_bytes(self, remote, offset, length, scratch):
            self.reads.append((remote, offset, length))
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                self.started.set()
                await asyncio.sleep(0.001)
                if self.block:
                    await asyncio.Event().wait()
                if self.fail:
                    raise OSError("read failed")
                with (root / remote.rsplit("/", 1)[-1]).open("rb") as handle:
                    handle.seek(offset)
                    return handle.read(length)
            finally:
                self.active -= 1

    downloader = FakeDownloader()
    config._remote_dir = "s3://bucket/data"
    config._downloader = downloader
    monkeypatch.setattr(config, "download_chunk_from_index", Mock(side_effect=AssertionError("whole chunk requested")))
    return downloader


def test_remote_projection_byte_budget_and_concurrency(arrays, monkeypatch):
    root, records = arrays
    dataset = StreamingDataset(str(root))
    remote = attach_remote(dataset, root, monkeypatch)

    async def run():
        fields = list(records[0])
        actual = await dataset.aread_window(0, 7, 5, fields, max_concurrent_reads=2)
        assert_values(actual, records[0], 7, 5, fields)
        # Cold metadata overhead is two offset/header reads + rank/shape per multidimensional field.
        payload = sum(
            value.numel() * value.element_size() if isinstance(value, torch.Tensor) else value.nbytes
            for value in actual.values()
        )
        assert sum(length for _, _, length in remote.reads) == payload + 8 + 16 + 20 + 16
        assert remote.peak == 2
        remote.reads.clear()
        actual = await dataset.aread_window(0, 9, 3, ["valid"])
        assert len(remote.reads) == 1
        assert remote.reads[0][2] == 3
        assert_values(actual, records[0], 9, 3, ["valid"])
        remote.reads.clear()
        assert await dataset.aread_window(0, 0, 1, []) == {}
        with pytest.raises(KeyError):
            await dataset.aread_window(0, 0, 1, ["valid", "missing"])
        with pytest.raises(ValueError, match="Duplicate"):
            await dataset.aread_window(0, 0, 1, ["valid", "valid"])
        with pytest.raises(IndexError, match="outside"):
            await dataset.aread_window(0, 29, 2, ["valid"])
        assert not remote.reads
        outputs = await asyncio.gather(*(dataset.aread_window(0, i, 2, fields) for i in range(5)))
        for i, output in enumerate(outputs):
            assert_values(output, records[0], i, 2, fields)
        remote.fail = True
        with pytest.raises(OSError, match="read failed"):
            await dataset.aread_window(0, 0, 2, fields)
        assert remote.active == 0
        remote.fail = False
        remote.block = True
        remote.started.clear()
        pending = asyncio.create_task(dataset.aread_window(0, 0, 2, fields))
        await remote.started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert remote.active == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    ("index", "start", "frames", "error"),
    [
        (-1, 0, 1, IndexError),
        (4, 0, 1, IndexError),
        (0, -1, 1, IndexError),
        (0, 0, 0, IndexError),
        (0, 29, 2, IndexError),
        (0, 0.5, 1, TypeError),
        (True, 0, 1, TypeError),
        (0, 0, "1", TypeError),
    ],
)
def test_invalid_windows(arrays, index, start, frames, error):
    root, _ = arrays
    with pytest.raises(error, match="."):
        StreamingDataset(str(root)).read_window(index, start, frames)


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_invalid_concurrency(arrays, concurrency):
    with pytest.raises(ValueError, match="positive integer"):
        StreamingDataset(str(arrays[0])).read_window(0, 0, 1, max_concurrent_reads=concurrency)


def test_unsupported_formats_and_misalignment(tmp_path):
    records = [
        {"a": np.ones(3), "b": np.ones(4), "text": "ignore me"},
        {"nested": {"a": np.ones(3)}},
        {"a": np.empty((3, 0))},
    ]
    for i, record in enumerate(records):
        root = tmp_path / str(i)
        cache = Cache(str(root), chunk_size=2)
        cache[0] = record
        cache.done()
        cache.merge()
        dataset = StreamingDataset(str(root))
        with pytest.raises(ValueError, match="."):
            dataset.read_window(0, 0, 1)
        if i == 0:
            np.testing.assert_array_equal(dataset.read_window(0, 0, 2, ["a"])["a"], np.ones(2))
            with pytest.raises(ValueError, match="frame-axis"):
                dataset.read_window(0, 0, 1, ["a", "b"])
            with pytest.raises(TypeError, match="sequence"):
                dataset.read_window(0, 0, 1, "a")


@pytest.mark.parametrize("option", ["compression", "encryption", "format"])
def test_reject_unsupported_chunk_formats(arrays, option):
    root, _ = arrays
    dataset = StreamingDataset(str(root))
    dataset.read_window(0, 0, 1, [])
    config = dataset.cache._reader.config
    config.config[option] = "mds" if option == "format" else "enabled"
    del dataset.cache._reader._window_reader
    with pytest.raises(ValueError, match="uncompressed"):
        dataset.read_window(0, 0, 1)


def test_subsample_and_split_mapping(arrays):
    root, records = arrays
    dataset = StreamingDataset(str(root))
    train, val = train_test_split(dataset, [0.5, 0.5])
    for subset in (train, val):
        for i in range(len(subset)):
            complete = subset[i]
            assert_values(subset.read_window(i, 1, 2), complete, 1, 2, list(complete))


def test_corrupt_metadata_and_truncated_payload(arrays):
    root, _ = arrays
    dataset = StreamingDataset(str(root))
    dataset.read_window(0, 0, 1)  # cache offsets and shapes
    chunk = next(root.glob("chunk-0-0.bin"))
    chunk.write_bytes(chunk.read_bytes()[:100])
    with pytest.raises(OSError, match="Short window read"):
        dataset.read_window(0, 10, 5)


@pytest.mark.parametrize("workers", [1, 2])
def test_temporal_optimize_and_three_range_customer_pattern(tmp_path, monkeypatch, workers):
    root = tmp_path / "temporal"
    optimize(
        make_record,
        [30, 17, 41, 23],
        output_dir=str(root),
        chunk_size=2,
        num_workers=workers,
        item_loader=TemporalArrayLoader(field_groups=[["clock", "valid"]]),
    )
    dataset = StreamingDataset(str(root), item_loader=TemporalArrayLoader())
    counts = dataset.frame_counts
    assert sorted(counts) == [17, 23, 30, 41]
    remote = attach_remote(dataset, root, monkeypatch)
    actual = dataset.read_window(0, 2, 7)
    assert_values(actual, make_record(counts[0]), 2, 7, list(make_record(counts[0])))
    assert len(remote.reads) == 3  # no metadata GET, even on a cold read
    assert sum(length for _, _, length in remote.reads) == 7 * (16 + 24 + 16)
    remote.reads.clear()
    subset = dataset.read_window(0, 1, 2, ["clock"])
    assert len(remote.reads) == 1
    assert remote.reads[0][2] == 32  # documented group overfetch
    torch.testing.assert_close(subset["clock"], torch.arange(1, 3))
    remote.reads.clear()
    with pytest.raises(IndexError, match="outside"):
        dataset.read_window(0, counts[0], 1)
    with pytest.raises(KeyError):
        dataset.read_window(0, 0, 1, ["missing"])
    assert dataset.read_window(0, 0, 1, []) == {}
    assert not remote.reads
    # Full-record reads, iteration, splits and serialization retain the existing contracts.
    plain = StreamingDataset(str(root), item_loader=TemporalArrayLoader())
    assert_values(plain[0], make_record(counts[0]), 0, counts[0], list(actual))
    for record in plain:
        assert_values(record, make_record(len(record["clock"])), 0, len(record["clock"]), list(actual))
    restored = pickle.loads(pickle.dumps(plain))  # noqa: S301
    assert_values(restored.read_window(0, 2, 7), make_record(counts[0]), 2, 7, list(actual))
    for split in train_test_split(StreamingDataset(str(root), item_loader=TemporalArrayLoader()), [0.5, 0.5]):
        for i, count in enumerate(split.frame_counts):
            assert_values(split.read_window(i, 1, 2), make_record(count), 1, 2, list(actual))


@pytest.mark.parametrize(
    ("groups", "full_reads", "valid_row_bytes"),
    [([["embedding", "clock", "tensor", "valid"]], 1, 56), ([], 4, 1)],
)
def test_temporal_all_fields_or_separate_packing(tmp_path, monkeypatch, groups, full_reads, valid_row_bytes):
    # Both packing choices preserve the API and values, while projection changes
    # physical I/O: a single selected bool still fetches an entire grouped row.
    cache = Cache(str(tmp_path), chunk_size=1, item_loader=TemporalArrayLoader(field_groups=groups))
    record = make_record(17)
    cache[0] = record
    cache.done()
    cache.merge()
    dataset = StreamingDataset(str(tmp_path), item_loader=TemporalArrayLoader())
    remote = attach_remote(dataset, tmp_path, monkeypatch)
    assert_values(dataset.read_window(0, 3, 7), record, 3, 7, list(record))
    assert len(remote.reads) == full_reads
    remote.reads.clear()
    assert_values(dataset.read_window(0, 15, 2, ["valid"]), record, 15, 2, ["valid"])
    assert len(remote.reads) == 1
    assert remote.reads[0][2] == 2 * valid_row_bytes


@pytest.mark.parametrize("groups", [[["missing"]], [["clock"], ["clock"]], [[]], ["clock"]])
def test_invalid_grouping(tmp_path, groups):
    with pytest.raises(ValueError, match="."):  # noqa: PT012
        cache = Cache(str(tmp_path), chunk_size=1, item_loader=TemporalArrayLoader(field_groups=groups))
        cache[0] = make_record(10)


def test_temporal_schema_validation(tmp_path):
    cache = Cache(str(tmp_path), chunk_size=3, item_loader=TemporalArrayLoader())
    cache[0] = make_record(10)
    changed = make_record(11)
    changed["valid"] = changed["valid"].astype(np.int32)
    with pytest.raises(ValueError, match="schema differs"):
        cache[1] = changed
    changed = make_record(11)
    changed["clock"] = torch.arange(5)
    with pytest.raises(ValueError, match="frame axis"):
        cache[1] = changed


def test_explicit_window_indices_are_global_under_ddp(arrays, monkeypatch):
    root, records = arrays
    dataset = StreamingDataset(str(root))
    monkeypatch.setattr(StreamingDataset, "__len__", lambda self: 1)
    assert_values(dataset.read_window(3, 2, 3), records[3], 2, 3, list(records[3]))


class WindowRequests(torch.utils.data.Dataset):
    def __init__(self, path):
        self.records = StreamingDataset(path, item_loader=TemporalArrayLoader())
        self.counts = self.records.frame_counts
        self.records.read_window(0, 0, 1)  # initialized parent must survive spawn/fork safely

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, index):
        return self.records.read_window(index, self.counts[index] - 2, 2)


@pytest.mark.parametrize("context", ["spawn", "fork"])
def test_window_dataloader_workers(tmp_path, context):
    import multiprocessing

    if context not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{context} is unavailable")
    cache = Cache(str(tmp_path), chunk_size=2, item_loader=TemporalArrayLoader())
    for i in range(4):
        cache[i] = make_record(10 + i)
    cache.done()
    cache.merge()
    loader = torch.utils.data.DataLoader(
        WindowRequests(str(tmp_path)),
        batch_size=None,
        num_workers=2,
        multiprocessing_context=context,
    )
    for i, record in enumerate(loader):
        torch.testing.assert_close(record["clock"], torch.arange(8 + i, 10 + i))


def test_temporal_full_record_loader_resume(tmp_path):
    from litdata import StreamingDataLoader

    cache = Cache(str(tmp_path), chunk_size=2, item_loader=TemporalArrayLoader())
    for i in range(6):
        cache[i] = make_record(10 + i)
    cache.done()
    cache.merge()
    first = StreamingDataLoader(StreamingDataset(str(tmp_path), item_loader=TemporalArrayLoader()), batch_size=1)
    iterator = iter(first)
    next(iterator)
    next(iterator)
    state = first.state_dict()
    expected = [batch["clock"].shape[-1] for batch in iterator]
    restored = StreamingDataLoader(StreamingDataset(str(tmp_path), item_loader=TemporalArrayLoader()), batch_size=1)
    restored.load_state_dict(state)
    assert [batch["clock"].shape[-1] for batch in restored] == expected
