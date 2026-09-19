"""Contract tests for the adaptable example; it is not a public dataset API."""

import asyncio
import copy

import numpy as np
import pytest

from examples.temporal_arrays.reader import ArrayWindowReader, write_records
from litdata.streaming.downloader import Downloader


def test_example_roundtrip_and_projection_byte_budget(tmp_path):
    records = [
        {
            "embedding": np.arange(180, dtype=np.float32).reshape(30, 2, 3),
            "clock": np.arange(30, dtype=np.int64),
            "valid": np.arange(30) % 2 == 0,
        },
        {
            "embedding": np.ones((17, 2, 3), dtype=np.float32),
            "clock": np.arange(17, dtype=np.int64),
            "valid": np.ones(17, dtype=bool),
        },
    ]
    root = tmp_path / "arrays"
    write_records(records, str(root), shard_bytes=300)

    async def run():
        reader = await ArrayWindowReader.open(str(root), str(tmp_path / "cache"))
        actual_read = reader.downloader.adownload_bytes
        reads = []

        async def tracked(path, offset, length, scratch):
            reads.append((offset, length))
            return await actual_read(path, offset, length, scratch)

        reader.downloader.adownload_bytes = tracked
        for record, start, frames, fields in [
            (0, 0, 1, ["valid"]),
            (0, 7, 17, ["embedding", "clock"]),
            (1, 0, 17, ["clock", "valid"]),
            (0, 29, 1, ["embedding"]),
        ]:
            reads.clear()
            values = await reader.read(record, start, frames, fields)
            assert list(values) == fields
            for name, actual in values.items():
                expected = records[record][name][start : start + frames]
                np.testing.assert_array_equal(actual, expected)
                assert actual.dtype == expected.dtype
                assert actual.flags.writeable
                assert actual.flags.owndata
            assert sum(length for _, length in reads) == sum(v.nbytes for v in values.values())
            assert len(reads) == len(fields)
        reads.clear()
        assert await reader.read(0, 0, 1, []) == {}
        for record, start, frames in [(-1, 0, 1), (2, 0, 1), (0, -1, 1), (0, 0, 0), (0, 29, 2)]:
            with pytest.raises(IndexError):
                await reader.read(record, start, frames, ["valid"])
        with pytest.raises(KeyError):
            await reader.read(0, 0, 1, ["valid", "unknown"])
        with pytest.raises(ValueError, match="Duplicate"):
            await reader.read(0, 0, 1, ["valid", "valid"])
        assert not reads
        index = {"version": 1, "records": copy.deepcopy(reader.records)}
        index["records"][0]["fields"]["valid"]["file"] = "../elsewhere.bin"
        with pytest.raises(ValueError, match="relative filenames"):
            ArrayWindowReader(reader.uri, index, reader.downloader, reader.cache_dir)

    asyncio.run(run())
    # Local reads never copy payload shards to the cache.
    assert not list((tmp_path / "cache").rglob("*.bin"))


def test_example_bounds_concurrency_and_drains_failed_reads(tmp_path):
    class FakeDownloader(Downloader):
        active = 0
        peak = 0
        fail = False

        async def adownload_bytes(self, path, offset, length, scratch):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.001)
                if self.fail and offset == 0:
                    raise OSError("read failed")
                return bytes(length)
            finally:
                self.active -= 1

    fields = {str(i): {"file": "data.bin", "offset": i, "shape": [1], "dtype": "|u1"} for i in range(6)}
    index = {"version": 1, "records": [{"frames": 1, "fields": fields}]}
    downloader = FakeDownloader("s3://bucket/data", str(tmp_path), [])

    async def run():
        reader = ArrayWindowReader("s3://bucket/data", index, downloader, str(tmp_path), max_concurrent_reads=2)
        await reader.read(0, 0, 1, list(fields))
        assert downloader.peak == 2
        downloader.fail = True
        with pytest.raises(OSError, match="read failed"):
            await reader.read(0, 0, 1, list(fields))
        assert downloader.active == 0

    asyncio.run(run())


def test_example_failed_write_does_not_publish_index(tmp_path):
    for name, record in [
        ("axis", {"a": np.ones(3), "b": np.ones(4)}),
        ("dtype", {"a": np.array([object()], dtype=object)}),
    ]:
        root = tmp_path / name
        with pytest.raises(ValueError, match="frame axis|fixed-width"):
            write_records([record], str(root))
        assert not (root / "index.json").exists()
