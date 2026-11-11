import asyncio
import os
import platform

import numpy as np
import pytest

from litdata import StreamingDataset, TemporalArrayLoader
from litdata.streaming import Cache

pytestmark = pytest.mark.skipif(os.name != "posix" or platform.system() != "Linux", reason="Linux NFS direct mode")


def test_direct_dataset_disables_mmap_and_ordinary_iteration(tmp_path, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "1")
    monkeypatch.setattr("litdata.utilities.direct_io._check_nfs", lambda fd: None)
    records = [{"x": np.arange(80, dtype=np.float32).reshape(20, 4)}]
    writer = Cache(str(tmp_path), chunk_size=1, item_loader=TemporalArrayLoader())
    writer[0] = records[0]
    writer.done()
    writer.merge()
    opened = []

    def fake_direct(path):
        opened.append(path)
        return open(path, "rb", buffering=0)

    monkeypatch.setattr("litdata.streaming.window.open_nfs_direct", fake_direct)
    dataset = StreamingDataset(str(tmp_path), item_loader=TemporalArrayLoader(), window_direct_io=True)
    np.testing.assert_array_equal(dataset.read_window(0, 3, 5)["x"], records[0]["x"][3:8])
    np.testing.assert_array_equal(asyncio.run(dataset.aread_window(0, 7, 3))["x"], records[0]["x"][7:10])
    assert opened
    assert dataset.cache._reader._window_reader._maps is None
    assert not dataset.cache._reader._posix_fast
    with pytest.raises(RuntimeError, match="read_window"):
        iter(dataset)
    with pytest.raises(RuntimeError, match="read_window"):
        dataset[0]
    dataset.cache._reader._window_reader.close()


def test_direct_dataset_rejects_non_nfs(tmp_path):
    with pytest.raises(OSError, match="NFS filesystem"):
        StreamingDataset(str(tmp_path), window_direct_io=True)


def test_real_nfs_direct_windows():
    mount = os.environ.get("LITDATA_TEST_NFS_DIR")
    if not mount:
        pytest.skip("Set LITDATA_TEST_NFS_DIR for real NFS window reads")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="litdata-window-test-", dir=mount) as directory:
        record = {"x": np.arange(800, dtype=np.float32).reshape(100, 8)}
        writer = Cache(directory, chunk_size=1, item_loader=TemporalArrayLoader())
        writer[0] = record
        writer.done()
        writer.merge()
        dataset = StreamingDataset(directory, item_loader=TemporalArrayLoader(), window_direct_io=True)
        try:
            np.testing.assert_array_equal(dataset.read_window(0, 7, 19)["x"], record["x"][7:26])
            np.testing.assert_array_equal(asyncio.run(dataset.aread_window(0, 2, 11))["x"], record["x"][2:13])
            assert dataset.cache._reader._window_reader._maps is None
        finally:
            if dataset.cache is not None:
                dataset.cache._reader._window_reader.close()
