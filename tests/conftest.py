import os
import shutil
import signal
import sys
import threading
from collections import OrderedDict
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch.distributed

from litdata import CombinedStreamingDataset, StreamingDataset
from litdata.constants import _POLARS_AVAILABLE
from litdata.streaming.cache import Cache
from litdata.streaming.reader import PrepareChunksThread
from litdata.utilities.dataset_utilities import get_default_cache_dir


@pytest.fixture(autouse=True, scope="session")
def teardown_process_group():
    """Ensures distributed process group gets closed before the next test runs."""
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture(autouse=True)
def disable_signals(monkeypatch):
    monkeypatch.setattr(signal, "signal", lambda *args, **kwargs: None)


@pytest.fixture
def mosaic_mds_index_data():
    return {
        "shards": [
            {
                "column_encodings": ["int", "jpeg"],
                "column_names": ["class", "image"],
                "column_sizes": [8, None],
                "compression": "zstd",
                "format": "mds",
                "hashes": [],
                "raw_data": {"basename": "shard.00000.mds", "bytes": 125824, "hashes": {}},
                "samples": 100,
                "size_limit": 67108864,
                "version": 2,
                "zip_data": {"basename": "shard.00000.mds.zstd", "bytes": 63407, "hashes": {}},
            }
        ],
        "version": 2,
    }


@pytest.fixture(scope="session")
def prepare_combined_dataset(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("combined_dataset")
    datasets = [str(tmpdir.join(f"dataset_{i}")) for i in range(2)]
    for dataset in datasets:
        cache = Cache(input_dir=dataset, chunk_bytes="64MB")
        for i in range(50):
            cache[i] = i
        cache.done()
        cache.merge()
    return datasets


@pytest.fixture
def combined_dataset(prepare_combined_dataset):
    dataset_1_path, dataset_2_path = prepare_combined_dataset
    dataset_1 = StreamingDataset(dataset_1_path)
    dataset_2 = StreamingDataset(dataset_2_path)
    return CombinedStreamingDataset(datasets=[dataset_1, dataset_2])


@pytest.fixture
def google_mock(monkeypatch):
    google = ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", google)
    google_cloud = ModuleType("cloud")
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud)
    google_cloud_storage = ModuleType("storage")
    monkeypatch.setitem(sys.modules, "google.cloud.storage", google_cloud_storage)
    google.cloud = google_cloud
    google.cloud.storage = google_cloud_storage
    return google


@pytest.fixture
def fsspec_mock(monkeypatch):
    fsspec = ModuleType("fsspec")
    monkeypatch.setitem(sys.modules, "fsspec", fsspec)
    return fsspec


@pytest.fixture
def obstore_mock(monkeypatch):
    obstore = ModuleType("obstore")
    monkeypatch.setitem(sys.modules, "obstore", obstore)
    obstore.store = Mock()
    return obstore


@pytest.fixture
def azure_mock(monkeypatch):
    azure = ModuleType("azure")
    monkeypatch.setitem(sys.modules, "azure", azure)
    azure_storage = ModuleType("storage")
    monkeypatch.setitem(sys.modules, "azure.storage", azure_storage)
    azure_storage_blob = ModuleType("storage")
    monkeypatch.setitem(sys.modules, "azure.storage.blob", azure_storage_blob)
    azure.storage = azure_storage
    azure.storage.blob = azure_storage_blob
    return azure


@pytest.fixture
def lightning_cloud_mock(monkeypatch):
    lightning_cloud = ModuleType("lightning_sdk.lightning_cloud")
    monkeypatch.setitem(sys.modules, "lightning_sdk.lightning_cloud", lightning_cloud)
    rest_client = ModuleType("rest_client")
    monkeypatch.setitem(sys.modules, "lightning_sdk.lightning_cloud.rest_client", rest_client)
    lightning_cloud.rest_client = rest_client
    rest_client.LightningClient = Mock()
    return lightning_cloud


@pytest.fixture
def lightning_sdk_mock(monkeypatch):
    lightning_sdk = ModuleType("lightning_sdk")
    monkeypatch.setitem(sys.modules, "lightning_sdk", lightning_sdk)
    return lightning_sdk


@pytest.fixture(autouse=True, scope="session")
def _thread_police():
    """Attempts stopping left-over threads to avoid test interactions.

    Adapted from PyTorch Lightning.

    """
    active_threads_before = set(threading.enumerate())
    yield
    active_threads_after = set(threading.enumerate())

    for thread in active_threads_after - active_threads_before:
        if isinstance(thread, PrepareChunksThread):
            thread.force_stop()
            continue

        stop = getattr(thread, "stop", None) or getattr(thread, "exit", None)
        if thread.daemon and callable(stop):
            # A daemon thread would anyway be stopped at the end of a program
            # We do it preemptively here to reduce the risk of interactions with other tests that run after
            stop()
            assert not thread.is_alive()
        elif thread.name == "QueueFeederThread":
            thread.join(timeout=20)
        else:
            if thread.name.startswith("pytest_timeout"):
                continue
            raise AssertionError(f"Test left zombie thread: {thread}")


# ==== fixtures for parquet ====
@pytest.fixture
def pq_data():
    return OrderedDict(
        {
            "name": ["Tom", "Jerry", "Micky", "Oggy", "Doraemon"],
            "weight": [57.9, 72.5, 53.6, 83.1, 69.4],  # (kg)
            "height": [1.56, 1.77, 1.65, 1.75, 1.63],  # (m)
        }
    )


@pytest.fixture
def write_pq_data(pq_data, tmp_path):
    if _POLARS_AVAILABLE:
        import polars as pl

        os.mkdir(os.path.join(tmp_path, "pq-dataset"))

        for i in range(5):
            df = pl.DataFrame(pq_data)
            file_path = os.path.join(tmp_path, "pq-dataset", f"tmp-{i}.parquet")
            df.write_parquet(file_path)


@pytest.fixture
def clean_pq_index_cache():
    """Ensures the PQ index cache is cleared before and after the test."""
    cache_path = os.path.join(get_default_cache_dir())

    # Cleanup before the test
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)

    yield

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)


@pytest.fixture
def huggingface_hub_mock(monkeypatch):
    huggingface_hub = ModuleType("huggingface_hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    hf_hub_download = ModuleType("hf_hub_download")
    monkeypatch.setitem(sys.modules, "huggingface_hub.hf_hub_download", hf_hub_download)
    huggingface_hub.hf_hub_download = hf_hub_download
    return huggingface_hub


@pytest.fixture
def huggingface_hub_fs_mock(monkeypatch, write_pq_data, tmp_path):
    huggingface_hub = ModuleType("huggingface_hub")
    hf_file_system = ModuleType("hf_file_system")

    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.HfFileSystem", hf_file_system)

    huggingface_hub.HfFileSystem = hf_file_system

    def mock_open(filename, mode="rb"):
        filename = filename.split("/")[-1]
        file_path = os.path.join(tmp_path, "pq-dataset", filename)
        assert os.path.exists(file_path), "file hf is trying to access, doesn't exist."
        return open(file_path, mode)

    def mock_ls(*args, **kwargs):
        files = sorted(os.listdir(os.path.join(tmp_path, "pq-dataset")))
        return [
            {
                "type": "file",
                "name": file_name,
                "size": os.path.getsize(os.path.join(tmp_path, "pq-dataset", file_name)),
            }
            for file_name in files
        ]

    def mock_hf_hub_download(repo_id, filename, cache_dir, repo_type, **kwargs):
        return os.path.join(tmp_path, "pq-dataset", os.path.basename(filename))

    hf_fs_mock = Mock()
    hf_fs_mock.ls = Mock(side_effect=mock_ls)
    hf_fs_mock.open = Mock(side_effect=mock_open)
    huggingface_hub.HfFileSystem = Mock(return_value=hf_fs_mock)
    huggingface_hub.hf_hub_download = Mock(side_effect=mock_hf_hub_download)

    return huggingface_hub


@pytest.fixture
def fsspec_pq_mock(monkeypatch, write_pq_data, tmp_path, fsspec_mock):
    def mock_open(filename, mode="rb"):
        filename = filename.split("/")[-1]
        print(f"{filename=}")
        file_path = os.path.join(tmp_path, "pq-dataset", filename)
        if mode.startswith("r"):
            assert os.path.exists(file_path), "file cloud is trying to access, doesn't exist."
        return open(file_path, mode)

    def mock_fsspec_ls(*args, **kwargs):
        file_list = []
        for file_name in os.listdir(os.path.join(tmp_path, "pq-dataset")):
            file_list.append(
                {
                    "type": "file",
                    "name": file_name,
                    "size": os.path.getsize(os.path.join(tmp_path, "pq-dataset", file_name)),
                }
            )
        return file_list

    fs_mock = Mock()
    fs_mock.ls = Mock(side_effect=mock_fsspec_ls)
    fs_mock.open = Mock(side_effect=mock_open)
    fsspec_mock.filesystem = Mock(return_value=fs_mock)

    return fsspec_mock
