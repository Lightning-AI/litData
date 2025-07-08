"""contains utility functions to return parquet files from local, s3, or gs."""

import io
import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import Any, Optional, Union
from urllib import parse

from litdata.constants import _FSSPEC_AVAILABLE, _HF_HUB_AVAILABLE, _INDEX_FILENAME, _PYARROW_AVAILABLE
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.utilities.env import _DistributedEnv


class ParquetDir(ABC):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[dict] = {},
        num_workers: int = 4,
    ):
        self.dir = _resolve_dir(dir_path)
        self.cache_path = cache_path
        self.storage_options = storage_options
        self.files: list[Any] = []
        self.num_workers = num_workers

    def __iter__(self) -> Generator[tuple[dict[str, Any], int], None, None]:
        """Iterate over the Parquet files and yield their metadata.

        Yields:
            Generator[Tuple[str, int], None, None]: A generator yielding tuples of file name, file path, and order.
        """
        if not self.files:
            raise RuntimeError(
                f"No Parquet files were found at '{self.dir.url or self.dir.path}'. "
                "Please verify that the provided path is correct and that it contains at least one .parquet file. "
                "If the files are located in a subdirectory, please specify the correct path."
            )

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.task, _file): (order, _file) for order, _file in enumerate(self.files)}
            for future in futures:
                file_metadata = future.result()
                order, _ = futures[future]
                yield file_metadata, order

    @abstractmethod
    def task(self, _file: Any) -> dict[str, Any]: ...

    @abstractmethod
    def write_index(self, chunks_info: list[dict[str, Any]], config: dict[str, Any]) -> None:
        """Write the index file to the cache directory.

        Args:
            chunks_info (List[Dict[str, Any]]): List of dictionaries containing chunk metadata.
            config (Dict[str, Any]): Configuration dictionary containing metadata.
        """
        assert self.cache_path is not None, "Cache path is not set."
        index_file_path = os.path.join(self.cache_path, _INDEX_FILENAME)

        # write to index.json file
        with open(index_file_path, "w") as f:
            data = {"chunks": chunks_info, "config": config, "updated_at": str(time())}
            json.dump(data, f, sort_keys=True)


class LocalParquetDir(ParquetDir):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[dict] = {},
        num_workers: int = 4,
    ):
        if not _PYARROW_AVAILABLE:
            raise ModuleNotFoundError(
                "The 'pyarrow' module is required for processing Parquet files. "
                "Please install it by running: `pip install pyarrow`"
            )
        super().__init__(dir_path, cache_path, storage_options, num_workers)

        for _f in os.listdir(self.dir.path):
            if _f.endswith(".parquet"):
                self.files.append(_f)

    def task(self, _file: str) -> dict[str, Any]:
        """Extract metadata from a Parquet file on the local filesystem."""
        import pyarrow.parquet as pq

        assert isinstance(_file, str)
        assert _file.endswith(".parquet")
        assert self.dir.path is not None
        assert self.dir.path != "", "Dir path can't be empty"

        file_path = os.path.join(self.dir.path, _file)
        parquet_file = pq.ParquetFile(file_path)
        num_rows = parquet_file.metadata.num_rows
        data_types = [str(col.type) for col in parquet_file.schema_arrow]
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)

        return {
            "file_name": file_name,
            "num_rows": num_rows,
            "file_size": file_size,
            "data_types": data_types,
        }

    def write_index(self, chunks_info: list[dict[str, Any]], config: dict[str, Any]) -> None:
        """Write the index file to the cache directory."""
        if self.cache_path is None:
            self.cache_path = self.dir.path
        super().write_index(chunks_info, config)


class CloudParquetDir(ParquetDir):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[dict] = None,
        num_workers: int = 4,
    ):
        if not _FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Indexing cloud parquet files depends on `fsspec`.", "Please, run: `pip install fsspec`"
            )
        if not _PYARROW_AVAILABLE:
            raise ModuleNotFoundError(
                "The 'pyarrow' module is required for processing Parquet files. "
                "Please install it by running: `pip install pyarrow`"
            )
        super().__init__(dir_path, cache_path, storage_options, num_workers)

        assert self.dir.url is not None, "Dir url can't be empty"

        import fsspec

        _CLOUD_PROVIDER = ("s3", "gs")

        for provider in _CLOUD_PROVIDER:
            if self.dir.url.startswith(provider):
                # Initialize the cloud filesystem
                self.fs = fsspec.filesystem(provider, *self.storage_options)
                print(f"using provider: {provider}")
                break

        # List all files and directories in the top-level of the specified directory
        for _f in self.fs.ls(self.dir.url, detail=True):
            if _f["type"] == "file" and _f["name"].endswith(".parquet"):
                self.files.append(_f)

    def task(self, _file: Any) -> dict[str, Any]:
        """Extract metadata from a Parquet file on the cloud filesystem without downloading the entire file."""
        import pyarrow.parquet as pq

        # Validate inputs
        if not isinstance(_file, dict) or "name" not in _file or "size" not in _file:
            raise ValueError(f"Invalid file object: {_file}")

        assert _file["name"].endswith(".parquet")

        file_name = os.path.basename(_file["name"])
        file_size = _file["size"]

        with self.fs.open(_file["name"], "rb") as f:
            # Read footer size (last 8 bytes: 4 for footer size + 4 for magic number)
            f.seek(file_size - 8)
            footer_size = int.from_bytes(f.read(4), "little")

            # seek to the start of the footer and read the footer data
            footer_start = file_size - footer_size - 8
            f.seek(footer_start)
            footer_data = f.read(file_size - footer_start)

            if len(footer_data) != file_size - footer_start:
                raise ValueError(f"Failed to read complete footer data from {file_name}")

        # Parse the footer data to extract schema information
        with io.BytesIO(footer_data) as footer_buffer:
            parquet_file = pq.ParquetFile(footer_buffer)
            data_types = [str(col.type) for col in parquet_file.schema_arrow]
            num_rows = parquet_file.metadata.num_rows

        return {
            "file_name": file_name,
            "num_rows": num_rows,
            "file_size": file_size,
            "data_types": data_types,
        }

    def write_index(self, chunks_info: list[dict[str, Any]], config: dict[str, Any]) -> None:
        """Write the index file and upload it to the cloud."""
        assert self.dir.url is not None
        with tempfile.TemporaryDirectory() as temp_dir:
            index_file_path = os.path.join(temp_dir, _INDEX_FILENAME)
            cloud_index_path = os.path.join(self.dir.url, _INDEX_FILENAME)

            # write to index.json file
            with open(index_file_path, "w") as f:
                data = {"chunks": chunks_info, "config": config, "updated_at": str(time())}
                json.dump(data, f, sort_keys=True)

            env = _DistributedEnv.detect()
            # Prevent multiple nodes & processes from uploading the same index file; only global rank 0 should proceed
            if env.global_rank != 0:
                return
            # upload index file to cloud
            with open(index_file_path, "rb") as local_file, self.fs.open(cloud_index_path, "wb") as cloud_file:
                for chunk in iter(lambda: local_file.read(4096), b""):  # Read in 4KB chunks
                    cloud_file.write(chunk)

            print(f"Index file successfully written to: {cloud_index_path}")


class HFParquetDir(ParquetDir):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[dict] = None,
        num_workers: int = 4,
    ):
        if not _HF_HUB_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Indexing HF depends on `huggingface_hub`.", "Please, run: `pip install huggingface_hub"
            )
        if not _PYARROW_AVAILABLE:
            raise ModuleNotFoundError(
                "The 'pyarrow' module is required for processing Parquet files. "
                "Please install it by running: `pip install pyarrow`"
            )
        super().__init__(dir_path, cache_path, storage_options, num_workers)

        assert self.dir.url is not None, "Dir url is not set."
        assert self.dir.url.startswith("hf"), "Dir url must start with 'hf://'."
        assert self.cache_path is not None, "Cache path is not set."

        # List all files and directories in the top-level of the specified directory
        from huggingface_hub import HfFileSystem

        self.fs = HfFileSystem()

        for _f in self.fs.ls(self.dir.url, detail=True):
            if isinstance(_f, dict) and _f["name"].endswith(".parquet"):
                self.files.append(_f)

    def task(self, _file: dict) -> dict[str, Any]:
        """Extract metadata from a Parquet file on Hugging Face Hub without downloading the entire file."""
        import pyarrow.parquet as pq

        # Validate inputs
        if not isinstance(_file, dict) or "name" not in _file or "size" not in _file:
            raise ValueError(f"Invalid file object: {_file}")

        assert _file["name"].endswith(".parquet")

        file_name = os.path.basename(_file["name"])
        file_size = _file["size"]

        with self.fs.open(_file["name"], "rb") as f:
            # Read footer size (last 8 bytes: 4 for footer size + 4 for magic number)
            f.seek(file_size - 8)
            footer_size = int.from_bytes(f.read(4), "little")

            # seek to the start of the footer and read the footer data
            footer_start = file_size - footer_size - 8
            f.seek(footer_start)
            footer_data = f.read(file_size - footer_start)

            if len(footer_data) != file_size - footer_start:
                raise ValueError(f"Failed to read complete footer data from {file_name}")

        # Parse the footer data to extract schema information
        with io.BytesIO(footer_data) as footer_buffer:
            parquet_file = pq.ParquetFile(footer_buffer)
            data_types = [str(col.type) for col in parquet_file.schema_arrow]
            num_rows = parquet_file.metadata.num_rows

        return {
            "file_name": file_name,
            "num_rows": num_rows,
            "file_size": file_size,
            "data_types": data_types,
        }

    def write_index(self, chunks_info: list[dict[str, Any]], config: dict[str, Any]) -> None:
        """Write the index file to the cache directory."""
        assert self.cache_path is not None
        assert self.dir.url is not None

        super().write_index(chunks_info, config)


def get_parquet_indexer_cls(
    dir_path: str,
    cache_path: Optional[str] = None,
    storage_options: Optional[dict] = {},
    num_workers: int = 4,
) -> ParquetDir:
    """Get the appropriate ParquetDir class based on the directory path scheme.

    Args:
        dir_path (str): Path to the directory containing the Parquet files.
        cache_path (Optional[str]): Local cache directory for storing temporary files.
        storage_options (Optional[Dict]): Additional storage options for accessing the Parquet files.
        remove_after_indexing (bool): Whether to remove files after indexing (default is True).
        num_workers (int): Number of workers to download metadata of Parquet files and index them.

    Returns:
        ParquetDir: An instance of the appropriate ParquetDir class.

    Raises:
        ValueError: If the provided `dir_path` does not have an associated ParquetDir class.
    """
    args = (dir_path, cache_path, storage_options, num_workers)

    obj = parse.urlparse(dir_path)

    if obj.scheme in ("local", ""):
        return LocalParquetDir(*args)
    if obj.scheme in ("gs", "s3"):
        return CloudParquetDir(*args)
    if obj.scheme == "hf":
        return HFParquetDir(*args)

    supported_schemes = ["local", "gs", "s3", "hf"]
    raise ValueError(
        f"The provided `dir_path` '{dir_path}' does not have an associated ParquetDir class. "
        f"Found scheme: '{obj.scheme}'. Supported schemes are: {', '.join(supported_schemes)}. "
        "Please provide a valid directory path with one of the supported schemes."
    )
