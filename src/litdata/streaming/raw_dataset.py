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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import urlparse

from torch.utils.data import Dataset

from litdata.constants import _ASYNCIO_AVAILABLE, _FSSPEC_AVAILABLE, _TQDM_AVAILABLE, _ZSTD_AVAILABLE
from litdata.streaming.downloader import Downloader, get_downloader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.utilities.dataset_utilities import generate_md5_hash, get_default_cache_dir

if not _ASYNCIO_AVAILABLE:
    raise ModuleNotFoundError(
        "The 'asyncio' package is required for streaming datasets. Please install it with `pip install asyncio`."
    )
else:
    import asyncio

logger = logging.getLogger(__name__)
SUPPORTED_PROVIDERS = ("s3", "gs", "azure")


@dataclass
class FileMetadata:
    """Metadata for a single file in the dataset."""

    path: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileMetadata":
        return cls(path=data["path"], size=data["size"])


class BaseIndexer(ABC):
    """Abstract base class for file indexing strategies."""

    @abstractmethod
    def discover_files(self, input_dir: str, storage_options: Optional[dict[str, Any]]) -> list[FileMetadata]:
        """Discover dataset files and return their metadata."""

    def build_or_load_index(
        self, input_dir: str, cache_dir: str, storage_options: Optional[dict[str, Any]]
    ) -> list[FileMetadata]:
        """Build or load a ZSTD-compressed index of file metadata."""
        if not _ZSTD_AVAILABLE:
            raise ModuleNotFoundError(str(_ZSTD_AVAILABLE))

        import zstd

        index_path = Path(cache_dir) / "index.json.zstd"

        # Try loading cached index if it exists
        if index_path.exists():
            try:
                with open(index_path, "rb") as f:
                    compressed_data = f.read()
                metadata = json.loads(zstd.decompress(compressed_data).decode("utf-8"))

                return [FileMetadata.from_dict(file_data) for file_data in metadata["files"]]
            except (FileNotFoundError, json.JSONDecodeError, zstd.ZstdError, KeyError) as e:
                logger.warning(f"Failed to load cached index from {index_path}: {e}")

        # Build fresh index
        logger.info(f"Building index for {input_dir} at {index_path}")
        files = self.discover_files(input_dir, storage_options)
        if not files:
            raise ValueError(f"No files found in {input_dir}")

        # Cache the index with ZSTD compression
        # TODO: upload the index to cloud storage
        try:
            metadata = {
                "source": input_dir,
                "files": [file.to_dict() for file in files],
                "created_at": time.time(),
            }
            with open(index_path, "wb") as f:
                f.write(zstd.compress(json.dumps(metadata).encode("utf-8")))
        except (OSError, zstd.ZstdError) as e:
            logger.warning(f"Error caching index to {index_path}: {e}")

        logger.info(f"Built index with {len(files)} files from {input_dir} at {index_path}")
        return files


class FileIndexer(BaseIndexer):
    """Indexes files recursively from cloud or local storage with optional extension filtering."""

    def __init__(
        self,
        max_depth: int = 5,
        extensions: Optional[list[str]] = None,
    ):
        self.max_depth = max_depth
        self.extensions = [ext.lower() for ext in (extensions or [])]

    def discover_files(self, input_dir: str, storage_options: Optional[dict[str, Any]]) -> list[FileMetadata]:
        """Discover dataset files and return their metadata."""
        parsed_url = urlparse(input_dir)

        if parsed_url.scheme in SUPPORTED_PROVIDERS:
            return self._discover_cloud_files(input_dir, storage_options)

        if not parsed_url.scheme or parsed_url.scheme == "file":
            return self._discover_local_files(input_dir)

        raise ValueError(
            f"Unsupported input directory scheme: {parsed_url.scheme}. Supported schemes are: {SUPPORTED_PROVIDERS}"
        )

    def _discover_cloud_files(self, input_dir: str, storage_options: Optional[dict[str, Any]]) -> list[FileMetadata]:
        """Recursively list files in a cloud storage bucket."""
        if not _FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(str(_FSSPEC_AVAILABLE))
        import fsspec

        obj = urlparse(input_dir)

        # TODO: Research on switching to 'obstore' for file listing to potentially improve performance.
        # Currently using 'fsspec' due to some issues with 'obstore' when handling multiple instances.
        fs = fsspec.filesystem(obj.scheme, **(storage_options or {}))
        files = fs.find(input_dir, maxdepth=self.max_depth, detail=True, withdirs=False)

        if _TQDM_AVAILABLE:
            from tqdm.auto import tqdm

            pbar = tqdm(desc="Discovering files", total=len(files))

        metadatas = []
        for _, file_info in files.items():
            if file_info.get("type") != "file":
                continue

            file_path = file_info["name"]
            if self._should_include_file(file_path):
                metadata = FileMetadata(
                    path=f"{obj.scheme}://{file_path}",
                    size=file_info.get("size", 0),
                )
                metadatas.append(metadata)
            if _TQDM_AVAILABLE:
                pbar.update(1)
        if _TQDM_AVAILABLE:
            pbar.close()
        return metadatas

    def _discover_local_files(self, input_dir: str) -> list[FileMetadata]:
        """Recursively list files in the local filesystem."""
        path = Path(input_dir)
        metadatas = []

        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue

            if self._should_include_file(str(file_path)):
                metadata = FileMetadata(
                    path=str(file_path),
                    size=file_path.stat().st_size,
                )
                metadatas.append(metadata)

        return metadatas

    def _should_include_file(self, file_path: str) -> bool:
        """Return True if file matches allowed extensions."""
        file_ext = Path(file_path).suffix.lower()
        return not self.extensions or file_ext in self.extensions


class CacheManager:
    """Manages file caching for remote datasets, preserving directory structure."""

    def __init__(
        self,
        input_dir: Union[str, Dir],
        cache_dir: Optional[str] = None,
        storage_options: Optional[dict] = None,
        cache_files: bool = False,
    ):
        self.input_dir = _resolve_dir(input_dir)
        self._input_dir_path = str(self.input_dir.path or self.input_dir.url)
        self.cache_files = cache_files

        self.cache_dir = self._create_cache_dir(self._input_dir_path, cache_dir)

        self.storage_options = storage_options or {}
        self._downloader: Optional[Downloader] = None

    @property
    def downloader(self) -> Downloader:
        """Lazily initialize the downloader."""
        if self._downloader is None:
            self._downloader = get_downloader(
                remote_dir=self._input_dir_path,
                cache_dir=self.cache_dir,
                chunks=[],
                storage_options=self.storage_options,
            )
        return self._downloader

    def _create_cache_dir(self, input_dir: str, cache_dir: Optional[str] = None) -> str:
        """Create cache directory if it doesn't exist."""
        if cache_dir is None:
            cache_dir = get_default_cache_dir()
        cache_path = os.path.join(cache_dir, generate_md5_hash(input_dir))
        os.makedirs(cache_path, exist_ok=True)
        return cache_path

    def get_local_path(self, remote_file_path: str) -> str:
        """Convert remote file path to its local cache location."""
        remote_base_path = self._input_dir_path.rstrip("/") + "/"
        if not remote_file_path.startswith(remote_base_path):
            raise ValueError(f"File path {remote_file_path} does not start with input dir {remote_base_path}")

        relative_path = remote_file_path[len(remote_base_path) :]
        local_path = Path(self.cache_dir) / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return str(local_path)

    async def download_file_async(self, file_path: str) -> bytes:
        """Asynchronously download and return file content."""
        if self.cache_files:
            local_path = self.get_local_path(file_path)
            if os.path.exists(local_path):
                return await asyncio.to_thread(Path(local_path).read_bytes)

        try:
            return await self.downloader.adownload_fileobj(file_path)
        except Exception as e:
            raise RuntimeError(f"Error downloading file {file_path}: {e}") from e


class StreamingRawDataset(Dataset):
    """Base class for streaming raw datasets.

    This class provides the core functionality for streaming raw data from a remote or local source,
    including file discovery, caching, and asynchronous downloading.

    To create a custom dataset, subclass this class and override the `setup` method
    to define the structure of your dataset items from the list of all discovered files.
    """

    def __init__(
        self,
        input_dir: str,
        cache_dir: Optional[str] = None,
        indexer: Optional[BaseIndexer] = None,
        storage_options: Optional[dict] = None,
        cache_files: bool = False,
        transform: Optional[Callable[[Union[bytes, list[bytes]]], Any]] = None,
    ):
        """Initialize StreamingRawDataset.

        Args:
            input_dir: Path to dataset root (e.g., 's3://bucket/dataset/').
            cache_dir: Directory for caching files (optional).
            indexer: Custom file indexer (default: FileIndexer).
            storage_options: Cloud storage options.
            cache_files: Whether to cache files locally (default: False).
            transform: A function to apply to each item. It will receive `bytes` for single-file
                       items or `List[bytes]` for grouped items.
        """
        self.input_dir = _resolve_dir(input_dir)
        self.cache_manager = CacheManager(self.input_dir, cache_dir, storage_options, cache_files)
        self.indexer = indexer or FileIndexer()
        self.storage_options = storage_options or {}
        self.transform = transform

        # Discover all files in the input directory.
        self.files: list[FileMetadata] = self.indexer.build_or_load_index(
            str(self.input_dir.path or self.input_dir.url), self.cache_manager.cache_dir, storage_options
        )
        logger.info(f"Discovered {len(self.files)} files.")

        # Transform the flat list of files into the desired item structure.
        self.items: Union[list[FileMetadata], list[list[FileMetadata]]] = self.setup(self.files)
        if not isinstance(self.items, list):
            raise TypeError(f"The setup method must return a list, but returned {type(self.items)}")
        logger.info(f"Dataset setup with {len(self.items)} items.")

    def setup(self, files: list[FileMetadata]) -> Union[list[FileMetadata], list[list[FileMetadata]]]:
        """Define the structure of the dataset from the list of discovered files.

        Override this method in a subclass to group or filter files into final dataset items.

        Args:
            files: A list of all `FileMetadata` objects discovered in the `input_dir`.

        Returns:
            The final structure of the dataset, which can be:
            - `List[FileMetadata]`: Each `FileMetadata` object is treated as a single item.
            - `List[List[FileMetadata]]`: Each inner list of `FileMetadata` objects is treated as a single item.
        """
        return files

    @lru_cache(maxsize=1)
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.items)

    def __getitem__(self, index: int) -> Any:
        """Get a single item by index."""
        if not (0 <= index < len(self)):
            raise IndexError(f"Index {index} out of range for dataset with length {len(self)}")

        item = self.items[index]
        if isinstance(item, FileMetadata):
            return asyncio.run(self._download_and_process_item(item.path))
        if isinstance(item, list):
            file_paths = [fm.path for fm in item]
            return asyncio.run(self._download_and_process_group(file_paths))
        raise TypeError(f"Dataset items must be of type FileMetadata or List[FileMetadata], but found {type(item)}")

    def __getitems__(self, indices: list[int]) -> list[Any]:
        """Asynchronously download a batch of items by indices."""
        # asyncio.run() handles loop creation, execution, and teardown cleanly.
        return asyncio.run(self._download_batch(indices))

    async def _download_batch(self, indices: list[int]) -> list[Any]:
        """Asynchronously download and process items."""
        batch_items = [self.items[i] for i in indices]
        coros = []
        for item in batch_items:
            if isinstance(item, FileMetadata):
                coros.append(self._download_and_process_item(item.path))
            elif isinstance(item, list):
                file_paths = [fm.path for fm in item]
                coros.append(self._download_and_process_group(file_paths))
            else:
                raise TypeError(
                    f"Dataset items must be of type FileMetadata or List[FileMetadata], but found {type(item)}"
                )
        return await asyncio.gather(*coros)

    async def _download_and_process_group(self, file_paths: list[str]) -> Any:
        """Download all files in a group, then apply the transform."""
        download_coros = [self.cache_manager.download_file_async(path) for path in file_paths]
        group_data: list[bytes] = await asyncio.gather(*download_coros)

        if self.transform:
            # The transform receives a list of bytes, corresponding to the list structure
            # of the item defined in setup(). This is true even if the list has only one element.
            return await asyncio.to_thread(self.transform, group_data)
        return group_data

    async def _download_and_process_item(self, file_path: str) -> Any:
        """Download a single file and apply the transform."""
        data: bytes = await self.cache_manager.download_file_async(file_path)
        if self.transform:
            # The transform receives a single bytes object, corresponding to the
            # single FileMetadata object structure of the item.
            return await asyncio.to_thread(self.transform, data)
        return data
