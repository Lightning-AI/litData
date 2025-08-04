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

import asyncio
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Union

from torch.utils.data import Dataset

from litdata.raw.indexer import BaseIndexer, FileIndexer, FileMetadata
from litdata.streaming.downloader import Downloader, get_downloader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.utilities.dataset_utilities import generate_md5_hash, get_default_cache_dir

logger = logging.getLogger(__name__)


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
        recompute_index: bool = False,
        transform: Optional[Callable[[Union[bytes, list[bytes]]], Any]] = None,
    ):
        """Initialize StreamingRawDataset.

        Args:
            input_dir: Path to dataset root (e.g., 's3://bucket/dataset/').
            cache_dir: Directory for caching files (optional).
            indexer: Custom file indexer (default: FileIndexer).
            storage_options: Cloud storage options.
            cache_files: Whether to cache files locally (default: False).
            recompute_index: Whether to recompute the index (default: False).
                If True, forces a re-scan of the input directory and rebuilds the index,
                ignoring any cached index files. This is useful when the dataset
                structure or files on the remote storage have changed.
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
            str(self.input_dir.path or self.input_dir.url),
            self.cache_manager.cache_dir,
            storage_options,
            recompute_index,
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
