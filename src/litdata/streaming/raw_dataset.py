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

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib import parse

import fsspec
from torch.utils.data import IterableDataset
from tqdm import tqdm

from litdata.streaming.downloader import get_downloader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.utilities.dataset_utilities import generate_md5_hash, get_default_cache_dir

logger = logging.getLogger(__name__)

INDEX_FILE_NAME = "index.txt"
CLASSES_FILE_NAME = "classes.txt"


class CacheManager:
    """Manages local file caching with directory structure preservation."""

    def __init__(self, cache_dir: str, remote_dir: str, storage_options: Optional[Dict] = None):
        self.cache_dir = cache_dir
        self.remote_dir = remote_dir
        self.storage_options = storage_options or {}
        self.downloader = None

        os.makedirs(cache_dir, exist_ok=True)

    def get_local_path(self, file_path: str, class_name: str) -> str:
        """Get local cache path maintaining class directory structure."""
        filename = os.path.basename(file_path)
        class_cache_dir = os.path.join(self.cache_dir, class_name)
        os.makedirs(class_cache_dir, exist_ok=True)
        return os.path.join(class_cache_dir, filename)

    def download_file(self, file_path: str, class_name: str) -> str:
        """Download file to cache and return local path."""
        local_path = self.get_local_path(file_path, class_name)

        # Return if already cached
        if os.path.exists(local_path):
            return local_path

        # Initialize downloader if needed
        if self.downloader is None:
            chunks = [{"filename": os.path.basename(file_path)}]
            self.downloader = get_downloader(
                remote_dir=os.path.dirname(file_path),
                cache_dir=os.path.dirname(local_path),
                chunks=chunks,
                storage_options=self.storage_options,
            )

        try:
            self.downloader.download_file(file_path, local_path)
            return local_path
        except Exception as e:
            logger.warning(f"Failed to download {file_path}: {e}")
            return file_path  # Return remote path as fallback


class StreamingRawDataset(IterableDataset):
    """Stream raw files from cloud storage with fast indexing and caching.

    Supports ImageFolder-style datasets:

    s3://bucket/dataset/
    ├── class_1/
    │   ├── file_001.jpg
    │   └── ...
    ├── class_2/
    │   ├── file_001.jpg
    │   └── ...
    └── ...

    Features:
    - Fast multithreaded indexing
    - Automatic index caching
    - Preloading for performance
    - PyTorch DataLoader compatible
    """

    def __init__(
        self,
        input_dir: Union[str, "Dir"],
        cache_dir: Optional[Union[str, "Dir"]] = None,
        index_workers: int = 8,
        max_preload_size: int = 10,
        storage_options: Optional[Dict] = None,
        **kwargs,
    ):
        """Initialize StreamingRawDataset.

        Args:
            input_dir: Path to dataset root (e.g. s3://bucket/dataset/)
            cache_dir: Directory for caching files (optional)
            index_workers: Number of threads for indexing (default: 8)
            max_preload_size: Maximum number of files to preload (default: 10)
            storage_options: Cloud storage options
            **kwargs: Additional arguments
        """
        # Resolve directories
        self.input_dir = _resolve_dir(input_dir)
        cache_dir = _resolve_dir(cache_dir) if cache_dir else None

        # Setup cache
        if cache_dir:
            cache_path = cache_dir.path or cache_dir.url
        else:
            dir_hash = generate_md5_hash(self.input_dir.url)
            cache_path = os.path.join(get_default_cache_dir(), dir_hash)

        self.cache_manager = CacheManager(cache_path, self.input_dir.url, storage_options)

        # Configuration
        self.index_workers = index_workers
        self.max_preload_size = max_preload_size
        self.storage_options = storage_options or {}

        # State
        self.classes: List[str] = []
        self.files: List[Tuple[str, str]] = []  # (file_path, class_name)
        self.fs: Optional[fsspec.AbstractFileSystem] = None

        # Preloading
        self._preload_executor = None
        self._preload_futures = {}

        # Build index
        self._build_index()
        if not self.files:
            raise ValueError(f"No files found in {self.input_dir.url}")

        logger.info(f"Initialized StreamingRawDataset with {len(self.files)} samples")

    def _build_index(self) -> None:
        """Build or load file index."""
        index_path = os.path.join(self.cache_manager.cache_dir, INDEX_FILE_NAME)
        classes_path = os.path.join(self.cache_manager.cache_dir, CLASSES_FILE_NAME)

        # Try loading cached index
        if os.path.exists(index_path) and os.path.exists(classes_path):
            self._load_cached_index(index_path, classes_path)
        else:
            self._build_fresh_index()
            self._save_index(index_path, classes_path)

    def _load_cached_index(self, index_path: str, classes_path: str) -> None:
        """Load index from cache."""
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if line and "," in line:
                    file_path, class_name = line.rsplit(",", 1)
                    self.files.append((file_path, class_name))

        with open(classes_path) as f:
            self.classes = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded cached index with {len(self.files)} samples")

    def _build_fresh_index(self) -> None:
        """Build fresh index by scanning cloud storage."""
        obj = parse.urlparse(self.input_dir.url)
        if obj.scheme not in ("s3", "gs"):
            raise ValueError(f"Unsupported provider: {obj.scheme}")

        self.fs = fsspec.filesystem(obj.scheme, **self.storage_options)

        # Get class directories
        folders = self.fs.ls(self.input_dir.url)
        self.classes = [os.path.basename(f) for f in folders if self.fs.isdir(f)]

        def list_class_files(class_name: str) -> List[Tuple[str, str]]:
            class_dir = f"{self.input_dir.url.rstrip('/')}/{class_name}"
            try:
                files = self.fs.ls(class_dir, detail=True)
                return [
                    (os.path.join(class_dir, os.path.basename(f["name"])), class_name)
                    for f in files
                    if f["type"] == "file"
                ]
            except Exception as e:
                logger.warning(f"Error listing {class_dir}: {e}")
                return []

        # Index in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.index_workers) as executor:
            futures = {executor.submit(list_class_files, cls): cls for cls in self.classes}
            for future in tqdm(as_completed(futures), total=len(self.classes), desc="Indexing"):
                results.extend(future.result())

        self.files = results
        logger.info(f"Indexed {len(self.files)} files")

    def _save_index(self, index_path: str, classes_path: str) -> None:
        """Save index to cache."""
        with open(index_path, "w") as f:
            for file_path, class_name in self.files:
                f.write(f"{file_path},{class_name}\n")

        with open(classes_path, "w") as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")

    def _init_preloading(self) -> None:
        """Initialize preloading executor."""
        if self._preload_executor is None and self.max_preload_size > 0:
            self._preload_executor = ThreadPoolExecutor(max_workers=4)
            # Start preloading first files
            for i in range(min(self.max_preload_size, len(self.files))):
                self._submit_preload(i)

    def _submit_preload(self, index: int) -> None:
        """Submit a file for preloading."""
        if index < len(self.files) and index not in self._preload_futures:
            file_path, class_name = self.files[index]
            future = self._preload_executor.submit(self.cache_manager.download_file, file_path, class_name)
            self._preload_futures[index] = future

    def _get_file(self, index: int) -> str:
        """Get local file path, downloading if necessary."""
        file_path, class_name = self.files[index]

        # Check if preloaded
        if index in self._preload_futures:
            future = self._preload_futures.pop(index)
            try:
                local_path = future.result(timeout=30)
                # Queue next file for preloading
                next_idx = index + self.max_preload_size
                if next_idx < len(self.files):
                    self._submit_preload(next_idx)
                return local_path
            except Exception as e:
                logger.warning(f"Preload failed for {index}: {e}")

        # Download directly
        return self.cache_manager.download_file(file_path, class_name)

    def load_sample(self, local_path: str, file_path: str, class_name: str, index: int) -> Any:
        """Load sample data. Override this method to customize loading.

        Args:
            local_path: Path to local cached file
            file_path: Original remote file path
            class_name: Class name
            index: Sample index

        Returns:
            Loaded sample data
        """
        # Default: return file content as bytes
        try:
            from PIL import Image

            image = Image.open(local_path)
            return {"index": index, "path": local_path, "image": image, "label": class_name}
        except Exception:
            # Fallback for remote files
            return {"path": file_path, "class_name": class_name, "index": index}

    def __iter__(self):
        """Iterate over dataset."""
        # Initialize preloading on first iteration
        if self._preload_executor is None:
            self._init_preloading()

        for i in range(len(self.files)):
            file_path, class_name = self.files[i]
            local_path = self._get_file(i)
            yield self.load_sample(local_path, file_path, class_name, i)

    def __getitem__(self, index: int) -> Any:
        """Get item by index."""
        if index >= len(self.files):
            raise IndexError(f"Index {index} out of range")

        if self._preload_executor is None:
            self._init_preloading()

        file_path, class_name = self.files[index]
        local_path = self._get_file(index)
        return self.load_sample(local_path, file_path, class_name, index)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.files)

    def get_class_counts(self) -> Dict[str, int]:
        """Get samples per class."""
        counts = {}
        for _, class_name in self.files:
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts


if __name__ == "__main__":
    # Example usage
    import time

    start = time.perf_counter()
    dataset = StreamingRawDataset(
        input_dir="s3://imagenet-1m-template/raw/train", index_workers=16, max_preload_size=20, cache_dir="raw_cache"
    )
    end = time.perf_counter()
    print(f"Dataset loaded in {end - start:.2f} seconds")
    print("sample files :", dataset.files[:3])

    print(f"Dataset: {len(dataset)} samples, {len(dataset.classes)} classes")

    # Test iteration
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        print(f"Sample {i}: {sample} ({len(sample) if isinstance(sample, bytes) else 'metadata'})")

    print("✅ Test completed")
