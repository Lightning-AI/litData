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
from typing import List, Optional, Tuple, Union
from urllib import parse

import fsspec
from torch.utils.data import IterableDataset
from tqdm import tqdm

from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.utilities.dataset_utilities import generate_md5_hash, get_default_cache_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


INDEX_FILE_NAME = "index.txt"
CLASSES_FILE_NAME = "classes.txt"


def _try_create_raw_cache_dir(input_dir: Optional[str], cache_dir: Optional[str] = None) -> Optional[str]:
    """Try to create the raw cache directory for the dataset.

    Args:
        input_dir (Optional[str]): The input directory for the dataset.
        cache_dir (Optional[str]): The cache directory to create.

    Returns:
        Optional[str]: The path to the created cache directory, or None if creation failed.
    """
    dir_url_hash = generate_md5_hash(input_dir or "")
    cache_dir = cache_dir if cache_dir is not None else get_default_cache_dir()

    # Create the cache directory path based on the input_dir hash
    cache_dir = os.path.join(cache_dir, dir_url_hash, dir_url_hash)

    # Create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


class StreamingRawDataset(IterableDataset):
    """Stream raw files from cloud storage with fast indexing and caching.

    Supports ImageFolder-style datasets with this structure:

    s3://bucket/dataset/
    ├── class_1/
    │   ├── file_001.jpg
    │   ├── file_002.jpg
    │   └── ...
    ├── class_2/
    │   ├── file_001.jpg
    │   └── ...
    └── ...

    Features:
    - Fast multithreaded indexing
    - Automatic index caching
    - Works with S3 and GCS
    - PyTorch DataLoader compatible
    """

    def __init__(
        self,
        input_dir: Union[str, "Dir"],
        cache_dir: Optional[Union[str, "Dir"]] = None,
        index_workers: int = 8,
        **kwargs,
    ):
        """Initialize StreamingRawDataset.

        Args:
            input_dir: Path to dataset root (local or cloud, e.g. s3://bucket/dataset/)
            cache_dir: Directory for caching files (optional)
            index_workers: Number of threads for indexing (default: 8)
            **kwargs: Additional arguments
        """
        # Resolve input directory
        input_dir = _resolve_dir(input_dir)
        cache_dir = _resolve_dir(cache_dir)

        # create cache directory if it doesn't exist
        cache_path = _try_create_raw_cache_dir(
            input_dir=input_dir.path if input_dir.path else input_dir.url,
            cache_dir=cache_dir.path if cache_dir else None,
        )
        if cache_path is not None:
            input_dir.path = cache_path

        self.input_dir = input_dir
        self.cache_dir = cache_dir
        self.index_workers = index_workers
        self.classes: List[str] = None
        self.files: List[str] = []
        self.fs: Optional[fsspec.AbstractFileSystem] = None

        # Index dataset files
        self._build_index()
        if not self.files:
            raise ValueError(f"No files found in {input_dir}")

        logger.info(f"Initialized StreamingRawDataset with {len(self.files)} samples")

    def _build_index(self) -> List[Tuple[str, str]]:
        """Build index of all files in the input_dir."""
        logger.info(f"Indexing files in {self.input_dir}...")
        self.index_cache_path = os.path.join(self.input_dir.path, INDEX_FILE_NAME)
        self.classes_cache_path = os.path.join(self.input_dir.path, CLASSES_FILE_NAME)

        if os.path.exists(self.index_cache_path) and os.path.exists(self.classes_cache_path):
            with open(self.index_cache_path) as f:
                files_list = f.readlines()
            self.files = [line.strip() for line in files_list if line.strip()]

            with open(self.classes_cache_path) as f:
                classes_list = f.readlines()
            self.classes = [line.strip() for line in classes_list if line.strip()]

            logger.info(f"Loaded index from {self.index_cache_path} with {len(self.files)} samples")
        else:
            _CLOUD_PROVIDER = ("s3", "gs")

            obj = parse.urlparse(self.input_dir.url)
            provider = obj.scheme
            if provider not in _CLOUD_PROVIDER:
                raise ValueError(
                    f"Unsupported cloud provider: {provider}. Supported providers are: {', '.join(_CLOUD_PROVIDER)}"
                )

            self.fs = fsspec.filesystem(provider)
            logger.info(f"Using provider: {provider}")

            # get classes
            folders = self.fs.ls(self.input_dir.url)
            self.classes = [os.path.basename(folder) for folder in folders if self.fs.isdir(folder)]
            logger.info(f"Found {len(self.classes)} classes")

            def list_files_for_class(class_name: str) -> List[Tuple[str, str]]:
                """List all files in a class directory."""
                class_dir = os.path.join(self.input_dir.url, class_name)
                files = self.fs.ls(class_dir, detail=True)
                return [os.path.join(class_dir, f["name"]) for f in files if f["type"] == "file"]

            # Index files in parallel
            results = []
            with ThreadPoolExecutor(max_workers=self.index_workers) as executor:
                future_to_class = {executor.submit(list_files_for_class, c): c for c in self.classes}
                for future in tqdm(as_completed(future_to_class), total=len(self.classes), desc="Indexing classes"):
                    try:
                        results.extend(future.result())
                    except Exception as e:
                        logger.warning(f"Error indexing class directory:{e}")

            self.files = results
            logger.info(f"Indexed {len(self.files)} files")
            # Save index to cache
            with open(self.index_cache_path, "w") as f:
                for file_path in self.files:
                    f.write(f"{file_path}\n")

            with open(self.classes_cache_path, "w") as f:
                for class_name in self.classes:
                    f.write(f"{class_name}\n")

            logger.info(f"Saved index to {self.index_cache_path} and classes to {self.classes_cache_path}")

    def __iter__(self):
        """Iterate over the dataset."""
        yield from self.files

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.files)


if __name__ == "__main__":
    # Example usage
    import time

    start_time = time.perf_counter()
    dataset = StreamingRawDataset(
        # input_dir="s3://grid-cloud-litng-ai-03/projects/01jpacd4y2yza88t23wf049m0t/datasets/caltech101/101_ObjectCategories",
        input_dir="s3://imagenet-1m-template/raw/train",
        # cache_dir="s3://my-bucket/my-cache",
        index_workers=16,
    )

    end_time = time.perf_counter()
    print(f"Dataset initialized in {end_time - start_time:.2f} seconds")
