"""Contains utility functions for indexing and streaming HF datasets."""

import os
import shutil
import tempfile
from contextlib import suppress
from typing import Optional

from filelock import FileLock, Timeout

from litdata.constants import _INDEX_FILENAME
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.dataset_utilities import _try_create_cache_dir, generate_md5_hash, get_default_cache_dir


def index_hf_dataset(dataset_url: str, cache_dir: Optional[str] = None) -> str:
    """Indexes a Hugging Face dataset and returns the path to the cache directory.

    Args:
        dataset_url (str): The URL of the Hugging Face dataset, starting with 'hf://'.
        cache_dir (Optional[str]): The directory for storing the cache and index. If None, a default location is used.

    Returns:
        str: The path to the cache directory containing the index file.

    Raises:
        ValueError: If the dataset URL does not start with 'hf://'.
    """
    if not dataset_url.startswith("hf://"):
        raise ValueError(
            f"Invalid Hugging Face dataset URL: {dataset_url}. "
            "URLs must start with 'hf://'. Please check the URL and try again."
        )

    # Acquire a file lock to guarantee exclusive access,
    # ensuring that multiple processes do not create the index simultaneously.
    with suppress(Timeout), FileLock(os.path.join(tempfile.gettempdir(), "hf_index.lock"), timeout=20):
        # Check for existing index in the cache
        cache_directory = _get_existing_cache(dataset_url, cache_dir)
        if cache_directory:
            print(f"Using existing index at {cache_directory}.")
            return cache_directory

        # Otherwise, create a new index file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_index_path = os.path.join(temp_dir, _INDEX_FILENAME)
            index_parquet_dataset(dataset_url, temp_dir, num_workers=os.cpu_count() or 4)

            # Prepare the cache directory and move the index file there
            cache_dir = _try_create_cache_dir(dataset_url, cache_dir, index_path=temp_index_path)
            assert cache_dir is not None
            cache_index_path = os.path.join(cache_dir, _INDEX_FILENAME)
            shutil.copyfile(temp_index_path, cache_index_path)
            print(f"Index created at {cache_index_path}.")

    return cache_dir  # type: ignore


def _get_existing_cache(dataset_url: str, cache_dir: Optional[str]) -> Optional[str]:
    """Checks if a cache directory with an index file exists for the given dataset URL.

    Args:
        dataset_url (str): The URL of the Hugging Face dataset.
        cache_dir (Optional[str]): The root directory for the cache.

    Returns:
        Optional[str]: The path to the existing cache directory if found, otherwise None.
    """
    # Determine the cache directory, preferring user-provided cache_dir if given
    cache_dir = cache_dir if cache_dir is not None else get_default_cache_dir()

    url_hash = generate_md5_hash(dataset_url)
    hashed_cache_path = os.path.join(cache_dir, url_hash)

    if not os.path.exists(hashed_cache_path):
        return None

    for subdir in os.listdir(hashed_cache_path):
        potential_cache_dir = os.path.join(hashed_cache_path, subdir)
        if os.path.exists(os.path.join(potential_cache_dir, _INDEX_FILENAME)):
            return potential_cache_dir

    return None
