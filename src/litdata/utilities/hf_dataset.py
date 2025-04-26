"""Contains utility functions for indexing and streaming HF datasets."""

import os
import tempfile
import shutil
from litdata.constants import _INDEX_FILENAME
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.dataset_utilities import _try_create_cache_dir, generate_md5_hash


def index_hf_dataset(hf_url: str, cache_dir: str = None) -> str:
    """Index a Hugging Face dataset and return the cache directory path.

    Args:
        hf_url (str): The URL of the Hugging Face dataset (must start with 'hf://').
        cache_dir (str, optional): The directory where the cache and index will be stored. Defaults to None.

    Returns:
        str: The path to the cache directory containing the index.

    Raises:
        ValueError: If the provided URL does not start with 'hf://'.
    """
    if not hf_url.startswith("hf://"):
        raise ValueError(
            f"Invalid Hugging Face dataset URL: {hf_url}. "
            "The URL should start with 'hf://'. Please check the URL and try again."
        )
    # If cache_dir is provided, check if the index already exists and return early if so
    if cache_dir:
        url_hash = generate_md5_hash(hf_url)
        cache_dir = os.path.join(cache_dir, url_hash)
        if os.path.exists(cache_dir):
            dirs = os.listdir(cache_dir)
            final_cache_dir = os.path.join(cache_dir, dirs[0] if dirs else "")
            if os.path.exists(os.path.join(final_cache_dir, _INDEX_FILENAME)):
                print(f"Index already exists at {final_cache_dir}.")
                return final_cache_dir
            else:
                print("Index not found in cache")
        else:
            print("cache not exist, creating a new one.", cache_dir)

    # Otherwise, create a new index file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_index_path = os.path.join(temp_dir, _INDEX_FILENAME)
        index_parquet_dataset(hf_url, temp_dir, num_workers=os.cpu_count() or 4)

        # Prepare the cache directory and move the index file there
        final_cache_dir = _try_create_cache_dir(hf_url, cache_dir, index_path=temp_index_path)
        cache_index_path = os.path.join(final_cache_dir, _INDEX_FILENAME)
        print(f"Indexing HF dataset from {hf_url} into {cache_index_path}.")
        shutil.copyfile(temp_index_path, cache_index_path)
        print(f"Index created at {cache_index_path}.")
        return final_cache_dir
