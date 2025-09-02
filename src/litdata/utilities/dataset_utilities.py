import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Optional

import numpy as np

from litdata.constants import _DEFAULT_CACHE_DIR, _DEFAULT_LIGHTNING_CACHE_DIR, _INDEX_FILENAME, _LITDATA_CACHE_DIR
from litdata.streaming.downloader import get_downloader
from litdata.streaming.item_loader import BaseItemLoader, TokensLoader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.utilities.subsample import shuffle_lists_together, subsample_filenames_and_roi


def wait_for_predicate(
    predicate: Callable[[], bool],
    timeout: float,
) -> bool:
    """Wait until the given predicate becomes True or the timeout expires.

    Args:
        predicate: A function returning a boolean condition to check.
        timeout: Maximum time (in seconds) to wait.

    Returns:
        True if predicate became True within timeout, else False.
    """
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def subsample_streaming_dataset(
    input_dir: Dir,
    cache_dir: Optional[Dir] = None,
    item_loader: Optional[BaseItemLoader] = None,
    subsample: float = 1.0,
    shuffle: bool = False,
    seed: int = 42,
    storage_options: Optional[dict] = {},
    session_options: Optional[dict] = {},
    index_path: Optional[str] = None,
    fnmatch_pattern: Optional[str] = None,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Subsample streaming dataset.

    But before doing that, we will do some preprocessing:
    - Make sure input_dir contains cache path and remote url.
    - Check if `index.json` file exists in cache path.
    - If not, download from remote url. If remote url doesn't contain `index.json` file, raise error.
    - Once downloaded, load chunks from `index.json` file.
    - Once chunks are ready, subsample (chunk filenames, region_of_interest).

    """
    subsampled_files: list[str] = []
    roi: list[tuple[int, int]] = []

    # Make sure input_dir contains cache path and remote url
    if _should_replace_path(input_dir.path):
        cache_path = _try_create_cache_dir(
            input_dir=input_dir.path if input_dir.path else input_dir.url,
            cache_dir=cache_dir.path if cache_dir else None,
            storage_options=storage_options,
            session_options=session_options,
            index_path=index_path,
        )
        if cache_path is not None:
            input_dir.path = cache_path

    assert input_dir.path is not None

    cache_index_filepath = os.path.join(input_dir.path, _INDEX_FILENAME)

    # Check if `index.json` file exists in cache path
    if not os.path.exists(cache_index_filepath) and isinstance(input_dir.url, str):
        assert input_dir.url is not None
        if index_path is not None:
            copy_index_to_cache_index_filepath(index_path, cache_index_filepath)
        else:
            downloader = get_downloader(input_dir.url, input_dir.path, [], storage_options, session_options)
            downloader.download_file(os.path.join(input_dir.url, _INDEX_FILENAME), cache_index_filepath)

    def path_exists(p: str) -> bool:
        return wait_for_predicate(lambda: os.path.exists(p), timeout=0.5)

    if not path_exists(input_dir.path):
        raise FileNotFoundError(f"The provided dataset path `{input_dir.path}` does not exist.")

    if os.path.exists(os.path.join(input_dir.path, _INDEX_FILENAME)):
        # load chunks from `index.json` file
        data = load_index_file(input_dir.path)
        original_chunks = data["chunks"]
    else:
        raise ValueError(
            f"The provided dataset `{input_dir.path}` doesn't contain any {_INDEX_FILENAME} file."
            "\n HINT: Did you successfully optimize a dataset to the provided `input_dir`?"
        )

    if fnmatch_pattern is not None:
        from fnmatch import fnmatch

        original_chunks = [chunk for chunk in original_chunks if fnmatch(chunk["filename"], fnmatch_pattern)]

    assert len(original_chunks) > 0, f"No chunks found in the `{input_dir}/index.json` file"

    # create a (chunk_start, chunk_end) list to indicate our subsample from where we can read.
    roi = generate_roi(original_chunks, item_loader)

    if math.isclose(subsample, 1.0):
        subsampled_files = [chnk["filename"] for chnk in original_chunks]

        return subsampled_files, roi

    final_files: list[str] = []
    final_roi: list[tuple[int, int]] = []

    random_seed_sampler = None
    if shuffle:
        random_seed_sampler = np.random.RandomState([seed])

    while subsample >= 1.0:
        # accumulate shuffled copies of the base into the final
        if random_seed_sampler is not None:
            original_chunks, roi = shuffle_lists_together(original_chunks, roi, random_seed_sampler)
        subsampled_files = [chnk["filename"] for chnk in original_chunks]
        final_files.extend(subsampled_files)
        final_roi.extend(roi)
        subsample -= 1.0

    if subsample > 0:
        # shuffle lists together
        if random_seed_sampler is not None:
            original_chunks, roi = shuffle_lists_together(original_chunks, roi, random_seed_sampler)

        num_items_to_subsample = int(sum([roi[1] - roi[0] for roi in roi]) * subsample)

        subsampled_files, roi, _, _ = subsample_filenames_and_roi(original_chunks, roi, num_items_to_subsample)
        final_files.extend(subsampled_files)
        final_roi.extend(roi)

    return final_files, final_roi


def _should_replace_path(path: Optional[str]) -> bool:
    """Whether the input path is a special path to be replaced."""
    if path is None or path == "":
        return True

    return (
        path.startswith("/teamspace/datasets/")
        or path.startswith("/teamspace/s3_connections/")
        or path.startswith("/teamspace/s3_folders/")
        or path.startswith("/teamspace/gcs_folders/")
        or path.startswith("/teamspace/gcs_connections/")
    )


def _read_updated_at(
    input_dir: Optional[Dir],
    storage_options: Optional[dict] = {},
    session_options: Optional[dict] = {},
    index_path: Optional[str] = None,
) -> str:
    """Read last updated timestamp from index.json file."""
    last_updation_timestamp = "0"
    index_json_content = None
    assert isinstance(input_dir, Dir)

    # Try to read index.json locally
    if input_dir.path is not None and os.path.exists(os.path.join(input_dir.path, _INDEX_FILENAME)):
        index_json_content = load_index_file(input_dir.path)
    # Try to read index.json remotely
    elif input_dir.url is not None:
        assert input_dir.url is not None
        # download index.json file and read last_updation_timestamp
        with tempfile.TemporaryDirectory() as tmp_directory:
            temp_index_filepath = os.path.join(tmp_directory, _INDEX_FILENAME)
            if index_path is not None:
                copy_index_to_cache_index_filepath(index_path, temp_index_filepath)
            else:
                downloader = get_downloader(input_dir.url, tmp_directory, [], storage_options, session_options)
                downloader.download_file(os.path.join(input_dir.url, _INDEX_FILENAME), temp_index_filepath)
            index_json_content = load_index_file(tmp_directory)

    if index_json_content is not None and "updated_at" in index_json_content:
        last_updation_timestamp = index_json_content["updated_at"]

    return last_updation_timestamp


def _clear_cache_dir_if_updated(input_dir_hash_filepath: str, updated_at_hash: str) -> None:
    """Clear the cache directory if it is outdated.

    If the directory at `input_dir_hash_filepath` exists and does not contain only a single subdirectory named
    `updated_at_hash`, the entire directory is deleted to prevent using stale or partial cache data.

    Args:
        input_dir_hash_filepath (str): Path to the hashed cache directory (e.g., /cache/chunks/{HASH(input_dir.url)}).
        updated_at_hash (str): The expected hash or timestamp for the current dataset state.
    """
    if os.path.exists(input_dir_hash_filepath):
        # check if it only contains one directory with updated_at_hash
        dir_list = os.listdir(input_dir_hash_filepath)
        if not (len(dir_list) == 1 and dir_list[0] == updated_at_hash):
            shutil.rmtree(input_dir_hash_filepath)


def generate_md5_hash(value: str) -> str:
    """Generate an MD5 hash for the given string value.

    Args:
        value (str): The input string to hash.

    Returns:
        str: The hexadecimal MD5 hash of the input string.
    """
    return hashlib.md5(value.encode()).hexdigest()  # noqa: S324


def get_default_cache_dir() -> str:
    """Return the default cache directory, using the Lightning cloud cache if running in a Lightning environment.

    Returns:
        str: The resolved default cache root directory.
    """
    if _LITDATA_CACHE_DIR is not None:
        return _LITDATA_CACHE_DIR
    is_lightning_cloud = "LIGHTNING_CLUSTER_ID" in os.environ and "LIGHTNING_CLOUD_PROJECT_ID" in os.environ
    return _DEFAULT_LIGHTNING_CACHE_DIR if is_lightning_cloud else _DEFAULT_CACHE_DIR


def _try_create_cache_dir(
    input_dir: Optional[str],
    cache_dir: Optional[str] = None,
    storage_options: Optional[dict] = {},
    session_options: Optional[dict] = {},
    index_path: Optional[str] = None,
) -> Optional[str]:
    """Prepare and return the cache directory for a dataset."""
    resolved_input_dir = _resolve_dir(input_dir)
    updated_at = _read_updated_at(resolved_input_dir, storage_options, session_options, index_path)

    # Fallback to a hash of the input_dir if updated_at is "0"
    if updated_at == "0" and input_dir is not None:
        updated_at = generate_md5_hash(input_dir)

    dir_url_hash = generate_md5_hash(resolved_input_dir.url or "")

    # Determine the cache directory, preferring user-provided cache_dir if given
    cache_dir = cache_dir if cache_dir is not None else get_default_cache_dir()

    input_dir_hash_path = os.path.join(cache_dir, dir_url_hash)
    _clear_cache_dir_if_updated(input_dir_hash_path, updated_at)
    cache_dir = os.path.join(input_dir_hash_path, updated_at)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def generate_roi(chunks: list[dict[str, Any]], item_loader: Optional[BaseItemLoader] = None) -> list[tuple[int, int]]:
    """Generates default region_of_interest for chunks."""
    roi = []

    if isinstance(item_loader, TokensLoader):
        for idx, chunk in enumerate(chunks):
            roi.append((0, chunk["dim"] // item_loader._block_size))
    else:
        for i, chunk in enumerate(chunks):
            end = chunk["chunk_size"]
            roi.append((0, end))

    return roi


def load_index_file(input_dir: str) -> dict[str, Any]:
    """Load index file from the specified input directory.

    This function supports loading both chunk-based and mds shard-based index files.
    For shard-based files, it adapts the format to be compatible with chunk-based processing.

    Args:
        input_dir (str): The directory containing the index file.

    Returns:
        Dict[str, Any]: The loaded and possibly adapted index data.

    Raises:
        FileNotFoundError: If the index file does not exist in the input directory.

    """
    index_filepath = os.path.join(input_dir, _INDEX_FILENAME)
    try:
        with open(index_filepath) as f:
            data = json.load(f)

        if "chunks" not in data and "shards" in data:
            # load mds shard-based index file and adapt to chunks format
            return adapt_mds_shards_to_chunks(data)

        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Index file not found at {index_filepath}.")


def adapt_mds_shards_to_chunks(data: dict[str, Any]) -> dict[str, Any]:
    """Adapt mds shard-based index data to chunk-based format for compatibility.
    For more details about MDS, refer to the MosaicML Streaming documentation: https://github.com/mosaicml/streaming.

    Args:
        data (Dict[str, Any]): The original index data containing shards.

    Returns:
        Dict[str, Any]: Adapted index data with chunks format.

    """
    chunks = []
    shards = data["shards"]
    for shard in shards:
        chunks.append(
            {
                "chunk_bytes": shard["zip_data"]["bytes"],
                "chunk_size": shard["samples"],
                "column_sizes": shard["column_sizes"],
                "dim": None,
                "filename": shard["zip_data"]["basename"],
            }
        )
    data["chunks"] = chunks

    data_spec = [
        1,
        {
            "type": "builtins.dict",
            "context": json.dumps(shards[0]["column_names"]),
            "children_spec": [{"type": None, "context": None, "children_spec": []} for _ in shards[0]["column_names"]],
        },
    ]
    data["config"] = {
        "chunk_bytes": sum(shard["zip_data"]["bytes"] for shard in shards),
        "chunk_size": sum(shard["samples"] for shard in shards),
        "compression": shards[0]["compression"],
        "data_format": shards[0]["column_encodings"],
        "format": shards[0]["format"],
        "data_spec": json.dumps(data_spec),
        "encryption": None,
    }
    return data


def copy_index_to_cache_index_filepath(index_path: str, cache_index_filepath: str) -> None:
    """Copy Index file from index_path to cache_index_filepath."""
    # If index_path is a directory, append "index.json"
    if os.path.isdir(index_path):
        index_path = os.path.join(index_path, _INDEX_FILENAME)
    # Check if the file exists before copying
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    # Copy the file to cache_index_filepath
    shutil.copy(index_path, cache_index_filepath)
