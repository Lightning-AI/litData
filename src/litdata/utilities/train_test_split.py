import logging
import os
from bisect import bisect_right
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from litdata import StreamingDataset
from litdata.constants import _INDEX_FILENAME
from litdata.utilities.dataset_utilities import load_index_file
from litdata.utilities.subsample import shuffle_lists_together, subsample_filenames_and_roi


def train_test_split(
    streaming_dataset: StreamingDataset, splits: list[float], seed: int = 42, shuffle: bool = True
) -> list[StreamingDataset]:
    """Splits a StreamingDataset into multiple subsets for training, testing, and validation.

    This function splits a StreamingDataset into multiple non-overlapping subsets based on the provided proportions.
    These subsets can be used for training, testing, and validation purposes.

    Args:
        streaming_dataset: An instance of StreamingDataset that needs to be split.
        splits: A list of floats representing the proportion of data to be allocated to each split
                             (e.g., [0.8, 0.1, 0.1] for 80% training, 10% testing, and 10% validation).
        seed: An integer used to seed the random number generator for reproducibility.
        shuffle: A boolean indicating whether to shuffle the data before splitting.

    Returns:
        List[StreamingDataset]: A list of StreamingDataset instances, where each element represents a split of the
                                original dataset according to the proportions specified in the 'splits' argument.

    Raises:
        ValueError: If any element in the 'splits' list is not a float between 0 (inclusive) and 1 (exclusive).
        ValueError: If the sum of the values in the 'splits' list is greater than 1.
        Exception: If the provided StreamingDataset is already a subsample (not currently supported).

    """
    if any(not isinstance(split, float) for split in splits):
        raise ValueError("Each split should be a float.")

    if not all(0 <= _f <= 1 for _f in splits):
        raise ValueError("Each Split should be a float with each value in [0,1].")

    if sum(splits) > 1:
        raise ValueError("Splits' sum must be less than 1.")

    # we need subsampled chunk filenames, original chunk file, and subsampled_roi

    dummy_streaming_dataset = deepcopy_dataset(streaming_dataset)
    # Note: We make this a set to speed up the computation of subsampled_chunks.
    dummy_subsampled_chunk_filename = set(dummy_streaming_dataset.subsampled_files)
    dummy_subsampled_roi = dummy_streaming_dataset.region_of_interest
    subsampled_chunks: list[dict[str, Any]] = []

    input_dir = dummy_streaming_dataset.input_dir
    assert input_dir.path

    if not os.path.exists(os.path.join(input_dir.path, _INDEX_FILENAME)):
        raise ValueError("Couldn't load original chunk file.")

    # load chunks from `index.json` file
    data = load_index_file(input_dir.path)

    # subsample the chunks defined in the index file
    original_chunks = data["chunks"]
    subsampled_chunks = [
        _org_chunk for _org_chunk in original_chunks if _org_chunk["filename"] in dummy_subsampled_chunk_filename
    ]

    new_datasets = [deepcopy_dataset(streaming_dataset) for _ in splits]

    dataset_length = sum([my_roi[1] - my_roi[0] for my_roi in dummy_subsampled_roi])

    if shuffle:
        subsampled_chunks, dummy_subsampled_roi = shuffle_lists_together(
            subsampled_chunks, dummy_subsampled_roi, np.random.RandomState([seed])
        )

    item_count_list = [int(dataset_length * split) for split in splits]

    if any(item_count == 0 for item_count in item_count_list):
        logging.warning("Warning: some splits are having item count 0, this will lead to empty datasets")

    for i, item_count in enumerate(item_count_list):
        curr_chunk_filename, curr_chunk_roi, left_chunks, left_roi = subsample_filenames_and_roi(
            subsampled_chunks, dummy_subsampled_roi, item_count
        )

        # update subsampled files & region_of_interest
        new_datasets[i].subsampled_files = curr_chunk_filename
        new_datasets[i].region_of_interest = curr_chunk_roi

        # reset dataset
        new_datasets[i].reset()

        subsampled_chunks = left_chunks
        dummy_subsampled_roi = left_roi

    return new_datasets


def subset_dataset(streaming_dataset: StreamingDataset, indices: Sequence[int] | slice) -> StreamingDataset:
    """Return a copy of ``streaming_dataset`` restricted to ``indices`` (global item order)."""
    new_dataset = deepcopy_dataset(streaming_dataset)

    if len(new_dataset.subsampled_files) != len(new_dataset.region_of_interest):
        raise ValueError("The provided dataset has mismatched subsampled_files and region_of_interest lengths.")

    dataset_length = sum(end - start for start, end in new_dataset.region_of_interest)

    if isinstance(indices, slice):
        indices = range(*indices.indices(dataset_length))

    if any(idx < 0 or idx >= dataset_length for idx in indices):
        raise ValueError(f"Subset indices must be in [0, {dataset_length - 1}] for the provided dataset.")

    chunk_starts: list[int] = []
    chunk_boundaries: list[tuple[str, int, int, int, int]] = []
    cursor = 0
    for filename, (roi_start, roi_end) in zip(new_dataset.subsampled_files, new_dataset.region_of_interest):
        chunk_len = roi_end - roi_start
        if chunk_len <= 0:
            continue
        chunk_starts.append(cursor)
        chunk_boundaries.append((filename, roi_start, roi_end, cursor, cursor + chunk_len))
        cursor += chunk_len

    new_subsampled_files: list[str] = []
    new_roi: list[tuple[int, int]] = []
    prev_chunk_idx: int | None = None

    for idx in indices:
        chunk_idx = bisect_right(chunk_starts, idx) - 1
        if chunk_idx < 0 or idx >= chunk_boundaries[chunk_idx][4]:
            raise ValueError(f"Index {idx} is out of bounds for the dataset.")

        filename, roi_start, _, global_start, _ = chunk_boundaries[chunk_idx]
        offset_in_chunk = roi_start + (idx - global_start)

        if new_roi and prev_chunk_idx == chunk_idx and offset_in_chunk == new_roi[-1][1]:
            new_roi[-1] = (new_roi[-1][0], new_roi[-1][1] + 1)
        else:
            new_subsampled_files.append(filename)
            new_roi.append((offset_in_chunk, offset_in_chunk + 1))

        prev_chunk_idx = chunk_idx

    new_dataset.subsampled_files = new_subsampled_files
    new_dataset.region_of_interest = new_roi
    new_dataset.reset()
    return new_dataset


def deepcopy_dataset(dataset: Any) -> Any:
    has_cache = dataset.cache is not None
    if has_cache:
        original_prepare_thread = dataset.cache._reader._prepare_thread
        original_force_download_queue = dataset.cache._reader._item_loader._force_download_queue
        dataset.cache._reader._prepare_thread = None
        dataset.cache._reader._item_loader._force_download_queue = None
    copied_dataset = deepcopy(dataset)
    if has_cache:
        dataset.cache._reader._prepare_thread = original_prepare_thread
        dataset.cache._reader._item_loader._force_download_queue = original_force_download_queue
    return copied_dataset
