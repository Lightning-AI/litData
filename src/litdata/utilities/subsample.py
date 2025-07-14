from typing import Any, Optional

import numpy as np


def shuffle_lists_together(
    list1: list[Any], list2: list[Any], random_seed_sampler: Optional[np.random.RandomState] = None, seed: int = 42
) -> tuple[list[Any], list[Any]]:
    """Shuffles list1 and applies the same shuffle order to list2.

    Args:
        list1: The first list to shuffle.
        list2: The second list to shuffle in correspondence with list1.
        random_seed_sampler: Random seed sampler to be used for shuffling
        seed: Seed to use in case random_seed_sampler is not provided

    Returns:
        A tuple containing the shuffled versions of list1 and list2.

    """
    # Sanity check
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same size.")

    if random_seed_sampler is None:
        random_seed_sampler = np.random.RandomState([seed])

    # Shuffle the first list
    shuffled_indices = list(range(len(list1)))

    shuffled_indices = random_seed_sampler.permutation(shuffled_indices).tolist()

    # Apply the same shuffle order to the second list
    shuffled_list1 = [list1[i] for i in shuffled_indices]
    shuffled_list2 = [list2[i] for i in shuffled_indices]

    # Return the shuffled lists
    return shuffled_list1, shuffled_list2


def subsample_filenames_and_roi(
    chunks: list[dict[str, Any]], roi_list: list[tuple[int, int]], item_count: int
) -> tuple[list[str], list[tuple[int, int]], list[dict[str, Any]], list[tuple[int, int]]]:
    assert len(chunks) == len(roi_list)

    if item_count == 0:
        return [], [], chunks, roi_list

    cumsum_sizes = np.cumsum([r[1] - r[0] for r in roi_list])

    match = np.argmax(cumsum_sizes >= item_count)

    exact_item_count_match = cumsum_sizes[match] == item_count

    subsampled_filenames = [c["filename"] for c in chunks[: match + 1]]
    subsampled_chunk_roi = roi_list[: match + 1]
    # bcoz tuple doesn't support item assignment
    subsampled_chunk_roi[-1] = (
        subsampled_chunk_roi[-1][0],
        subsampled_chunk_roi[-1][1] - (cumsum_sizes[match] - item_count),
    )

    assert sum(_chnk[1] - _chnk[0] for _chnk in subsampled_chunk_roi) == item_count

    if exact_item_count_match:
        match += 1  # start from next chunk
    left_over_chunks = chunks[match:]
    left_over_chunk_roi = roi_list[match:]

    if not exact_item_count_match:
        # bcoz tuple doesn't support item assignment
        left_over_chunk_roi[0] = subsampled_chunk_roi[-1][1], left_over_chunk_roi[0][1]  # start from next chunk

    return (
        subsampled_filenames,
        subsampled_chunk_roi,
        left_over_chunks,
        left_over_chunk_roi,
    )
