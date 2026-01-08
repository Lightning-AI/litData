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

from collections import defaultdict
from typing import Any


def _pack_greedily(items: list[Any], weights: list[int], num_bins: int) -> tuple[dict[int, list[Any]], dict[int, int]]:
    """Greedily pack items with given weights into bins such that the total weight of each bin is roughly equally
    distributed among all bins.

    Returns:
        bin_contents: A dictionary mapping bin IDs to lists of items.
        bin_weights: A dictionary mapping bin IDs to the total weight of items in each bin.

    Example:
        >>> bin_contents, bin_weights = _pack_greedily(
                items=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
                weights=[4, 1, 2, 5, 8, 7, 3, 6, 9],
                num_bins=3,
            )
        >>> assert bin_contents == {0: ["I", "A", "G"], 1: ["E", "D", "C"], 2: ["F", "H", "B"]}
        >>> assert bin_weights == {0: 16, 1: 15, 2: 14}

    """
    if len(items) != len(weights):
        raise ValueError(f"Items and weights must have the same length, got {len(items)} and {len(weights)}.")
    if any(w <= 0 for w in weights):
        raise ValueError(f"All weights must be positive. Found: {weights}")

    sorted_items_and_weights = sorted(zip(items, weights), key=lambda x: x[1], reverse=True)
    bin_contents = defaultdict(list)
    bin_weights = dict.fromkeys(range(num_bins), 0)

    for item, weight in sorted_items_and_weights:
        min_bin_id = min(bin_weights, key=(lambda x: bin_weights[x]), default=0)
        bin_contents[min_bin_id].append(item)
        bin_weights[min_bin_id] += weight

    return bin_contents, bin_weights
