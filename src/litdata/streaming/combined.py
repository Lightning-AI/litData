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
import random
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import Any, Literal, Optional, Union

from litdata.debugger import ChromeTraceColors, _get_log_msg
from litdata.streaming.dataset import StreamingDataset
from litdata.utilities.base import (
    __NUM_SAMPLES_YIELDED_KEY__,
    __SAMPLES_KEY__,
    _BaseStreamingDatasetWrapper,
)
from litdata.utilities.env import _WorkerEnv

logger = logging.getLogger("litdata.streaming.combined")


class BatchingMethod:
    STRATIFIED = "stratified"
    PER_STREAM = "per_stream"


BatchingMethodType = Literal["stratified", "per_stream"]


class CombinedStreamingDataset(_BaseStreamingDatasetWrapper):
    """Enables to stream data from multiple StreamingDataset with the sampling ratio of
    your choice.

    Additionally, the `CombinedStreamingDataset` keeps track of the number of samples fetched to enable reusability
    of the datasets.

    Note that due to the random sampling, the number of samples returned from the iterator is variable and a function
    of the given seed. The combined dataset will raise a StopIteration as soon as any of the datasets is exhausted.

    """

    def __init__(
        self,
        datasets: list[StreamingDataset],
        seed: int = 42,
        weights: Optional[Sequence[float]] = None,
        iterate_over_all: bool = True,
        batching_method: BatchingMethodType = "stratified",
        force_override_state_dict: bool = False,
    ) -> None:
        """Enable to stream data from multiple StreamingDataset with the sampling ratio of your choice.

        Args:
            datasets: The list of the StreamingDataset to use.
            seed: The random seed to initialize the sampler
            weights: The sampling ratio for the datasets
            iterate_over_all: When iterate_over_all is True, the combined dataset iterates over all the datasets.
                Otherwise, it stops as soon as one raises a StopIteration.
            batching_method (str, optional): When batching_method is set to "stratified" (default),
                batches will include samples from all datasets. On the other hand, when batching_method is "per_stream",
                batches will consist of samples from a single dataset,  which is selected randomly.
            force_override_state_dict: Boolean flag for allowing local arguments to override a loaded state dict.

        """
        self._check_datasets(datasets)

        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        self._iterate_over_all = iterate_over_all
        self._force_override_state_dict = force_override_state_dict

        if iterate_over_all and weights:
            raise ValueError(
                "When `iterate_over_all` is set to True, the weights argument shouldn't be provided.",
                " Instead, it will be computed from the inverse of the dataset length.",
            )

        self._iterate_over_all = iterate_over_all

        if weights is None:
            # Weighted based on the dataset length
            dataset_lens = [len(d) for d in datasets]
            total_len = sum(dataset_lens)
            assert total_len > 0
            self._weights = [len / total_len for len in dataset_lens]
        else:
            weights_sum = sum(weights)
            self._weights = [w / weights_sum for w in weights]

        self._iterator: Optional[_CombinedDatasetIterator] = None
        self._use_streaming_dataloader = False
        self._num_samples_yielded: Optional[dict[int, list[int]]] = None
        self._current_epoch = 0
        self.num_workers = 1
        self.batch_size = 1
        self._batching_method: BatchingMethodType = batching_method

    def get_len(self, num_workers: int, batch_size: int) -> Optional[int]:
        self.num_workers = num_workers
        self.batch_size = batch_size
        if self._iterate_over_all:
            return self._get_total_length()
        return None

    def __len__(self) -> Optional[int]:
        return self.get_len(1, 1)

    # total length of the datasets
    def _get_total_length(self) -> int:
        return sum(self._get_len(d) for d in self._datasets)

    def set_epoch(self, current_epoch: int) -> None:
        """Set the current epoch to the datasets on epoch starts.

        When using the StreamingDataLoader, this is done automatically

        """
        self._current_epoch = current_epoch
        for dataset in self._datasets:
            dataset.set_epoch(current_epoch)

    def __iter__(self) -> Iterator[Any]:
        assert self._weights

        worker_env = _WorkerEnv.detect()

        num_samples_yielded = None

        if self._num_samples_yielded is not None and worker_env.rank in self._num_samples_yielded:
            num_samples_yielded = self._num_samples_yielded.get(worker_env.rank, 0)

        self._iterator = _CombinedDatasetIterator(
            self._datasets,
            self._seed,
            self._weights,
            self._use_streaming_dataloader,
            num_samples_yielded,
            self.batch_size,
            self._batching_method,
            self._iterate_over_all,
        )
        return self._iterator

    def state_dict(
        self, num_workers: int, batch_size: int, num_samples_yielded: Optional[list[int]] = None
    ) -> dict[str, Any]:
        if self._iterator is None:
            if num_samples_yielded is None:
                return {}
            return _state_dict(self._datasets, num_samples_yielded, num_workers, batch_size)
        return self._iterator.state_dict(num_workers, batch_size)


class _CombinedDatasetIterator(Iterator):
    def __init__(
        self,
        datasets: list[StreamingDataset],
        seed: int,
        weights: Sequence[Optional[float]],
        use_streaming_dataloader: bool,
        num_samples_yielded: Any,
        batch_size: Union[int, Sequence[int]],
        batching_method: BatchingMethodType,
        iterate_over_all: bool = False,
    ) -> None:
        self._datasets = datasets
        self._dataset_iters = [iter(dataset) for dataset in datasets]
        self._dataset_indexes: list[Optional[int]] = list(range(len(datasets)))
        self._num_samples_yielded = num_samples_yielded or [0 for _ in range(len(datasets))]
        self._original_weights = deepcopy(weights)
        self._weights = deepcopy(weights)
        self._rng = random.Random(seed)  # noqa: S311
        self._iterate_over_all = iterate_over_all
        self._batching_method = batching_method
        # Batch size can be an int (applied to all datasets) or a sequence providing
        # a specific batch size per dataset.
        self._batch_size = batch_size
        from collections.abc import Sequence as _Sequence

        # Validate when a sequence is provided
        if isinstance(batch_size, _Sequence) and len(batch_size) != len(datasets):
            raise ValueError("When providing a sequence of batch sizes, its length must match the number of datasets.")
        self._is_done = False

        if num_samples_yielded is not None:
            self._num_samples_yielded = num_samples_yielded
            for _ in range(sum(num_samples_yielded)):
                choice_indexes: list[int] = [index for index in self._dataset_indexes if index is not None]
                choice_weights: list[float] = [w for w in self._weights if w is not None]
                self._rng.choices(choice_indexes, weights=choice_weights, k=1)

        self._use_streaming_dataloader = use_streaming_dataloader
        self._is_done = False

        # Track the number of samples yielded in the current (DataLoader) batch
        # and the active dataset index. This is used only when batching_method is
        # set to "per_stream".  With per-dataset batch sizes we look up the limit
        # dynamically based on ``self._batch_size`` if it is a sequence.
        self._samples_yielded_in_batch = 0
        self._cur_dataset_index = -1

        logger.debug(
            _get_log_msg({"name": "iterating_combined_dataset", "ph": "B", "cname": ChromeTraceColors.LIGHT_BLUE})
        )

    def __next__(self) -> Any:
        if self._iterate_over_all:
            while True:
                try:
                    indexes_left = [index for index in self._dataset_indexes if index is not None]
                    if len(indexes_left) > 1:
                        dataset_index = self._get_dataset_index()
                    elif len(indexes_left) == 1:
                        dataset_index = indexes_left[0]
                    return self._get_sample(dataset_index)
                except StopIteration as e:
                    if len(indexes_left) == 1:
                        self._dataset_indexes = list(range(len(self._datasets)))
                        self._weights = deepcopy(self._original_weights)
                        logger.debug(
                            _get_log_msg(
                                {"name": "iterating_combined_dataset", "ph": "E", "cname": ChromeTraceColors.LIGHT_BLUE}
                            )
                        )
                        raise e

                    self._dataset_indexes[dataset_index] = None
                    self._weights[dataset_index] = None  # type: ignore
                    new_sum = sum([w for w in self._weights if w is not None])
                    self._weights = [None if w is None else w / new_sum for w in self._weights]

        # stop on the first iteration
        return self._get_sample(self._get_dataset_index())

    def _get_dataset_index(self) -> int:
        if self._batching_method == BatchingMethod.STRATIFIED:
            # For every sample, randomly select a dataset (weighted)
            dataset_idx = self._set_new_dataset_index()
        elif self._batching_method == BatchingMethod.PER_STREAM:
            # For each batch, pick a dataset and stick with it until the
            # desired number of samples for that dataset have been yielded.

            from collections.abc import Sequence as _Sequence

            if self._cur_dataset_index == -1:
                # Start of iteration or after switching dataset
                self._cur_dataset_index = self._set_new_dataset_index()
                self._samples_yielded_in_batch = 0

            dataset_idx = self._cur_dataset_index

            # Determine the batch-size limit for the current dataset
            limit = self._batch_size[dataset_idx] if isinstance(self._batch_size, _Sequence) else self._batch_size

            if self._samples_yielded_in_batch >= limit:
                # Current dataset reached its quota; pick a *different* dataset if possible
                candidate_idx = self._cur_dataset_index
                if len([i for i in self._dataset_indexes if i is not None]) > 1:
                    while candidate_idx == self._cur_dataset_index:
                        candidate_idx = self._set_new_dataset_index()
                # Update tracking
                self._cur_dataset_index = candidate_idx
                self._samples_yielded_in_batch = 0
                dataset_idx = self._cur_dataset_index
                # Re-compute limit for the new dataset
                if isinstance(self._batch_size, _Sequence):
                    limit = self._batch_size[dataset_idx]

            self._samples_yielded_in_batch += 1
        else:
            raise ValueError(f"Invalid batching method: {self._batching_method}")
        return dataset_idx

    def _set_new_dataset_index(self) -> int:
        # randomly select a dataset index
        indexes = [index for index in self._dataset_indexes if index is not None]
        weights = [w for w in self._weights if w is not None]
        (dataset_index,) = self._rng.choices(indexes, weights=weights, k=1)
        return dataset_index

    def _get_sample(self, dataset_index: int) -> Any:
        # get the sample
        sample = next(self._dataset_iters[dataset_index])

        # keep track the sample was fetched
        self._num_samples_yielded[dataset_index] += 1

        # return a new sample
        if self._use_streaming_dataloader:
            return {
                __SAMPLES_KEY__: sample,
                __NUM_SAMPLES_YIELDED_KEY__: self._num_samples_yielded,
            }
        return sample

    def state_dict(self, num_workers: int = 0, batch_size: int = 1) -> dict[str, Any]:
        return _state_dict(self._datasets, self._num_samples_yielded, num_workers, batch_size)


def _state_dict(
    datasets: list[StreamingDataset], num_samples_yielded: list[int], num_workers: int = 0, batch_size: int = 1
) -> dict[str, Any]:
    return {
        str(dataset_idx): dataset.state_dict(
            num_samples_yielded=num_samples_yielded[dataset_idx], num_workers=num_workers, batch_size=batch_size
        )
        for dataset_idx, dataset in enumerate(datasets)
    }
