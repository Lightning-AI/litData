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

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, Optional, Union

from torch.utils.data import IterableDataset

from litdata.streaming.dataset import StreamingDataset

__NUM_CYCLES_KEY__ = "__NUM_CYCLES__"
__NUM_SAMPLES_YIELDED_KEY__ = "__NUM_SAMPLES_YIELDED__"
__SAMPLES_KEY__ = "__SAMPLES__"


class _BaseStreamingDatasetWrapper(IterableDataset, ABC):
    # Base class for datasets that wrap multiple streaming datasets
    # This includes CombinedStreamingDataset and ParallelStreamingDataset

    _datasets: list[StreamingDataset]
    _current_epoch: int
    batch_size: Union[int, Sequence[int]]
    num_workers: int
    _force_override_state_dict: bool
    _use_streaming_dataloader: bool
    _num_samples_yielded: Optional[dict[int, list[int]]] = None

    def set_shuffle(self, shuffle: bool) -> None:
        """Set the current shuffle to the datasets."""
        for dataset in self._datasets:
            dataset.set_shuffle(shuffle)

    def set_batch_size(self, batch_size: Union[int, Sequence[int]]) -> None:
        """Set the current batch size.

        This method now supports either:

        1. a single ``int`` applied to all wrapped datasets (previous behaviour), or
        2. a ``Sequence[int]`` that specifies one batch size per wrapped dataset.

        The length of the sequence must match the number of wrapped datasets.
        """
        # Defer the import to avoid overhead when not required
        from collections.abc import Sequence

        self.batch_size = batch_size  # store as-is for later access

        if isinstance(batch_size, Sequence):
            if len(batch_size) != len(self._datasets):
                raise ValueError(
                    "The length of `batch_size` must match the number of datasets when passing a sequence."
                )
            for bs, dataset in zip(batch_size, self._datasets):
                dataset.set_batch_size(bs)
        else:
            for dataset in self._datasets:
                dataset.set_batch_size(int(batch_size))

    def set_num_workers(self, num_workers: int) -> None:
        """Set the current number of workers to the datasets."""
        self.num_workers = num_workers
        for dataset in self._datasets:
            dataset.set_num_workers(num_workers)

    def set_drop_last(self, drop_last: bool) -> None:
        """Set the current drop_last to the datasets."""
        for dataset in self._datasets:
            dataset.set_drop_last(drop_last)

    def reset_state_dict(self) -> None:
        """Reset the state of the dataset."""
        for dataset in self._datasets:
            dataset.reset_state_dict()

    def _check_datasets(self, datasets: list[StreamingDataset]) -> None:
        if any(not isinstance(d, StreamingDataset) for d in datasets):
            raise RuntimeError("The provided datasets should be instances of the StreamingDataset.")

    def _set_use_streaming_dataloader(self, use_streaming_dataloader: bool) -> None:
        # Used to prevent returning num_samples_yielded when using PyTorch DataLoader
        self._use_streaming_dataloader = use_streaming_dataloader

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return

        if len(state_dict["dataset"]) != len(self._datasets):
            if not self._force_override_state_dict:
                raise RuntimeError(
                    f"The provided state doesn't match the current number of datasets: {self._datasets}."
                )
            if len(state_dict["dataset"]) > len(self._datasets):
                raise RuntimeError(
                    "Currently it's only possible to add datasets to the end of the dataset list when overriding state."
                )

        for dataset_idx, dataset in enumerate(self._datasets):
            if str(dataset_idx) in state_dict["dataset"]:
                dataset.load_state_dict(state_dict["dataset"][str(dataset_idx)])

            elif not self._force_override_state_dict:
                raise RuntimeError(f"The provided state doesn't contain the index {dataset_idx}.")

        # Used to iterate over the sampler to avoid sampling the same samples
        if self._use_streaming_dataloader:
            self._num_samples_yielded = state_dict["num_samples_yielded"]

    def _get_len(self, d: Any) -> int:
        # mypy: ``self.batch_size`` can be a ``Sequence[int]`` now, but the
        # underlying datasets still expect a plain ``int`` for their
        # ``get_len`` signature.  We pass an `int` in both cases and use the
        # first element of the sequence when a per-dataset list is provided.

        from collections.abc import Sequence

        if isinstance(self.batch_size, Sequence):
            bs_int: int = int(self.batch_size[0] if self.batch_size else 1)
        else:
            bs_int = int(self.batch_size)

        if isinstance(d, StreamingDataset):
            return d.get_len(self.num_workers, bs_int)
        return len(d)

    @abstractmethod
    def set_epoch(self, current_epoch: int) -> None: ...

    @abstractmethod
    def get_len(self, num_workers: int, batch_size: int) -> Optional[int]: ...

    @abstractmethod
    def __len__(self) -> Optional[int]: ...

    @abstractmethod
    def state_dict(
        self, num_workers: int, batch_size: int, num_samples_yielded: Optional[list[int]] = None
    ) -> dict[str, Any]: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Any]: ...
