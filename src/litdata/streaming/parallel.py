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

import hashlib
import inspect
import logging
import random
from collections.abc import Iterator
from copy import deepcopy
from typing import Any, Literal, Optional, Protocol, Union

import numpy as np
import torch

from litdata.streaming.dataset import StreamingDataset
from litdata.utilities.base import (
    __NUM_CYCLES_KEY__,
    __NUM_SAMPLES_YIELDED_KEY__,
    __SAMPLES_KEY__,
    _BaseStreamingDatasetWrapper,
)
from litdata.utilities.env import _WorkerEnv

logger = logging.getLogger("litdata.streaming.parallel")

RandomGenerator = Union[random.Random, np.random.Generator, torch.Generator]
GeneratorName = Literal["random", "numpy", "torch"]


class Transform(Protocol):
    def __call__(self, samples: tuple[Any, ...], rng: Optional[dict[GeneratorName, RandomGenerator]] = None) -> Any: ...


class ParallelStreamingDataset(_BaseStreamingDatasetWrapper):
    """Enables to stream data from multiple StreamingDataset in parallel.

    By default, the yielded samples are tuples where the `n`-th element is a sample from the `n`-th dataset.

    Additionally, the parallel dataset keeps track of the number of samples fetched to enable reusability of the
    datasets.

    The parallel dataset can be configured to raise a ``StopIteration`` as soon as any of the datasets is exhausted, or
    to cycle through the datasets until a given number of samples are yielded. When cycling and using a
    ``StreamingDataLoader``, the ``resume`` option can be used to either yield the same ``length`` samples in each
    epoch, or to resume the dataset from where it left off in the previous epoch.

    New data can be generated on-the-fly from a sample from each dataset by providing a ``transform`` function. This
    function can take a single tuple argument containing a sample from each dataset, and optionally a dictionary of
    random number generators which are seeded using the current state of the dataset. The keys of this dictionary are
    ``"random"``, ``"numpy"`` and ``"torch"``, and the values are instances of ``random.Random``,
    ``numpy.random.Generator`` and ``torch.Generator`` respectively. This is useful if the data transformation
    requires random number generation which should be resumable.

    Example:
        >>> def transform(samples):
        >>>     sample_1, sample_2 = samples
        >>>     return sample_1 + sample_2
        ...
        >>> # or using random number generators
        >>> def transform(samples, rngs):
        >>>     sample_1, sample_2 = samples
        >>>     rng = rngs["random"]
        >>>     return rng.random() * sample_1 + rng.random() * sample_2
        ...
        >>> dset_1 = StreamingDataset(...)
        >>> dset_2 = StreamingDataset(...)
        >>> parallel_dset = ParallelStreamingDataset(
        >>>     datasets=[dset_1, dset_2],
        >>>     transform=transform,
        >>> )

    """

    def __init__(
        self,
        datasets: list[StreamingDataset],
        length: Optional[Union[int, float]] = None,
        force_override_state_dict: bool = False,
        transform: Optional[Transform] = None,
        seed: int = 42,
        resume: bool = True,
        reset_rngs: bool = False,
    ) -> None:
        """Enable to stream data from multiple StreamingDataset in parallel.

        Args:
            datasets: The list of the StreamingDataset to use.
            length: The number of samples to yield. If ``None``, the datasets are iterated over until one of them is
                exhausted. If an integer, the datasets are cycled until ``length`` samples are yielded. Can be
                ``float("inf")`` for an infinite dataset.
            force_override_state_dict: Boolean flag for allowing local arguments to override a loaded state dict.
            transform: A function to apply to the samples yielded by the datasets to generate new data. Takes as
                argument a tuple containing one sample from each dataset, and optionally a dictionary of random
                number generators which are seeded using the current state of the dataset.
            seed: Seed for the random number generators provided to ``transform``.
            resume: If ``True`` and ``length`` is not ``None``, tells the dataloader to resume the dataset from where it
                left off in the previous epoch. If ``False``, the same ``length`` samples are yielded in each epoch.
                Ignored if ``length`` is ``None``.
            reset_rngs: If ``True``, the random number generators provided to ``transform`` are reset to their initial
                state at the beginning of each epoch. Together with ``resume=False``, this allows to produce the same
                samples in each epoch.
        """
        self._check_datasets(datasets)

        if length is not None and not isinstance(length, int) and length != float("inf"):
            raise ValueError(f"`length` must be `None`, an integer, or `float('inf')`, got {length}.")

        transform_nargs = None
        if transform is not None:
            transform_nargs = sum(
                p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
                for p in inspect.signature(transform).parameters.values()
            )
            if transform_nargs not in (1, 2):
                raise ValueError(f"transform function must take 1 or 2 arguments, got {transform_nargs} instead.")

        self._datasets = datasets
        self._length = length
        self._force_override_state_dict = force_override_state_dict
        self._transform = transform
        self._transform_nargs = transform_nargs
        self._seed = seed
        self._reset_rngs = reset_rngs
        self._iterator: Optional[_ParallelDatasetIterator] = None
        self._use_streaming_dataloader = False
        self._num_samples_yielded: Optional[dict[int, list[int]]] = None
        self._num_cycles: Optional[dict[int, list[int]]] = None
        self._current_epoch = 0
        self.num_workers = 1
        self.batch_size = 1
        self.resume = resume

        if length is not None:
            for dataset in self._datasets:
                if isinstance(dataset, StreamingDataset):
                    dataset.set_epoch(1)

    def is_cycling(self) -> bool:
        if self._length is None:
            return False
        if isinstance(self._length, int) or self._length == float("inf"):
            return True
        raise ValueError(f"ParallelStreamingDataset length must be None, int, or float('inf'), got {self._length}.")

    def is_infinite(self) -> bool:
        if self._length is None or isinstance(self._length, int):
            return False
        if self._length == float("inf"):
            return True
        raise ValueError(f"ParallelStreamingDataset length must be None, int, or float('inf'), got {self._length}.")

    def set_epoch(self, current_epoch: int) -> None:
        self._current_epoch = current_epoch
        if self.is_cycling() and self.resume:
            # do not set the epoch as cycling datasets have their own epoch counter
            return
        for dataset in self._datasets:
            dataset.set_epoch(current_epoch)

    def update_epoch_counters(self, num_cycles: list[int]) -> None:
        """Update the epoch counter of the wrapped datasets when cycling."""
        if self.is_cycling():
            assert len(num_cycles) == len(self._datasets)
            for i_cycle, dset in zip(num_cycles, self._datasets):
                # epoch counter starts at 1 while cycle counter starts at 0
                if dset.current_epoch != i_cycle + 1:
                    # do not call dset.set_epoch as it is ignored if the dataset has non-None _state_dict attribute
                    dset.current_epoch = i_cycle + 1

    def get_len(self, num_workers: int, batch_size: int) -> Optional[int]:
        self.num_workers = num_workers
        self.batch_size = batch_size
        # initialize lengths even if self._length is not None to call self._get_len() on all the wrapped datasets and
        # set their num_workers and batch_size attributes
        lengths = self.get_all_lens()
        if self._length is None:
            return min(lengths)
        if self._length == float("inf"):
            return None
        if isinstance(self._length, int):
            return self._length
        raise ValueError(f"ParallelStreamingDataset length must be None, int, or float('inf'), got {self._length}.")

    def get_all_lens(self) -> list[int]:
        return [self._get_len(d) for d in self._datasets]

    def __iter__(self) -> Iterator[Any]:
        if self.is_cycling() and not self.resume:
            self.set_epoch(1)

        worker_env = _WorkerEnv.detect()

        num_samples_yielded = None
        if self._num_samples_yielded is not None and worker_env.rank in self._num_samples_yielded:
            num_samples_yielded = self._num_samples_yielded.get(worker_env.rank, 0)

        num_cycles = None
        if self._num_cycles is not None and worker_env.rank in self._num_cycles:
            num_cycles = self._num_cycles.get(worker_env.rank, 0)

        # convert the length option to the corresponding number of samples for the current worker
        length = self._length
        if isinstance(length, int):
            length = length // worker_env.world_size + (worker_env.rank < length % worker_env.world_size)

        # convert the true length of each dataset i.e. the cycle length to the corresponding value for current worker
        tot_dset_lengths = self.get_all_lens()
        dset_lengths = [
            dl // worker_env.world_size + (worker_env.rank < dl % worker_env.world_size) for dl in tot_dset_lengths
        ]

        # compute random seed based on the current dataset state and initialize generators
        tot_samples_yielded, tot_cycles = self.get_num_samples_yielded()
        tot_samples_yielded = [0 if dl == 0 else s % dl for s, dl in zip(tot_samples_yielded, tot_dset_lengths)]
        if self._reset_rngs:
            state = (self._seed, worker_env.rank, *tot_samples_yielded)
        elif self._length is None:
            state = (self._seed, worker_env.rank, *tot_samples_yielded, self._current_epoch)
        else:
            state = (self._seed, worker_env.rank, *tot_samples_yielded, *tot_cycles)
        # produce a seed from the state in a stable way; there might be a better way to do this
        seed = int(hashlib.sha256(str(state).encode()).hexdigest(), 16) % (2**32 - 1)
        rngs: dict[GeneratorName, RandomGenerator] = {
            "random": random.Random(seed),  # noqa: S311
            "numpy": np.random.default_rng(seed),
            "torch": torch.Generator().manual_seed(seed),
        }

        self._iterator = _ParallelDatasetIterator(
            self._datasets,
            self._use_streaming_dataloader,
            num_samples_yielded,
            num_cycles,
            length,
            dset_lengths,
            self._transform,
            self._transform_nargs,
            rngs,
        )
        return self._iterator

    def __len__(self) -> Optional[int]:
        # ``batch_size`` may be a sequence when per-dataset values were set on
        # the wrapper.  For length estimation we only need a scalar; we take
        # the first element if a sequence is provided.
        from collections.abc import Sequence

        bs_int: int = int(self.batch_size[0]) if isinstance(self.batch_size, Sequence) else int(self.batch_size)
        return self.get_len(self.num_workers, bs_int if bs_int else 1)

    def get_num_samples_yielded(
        self,
        num_samples_yielded: Optional[dict[int, list[int]]] = None,
        num_cycles: Optional[dict[int, list[int]]] = None,
    ) -> tuple[list[int], list[int]]:
        """Get the number of samples yielded and the number of cycles for each dataset across workers.

        Get the total number of samples yielded by each dataset across workers since it was last cycled, and the number
        of times each dataset was cycled.

        Args:
            num_samples_yielded: The number of samples yielded by each dataset and each worker. Keys are the worker
                ranks and values are the number of samples yielded by each dataset.
            num_cycles: The number of times each dataset was cycled in each worker. Keys are the worker ranks and values
                are the number of times each dataset was cycled.

        Returns:
            A tuple of two lists: the total number of samples yielded by each dataset across workers, and the number of
            times each dataset was cycled.
        """
        num_samples_yielded = num_samples_yielded or self._num_samples_yielded or {}
        num_cycles = num_cycles or self._num_cycles or {}
        assert num_samples_yielded.keys() == num_cycles.keys()
        assert all(len(s) == len(c) for s, c in zip(num_samples_yielded.values(), num_cycles.values()))
        output = [0 for _ in range(len(self._datasets))]
        cycles = [0 for _ in range(len(self._datasets))]
        for i, (num_samples_yielded, num_cycles) in enumerate(
            zip(zip(*num_samples_yielded.values()), zip(*num_cycles.values()))
        ):
            cycles[i] = max(num_cycles)
            output[i] = sum(s for (s, c) in zip(num_samples_yielded, num_cycles) if c == cycles[i])
        return output, cycles

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if self._use_streaming_dataloader:
            self._num_cycles = state_dict["num_cycles"]

    def state_dict(
        self, num_workers: int, batch_size: int, num_samples_yielded: Optional[list[int]] = None
    ) -> dict[str, Any]:
        if self._iterator is None and num_samples_yielded is None:
            return {}
        num_samples_yielded = num_samples_yielded or [0 for _ in range(len(self._datasets))]
        return {
            str(dataset_idx): deepcopy(
                dataset.state_dict(
                    num_samples_yielded=num_samples_yielded[dataset_idx], num_workers=num_workers, batch_size=batch_size
                )
            )
            for dataset_idx, dataset in enumerate(self._datasets)
        }


class _ParallelDatasetIterator(Iterator):
    def __init__(
        self,
        datasets: list[StreamingDataset],
        use_streaming_dataloader: bool,
        num_samples_yielded: Any,
        num_cycles: Any,
        length: Optional[Union[int, float]],
        dset_lengths: list[int],
        transform: Optional[Transform],
        transform_nargs: Optional[int],
        rngs: dict[GeneratorName, RandomGenerator],
    ) -> None:
        self._datasets = datasets
        self._dataset_iters = [iter(dataset) for dataset in datasets]
        self._num_samples_yielded = num_samples_yielded or [0 for _ in range(len(datasets))]
        self._num_cycles = num_cycles or [0 for _ in range(len(datasets))]
        self._length = length
        self._use_streaming_dataloader = use_streaming_dataloader
        self._transform = transform
        self._transform_nargs = transform_nargs
        self._rngs = rngs
        self._count = 0
        if isinstance(self._length, int) and self._length > 0:
            # infer counter resume value from the number of times we cycled, the number of samples yielded in the
            # current cycle, the dataset length i.e. cycle length, and the length option
            self._count = (dset_lengths[0] * self._num_cycles[0] + self._num_samples_yielded[0]) % self._length
            assert all(
                (dset_lengths[i] * self._num_cycles[i] + self._num_samples_yielded[i]) % self._length == self._count
                for i in range(1, len(dset_lengths))
            )
        else:
            self._count = 0

    def transform(self, samples: tuple[Any, ...]) -> Any:
        if self._transform is None:
            return samples
        assert self._transform_nargs is not None
        if self._transform_nargs == 1:
            return self._transform(samples)
        if self._transform_nargs == 2:
            return self._transform(samples, self._rngs)
        raise RuntimeError(f"transform function must take 1 or 2 arguments, got {self._transform_nargs} instead.")

    def __next__(self) -> Union[Any, dict[str, Any]]:
        if self._length is not None and self._count >= self._length:
            raise StopIteration
        samples, _resets = zip(*[self._get_sample(i) for i in range(len(self._datasets))])
        # update _num_samples_yielded and _num_cycles only if samples were successfully fetched from all datasets
        for i, _reset in enumerate(_resets):
            self._num_samples_yielded[i] = 1 if _reset else self._num_samples_yielded[i] + 1
            self._num_cycles[i] = self._num_cycles[i] + 1 if _reset else self._num_cycles[i]
        self._count += 1
        samples = self.transform(samples)
        if self._use_streaming_dataloader:
            return {
                __SAMPLES_KEY__: samples,
                __NUM_SAMPLES_YIELDED_KEY__: self._num_samples_yielded,
                __NUM_CYCLES_KEY__: self._num_cycles,
            }
        return samples

    def _get_sample(self, dataset_index: int) -> tuple[Any, bool]:
        _reset = False
        try:
            sample = next(self._dataset_iters[dataset_index])
        except StopIteration as e:
            if self._length is None:
                raise e
            self._dataset_iters[dataset_index] = iter(self._datasets[dataset_index])
            _reset = True
            try:
                sample = next(self._dataset_iters[dataset_index])
            except StopIteration as e:
                # The dataset is empty or this worker got 0 samples assigned. Either way raise the StopIteration.
                raise e
        return sample, _reset
