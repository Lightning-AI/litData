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
from multiprocessing import Manager, Queue
from typing import Any, Dict, List, Tuple, Union


class WorkerItemProvider:
    """Helper class for providing items to the worker."""

    def __init__(self, items: List[List[Any]], num_downloaders: int, num_workers: int) -> None:
        self.manager = Manager()
        self.items = items
        self.num_downloaders = num_downloaders
        self.num_workers = num_workers

        self.paths: Dict[int, List[List[str]]] = self.manager.dict()
        self.ready_to_process_item_queue: Dict[int, Queue] = self.manager.dict()
        self.ready_to_process_shared_queue: Queue = self.manager.Queue()

        # Initialize queues & paths, can't use defaultdict as it is not serializable
        self._initialize()

    def _initialize(self):
        for worker_index in range(self.num_workers):
            self.paths[worker_index] = []
            self.ready_to_process_item_queue[worker_index] = self.manager.Queue()

    def set_items(self, index: Union[int, Tuple[int, int]], item: List[Any]) -> None:
        if isinstance(index, int):
            self.items[index] = item
        elif isinstance(index, tuple):
            worker_index, item_index = index
            self.items[worker_index][item_index] = item
        else:
            raise ValueError(f"Invalid index type: Expected (int or Tuple[int, int]), but got {type(index)}")

    def get_items(self, index: Union[int, Tuple[int, int]]) -> List[Any]:
        if isinstance(index, int):
            return self.items[index]
        if isinstance(index, tuple):
            worker_index, item_index = index
            return self.items[worker_index][item_index]
        raise ValueError(f"Invalid index type: Expected (int or Tuple[int, int]), but got {type(index)}")

    def prepare_ready_to_use_queue(self, use_shared_queue: bool, worker_index: int) -> None:
        """Default (if not using downloaders), Prepares the queue for the worker to use."""
        target_queue = (
            self.ready_to_process_shared_queue if use_shared_queue else self.ready_to_process_item_queue[worker_index]
        )

        for index, _ in enumerate(self.items[worker_index]):
            target_queue.put_nowait((worker_index, index))

        sentinel = None

        # A worker stops only after it has received `num_downloaders` sentinel values (None).
        #
        # When using a **shared queue**, sentinels are inserted by each worker into the same queue.
        # This means the queue ends up with `num_downloaders * num_workers` sentinel values.
        #
        # The goal is to ensure that **each worker eventually sees `num_downloaders` sentinel values**,
        # so it knows that no more real items will arrive.
        #
        # Example:
        #   Suppose we have:
        #     - 2 workers (Worker 0 and Worker 1)
        #     - 2 downloaders per worker
        #
        #   Shared queue content might look like this:
        #     [(0,0), (0,1), (0,2), None, None,
        #      (1,0), (1,1), (1,2), None, None]
        #
        #   In the above example, each tuple represents: (worker_index, item_index)
        #   having `worker_index` as well helps to use same code for both shared and non-shared queues.
        #
        #   In this case:
        #     - Worker 0 might process: (0,0), (0,2), None, (1,1), None → stops after 2 Nones
        #     - Worker 1 might process: (0,1), None, (1,0), (1,2), None → stops after 2 Nones
        #
        #   Each worker sees 2 sentinel values → termination condition is met.
        #
        # ---
        #
        # When using `non-shared queues`, each worker has its own queue and will contain only its own sentinels.

        for _ in range(self.num_downloaders):
            target_queue.put_nowait(sentinel)
