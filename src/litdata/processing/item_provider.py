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
from multiprocessing import Queue
from typing import Any, Dict, List, Tuple, Union


class WorkerItemProvider:
    """Helper class for providing items to the worker."""

    def __init__(self, items: List[List[Any]], num_downloaders: int) -> None:
        self.items = items
        self.paths: Dict[int, List[List[str]]] = defaultdict(list)
        self.num_downloaders = num_downloaders
        self.ready_to_process_item: Dict[int, Queue] = defaultdict(Queue)
        self.ready_to_process_shared_queue: Queue = Queue()

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
        for index, _ in enumerate(self.items[worker_index]):
            if use_shared_queue:
                self.ready_to_process_shared_queue.put_nowait((worker_index, index))
            else:
                self.ready_to_process_item[worker_index].put_nowait((worker_index, index))

        sentinel = None

        target_queue = (
            self.ready_to_process_shared_queue if use_shared_queue else self.ready_to_process_item[worker_index]
        )

        for _ in range(self.num_downloaders):
            target_queue.put_nowait(sentinel)
