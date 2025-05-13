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
from typing import Any, Dict, List, Tuple


class WorkerItemProvider:
    """Helper class for providing items to the worker."""

    def __init__(self, items: List[List[Any]]):
        self.index = 0
        self.items = items
        self.ready_to_process_item: Dict[int, Queue] = defaultdict(Queue)
        self.ready_to_process_shared_queue: Queue = Queue()

    def get_next(self, index: int) -> Any:
        """Get the next item from the provider.

        Returns:
            Any: The next item from the provider.
        """
        try:
            return self.ready_to_process_item[index].get()
        except KeyError:
            raise KeyError(f"Item with index {index} not found in ready_to_process_item.")
        except Exception as e:
            raise RuntimeError(f"Error while getting item from `ready_to_process_item`: {e}")

    def get_next_shared(self) -> Any:
        """Get the next item from the provider.

        Returns:
            Any: The next item from the shared queue.
        """
        try:
            return self.ready_to_process_shared_queue.get()
        except Exception as e:
            raise RuntimeError(f"Error while getting item from `ready_to_process_shared_queue`: {e}")

    def set_items(self, index: int, item: List[Any]) -> None:
        self.items[index] = item

    def get_items(self, index: int) -> List[Any]:
        return self.items[index]

    def prepare_ready_to_use_queue(self, use_shared: bool = False) -> None:
        for index, items in enumerate(self.items):
            for _item in items:
                if use_shared:
                    self.ready_to_process_shared_queue.put(_item)
                else:
                    self.ready_to_process_item[index].put(_item)


class StaticPartitionProvider(WorkerItemProvider):
    """Provides items from a static partition."""

    def get_next_item(self) -> Tuple[int, Any]:
        if self.index >= len(self.items):
            return -1, None
        item = self.items[self.index]
        self.index += 1
        return (self.index - 1, item)

    def get_item_with_path(self, index: int) -> Tuple[int, Any]:
        item_with_path = self.items_with_paths[index]
        return index, item_with_path

    def set_item_with_path(self, index: int, item_with_path: Any) -> None:
        self.items_with_paths[index] = item_with_path

    def __len__(self) -> int:
        return len(self.items)


class SharedQueueProvider(WorkerItemProvider):
    """Provides items from a shared queue."""

    def __init__(self, items: List[Any]):
        self.items = items

    def get_next_item(self) -> Any:
        raise NotImplementedError("SharedQueueProvider does not support get_next_item() method.")
