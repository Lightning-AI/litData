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
from typing import Any, Dict, List


class WorkerItemProvider:
    """Helper class for providing items to the worker."""

    def __init__(self, items: List[List[Any]]):
        self.index = 0
        self.items = items
        self.ready_to_process_item: Dict[int, Queue] = defaultdict(Queue)
        self.ready_to_process_shared_queue: Queue = Queue()

    def set_items(self, index: int, item: List[Any]) -> None:
        self.items[index] = item

    def get_items(self, index: int) -> List[Any]:
        return self.items[index]

    def prepare_ready_to_use_queue(self, use_shared: bool = False) -> None:
        for index, items in enumerate(self.items):
            for _item_index, _ in enumerate(items):
                if use_shared:
                    self.ready_to_process_shared_queue.put_nowait(_item_index)
                else:
                    self.ready_to_process_item[index].put_nowait(_item_index)
