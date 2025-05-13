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
from multiprocessing import Queue
from typing import Any


class WorkerItemProvider(ABC):
    """Abstract base class for providing items to the worker.
    This class is used to define a common interface for different item providers.
    """

    @abstractmethod
    def get_next_item(self) -> Any:
        pass


class StaticPartitionProvider(WorkerItemProvider):
    """Provides items from a static partition."""

    def __init__(self, items: list):
        self.items = items
        self.index = 0

    def get_next_item(self) -> Any:
        if self.index >= len(self.items):
            return None
        item = self.items[self.index]
        self.index += 1
        return item

    def __len__(self) -> int:
        return len(self.items)


class SharedQueueProvider(WorkerItemProvider):
    """Provides items from a shared queue."""

    def __init__(self, queue: Queue):
        self.queue = queue

    def get_next_item(self) -> Any:
        try:
            return self.queue.get_nowait()
        except Exception:
            return None
