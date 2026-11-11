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

"""Public exception types for streaming / optimize I/O."""

from __future__ import annotations


class ChunkWaitTimeoutError(FileNotFoundError):
    """Timed out waiting for a chunk file to appear on disk.

    Subclasses ``FileNotFoundError`` so existing ``except FileNotFoundError``
    handlers still catch it. The message is a timeout, not a missing object.
    """

    def __init__(self, path: str, waited_s: float) -> None:
        self.path = path
        self.waited_s = waited_s
        super().__init__(
            f"Timed out after {waited_s:.0f}s waiting for chunk {path}. "
            "This is usually a hung or crashed download/decompress "
            "(PrepareChunksThread), not a missing object. Check stderr for "
            "'[litdata] PrepareChunksThread CRASHED' and enable_tracer() "
            "`crash` events. num_workers=0 succeeding while num_workers>0 "
            "fails often means obstore-after-fork or a Session TypeError."
        )
