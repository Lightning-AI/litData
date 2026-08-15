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

"""Public helpers for finishing a dataset written outside ``optimize()``."""

from __future__ import annotations

import os

from litdata.constants import _INDEX_FILENAME
from litdata.streaming.writer import BinaryWriter, _is_node_index_file, _is_worker_index_file


def is_complete_dataset(input_dir: str) -> bool:
    """Return True when ``index.json`` exists (the dataset is ready to stream)."""
    return os.path.isfile(os.path.join(input_dir, _INDEX_FILENAME))


def complete_dataset(input_dir: str) -> str:
    """Merge worker/node index shards into ``index.json`` if needed.

    ``optimize()`` already does this. Use this when you write chunks yourself
    (``Cache`` / ``BinaryWriter``) or when a multi-process job left
    ``{rank}.index.json`` files and no merged index.

    Returns ``input_dir``. No-op if ``index.json`` is already present.
    """
    if is_complete_dataset(input_dir):
        return input_dir

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {input_dir}")

    files = os.listdir(input_dir)
    shards = [f for f in files if _is_worker_index_file(f) or _is_node_index_file(f)]
    if not shards:
        raise FileNotFoundError(
            f"The provided dataset `{input_dir}` has no {_INDEX_FILENAME} and no worker index shards "
            f"({{rank}}.{_INDEX_FILENAME}). Did optimize() finish?"
        )

    writer = BinaryWriter(cache_dir=input_dir, chunk_bytes="64MB")
    writer._merge_no_wait()
    return input_dir
