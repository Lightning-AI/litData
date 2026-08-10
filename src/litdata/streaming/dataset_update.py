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

"""In-place keyed updates for optimized LitData datasets.

Requires a ``keys/`` sidecar (``keys/shard-*.parquet``) produced by
``optimize(..., key_fn=...)`` or ``build_keys_index``.
Only local dataset directories are supported in v1.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from time import time
from typing import Any

from litdata.constants import _INDEX_FILENAME, _KEYS_DIRNAME
from litdata.streaming.cache import Cache
from litdata.streaming.resolver import _resolve_dir
from litdata.streaming.writer import BinaryWriter
from litdata.utilities.keys_index import (
    KeyIndex,
    _atomic_replace,
    has_keys_index,
    normalize_key,
)

_CHUNK_NAME_RE = re.compile(
    r"^chunk-(?P<rank>\d+)-(?P<chunk_index>\d+)"
    r"(?:-u(?P<update>\d+))?"
    r"(?:\.(?P<compression>[^.]+))?\.bin$"
)


def dataset_update(input_dir: str) -> DatasetUpdate:
    """Open a keyed update session for an optimized dataset.

    Example::

        with dataset_update("optimized_data") as update:
            update["sample-id"] = {"x": 1, "y": 2}
            update.commit()
    """
    return DatasetUpdate(input_dir)


class DatasetUpdate:
    """Session-style context manager for keyed sample replaces.

    Stage changes with ``update[key] = sample``, then call :meth:`commit`.
    Exiting the context without ``commit()`` discards pending changes.
    After a successful commit, the session is closed for further writes.
    """

    def __init__(self, input_dir: str) -> None:
        resolved = _resolve_dir(input_dir)
        # Studio ``lightning_storage`` paths often resolve with both a local FUSE
        # ``path`` and a remote ``url``. Prefer the local path so commits write
        # through the mount. Pure remote URLs (no local dir) are not supported yet.
        if resolved.path is None or not os.path.isdir(resolved.path):
            if resolved.url is not None:
                raise NotImplementedError(
                    "dataset_update() requires a local dataset directory "
                    "(or a lightning_storage FUSE path). "
                    f"Got remote-only url={resolved.url!r}."
                )
            raise FileNotFoundError(f"Dataset directory not found: {input_dir}")

        self._dir = resolved.path
        index_path = os.path.join(self._dir, _INDEX_FILENAME)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Missing {_INDEX_FILENAME} in {self._dir}. Did you run optimize()?")
        if not has_keys_index(self._dir):
            raise FileNotFoundError(
                f"Missing {_KEYS_DIRNAME}/ key index in {self._dir}. "
                "Re-run optimize(..., key_fn=...) or call build_keys_index(dir, key_fn)."
            )

        with open(index_path, encoding="utf-8") as f:
            self._index: dict[str, Any] = json.load(f)
        self._key_index = KeyIndex(self._dir)
        self._pending: dict[str | int, Any] = {}
        self._entered = False
        self._committed = False

        # Build global_index → (chunk_list_index, local_offset) and chunk intervals
        self._chunk_starts: list[int] = []
        start = 0
        for chunk in self._index["chunks"]:
            self._chunk_starts.append(start)
            start += int(chunk["chunk_size"])
        self._length = start

    def __enter__(self) -> DatasetUpdate:
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Like SQLAlchemy Session: no implicit commit — uncommitted work is dropped.
        try:
            self._pending.clear()
        finally:
            self._entered = False
            self._key_index.close()

    def __setitem__(self, key: Any, sample: Any) -> None:
        self.replace(key, sample)

    def replace(self, key: Any, sample: Any) -> None:
        if self._committed:
            raise RuntimeError("Cannot modify dataset_update after commit().")
        # Existence is validated on commit via a single batched parquet scan.
        self._pending[normalize_key(key)] = sample

    def commit(self) -> None:
        """Persist pending keyed replaces. Further modifications are rejected."""
        if self._committed:
            raise RuntimeError("dataset_update session already committed.")
        if not self._pending:
            self._committed = True
            return

        resolved = self._key_index.resolve_many(list(self._pending.keys()))
        missing = [k for k in self._pending if k not in resolved]
        if missing:
            raise KeyError(f"Unknown dataset key: {missing[0]!r}")

        # Group pending updates by chunk index in index["chunks"]
        updates_by_chunk: dict[int, dict[int, Any]] = defaultdict(dict)
        for key, sample in self._pending.items():
            global_index, chunk_i, _chunk_off = resolved[key]
            if chunk_i < 0:
                chunk_i, local_i = self._locate(global_index)
            else:
                local_i = global_index - self._chunk_starts[chunk_i]
            updates_by_chunk[chunk_i][local_i] = sample

        config = self._index["config"]
        compression = config.get("compression")

        for chunk_i, local_updates in updates_by_chunk.items():
            self._rewrite_chunk(chunk_i, local_updates, compression, config)

        self._index["updated_at"] = str(time())
        index_path = os.path.join(self._dir, _INDEX_FILENAME)
        tmp_index = f"{index_path}.tmp"
        with open(tmp_index, "w", encoding="utf-8") as f:
            json.dump(self._index, f, sort_keys=True)
        _atomic_replace(tmp_index, index_path)

        # keys/ store unchanged for whole-sample replace (same key set / indices)
        self._pending.clear()
        self._committed = True
        self._sync_studio_read_cache()

    def _sync_studio_read_cache(self) -> None:
        """Copy the updated dataset into the Studio chunk cache.

        Paths under ``/teamspace/lightning_storage/`` (and similar) are rewritten by
        ``StreamingDataset`` to ``/cache/chunks/...``, which is filled from the remote
        URL. FUSE writes are not always visible on R2 immediately, so without this
        sync a post-commit read can still see stale chunks.
        """
        from litdata.utilities.dataset_utilities import _should_replace_path, _try_create_cache_dir

        if not _should_replace_path(self._dir):
            return
        cache_path = _try_create_cache_dir(self._dir)
        if cache_path is None:
            return

        for name in os.listdir(self._dir):
            src = os.path.join(self._dir, name)
            dst = os.path.join(cache_path, name)
            if name == "keys" and os.path.isdir(src):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                shutil.copy2(src, dst)

    def _locate(self, global_index: int) -> tuple[int, int]:
        if global_index < 0 or global_index >= self._length:
            raise IndexError(global_index)
        # linear scan is fine for typical chunk counts; chunks are contiguous
        for chunk_i, start in enumerate(self._chunk_starts):
            size = int(self._index["chunks"][chunk_i]["chunk_size"])
            if start <= global_index < start + size:
                return chunk_i, global_index - start
        raise IndexError(global_index)

    def _rewrite_chunk(
        self,
        chunk_i: int,
        local_updates: dict[int, Any],
        compression: str | None,
        config: dict[str, Any],
    ) -> None:
        chunk_info = self._index["chunks"][chunk_i]
        filename = chunk_info["filename"]
        match = _CHUNK_NAME_RE.match(filename)
        if not match:
            raise ValueError(f"Unrecognized chunk filename: {filename}")
        rank = int(match.group("rank"))
        chunk_index = int(match.group("chunk_index"))
        chunk_size = int(chunk_info["chunk_size"])
        global_start = self._chunk_starts[chunk_i]

        cache = Cache(self._dir, chunk_bytes=1)
        samples: list[Any]
        try:
            samples = [cache[global_start + i] for i in range(chunk_size)]
        finally:
            # Windows cannot replace a chunk that still has an open mmap/handle.
            item_loader = getattr(cache._reader, "_item_loader", None)
            close_open = getattr(item_loader, "_close_open_chunk", None)
            if callable(close_open):
                close_open()
            elif item_loader is not None and hasattr(item_loader, "close"):
                item_loader.close(chunk_i)
            del cache

        for local_i, sample in local_updates.items():
            samples[local_i] = sample

        tmp_dir = tempfile.mkdtemp(prefix="litdata-update-")
        try:
            writer = BinaryWriter(
                tmp_dir,
                chunk_size=len(samples),
                compression=compression,
                chunk_index=chunk_index,
            )
            writer._rank = rank

            for i, sample in enumerate(samples):
                writer.add_item(i, sample)
            writer.done()

            # Ensure rewritten samples stay schema-compatible with the dataset.
            new_config = writer.get_config()
            if new_config.get("data_format") != config.get("data_format"):
                raise ValueError(
                    "Updated sample is incompatible with the dataset data_format. "
                    f"Expected {config.get('data_format')}, got {new_config.get('data_format')}."
                )

            new_chunk_path = os.path.join(tmp_dir, filename)
            if not os.path.isfile(new_chunk_path):
                # BinaryWriter may have used a slightly different name if compression unset
                produced = [f for f in os.listdir(tmp_dir) if f.startswith("chunk-") and f.endswith(".bin")]
                if len(produced) != 1:
                    raise RuntimeError(f"Expected one rewritten chunk, found {produced} in {tmp_dir}")
                new_chunk_path = os.path.join(tmp_dir, produced[0])
                filename = produced[0]

            dest = os.path.join(self._dir, filename)
            tmp_dest = dest + ".tmp"
            shutil.copyfile(new_chunk_path, tmp_dest)
            try:
                _atomic_replace(tmp_dest, dest)
            except PermissionError:
                # Destination still locked (e.g. another StreamingDataset mmap on Windows).
                # Publish under a new filename and retarget the index entry.
                compression_part = match.group("compression")
                update_token = int(time() * 1000) % 1_000_000_000
                if compression_part:
                    filename = f"chunk-{rank}-{chunk_index}-u{update_token}.{compression_part}.bin"
                else:
                    filename = f"chunk-{rank}-{chunk_index}-u{update_token}.bin"
                dest = os.path.join(self._dir, filename)
                _atomic_replace(tmp_dest, dest)
                old_path = os.path.join(self._dir, chunk_info["filename"])
                with contextlib.suppress(OSError, PermissionError):
                    if old_path != dest and os.path.isfile(old_path):
                        os.remove(old_path)

            new_size = os.path.getsize(dest)
            chunk_info["filename"] = filename
            chunk_info["chunk_bytes"] = new_size
            chunk_info["chunk_size"] = len(samples)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
