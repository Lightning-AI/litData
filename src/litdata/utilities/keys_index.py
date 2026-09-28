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

"""Parquet key / metadata sidecar for optimized datasets (Polars-backed).

Default on-disk layout::

    index.json          # includes a ``keys`` section (num_shards / sharding)
    keys/
      shard-00000.parquet
      shard-00001.parquet  # when num_shards > 1

Each shard parquet is sorted by ``key`` and contains:

* ``key`` — str or int64 (opaque entity id)
* ``index`` — int64 global sample index
* ``chunk_index`` — int32 chunk list index in ``index.json`` (optional until enriched)
* ``chunk_offset`` — int32 value passed as ``ChunkedIndex.index`` (optional until enriched)

With ``num_shards > 1``, rows are assigned with a stable hash of ``key``.
Legacy single-file ``keys.parquet`` is still readable.

Written by ``optimize(..., key_fn=...)`` / ``build_keys_index``. Used by
``dataset_update`` and ``StreamingDataset.get_by_key`` / ``dataset[str_key]``.

Remote datasets keep shards in object storage. Lookups use
``scan_parquet(s3://.../keys/shard-*.parquet)`` with predicate pushdown so only
parquet footer + matching row groups are fetched — shards are not downloaded
wholesale into the chunk cache. Shard layout is described in ``index.json``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from time import sleep
from typing import Any
from urllib import parse

from litdata.constants import (
    _DEFAULT_KEYS_NUM_SHARDS,
    _INDEX_FILENAME,
    _KEYS_DIRNAME,
    _KEYS_FILENAME,
    _KEYS_SHARD_TEMPLATE,
    _POLARS_AVAILABLE,
    _RANK_KEYS_SUFFIX,
    _SUPPORTED_PROVIDERS,
)


def _require_polars() -> Any:
    if not _POLARS_AVAILABLE:
        raise ModuleNotFoundError(
            "Polars is required for optimize(key_fn=...) / dataset_update. Install with `pip install 'polars>1.0.0'`."
        )
    import polars as pl

    return pl


def _atomic_replace(tmp_path: str, dest_path: str) -> None:
    """Replace ``dest_path`` with ``tmp_path``, retrying Windows file-lock races.

    On Windows, ``os.replace`` fails with ``PermissionError`` if another handle
    still has ``dest_path`` open (or antivirus briefly locks it). Retry with a
    short backoff instead of failing the write.
    """
    last_err: PermissionError | None = None
    for _ in range(20):
        try:
            os.replace(tmp_path, dest_path)
            return
        except PermissionError as e:
            last_err = e
            sleep(0.05)
    assert last_err is not None
    raise last_err


def normalize_key(key: Any) -> str | int:
    """Normalize a user key to str or int."""
    if isinstance(key, bool) or key is None:
        raise TypeError(f"Unsupported key type: {type(key)!r} ({key!r})")
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        return key
    if isinstance(key, (bytes, bytearray, memoryview)):
        return bytes(key).decode("utf-8")
    try:
        import numpy as np

        if isinstance(key, np.integer):
            return int(key)
    except ImportError:
        pass
    raise TypeError(f"Unsupported key type: {type(key)!r}. Use str or int.")


def _natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def key_shard(key: Any, num_shards: int) -> int:
    """Stable shard id for ``key`` in ``[0, num_shards)``."""
    if num_shards <= 1:
        return 0
    nkey = normalize_key(key)
    digest = hashlib.blake2b(str(nkey).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % num_shards


def _is_remote_uri(path: str) -> bool:
    return parse.urlparse(path).scheme in _SUPPORTED_PROVIDERS


def _join_uri(base: str, *parts: str) -> str:
    if _is_remote_uri(base):
        return "/".join([base.rstrip("/")] + [p.strip("/") for p in parts])
    return os.path.join(base, *parts)


def keys_dir(dataset_dir: str) -> str:
    return _join_uri(dataset_dir, _KEYS_DIRNAME)


def shard_filename(shard: int) -> str:
    return _KEYS_SHARD_TEMPLATE.format(shard)


def shard_path(dataset_dir: str, shard: int = 0) -> str:
    return _join_uri(keys_dir(dataset_dir), shard_filename(shard))


def keys_path(dataset_dir: str) -> str:
    """Path/URI to the default shard (or legacy file if that is all that exists)."""
    default_shard = shard_path(dataset_dir, 0)
    if _is_remote_uri(dataset_dir):
        return default_shard
    if os.path.isfile(default_shard) or os.path.isdir(keys_dir(dataset_dir)):
        return default_shard
    legacy = os.path.join(dataset_dir, _KEYS_FILENAME)
    if os.path.isfile(legacy):
        return legacy
    return default_shard


def has_keys_index(dataset_dir: str, storage_options: dict[str, Any] | None = None) -> bool:
    if _is_remote_uri(dataset_dir):
        from litdata.streaming.fs_provider import _get_fs_provider

        fs = _get_fs_provider(dataset_dir, storage_options)
        return fs.exists(shard_path(dataset_dir, 0)) or fs.exists(_join_uri(dataset_dir, _KEYS_FILENAME))
    return os.path.isfile(shard_path(dataset_dir, 0)) or os.path.isfile(os.path.join(dataset_dir, _KEYS_FILENAME))


def keys_config(num_shards: int, sharding: str = "hash") -> dict[str, Any]:
    """Build the ``keys`` section stored inside ``index.json``."""
    return {
        "version": 1,
        "num_shards": int(num_shards),
        "sharding": sharding if num_shards > 1 else "none",
    }


def set_keys_config_in_index(
    dataset_dir: str,
    num_shards: int,
    sharding: str = "hash",
) -> None:
    """Write / update the ``keys`` section on a local ``index.json``."""
    index_path = os.path.join(dataset_dir, _INDEX_FILENAME)
    if not os.path.isfile(index_path):
        return
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)
    data["keys"] = keys_config(num_shards, sharding=sharding)
    tmp = f"{index_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True)
    _atomic_replace(tmp, index_path)


def read_keys_config(dataset_dir: str, storage_options: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Read the ``keys`` section from ``index.json`` (local or remote)."""
    index_path = _join_uri(dataset_dir, _INDEX_FILENAME)
    if _is_remote_uri(dataset_dir):
        from litdata.streaming.fs_provider import _get_fs_provider

        fs = _get_fs_provider(dataset_dir, storage_options)
        if not fs.exists(index_path):
            return None
        data = _read_remote_json(index_path, storage_options)
        cfg = data.get("keys")
        return cfg if isinstance(cfg, dict) else None

    if not os.path.isfile(index_path):
        return None
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("keys")
    return cfg if isinstance(cfg, dict) else None


def _infer_keys_config_from_shards(dataset_dir: str) -> dict[str, Any]:
    shards = list_shard_files(keys_dir(dataset_dir))
    n = max(len(shards), 1)
    return keys_config(n, sharding="hash" if n > 1 else "none")


def list_shard_files(keys_directory: str) -> list[str]:
    if not os.path.isdir(keys_directory):
        return []
    files = [
        os.path.join(keys_directory, name)
        for name in os.listdir(keys_directory)
        if name.startswith("shard-") and name.endswith(".parquet")
    ]
    return sorted(files, key=lambda p: _natural_key(os.path.basename(p)))


def list_key_parquet_files(dataset_dir: str) -> list[str]:
    """Parquet files that make up the key index (sharded store or legacy file)."""
    shards = list_shard_files(keys_dir(dataset_dir))
    if shards:
        return shards
    legacy = os.path.join(dataset_dir, _KEYS_FILENAME)
    if os.path.isfile(legacy):
        return [legacy]
    return []


def _read_remote_json(url: str, storage_options: dict[str, Any] | None) -> dict[str, Any]:
    from litdata.streaming.fs_provider import _get_fs_provider

    fs = _get_fs_provider(url, storage_options)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        local_path = tmp.name
    try:
        fs.download_file(url, local_path)
        with open(local_path, encoding="utf-8") as f:
            return json.load(f)
    finally:
        with contextlib.suppress(OSError):
            os.remove(local_path)


def _resolve_remote_key_paths(dataset_url: str, storage_options: dict[str, Any] | None) -> tuple[list[str], int, str]:
    """Resolve remote shard URIs using ``index.json``'s ``keys`` section."""
    from litdata.streaming.fs_provider import _get_fs_provider

    fs = _get_fs_provider(dataset_url, storage_options)
    shard0_url = shard_path(dataset_url, 0)
    legacy_url = _join_uri(dataset_url, _KEYS_FILENAME)

    if fs.exists(shard0_url):
        cfg = read_keys_config(dataset_url, storage_options) or {"num_shards": 1, "sharding": "none"}
        num_shards = int(cfg.get("num_shards", 1))
        sharding = str(cfg.get("sharding", "hash" if num_shards > 1 else "none"))
        paths = [shard_path(dataset_url, i) for i in range(num_shards)]
        return paths, num_shards, sharding

    if fs.exists(legacy_url):
        return [legacy_url], 1, "none"

    raise FileNotFoundError(f"No key index under {dataset_url!r} (expected {_KEYS_DIRNAME}/ or {_KEYS_FILENAME})")


def _resolve_key_paths(path: str, storage_options: dict[str, Any] | None = None) -> tuple[list[str], int, str]:
    """Resolve a dataset dir/URL, ``keys/`` dir, or parquet file into shard paths/URIs."""
    if _is_remote_uri(path):
        # Remote parquet file directly, or dataset / keys prefix.
        if path.rstrip("/").endswith(".parquet"):
            return [path], 1, "none"
        if path.rstrip("/").endswith(f"/{_KEYS_DIRNAME}") or path.rstrip("/").endswith(_KEYS_DIRNAME):
            # keys/ URL without dataset root — read manifest beside shards
            dataset_url = path.rstrip("/").rsplit("/", 1)[0]
            return _resolve_remote_key_paths(dataset_url, storage_options)
        return _resolve_remote_key_paths(path, storage_options)

    if os.path.isfile(path):
        return [path], 1, "none"

    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    # Dataset directory or a keys/ directory with shards.
    direct_shards = list_shard_files(path)
    if direct_shards:
        keys_directory = path
        dataset_root = os.path.dirname(path) if os.path.basename(path.rstrip(os.sep)) == _KEYS_DIRNAME else path
    else:
        nested = keys_dir(path)
        nested_shards = list_shard_files(nested)
        if nested_shards:
            keys_directory = nested
            dataset_root = path
        else:
            legacy = os.path.join(path, _KEYS_FILENAME)
            if os.path.isfile(legacy):
                return [legacy], 1, "none"
            raise FileNotFoundError(f"No key index under {path!r} (expected {_KEYS_DIRNAME}/ or {_KEYS_FILENAME})")

    shards = list_shard_files(keys_directory)
    if not shards:
        raise FileNotFoundError(f"No shard-*.parquet files in {keys_directory}")
    cfg = read_keys_config(dataset_root, storage_options) or _infer_keys_config_from_shards(dataset_root)
    num_shards = int(cfg.get("num_shards", len(shards)))
    sharding = str(cfg.get("sharding", "hash" if num_shards > 1 else "none"))
    return shards, num_shards, sharding


def _polars_cloud_paths_and_options(
    paths: Sequence[str], storage_options: dict[str, Any] | None
) -> tuple[list[str], dict[str, Any] | None]:
    """Adapt URIs/credentials for Polars ``scan_parquet`` (object_store)."""
    if not paths or not _is_remote_uri(paths[0]):
        return list(paths), None

    opts = dict(storage_options or {})
    scheme = parse.urlparse(paths[0]).scheme

    # R2: Polars talks S3 protocol; resolve Lightning data-connection creds if needed.
    if scheme == "r2":
        if opts.get("data_connection_id") and not opts.get("aws_access_key_id"):
            from litdata.streaming.client import R2Client

            client = R2Client(storage_options=opts)
            creds = client.get_r2_bucket_credentials(opts["data_connection_id"])
            opts = {k: v for k, v in opts.items() if k != "data_connection_id"}
            opts.update(creds)
        paths = ["s3://" + p[len("r2://") :] for p in paths]

    # Normalize endpoint key for object_store / Polars.
    if "endpoint_url" in opts and "aws_endpoint_url" not in opts:
        opts["aws_endpoint_url"] = opts.pop("endpoint_url")
    opts.pop("data_connection_id", None)

    return list(paths), (opts or None)


def _to_dataframe(
    keys: Sequence[str | int],
    indices: Sequence[int],
    chunk_indices: Sequence[int] | None = None,
    chunk_offsets: Sequence[int] | None = None,
) -> Any:
    pl = _require_polars()
    if len(keys) != len(indices):
        raise ValueError("keys and indices must have the same length")

    if keys and all(isinstance(k, int) and not isinstance(k, bool) for k in keys):
        key_series = pl.Series("key", list(keys), dtype=pl.Int64)
    else:
        key_series = pl.Series("key", [str(k) for k in keys], dtype=pl.Utf8)

    data: dict[str, Any] = {
        "key": key_series,
        "index": pl.Series("index", list(indices), dtype=pl.Int64),
    }
    if chunk_indices is not None:
        data["chunk_index"] = pl.Series("chunk_index", list(chunk_indices), dtype=pl.Int32)
    if chunk_offsets is not None:
        data["chunk_offset"] = pl.Series("chunk_offset", list(chunk_offsets), dtype=pl.Int32)
    return pl.DataFrame(data)


def save_keys(
    path: str,
    keys: Sequence[str | int],
    indices: Sequence[int] | None = None,
    chunk_indices: Sequence[int] | None = None,
    chunk_offsets: Sequence[int] | None = None,
    *,
    sort_by_key: bool = True,
) -> None:
    """Write a single keys parquet file via Polars."""
    if indices is None:
        indices = list(range(len(keys)))
    df = _to_dataframe(keys, indices, chunk_indices, chunk_offsets)

    n_unique = df["key"].n_unique()
    if n_unique != df.height:
        raise ValueError("Duplicate keys are not allowed in the keys sidecar.")

    if sort_by_key and df.height > 0:
        df = df.sort("key")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    df.write_parquet(tmp, compression="zstd", row_group_size=min(1_000_000, max(df.height, 1)))
    _atomic_replace(tmp, path)


def write_keys_store(
    dataset_dir: str,
    keys: Sequence[str | int],
    indices: Sequence[int] | None = None,
    chunk_indices: Sequence[int] | None = None,
    chunk_offsets: Sequence[int] | None = None,
    *,
    num_shards: int = _DEFAULT_KEYS_NUM_SHARDS,
) -> str:
    """Write ``keys/shard-*.parquet`` and record layout in ``index.json`` ``keys`` section."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if indices is None:
        indices = list(range(len(keys)))
    if len(keys) != len(indices):
        raise ValueError("keys and indices must have the same length")

    df = _to_dataframe(keys, indices, chunk_indices, chunk_offsets)
    n_unique = df["key"].n_unique()
    if n_unique != df.height:
        raise ValueError("Duplicate keys are not allowed in the keys sidecar.")

    out_dir = keys_dir(dataset_dir)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if num_shards == 1:
        path = shard_path(dataset_dir, 0)
        if df.height > 0:
            df = df.sort("key")
        tmp = f"{path}.tmp"
        df.write_parquet(tmp, compression="zstd", row_group_size=min(1_000_000, max(df.height, 1)))
        _atomic_replace(tmp, path)
    else:
        shard_ids = [key_shard(k, num_shards) for k in df["key"].to_list()]
        pl = _require_polars()
        df = df.with_columns(pl.Series("_shard", shard_ids, dtype=pl.Int32))
        for shard in range(num_shards):
            part = df.filter(pl.col("_shard") == shard).drop("_shard")
            if part.height > 0:
                part = part.sort("key")
            path = shard_path(dataset_dir, shard)
            tmp = f"{path}.tmp"
            part.write_parquet(tmp, compression="zstd", row_group_size=min(1_000_000, max(part.height, 1)))
            _atomic_replace(tmp, path)

    set_keys_config_in_index(dataset_dir, num_shards=num_shards)
    # Drop legacy single-file sidecar if present so readers prefer the store.
    legacy = os.path.join(dataset_dir, _KEYS_FILENAME)
    if os.path.isfile(legacy):
        os.remove(legacy)
    return out_dir


def save_rank_keys(path: str, index_key_pairs: Sequence[tuple[int, str | int]]) -> None:
    """Write a per-rank keys file from explicit ``(sample_index, key)`` pairs."""
    if not index_key_pairs:
        save_keys(path, [], indices=[])
        return
    ordered = sorted(index_key_pairs, key=lambda x: x[0])
    indices = [i for i, _ in ordered]
    keys = [k for _, k in ordered]
    if indices != list(range(len(indices))):
        raise ValueError(f"Rank key indexes must be contiguous from 0. Found {indices[:5]}... (len={len(indices)})")
    save_keys(path, keys, indices=indices, sort_by_key=False)


def _chunk_columns_for_indices(
    global_indices: Sequence[int], index_json: dict[str, Any]
) -> tuple[list[int], list[int]]:
    chunk_starts: list[int] = []
    start = 0
    for chunk in index_json["chunks"]:
        chunk_starts.append(start)
        start += int(chunk["chunk_size"])

    chunk_indices: list[int] = []
    chunk_offsets: list[int] = []
    for gidx in global_indices:
        for ci, cstart in enumerate(chunk_starts):
            csize = int(index_json["chunks"][ci]["chunk_size"])
            if cstart <= gidx < cstart + csize:
                chunk_indices.append(ci)
                # For default (non-subsampled) layouts this matches
                # ChunksConfig._get_chunk_index_from_index's first return value.
                chunk_offsets.append(int(gidx))
                break
        else:
            raise IndexError(f"Sample index {gidx} out of range for chunk layout")
    return chunk_indices, chunk_offsets


def enrich_keys_with_chunks(path: str, index_json: dict[str, Any]) -> None:
    """Add ``chunk_index`` / ``chunk_offset`` columns to a parquet file or keys store."""
    pl = _require_polars()

    # Keys store directory (dataset dir or keys/)
    if os.path.isdir(path):
        try:
            paths, num_shards, _sharding = _resolve_key_paths(path)
        except FileNotFoundError:
            paths, num_shards = [], _DEFAULT_KEYS_NUM_SHARDS
        if not paths:
            return
        # Re-load rows, enrich, rewrite store under the dataset dir.
        dataset_dir = path if os.path.basename(path.rstrip(os.sep)) != _KEYS_DIRNAME else os.path.dirname(path)
        df = pl.concat([pl.read_parquet(p) for p in paths])
        chunk_indices, chunk_offsets = _chunk_columns_for_indices(df["index"].to_list(), index_json)
        write_keys_store(
            dataset_dir,
            df["key"].to_list(),
            indices=df["index"].to_list(),
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            num_shards=num_shards,
        )
        return

    df = pl.read_parquet(path)
    chunk_indices, chunk_offsets = _chunk_columns_for_indices(df["index"].to_list(), index_json)
    save_keys(
        path,
        df["key"].to_list(),
        indices=df["index"].to_list(),
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )


class KeyIndex:
    """Lazy Parquet key lookup — no full in-memory key map.

    ``path`` may be a local dataset directory, a ``keys/`` directory, a parquet
    file, or a remote dataset URL (``s3://``, ``gs://``, ``r2://``).

    Lookups use ``scan_parquet`` + predicate pushdown (and hash routing when
    sharded). For remote URLs this fetches only parquet metadata / matching row
    groups — not the whole shard into the local chunk cache.
    """

    def __init__(self, path: str, storage_options: dict[str, Any] | None = None) -> None:
        _require_polars()
        self.path = path
        self._storage_options = storage_options
        self._paths, self._num_shards, self._sharding = _resolve_key_paths(path, storage_options)
        schema_names = list(self._scan(self._paths[0]).collect_schema().names())
        self._has_chunks = "chunk_index" in schema_names and "chunk_offset" in schema_names
        self._count: int | None = None

    def close(self) -> None:
        return None

    def __enter__(self) -> KeyIndex:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _scan(self, paths: str | Sequence[str] | None = None) -> Any:
        pl = _require_polars()
        target = self._paths if paths is None else paths
        target_list = [target] if isinstance(target, str) else list(target)
        cloud_paths, cloud_opts = _polars_cloud_paths_and_options(target_list, self._storage_options)
        if cloud_opts is not None:
            return pl.scan_parquet(cloud_paths if len(cloud_paths) > 1 else cloud_paths[0], storage_options=cloud_opts)
        return pl.scan_parquet(cloud_paths if len(cloud_paths) > 1 else cloud_paths[0])

    def _paths_for_keys(self, keys: Sequence[Any]) -> list[str]:
        if self._num_shards <= 1 or self._sharding == "none":
            return self._paths
        by_shard: dict[int, None] = {}
        for key in keys:
            by_shard[key_shard(key, self._num_shards)] = None
        return [self._paths[s] for s in sorted(by_shard) if s < len(self._paths)]

    def __len__(self) -> int:
        if self._count is None:
            pl = _require_polars()
            self._count = int(self._scan().select(pl.len()).collect().item())
        return self._count

    def __contains__(self, key: Any) -> bool:
        return self.get(key) is not None

    def _lookup_rows(self, keys: Sequence[Any]) -> Any:
        """Fetch rows for the given keys."""
        pl = _require_polars()
        nkeys = [normalize_key(k) for k in keys]
        cols = ["key", "index"]
        if self._has_chunks:
            cols.extend(["chunk_index", "chunk_offset"])
        paths = self._paths_for_keys(nkeys)
        return self._scan(paths).filter(pl.col("key").is_in(nkeys)).select(cols).collect()

    def get(self, key: Any, default: int | None = None) -> int | None:
        df = self._lookup_rows([key])
        if df.height == 0:
            return default
        return int(df["index"][0])

    def __getitem__(self, key: Any) -> int:
        idx = self.get(key)
        if idx is None:
            raise KeyError(key)
        return idx

    def resolve(self, key: Any) -> tuple[int, int, int]:
        """Return ``(global_index, chunk_index, chunk_offset)`` for ``key``."""
        df = self._lookup_rows([key])
        if df.height == 0:
            raise KeyError(normalize_key(key))
        gidx = int(df["index"][0])
        if self._has_chunks:
            return gidx, int(df["chunk_index"][0]), int(df["chunk_offset"][0])
        return gidx, -1, -1

    def resolve_many(self, keys: Sequence[Any]) -> dict[Any, tuple[int, int, int]]:
        """Batch resolve keys (routed per shard when sharded)."""
        if not keys:
            return {}
        nkeys = [normalize_key(k) for k in keys]
        if self._num_shards <= 1 or self._sharding == "none":
            df = self._lookup_rows(nkeys)
        else:
            pl = _require_polars()
            grouped: dict[int, list[Any]] = defaultdict(list)
            for key in nkeys:
                grouped[key_shard(key, self._num_shards)].append(key)
            cols = ["key", "index"]
            if self._has_chunks:
                cols.extend(["chunk_index", "chunk_offset"])
            frames = []
            for shard, shard_keys in grouped.items():
                if shard >= len(self._paths):
                    continue
                frames.append(
                    self._scan(self._paths[shard]).filter(pl.col("key").is_in(shard_keys)).select(cols).collect()
                )
            df = pl.concat(frames) if frames else pl.DataFrame({c: [] for c in cols})

        out: dict[Any, tuple[int, int, int]] = {}
        for row in df.iter_rows(named=True):
            key = row["key"]
            if self._has_chunks:
                out[key] = (int(row["index"]), int(row["chunk_index"]), int(row["chunk_offset"]))
            else:
                out[key] = (int(row["index"]), -1, -1)
        return out

    def key_at(self, index: int) -> str | int:
        """Return the key for a global sample index (filter — avoid on hot paths)."""
        pl = _require_polars()
        matched = self._scan().filter(pl.col("index") == index).select("key").collect()
        if matched.height == 0:
            raise IndexError(index)
        return matched["key"][0]

    def keys(self) -> Iterator[str | int]:
        """Stream keys from parquet. Avoid on huge sidecars — materializes column."""
        yield from self._scan().select("key").collect().to_series().to_list()

    def indices(self) -> list[int]:
        """Load all global indexes. Avoid on huge sidecars."""
        return self._scan().select("index").collect().to_series().to_list()


def load_keys(path: str) -> list[str | int]:
    """Load all keys into a list (tests / small datasets only — not for millions of keys)."""
    with KeyIndex(path) as idx:
        return list(idx.keys())


def load_key_index(path: str) -> KeyIndex:
    return KeyIndex(path)


def merge_rank_key_files(
    cache_dir: str,
    output_filename: str | None = None,
    *,
    num_shards: int = _DEFAULT_KEYS_NUM_SHARDS,
) -> str | None:
    """Merge ``{rank}.keys.parquet`` in rank order.

    By default writes the sharded store under ``cache_dir/keys/``.
    When ``output_filename`` is set (multi-node partials), writes a single flat
    parquet at ``cache_dir/output_filename`` instead.
    """
    pl = _require_polars()
    files = [f for f in os.listdir(cache_dir) if f.endswith(_RANK_KEYS_SUFFIX)]
    if not files:
        return None

    frames = []
    global_base = 0
    for filename in sorted(files, key=_natural_key):
        filepath = os.path.join(cache_dir, filename)
        df = pl.read_parquet(filepath)
        os.remove(filepath)
        if df.height == 0:
            continue
        df = df.with_columns((pl.col("index") + global_base).alias("index"))
        global_base += df.height
        frames.append(df.select(["key", "index"]))

    merged = pl.concat(frames) if frames else pl.DataFrame({"key": [], "index": []})
    keys = merged["key"].to_list() if merged.height else []
    indices = merged["index"].to_list() if merged.height else []

    if output_filename is not None:
        out_path = os.path.join(cache_dir, output_filename)
        save_keys(out_path, keys, indices=indices, sort_by_key=True)
        return out_path

    return write_keys_store(cache_dir, keys, indices=indices, num_shards=num_shards)


def concatenate_key_files(
    paths: Sequence[str],
    dataset_dir: str,
    *,
    num_shards: int = _DEFAULT_KEYS_NUM_SHARDS,
) -> str:
    """Concatenate key parquet files into ``dataset_dir/keys/``."""
    pl = _require_polars()
    frames = []
    base = 0
    for filepath in paths:
        df = pl.read_parquet(filepath).select(["key", "index"])
        if df.height == 0:
            continue
        local_min = int(df["index"].min())
        df = df.with_columns((pl.col("index") - local_min + base).alias("index"))
        base += df.height
        frames.append(df)

    merged = pl.concat(frames) if frames else pl.DataFrame({"key": [], "index": []})
    return write_keys_store(
        dataset_dir,
        merged["key"].to_list() if merged.height else [],
        indices=merged["index"].to_list() if merged.height else [],
        num_shards=num_shards,
    )


def iter_key_indexes(
    input_dir: str,
    key_fn: Callable[[Any], Any],
    *,
    verbose: bool = True,
) -> Iterator[tuple[str | int, int]]:
    """Yield ``(key, global_index)`` by scanning an already-optimized dataset.

    Walks chunks sequentially through :class:`~litdata.streaming.cache.Cache`
    (keeps each chunk mmap warm) instead of per-sample ``dataset[i]`` lookups.

    Does not materialize a full key→index map. Prefer :func:`build_keys_index`
    when you need a durable sidecar for ``dataset_update`` / keyed reads.
    """
    from litdata.constants import _TQDM_AVAILABLE
    from litdata.streaming.cache import Cache
    from litdata.streaming.resolver import _resolve_dir
    from litdata.streaming.sampler import ChunkedIndex

    resolved = _resolve_dir(input_dir)
    if resolved.path is None or not os.path.isdir(resolved.path):
        raise FileNotFoundError(f"Dataset directory not found: {input_dir}")

    index_path = os.path.join(resolved.path, _INDEX_FILENAME)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Missing {_INDEX_FILENAME} in {resolved.path}. Did you run optimize()?")

    cache = Cache(resolved.path, chunk_bytes=1)
    intervals = cache.get_chunk_intervals()
    total = len(cache)

    pbar: Any = None
    if verbose and _TQDM_AVAILABLE:
        from tqdm.auto import tqdm as _tqdm

        pbar = _tqdm(total=total, desc="Building keys index", unit="sample")

    global_index = 0
    try:
        for chunk_index, interval in enumerate(intervals):
            begin = int(interval.roi_start_idx)
            end = int(interval.roi_end_idx)
            chunk_size = end - begin
            for offset in range(chunk_size):
                sample = cache[
                    ChunkedIndex(
                        index=begin + offset,
                        chunk_index=chunk_index,
                        chunk_size=chunk_size,
                    )
                ]
                yield normalize_key(key_fn(sample)), global_index
                global_index += 1
                if pbar is not None:
                    pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        # Release Windows file locks from mmap before callers rewrite chunks.
        item_loader = getattr(cache._reader, "_item_loader", None)
        close_open = getattr(item_loader, "_close_open_chunk", None)
        if callable(close_open):
            close_open()
        elif item_loader is not None and hasattr(item_loader, "close") and intervals:
            item_loader.close(len(intervals) - 1)

    if global_index != total:
        raise RuntimeError(f"Key scan produced {global_index} samples but index reports {total}.")


def build_keys_index(
    input_dir: str,
    key_fn: Callable[[Any], Any],
    *,
    output_dir: str | None = None,
    overwrite: bool = False,
    verbose: bool = True,
    num_shards: int = _DEFAULT_KEYS_NUM_SHARDS,
) -> str:
    """Scan an optimized dataset with ``key_fn`` and write ``keys/shard-*.parquet``.

    Use this to backfill a key sidecar for datasets produced without
    ``optimize(..., key_fn=...)``. Returns the path to the ``keys/`` directory.
    """
    from litdata.streaming.resolver import _resolve_dir

    resolved = _resolve_dir(input_dir)
    if resolved.path is None or not os.path.isdir(resolved.path):
        raise FileNotFoundError(f"Dataset directory not found: {input_dir}")

    dataset_dir = output_dir or resolved.path
    out = keys_dir(dataset_dir)
    if has_keys_index(dataset_dir) and not overwrite:
        raise FileExistsError(f"Key index already exists under {dataset_dir}. Pass overwrite=True to replace it.")

    keys: list[str | int] = []
    indices: list[int] = []
    for key, index in iter_key_indexes(resolved.path, key_fn, verbose=verbose):
        keys.append(key)
        indices.append(index)

    write_keys_store(dataset_dir, keys, indices=indices, num_shards=num_shards)

    # Load+close before enrich: enrich rewrites index.json via write_keys_store, and
    # Windows cannot os.replace a file that still has an open handle.
    index_path = os.path.join(resolved.path, _INDEX_FILENAME)
    with open(index_path, encoding="utf-8") as f:
        index_json = json.load(f)
    enrich_keys_with_chunks(dataset_dir, index_json)
    return out
