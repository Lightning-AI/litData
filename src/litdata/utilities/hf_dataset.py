"""Contains utility functions for indexing and streaming HF datasets."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from fnmatch import fnmatch
from functools import partial
from typing import Any

from litdata.constants import _INDEX_FILENAME
from litdata.streaming.serializers import JsonLeaf
from litdata.streaming.writer import index_parquet_dataset
from litdata.types import types_from_arrow, wrap_for_pytree
from litdata.utilities.dataset_utilities import generate_md5_hash, get_default_cache_dir
from litdata.utilities.torch_utils import is_local_rank_0, maybe_barrier


def _hf_cache_root(cache_dir: str | None = None) -> str:
    """Root for persisted HF index/parquet — sibling of the chunk cache, not inside it.

    ``optimize`` deletes ``get_default_cache_dir()`` (``/cache/chunks`` on Studio).
    Parquet stored there vanished mid-convert.
    """
    if cache_dir is not None:
        return cache_dir
    env = os.getenv("LITDATA_HF_CACHE_DIR")
    if env:
        return env
    chunks = os.path.normpath(get_default_cache_dir())
    parent = os.path.dirname(chunks)
    return parent if parent and parent != os.sep else chunks


def hf_index_cache_path(dataset_url: str, cache_dir: str | None = None) -> str:
    """Stable directory for a dataset's ``index.json`` (no timestamp subdirectory)."""
    return os.path.join(_hf_cache_root(cache_dir), "hf-index", generate_md5_hash(dataset_url))


def hf_parquet_cache_path(dataset_url: str, cache_dir: str | None = None) -> str:
    """Stable directory for persisted Hub parquet files (reused across optimize / stream)."""
    return os.path.join(_hf_cache_root(cache_dir), "hf-parquet", generate_md5_hash(dataset_url))


def index_hf_dataset(dataset_url: str, cache_dir: str | None = None, storage_options: dict | None = None) -> str:
    """Indexes a Hugging Face dataset and returns the path to the cache directory.

    The index is persisted at ``{cache_dir}/hf-index/<url_hash>/index.json`` so later
    ``StreamingDataset`` / ``index_hf_dataset`` calls reuse it instead of re-scanning
    Hub footers. A pre-written ``{cache_dir}/index.json`` is also reused.

    Args:
        dataset_url (str): The URL of the Hugging Face dataset, starting with 'hf://'.
        cache_dir (Optional[str]): The directory for storing the cache and index. If None, a default location is used.
        storage_options (Optional[dict]): Passed to ``HfFileSystem`` (e.g. ``token``).

    Returns:
        str: The path to the cache directory containing the index file.

    Raises:
        ValueError: If the dataset URL does not start with 'hf://'.
    """
    if not dataset_url.startswith("hf://"):
        raise ValueError(
            f"Invalid Hugging Face dataset URL: {dataset_url}. "
            "URLs must start with 'hf://'. Please check the URL and try again."
        )

    cache_directory = _get_existing_cache(dataset_url, cache_dir)
    if cache_directory:
        if is_local_rank_0():
            print(f"Using existing index at {cache_directory}.")
        return cache_directory

    maybe_barrier()

    dest = hf_index_cache_path(dataset_url, cache_dir)
    os.makedirs(dest, exist_ok=True)
    dest_index = os.path.join(dest, _INDEX_FILENAME)
    if os.path.isfile(dest_index):
        return dest

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_index_path = os.path.join(temp_dir, _INDEX_FILENAME)
        index_parquet_dataset(
            dataset_url, temp_dir, storage_options=storage_options or {}, num_workers=os.cpu_count() or 4
        )

        if is_local_rank_0():
            print(f"Creating cache directory at {dest}.")
            shutil.copyfile(temp_index_path, dest_index)
            print(f"Index created at {dest_index}.")

    maybe_barrier()

    return dest


def _optimize_hf_output_has_index(output_dir: str, storage_options: dict | None = None) -> bool:
    """True when ``output_dir`` already has ``index.json`` (local or remote)."""
    from litdata.processing.utilities import construct_storage_options, read_index_file_content
    from litdata.streaming.resolver import _resolve_dir

    resolved = _resolve_dir(output_dir)
    merged = construct_storage_options(storage_options or {}, resolved)
    return read_index_file_content(resolved, merged) is not None


def optimize_hf(
    name: str,
    output_dir: str,
    revision: str | None = None,
    split: str | None = None,
    config: str | None = None,
    chunk_size: int | None = None,
    chunk_bytes: int | str | None = None,
    fn: Callable[[dict[str, Any]], Any] | None = None,
    num_workers: int | None = None,
    columns: list[str] | None = None,
    overwrite: bool = False,
    cache_dir: str | None = None,
    storage_options: dict | None = None,
    **optimize_kwargs: Any,
) -> str:
    """Convert a Hugging Face dataset into LitData chunks (``chunk-*.bin`` + ``index.json``).

    1. Resolve Hub parquet (native layout or ``refs/convert/parquet``).
    2. Index once and persist ``index.json`` under ``{hf_cache}/hf-index/<hash>/``.
    3. Download each parquet into ``{hf_cache}/hf-parquet/<hash>/`` (skip if the size matches).
       That tree is a *sibling* of the chunk cache so ``optimize`` cannot delete it.
    4. ``optimize`` from ``hf://`` URLs (workers re-download if a persist file is missing)
       into **256MB** chunks unless ``chunk_size`` / ``chunk_bytes`` is set.
    5. Variable-length lists/dicts are stored as one JSON leaf so samples keep a
       stable ``data_format`` (SQuAD answers, UltraChat messages, …). Hub
       ``{bytes, path}`` media stays Arrow binary (no ``Image`` / ``Audio`` /
       ``Video`` wrap).

    Later calls reuse ``output_dir`` when it already has ``index.json``
    (local path, ``s3://`` / ``gs://`` / ``r2://``, or lightning storage).

    Example::

        ld.optimize_hf("stanfordnlp/imdb", output_dir="imdb-opt", split="train")
        ds = ld.StreamingDataset("imdb-opt")

    Args:
        name: Hub id (``org/name``) or an explicit ``hf://...`` URL.
        output_dir: Directory for optimized chunks (local, s3/gs/r2, or
            ``/teamspace/lightning_storage/...``).
        revision: Git revision (``main``, ``refs/convert/parquet``, a commit). Default: auto.
        split: Optional split (``train``, ``train_sft``, …). Omit to convert every parquet found.
        config: Optional dataset config / subset (``plain_text``, ``main``, …).
        chunk_size: Samples per chunk. Prefer ``chunk_bytes`` (256MB) for remote I/O.
        chunk_bytes: Chunk size in bytes (default ``256MB`` when ``chunk_size`` is omitted).
        fn: Optional picklable ``row dict -> sample``. Default writes the row as-is.
        num_workers: Optimize workers. Default is ``min(n_files, cpu count)``.
        columns: Optional parquet column projection.
        overwrite: Rebuild even if ``output_dir/index.json`` exists.
        cache_dir: Root for the persisted HF index and parquet files. Default is
            a sibling of the chunk cache (``/cache/hf-*`` on Studio, not ``/cache/chunks``).
        storage_options: Hub token / extras (or set ``HF_TOKEN``).
        **optimize_kwargs: Extra ``optimize`` kwargs (``compression``, ``mode``, …).

    Returns:
        ``output_dir``.
    """
    if not overwrite and _optimize_hf_output_has_index(output_dir, storage_options):
        if is_local_rank_0():
            print(f"Using existing optimized dataset at {output_dir}.")
        return output_dir

    dataset_url = resolve_hf_dataset_url(
        name, revision=revision, split=split, config=config, storage_options=storage_options
    )
    index_url, _pattern = _index_url_and_pattern(dataset_url)
    persist_dir = hf_parquet_cache_path(index_url, cache_dir)
    inputs = _prepare_optimize_inputs(dataset_url, cache_dir, storage_options)
    if not inputs:
        raise FileNotFoundError(f"No parquet files found for {name!r} (resolved {dataset_url}).")

    from litdata.processing.functions import optimize

    workers = num_workers if num_workers is not None else min(len(inputs), os.cpu_count() or 4)
    kwargs = dict(optimize_kwargs)
    kwargs.setdefault("mode", "overwrite" if overwrite else None)
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    else:
        kwargs["chunk_bytes"] = chunk_bytes if chunk_bytes is not None else "256MB"

    optimize(
        fn=partial(
            _optimize_hf_file,
            columns=columns,
            transform=fn,
            storage_options=storage_options,
            persist_dir=persist_dir,
            index_url=index_url,
        ),
        inputs=inputs,
        output_dir=output_dir,
        num_workers=workers,
        storage_options=storage_options or {},
        **kwargs,
    )
    return output_dir


def resolve_hf_dataset_url(
    name: str,
    revision: str | None = None,
    split: str | None = None,
    config: str | None = None,
    storage_options: dict | None = None,
) -> str:
    """Pick an ``hf://`` parquet prefix for a Hub dataset id."""
    if name.startswith("hf://"):
        return name

    repo = name.removeprefix("datasets/")
    candidates: list[str] = []
    if revision:
        prefix = f"hf://datasets/{repo}@{revision}"
        if config and split:
            candidates.append(f"{prefix}/{config}/{split}")
        if split:
            candidates.append(f"{prefix}/{split}")
        if config:
            candidates.append(f"{prefix}/{config}")
        candidates.append(prefix)

    configs = [c for c in (config, "default", "plain_text") if c]
    for cfg in configs:
        if split:
            candidates.append(f"hf://datasets/{repo}@refs/convert/parquet/{cfg}/{split}")
        candidates.append(f"hf://datasets/{repo}@refs/convert/parquet/{cfg}")
    if split:
        candidates.extend(
            [
                f"hf://datasets/{repo}/data/{split}-*.parquet",
                f"hf://datasets/{repo}/data/{split}*.parquet",
                f"hf://datasets/{repo}/{split}-*.parquet",
            ]
        )
    candidates.extend(
        [
            f"hf://datasets/{repo}@refs/convert/parquet",
            f"hf://datasets/{repo}/data",
            f"hf://datasets/{repo}",
        ]
    )

    seen: set[str] = set()
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        if _list_hf_parquet_urls(url, storage_options):
            return url

    raise FileNotFoundError(
        f"No parquet files found for dataset {name!r} (revision={revision!r}, split={split!r}, config={config!r})."
    )


def _prepare_optimize_inputs(
    dataset_url: str,
    cache_dir: str | None,
    storage_options: dict | None,
) -> list[str]:
    """Index + persist parquet, return ``hf://`` URLs so ``optimize`` does not remap local files."""
    index_url, pattern = _index_url_and_pattern(dataset_url)
    index_dir = index_hf_dataset(index_url, cache_dir, storage_options)
    chunks = _chunks_from_index(index_dir, pattern)
    if not chunks:
        return _list_hf_parquet_urls(dataset_url, storage_options)
    _persist_hf_parquet_files(index_url, chunks, cache_dir, storage_options)
    return [f"{index_url.rstrip('/')}/{chunk['filename']}" for chunk in chunks]


def _index_url_and_pattern(dataset_url: str) -> tuple[str, str | None]:
    base = os.path.basename(dataset_url)
    if base.endswith(".parquet"):
        return dataset_url.rsplit("/", 1)[0], base
    return dataset_url, None


def _chunks_from_index(index_dir: str, pattern: str | None) -> list[dict[str, Any]]:
    from litdata.utilities.dataset_utilities import load_index_file

    data = load_index_file(index_dir)
    chunks = list(data.get("chunks") or [])
    if pattern is None:
        return chunks
    return [
        chunk
        for chunk in chunks
        if fnmatch(chunk.get("filename", ""), pattern) or fnmatch(os.path.basename(chunk.get("filename", "")), pattern)
    ]


def _persist_hf_parquet_files(
    index_url: str,
    chunks: list[dict[str, Any]],
    cache_dir: str | None,
    storage_options: dict | None,
) -> list[str]:
    """Download indexed parquet files into ``hf-parquet/<hash>/``, skipping complete copies."""
    from litdata.streaming.downloader import get_downloader

    dest_root = hf_parquet_cache_path(index_url, cache_dir)
    os.makedirs(dest_root, exist_ok=True)
    downloader = None
    local_paths: list[str] = []
    for chunk in chunks:
        filename = chunk["filename"]
        local = os.path.join(dest_root, filename)
        expected = chunk.get("chunk_bytes")
        if expected and os.path.isfile(local) and os.path.getsize(local) == expected:
            local_paths.append(local)
            continue
        if downloader is None:
            downloader = get_downloader(index_url, dest_root, [], storage_options or {})
        os.makedirs(os.path.dirname(local) or dest_root, exist_ok=True)
        remote = f"{index_url.rstrip('/')}/{filename}"
        if is_local_rank_0():
            print(f"Persisting {filename} -> {local}")
        downloader.download_file(remote, local)
        local_paths.append(local)
    return local_paths


def _list_hf_parquet_urls(dataset_url: str, storage_options: dict | None) -> list[str]:
    from litdata.utilities.hf_fs import get_hf_filesystem, hf_relative_name, list_hf_parquet_files

    listed_url = dataset_url
    pattern = None
    base = os.path.basename(dataset_url)
    if base.endswith(".parquet") and any(ch in base for ch in "*?["):
        listed_url, pattern = dataset_url.rsplit("/", 1)
    elif base.endswith(".parquet"):
        return [dataset_url]

    files = list_hf_parquet_files(get_hf_filesystem(storage_options), listed_url)
    urls: list[str] = []
    for rec in files:
        name = str(rec.get("name") or "")
        if pattern and not (fnmatch(name, pattern) or fnmatch(os.path.basename(name), pattern)):
            continue
        if name.startswith("hf://"):
            urls.append(name)
        else:
            rel = hf_relative_name(listed_url, name)
            urls.append(f"{listed_url.rstrip('/')}/{rel}")
    return urls


def _persist_relpath(url: str, index_url: str | None) -> str:
    if index_url:
        prefix = index_url.rstrip("/") + "/"
        if url.startswith(prefix):
            return url[len(prefix) :]
    return os.path.basename(url)


def _materialize_hf_parquet(
    url: str,
    storage_options: dict | None = None,
    persist_dir: str | None = None,
    index_url: str | None = None,
) -> str:
    local = None
    if persist_dir:
        local = os.path.join(persist_dir, _persist_relpath(url, index_url))
        if os.path.isfile(local) and os.path.getsize(local) > 0:
            return local

    if os.path.isfile(url):
        return url

    if not url.startswith("hf://"):
        raise FileNotFoundError(f"Parquet file is gone and is not an hf:// URL: {url}")

    from huggingface_hub import hf_hub_download

    from litdata.utilities.hf_fs import hf_token, parse_hf_url

    repo_id, revision, path = parse_hf_url(url)
    downloaded = hf_hub_download(
        repo_id,
        path,
        repo_type="dataset",
        revision=revision,
        token=hf_token(storage_options),
    )
    if local:
        os.makedirs(os.path.dirname(local) or persist_dir or ".", exist_ok=True)
        tmp = f"{local}.tmp.{os.getpid()}"
        shutil.copy2(downloaded, tmp)
        os.replace(tmp, local)
        return local
    return downloaded


def _identity(value: Any) -> Any:
    return value


class _PickleLeaf:
    """Pytree leaf for nested values that are not JSON-serializable."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __reduce__(self) -> tuple[Any, tuple[Any]]:
        return (_identity, (self.value,))


def _empty_for_pa_type(pa_type: Any) -> Any:
    import pyarrow as pa

    if pa.types.is_dictionary(pa_type):
        return _empty_for_pa_type(pa_type.value_type)
    is_list_type = pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type)
    is_fixed = getattr(pa.types, "is_fixed_size_list", lambda _: False)(pa_type)
    if is_list_type or is_fixed:
        return []
    if pa.types.is_struct(pa_type) or pa.types.is_map(pa_type):
        return {}
    if pa.types.is_integer(pa_type):
        return 0
    if pa.types.is_floating(pa_type):
        return 0.0
    if pa.types.is_boolean(pa_type):
        return False
    if pa.types.is_binary(pa_type) or pa.types.is_large_binary(pa_type):
        return b""
    return ""


def _nested_leaf(value: Any) -> Any:
    try:
        json.dumps(value)
        return JsonLeaf(value)
    except (TypeError, ValueError):
        return _PickleLeaf(value)


def _has_list(value: Any) -> bool:
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return any(_has_list(item) for item in value.values())
    return False


def _schema_empties(parquet_file: Any, columns: list[str] | None) -> dict[str, Any]:
    schema = parquet_file.schema_arrow
    names = columns if columns is not None else [field.name for field in schema]
    empties: dict[str, Any] = {}
    for name in names:
        try:
            empties[name] = _empty_for_pa_type(schema.field(name).type)
        except KeyError:
            empties[name] = ""
    return empties


def _wrap_nested(value: Any) -> Any:
    if isinstance(value, list) or (isinstance(value, dict) and _has_list(value)):
        return _nested_leaf(value)
    if isinstance(value, dict):
        return {key: _wrap_nested(item) for key, item in value.items()}
    return value


def _stabilize_hf_row(
    row: dict[str, Any],
    empties: dict[str, Any],
    types: Any = None,
) -> dict[str, Any]:
    """Coerce nulls and wrap lists so ``optimize`` sees a stable pytree."""
    if types is not None:
        return wrap_for_pytree(row, types, wrap_leaf=JsonLeaf)
    keys = list(empties) if empties else list(row)
    out: dict[str, Any] = {}
    for key in keys:
        value = row.get(key, empties.get(key, ""))
        if value is None:
            value = empties.get(key, "")
        out[key] = _wrap_nested(value)
    for key, value in row.items():
        if key not in out:
            out[key] = _wrap_nested("" if value is None else value)
    return out


def _wrap_hf_media_features(sample: Any) -> Any:
    """Leave Hub ``{bytes, path}`` media as dicts for the Arrow IPC footer.

    ``bytes`` is an Arrow binary column and ``path`` a string. Do not wrap as
    ``Image`` / ``Audio`` / ``Video`` — that decodes on read and skips the
    footer. Users who pass ``PIL.Image`` or ``litdata.Image`` explicitly still
    use pytree serializers.
    """
    return sample


def _optimize_hf_file(
    url: str,
    columns: list[str] | None = None,
    transform: Callable[[dict[str, Any]], Any] | None = None,
    storage_options: dict | None = None,
    persist_dir: str | None = None,
    index_url: str | None = None,
) -> Iterator[Any]:
    import pyarrow.parquet as pq

    path = _materialize_hf_parquet(url, storage_options, persist_dir, index_url)
    parquet_file = pq.ParquetFile(path)
    empties = _schema_empties(parquet_file, columns)
    types = types_from_arrow(parquet_file.schema_arrow, columns)
    for batch in parquet_file.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            sample = transform(row) if transform is not None else row
            if isinstance(sample, dict):
                sample = _stabilize_hf_row(sample, empties, types)
            yield _wrap_hf_media_features(sample)


def _get_existing_cache(dataset_url: str, cache_dir: str | None) -> str | None:
    """Checks if a cache directory with an index file exists for the given dataset URL.

    Args:
        dataset_url (str): The URL of the Hugging Face dataset.
        cache_dir (Optional[str]): The root directory for the cache.

    Returns:
        Optional[str]: The path to the existing cache directory if found, otherwise None.
    """
    chunks_root = cache_dir if cache_dir is not None else get_default_cache_dir()

    if os.path.isfile(os.path.join(chunks_root, _INDEX_FILENAME)):
        return chunks_root

    stable = hf_index_cache_path(dataset_url, cache_dir)
    if os.path.isfile(os.path.join(stable, _INDEX_FILENAME)):
        return stable

    # Previous default: index lived under the chunk cache (deleted by optimize).
    if cache_dir is None:
        legacy_stable = os.path.join(get_default_cache_dir(), "hf-index", generate_md5_hash(dataset_url))
        if os.path.isfile(os.path.join(legacy_stable, _INDEX_FILENAME)):
            return legacy_stable

    # Legacy layout: {cache_dir}/{url_hash}/{updated_at}/index.json
    hashed_cache_path = os.path.join(chunks_root, generate_md5_hash(dataset_url))
    if not os.path.exists(hashed_cache_path):
        return None

    for subdir in os.listdir(hashed_cache_path):
        potential_cache_dir = os.path.join(hashed_cache_path, subdir)
        if os.path.exists(os.path.join(potential_cache_dir, _INDEX_FILENAME)):
            return potential_cache_dir

    return None
