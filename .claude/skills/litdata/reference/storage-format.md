# Cache, BinaryWriter, BinaryReader, storage format, FsProvider, sampler

Contributor deep dive for the **optimized** LitData binary format and the classes that own it. User-facing cache knobs → [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md). Read-path shuffle → [streaming.md](streaming.md). Paths → [resolver.md](resolver.md).

```
optimize / CacheDataLoader (write)
  sample → BinaryWriter → chunk-*.bin + {rank}.index.json → merge → index.json
  (upload via FsProvider)

StreamingDataset (read)
  index.json → ChunksConfig → BinaryReader (+ PrepareChunksThread / Downloader)
  → item_loader deserialize → sample
```

`Cache` (`streaming/cache.py`) is the thin facade that owns **both** a `BinaryWriter` and a `BinaryReader` over the same `cache_dir`.

______________________________________________________________________

## 1. `Cache` (`streaming/cache.py`)

Users almost never construct this for streaming — `StreamingDataset._create_cache` does. Still used directly by `optimize` workers and by `CacheDataLoader` / `CacheDataset`.

| Concern                      | Method / property                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| Write one sample             | `__setitem__(i, data)` / `_add_item(i, data)` → writer                                         |
| Finish worker chunks         | `done()` → flush remaining + write `{rank}.index.json`                                         |
| Merge indexes                | `merge(num_workers)` / `_merge_no_wait(...)` → writer                                          |
| Read one sample              | `__getitem__(int \| ChunkedIndex)` → reader (`int` promoted via `_get_chunk_index_from_index`) |
| Length / intervals           | `__len__`, `get_chunk_intervals()` → reader                                                    |
| Phase gate                   | `filled` — `True` when `index.json` exists in `cache_dir`                                      |
| Rank                         | `rank` = `global_rank * worker_world + worker_rank`                                            |
| Checkpoint (optimize resume) | `save_checkpoint` / `checkpoint_dir`                                                           |

Ctor wires:

- `input_dir` → `_resolve_dir` → `path` = cache root, `url` = remote for the reader
- Writer gets `chunk_size` / `chunk_bytes`, compression, encryption, serializers, `item_loader`, `writer_chunk_index`
- Reader gets `max_cache_size`, `max_pre_download`, `subsampled_files` / `region_of_interest`, `storage_options` / `session_options`, `on_demand_bytes`

**Invariant:** `filled` means the **merged** `index.json` is present. Per-rank `{rank}.index.json` files alone do **not** count as filled for the `Cache` property (that checks `_INDEX_FILENAME` only). Writer’s own `filled` checks that enough per-rank indexes exist before merge.

______________________________________________________________________

## 2. On-disk storage format

### Chunk file (`chunk-{rank}-{chunk_index}[.compression].bin`)

Binary layout (`BinaryWriter._create_chunk`, `writer.py` ~218):

```
+-----------+------------------+-----------+
| num_items | offset_array     | item_data |
| uint32    | uint32[N+1]      | bytes     |
| 4 bytes   | 4*(N+1) bytes    | variable  |
+-----------+------------------+-----------+
```

- `offsets[0]` = header size = `4 + 4*(N+1)` (start of first item).
- Item `i` occupies bytes `[offsets[i], offsets[i+1])`.
- Optional **sample** encryption: each item’s payload encrypted before packing (`EncryptionLevel.SAMPLE`).
- Optional **chunk** encryption: whole `num_items‖offsets‖data` encrypted after pack (`EncryptionLevel.CHUNK`).
- Optional **compression**: applied to the whole file bytes at write (`write_chunk_to_file`); filename includes the algorithm, e.g. `chunk-0-3.zstd.bin`. Reader decompresses to a sibling path without the compression infix before mmap/read.

Item bytes themselves are **item-loader specific**. Default `PyTreeLoader.encode_data` packs flattened serializer leaves (sizes + payload). `TokensLoader` / `ParquetLoader` differ — loaders own both encode and decode.

### Per-chunk metadata (inside `index.json` → `chunks[]`)

Each entry written by the writer:

| Field         | Meaning                                                                                 |
| ------------- | --------------------------------------------------------------------------------------- |
| `filename`    | e.g. `chunk-0-0.bin` or `chunk-0-0.zstd.bin`                                            |
| `chunk_bytes` | Size of the **on-disk** (possibly encrypted/compressed) file                            |
| `chunk_size`  | Number of items in the chunk (`N`)                                                      |
| `dim`         | Optional: sum of per-item dims (TokensLoader uses dim = token count); else often `null` |

### Index files

| File                     | Who writes                                           | Role                                 |
| ------------------------ | ---------------------------------------------------- | ------------------------------------ |
| `{rank}.index.json`      | Each writer worker’s `done()` → `write_chunks_index` | `{"chunks": [...], "config": {...}}` |
| `{node_rank}-index.json` | Multi-node merge (`_merge_no_wait(node_rank=…)`)     | Per-node partial merge               |
| `index.json`             | Rank-0 merge (`merge` / `_merge_no_wait`)            | Canonical dataset index              |

Merged `index.json` shape:

```json
{
  "chunks": [ { "filename", "chunk_bytes", "chunk_size", "dim" }, ... ],
  "config": {
    "compression": null | "zstd" | ...,
    "chunk_size": ...,
    "chunk_bytes": ...,
    "data_format": ["jpeg", "int", ...],
    "data_spec": "<treespec dump or null>",
    "encryption": null | { ... state_dict ... },
    "item_loader": "PyTreeLoader"
  },
  "updated_at": "<time() string>"
}
```

- **`data_format`**: ordered serializer names for flattened leaves (fixed after first sample).
- **`data_spec`**: pytree treespec string for `tree_unflatten` on read.
- **`item_loader`**: class **name** string; `ChunksConfig._validate_item_loader` requires the reader’s loader class to match.
- Merge asserts all per-rank `config` dicts are **identical**; then concatenates `chunks` in natural-sorted index-file order and deletes the per-rank files.

Raw datasets use a **different** index: `index.json.zstd` (`raw/indexer.py`) — do not conflate with optimized `index.json`.

### `Interval` (`item_loader.py`)

```text
Interval(chunk_start, roi_start_idx, roi_end_idx, chunk_end)
```

Global item index ranges per chunk (with optional ROI for subsample / `train_test_split`). `ChunksConfig` maps global index → `(chunk_index, local offset)` via these intervals. Prefetch / shared-chunk deletion also reason about chunk indexes from here.

______________________________________________________________________

## 3. `BinaryWriter` (`streaming/writer.py`)

**Purpose:** serialize samples in **strict ascending index order**, pack them into rolling chunks, emit per-rank index JSON, merge.

### Lifecycle

1. `add_item(index, sample)` / `__setitem__` — serialize → buffer in `_serialized_items`.
2. When buffer hits `chunk_bytes` or `chunk_size` (`_should_write`), `write_chunk` → `_create_chunk` → `write_chunk_to_file`.
3. `done()` — flush remainder (`on_done=True` allows non-contiguous flush of leftover keys), write `{rank}.index.json`.
4. `merge(num_workers)` — only **rank 0** waits for all `{*}.index.json`, then `_merge_no_wait`; other ranks spin until final `index.json` appears.

### Serialization path

```
sample → tree_flatten → per-leaf Serializer (can_serialize order)
      → item_loader.encode_data(data, sizes, flattened)
      → optional sample encryption
      → Item(index, data, bytes, dim)
```

- Exactly one of `chunk_size` / `chunk_bytes` required at ctor (same rule as `optimize`).
- First sample freezes `data_format` + `data_spec`; later samples must stay compatible (`_serialize_with_data_format`).
- Oversized single item: still written; warns if > `chunk_bytes`.
- Rank for filenames: `DATA_OPTIMIZER_GLOBAL_RANK` if set, else `global_rank * workers + worker_rank`.

### Filenames

- Uncompressed: `chunk-{rank}-{chunk_index}.bin`
- Compressed: `chunk-{rank}-{chunk_index}.{compression}.bin`

### Checkpointing

`save_checkpoint` supports optimize `use_checkpoint` resume (partial chunk info under `checkpoints/`). Prefer [processing.md](processing.md) for the job-level story.

______________________________________________________________________

## 4. `BinaryReader` + `ChunksConfig`

### `ChunksConfig` (`streaming/config.py`)

Loads `index.json` from `cache_dir` (downloads it once if `remote_dir` set via downloader). Responsibilities:

- Own `chunks` list (+ optional `subsampled_files` filter)
- Restore `data_spec`, `setup` item_loader, `generate_intervals`
- Map `ChunkedIndex` / global index → `(local_path, begin, filesize_bytes)`
- Decompress compressed chunks to local cache (`try_decompress`)
- Refcount helpers: `increment_local_lock` / `decrement_local_lock`, `can_delete`, `skip_chunk_indexes_deletion`, `_shared_chunk_indexes`
- On-demand byte ranges: `download_chunk_bytes_from_index`

### `BinaryReader` (`streaming/reader.py`)

Inverse of the writer for the **read** path:

1. Lazy-load `ChunksConfig`.
2. Ensure `PrepareChunksThread` is running; queue downloads for upcoming chunk indexes (from `ChunkedIndex.chunk_indexes` when provided).
3. Wait until the `.bin` exists (item loader busy-wait; force re-download after `FORCE_DOWNLOAD_TIME`; fail after `MAX_WAIT_TIME`).
4. `item_loader.load_item_from_chunk(...)` → deserialize.
5. On chunk change: decrement lock + enqueue delete of previous chunk; on `is_last_index` stop prepare thread.

**Downloads use `Downloader` (read path), not `FsProvider`.** Prefetch / eviction / async gather → [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md).

`on_demand_bytes`: range-GET a single sample when remote + PyTreeLoader + no compression/encryption; iteration disables this so full chunks are cached.

______________________________________________________________________

## 5. `FsProvider` (`streaming/fs_provider.py`) — write / management I/O

**Not** the training download path. Used when optimizing, merging, checking empty remotes, deleting overwrite targets, uploading indexes/chunks, copying between remotes.

### Vs `Downloader`

|         | **FsProvider**                                   | **Downloader**                         |
| ------- | ------------------------------------------------ | -------------------------------------- |
| Module  | `fs_provider.py`                                 | `downloader.py`                        |
| When    | Write / manage remote datasets                   | Stream / prefetch chunks into cache    |
| Schemes | **`s3`**, **`gs`**, **`r2` only**                | Also `azure`, `hf`, `local`, …         |
| Factory | `_get_fs_provider(url, storage_options)`         | `get_downloader(...)` / `_DOWNLOADERS` |
| Clients | `S3Client` / `R2Client` / `google.cloud.storage` | obstore/boto3/etc. per subclass        |

`azure://` and `hf://` streaming work via downloaders; **FsProvider will raise** `Unsupported scheme` for them — optimize/upload paths that need Azure/HF must go through other helpers or aren’t covered by this ABC.

### API surface

| Method                                 | Typical use                                                     |
| -------------------------------------- | --------------------------------------------------------------- |
| `upload_file(local, remote)`           | Chunk / index upload (`data_processor`)                         |
| `download_file` / `download_directory` | Peer node indexes, checkpoint restore                           |
| `exists` / `is_empty`                  | Refuse non-empty output unless append/overwrite (`resolver.py`) |
| `delete_file_or_directory`             | `mode="overwrite"`, checkpoint cleanup                          |
| `copy`                                 | `merge_datasets` remote copy (`functions.py`)                   |
| `list_directory`                       | **Not implemented** on current providers (raises)               |

Helpers: `get_bucket_and_path(url, expected_scheme)`, `not_supported_provider` (scheme not in `_SUPPORTED_PROVIDERS`).

`storage_options` (and resolver-merged `data_connection_id`) flow into the provider ctor. R2 subclasses S3 but swaps in `R2Client` and re-implements scheme-specific path parsing (`r2://`).

### Call sites (grep landmarks)

- `resolver.py` — empty / exists / delete before write
- `processing/data_processor.py` — download inputs, upload chunks, multi-node index gather, checkpoints
- `processing/functions.py` — `merge_datasets` copy/upload
- `processing/utilities.py` — fetch remote `index.json`

______________________________________________________________________

## 6. Sampler (`streaming/sampler.py`)

**Two different things live here.** Do not confuse with `shuffle.py` (StreamingDataset epoch shuffle).

### `ChunkedIndex` (dataclass) — **used on the read path**

```python
ChunkedIndex(
    index,            # global sample index
    chunk_index,      # which chunk file
    chunk_size=None,
    chunk_indexes=None,  # optional list: chunks this worker may prefetch / keep
    is_last_index=False, # signal PrepareChunksThread shutdown
)
```

Docstring form of `chunk_indexes` historically described interval-style ranges; in practice StreamingDataset / CacheBatchSampler often pass a **list of chunk indexes** this worker owns for mmap / prefetch hints (`set_mmap_allowed_chunks`, prepare-thread queue priming).

`StreamingDataset.__next__` builds `ChunkedIndex` → `Cache.__getitem__` → `BinaryReader.read`. Random access with a plain `int` is promoted inside `Cache.__getitem__` via `_get_chunk_index_from_index`.

### `CacheBatchSampler` — **write / `CacheDataLoader` only**

Used by `CacheDataLoader` (`dataloader.py` ~261), which wraps a **map-style** dataset + `Cache` to binarize on the fly while training. **Not** used by `StreamingDataLoader` / `StreamingDataset` (those are `IterableDataset` and shuffle via `shuffle.py`).

Behavior depends on `cache.filled`:

| Phase                                    | Behavior                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Not filled** (writing)                 | Ignore `shuffle`. Partition indices by replica then worker into **contiguous** ranges so each worker sees successive indexes (required for `BinaryWriter`). Validate with `_validate()` — raises if any worker’s indices aren’t consecutive. Round-robin yield batches across workers (`__iter_indices_per_workers__`).                           |
| **Filled** (reading via CacheDataLoader) | Shuffle chunk order (`np.random.permutation`). Distribute chunks across replicas (`index % num_replicas == rank`), then across workers (`i % num_workers`). Within each chunk, permute item indexes in `[roi_start, roi_end)`. First `ChunkedIndex` in a worker’s stream may carry `chunk_indexes=` = that worker’s full chunk list for prefetch. |

Distributed write path: `replica_size = dataset_size // num_replicas`, then `worker_size = dataset_size // (num_replicas * num_workers)`; last replica/worker takes the remainder.

`CacheDataLoader` refuses external `sampler` / `batch_sampler`, refuses `IterableDataset`, and logs that shuffle is ignored while caching.

______________________________________________________________________

## 7. Agent checklist

When editing these modules:

1. **Format change** → update Writer `_create_chunk`, Reader/item_loader decode, and this doc; bump / compatibility story for old `index.json`.
2. **New serializer** → register in `_SERIALIZERS` order; first-sample `data_format` must stay stable across workers (merge config equality).
3. **New cloud scheme for optimize upload** → extend **both** `_get_fs_provider` and downloaders if streaming should work; tests for empty/exists/delete.
4. **Sampler vs shuffle** → read-path bugs → `shuffle.py` + `ChunkedIndex` construction in `dataset.py`; write-order bugs → `CacheBatchSampler` contiguous partitions.
5. **Shared chunks** → deletion / `.cnt` / `skip_chunk_indexes_deletion` in [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md), not sampler.
6. **Never** use `FsProvider` inside `PrepareChunksThread` — downloads stay on `Downloader` (+ async prefetch).

______________________________________________________________________

## Related files

| File                       | Role                                       |
| -------------------------- | ------------------------------------------ |
| `streaming/cache.py`       | Facade                                     |
| `streaming/writer.py`      | BinaryWriter + merge                       |
| `streaming/reader.py`      | BinaryReader + PrepareChunksThread         |
| `streaming/config.py`      | ChunksConfig / index load                  |
| `streaming/item_loader.py` | Per-format encode/decode + intervals       |
| `streaming/fs_provider.py` | Remote write/management                    |
| `streaming/downloader.py`  | Remote read/prefetch                       |
| `streaming/sampler.py`     | ChunkedIndex + CacheBatchSampler           |
| `streaming/dataloader.py`  | `CacheDataLoader` vs `StreamingDataLoader` |
| `streaming/shuffle.py`     | StreamingDataset shuffle (not sampler.py)  |
