# The streaming (read) pipeline

All paths under `src/litdata/streaming/` unless noted. This is the inverse of the write pipeline: **cloud storage → download → local chunk file → mmap/seek + deserialize → collate → batch.**

## Key classes

| Layer                                      | Class                                  | File                             |
| ------------------------------------------ | -------------------------------------- | -------------------------------- |
| User dataset (`IterableDataset`)           | `StreamingDataset`                     | `dataset.py:51`                  |
| DataLoader (subclasses torch `DataLoader`) | `StreamingDataLoader`                  | `dataloader.py:559`              |
| Read/write facade                          | `Cache`                                | `cache.py:35`                    |
| Reader + background download thread        | `BinaryReader` / `PrepareChunksThread` | `reader.py:312` / `reader.py:50` |
| Index/chunk metadata                       | `ChunksConfig`                         | `config.py:33`                   |
| Chunk→item decoding                        | `BaseItemLoader` + subclasses          | `item_loader.py`                 |
| Backend download                           | `Downloader` subclasses                | `downloader.py`                  |
| Chunk→worker assignment / shuffle          | `Shuffle` (`NoShuffle`/`FullShuffle`)  | `shuffle.py`                     |
| Item index                                 | `ChunkedIndex` (dataclass)             | `sampler.py:24`                  |

## Runtime flow (single item)

1. `StreamingDataLoader.__init__` (`dataloader.py:614`) pushes `batch_size`/`num_workers`/`shuffle`/`drop_last` into the dataset via `set_*` methods, then calls torch `DataLoader.__init__`. Because `StreamingDataset` is an `IterableDataset`, torch uses its `__iter__`/`__next__` (no external sampler).
2. `StreamingDataLoader.__iter__` (`dataloader.py:687`) sets the epoch, resets per-epoch counters, delegates to `super().__iter__()`, and wraps the iterator to track `_num_samples_yielded_streaming` for checkpointing.
3. `StreamingDataset.__iter__` (`dataset.py:357`) detects `_WorkerEnv`/`_DistributedEnv`, builds the `Cache` (`_create_cache`, `dataset.py:271`), builds the `Shuffle` (`_create_shuffler`, `dataset.py:330`), and computes **this worker's** chunk list + intervals via `shuffler.get_chunks_and_intervals_per_workers(...)`. Slices by `worker_rank = global_rank * worker_world_size + worker_rank` (`dataset.py:377`).
4. `StreamingDataset.__next__` (`dataset.py:504`) pops the next global index from `upcoming_indexes` (refilled per-chunk, shuffled in-chunk by `shuffler.__call__`), wraps it in a `ChunkedIndex`, calls `__getitem__`.
5. `__getitem__` (`dataset.py:481`) → `Cache.__getitem__` (`cache.py:151`) → `BinaryReader.read` (`reader.py:438`).
6. `BinaryReader.read` lazily loads `ChunksConfig` (`_try_load_config`, `reader.py:384`), resolves `(chunk_filepath, begin, filesize_bytes)` via `ChunksConfig.__getitem__` (`config.py:289`), queues the `PrepareChunksThread` (`reader.py:398`), then asks the item loader to decode. It also drives chunk lifetime: decrement lock + queue deletion of the previous chunk, and on `is_last_index` stops the thread.
7. Batching: torch collates via `StreamingDataLoaderCollateFn` (`dataloader.py:502`), which unwraps the `__SAMPLES__`/`__NUM_SAMPLES_YIELDED__` envelope.

**Prefetching/eviction**: `PrepareChunksThread` (`reader.py:50`) is a per-worker daemon thread. It pre-downloads up to `max_pre_download` chunks, pre-loads them, and deletes consumed chunks once `max_cache_size` is exceeded (`_maybe_delete_chunks`, `reader.py:206`). Cross-worker refcounting uses `.cnt` files + `FileLock`.

**On-demand mode**: if `on_demand_bytes` is true and the data is remote + PyTreeLoader + no compression/encryption, `read` fetches only the item's byte range (HTTP range GET) via `read_item_bytes` (`reader.py:535`) → `ChunksConfig.download_chunk_bytes_from_index` (`config.py:160`). Iterating flips `on_demand_bytes=False` (`dataset.py:365`) so full chunks are cached during training; it flips back on `StopIteration`.

## On-disk format / Cache / Writer / Reader / FsProvider / sampler

**Full deep dive:** [storage-format.md](storage-format.md) — `Cache` facade, binary chunk layout, `index.json` schema, `BinaryWriter`/`BinaryReader`/`ChunksConfig`, `FsProvider` vs `Downloader`, `ChunkedIndex` + `CacheBatchSampler`.

Sketch:

```
+-----------+----------------+-----------+
| num_items | offset_array   | item_data |   → chunk-{rank}-{idx}[.zstd].bin
| uint32    | uint32[N+1]    | bytes     |
+-----------+----------------+-----------+
```

`BinaryWriter` → `{rank}.index.json` → merge → `index.json`. `ChunksConfig` + `BinaryReader` consume that on the read path. Prefetch/eviction → [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md).

## Item loaders (`item_loader.py`) — own byte layout AND interval math

- **`PyTreeLoader`** (`item_loader.py`) — default for LitData chunks. Offset table + pytree deserialize; encryption + MDS.
- **`TokensLoader`** (`item_loader.py`) — NLP token windows (`block_size`); `np.memmap` + `torch.frombuffer`.
- **`ParquetLoader`** (`item_loader.py` ~`:851`) — parquet files as “chunks”. Needs `polars>1.0` + `pyarrow`. Returns **row dicts**.
  - `low_memory=True` (default): pyarrow row groups → polars rows; evict group when consumed.
  - `low_memory=False`: `pl.scan_parquet(...).collect()` whole file; enables `pre_load_chunk`.
  - HF (`hf://`) auto-wires this loader (`dataset.py`); other schemes must pass it explicitly and match `index.json`.
  - `StreamingDataLoader` forbids fork with this loader + `num_workers>0` (`dataloader.py` ~`:630`) → `spawn` / `forkserver`.
  - Indexing: `index_parquet_dataset` / `index_hf_dataset` → `utilities/parquet.py` dispatch (`local` / `s3` / `gs` / `hf` only). User cookbook: [using-litdata.md](using-litdata.md) §10.

`ChunksConfig._validate_item_loader` raises if the passed loader class ≠ `config["item_loader"]`.

## Backends

- **`downloader.py`** — read path. `Downloader` ABC (`:41`); subclasses `S3Downloader`, `R2Downloader`, `GCPDownloader`, `AzureDownloader`, `HFDownloader`, `LocalDownloader`. Selected by URL prefix via `_DOWNLOADERS` registry + `get_downloader` (`:658`); register new ones with `register_downloader` (`:632`).
- **S3 path prefers obstore**: `obstore` is a **hard** install dependency (`requirements.txt`). Chunk GETs stream via obstore when usable; boto3 remains the fallback. **`index.json` always uses boto3** so the DataLoader parent does not start tokio before fork; workers then lazy-init obstore. If the parent *did* start obstore, workers fall back to boto3 (`obstore_usable()` — re-creating `S3Store` is not enough; tokio is process-global). Pickle drops `_store` / `_store_pid`. Stream chunk size: `LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB` (default **8** MiB).
- **Obstore credentials**: `_build_obstore_s3_store` copies endpoint/region from the existing `S3Client`/`R2Client` and uses a credential provider. **Never** unpack `storage_options` (`data_connection_id`, `endpoint_url`, …) into `boto3.Session` — that `TypeError` used to kill `PrepareChunksThread` and surface as a 120s `FileNotFoundError` on Studio R2.
- **Async chunk prefetch** (`async_prefetch.py`): **not** an async DataLoader. Overlaps remote chunk downloads inside `PrepareChunksThread` via `asyncio.gather`. Default **on for remote**, off for local-only; `LITDATA_ASYNC_CHUNK_PREFETCH=0/1` overrides. When on, raises `max_pre_download` floor to 4 (`LITDATA_ASYNC_MIN_PRE_DOWNLOAD`). See [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md) and [env-vars.md](env-vars.md).
- **`fs_provider.py`** — write/management only (`s3`/`gs`/`r2`). Details + vs Downloader: [storage-format.md](storage-format.md) §5.
- **`resolver.py`** — path/URL resolution (NOT I/O backend selection). `_resolve_dir` (`:50`) → `Dir(path, url, data_connection_id)`. Full map: [resolver.md](resolver.md), [lightning-studio.md](lightning-studio.md).
- **`client.py`** — `S3Client`/`R2Client` wrap boto3 with credential refresh + temporary project-role credentials from the Lightning control plane.
- **`compression.py`** — `Compressor` ABC + `ZSTDCompressor`, registered into `_COMPRESSORS`. Decompression in `ChunksConfig.try_decompress` (`config.py:182`).
- **`serializers.py`** — `Serializer` ABC; ordered `_SERIALIZERS` dict (`:553`, order matters because `can_serialize` is tried top-to-bottom). Includes `str/bool/int/float/video/tifffile/pil/jpeg/numpy/tensor/pickle` (catch-all). Users pass custom serializers via `StreamingDataset(serializers=...)`.

## Shuffling & sharding — two independent stages

**Stage A — chunk→worker assignment** (`Shuffle.get_chunks_and_intervals_per_workers`):

- `NoShuffle` (`shuffle.py:60`) keeps order; `FullShuffle` (`shuffle.py:83`) permutes deterministically with `np.random.RandomState([seed, seed_shift])` where `seed_shift = 1` for multi-node else `current_epoch`.
- Both call `_associate_chunks_and_intervals_to_workers` (`utilities/shuffle.py:65`), which greedily fills each worker's item budget and **splits chunk intervals** across worker boundaries — so one chunk can be shared by multiple workers (and re-downloaded by each).
- Multi-node epoch>1 adds an intra-node reshuffle (`_intra_node_chunk_shuffle`).

**Stage B — in-chunk item order** (`Shuffle.__call__`): `FullShuffle` permutes with `np.random.RandomState([seed, num_chunks, current_epoch, chunk_index])` (`shuffle.py:140`), deterministic per (chunk, epoch).

**`sampler.py` vs `shuffle.py`:** `ChunkedIndex` is used on the **StreamingDataset read path**. `CacheBatchSampler` is **only** for `CacheDataLoader` (map-dataset → write-while-train). Epoch shuffle for streaming lives in `shuffle.py`. Full logic: [storage-format.md](storage-format.md) §6.

## Resume / checkpointing

`_replay_sampling` (`dataset.py:755`) + `_replay_chunks_sampling` (`dataset.py:778`) reconstruct where each worker stopped from `num_samples_yielded`/`num_workers`/`batch_size`. `_resume` (`dataset.py:425`) re-shuffles the current chunk and skips consumed indexes. `_validate_state_dict` (`dataset.py:607`) hard-fails if shuffle/num_workers/seed/input_dir/item_loader/drop_last changed (unless `force_override_state_dict`). `state_dict` raises if called inside a worker (`dataset.py:574`).

## CombinedStreamingDataset & ParallelStreamingDataset

Both subclass `_BaseStreamingDatasetWrapper` (`utilities/base.py:27`) which fans `set_*`/`load_state_dict` to all wrapped datasets and carries the `__SAMPLES__`/`__NUM_SAMPLES_YIELDED__`/`__NUM_CYCLES__` envelope.

- **`CombinedStreamingDataset`** (`combined.py:40`) — samples **one** dataset per step by weight (seeded `random.Random`). `iterate_over_all=True` iterates until all exhausted, re-normalizing weights; `batching_method` is `"stratified"` (mix within a batch) or `"per_stream"` (whole batch from one dataset). Weights default to inverse dataset length.
- **`ParallelStreamingDataset`** (`parallel.py:44`) — pulls **one sample from every dataset per step**, yields a tuple or `transform(samples[, rngs])`. Supports cycling (`length` = None/int/`inf`), resumable per-worker RNGs seeded via SHA-256 from `(seed, worker_rank, samples_yielded, cycles)`, and `reset_rngs`.

## Gotchas (read before editing)

- **Item loader coupling**: `encode_data` (write) and `load_item_from_chunk`/`generate_intervals` (read) must stay consistent. Header size `len(data_format)*4` and the `(1+(index-begin))*4` offset formula are load-bearing.
- **Chunk lifetime is fragile**: deletion gated by `.cnt` refcount + `FileLock`; `skip_chunk_indexes_deletion` (`config.py:117`) prevents deleting a chunk a same-node worker still needs. Deletion is deferred until after the next item loads (avoids mmap segfaults). Windows requires `close()` before `os.remove`.
- **ParquetLoader + fork deadlock**: `StreamingDataLoader` raises if `ParquetLoader` is used with `num_workers>0` under `fork` — use `spawn`/`forkserver` (`dataloader.py:629`).
- **`num_workers=0` counts as 1 worker** internally (`dataset.py`).
- **Obstore is not fork-safe**: parent must not start tokio (`index.json` is boto3). Symptom: `FileNotFoundError` on `chunk-*.bin` after `MAX_WAIT_TIME` only when `num_workers>0`. [debugging.md](debugging.md).
- **PrepareChunksThread crash visibility**: `_report_crash` prints to stderr, emits tracer `crash` (`ph: I`), stashes the exception so waiters raise `RuntimeError` immediately instead of timing out as `FileNotFoundError`.
- **Datasets are immutable**: resolver refuses to write into a non-empty output dir unless `append`/`overwrite` (`resolver.py`).
- **Two dataloaders exist**: `CacheDataLoader` (`dataloader.py:260`, write/cache path) vs `StreamingDataLoader` (read path). Don't cross-wire them.
- Missing `index.json` in `_create_cache` raises a "did you optimize?" error (`dataset.py:322`).

## Runnable examples (from README)

```python
# Stream optimized data (README:328)
from litdata import StreamingDataset, StreamingDataLoader
dataset = StreamingDataset('s3://my-bucket/my-data', shuffle=True, drop_last=True)
dataloader = StreamingDataLoader(dataset, batch_size=64)
for batch in dataloader:
    ...

# Custom S3-compatible endpoint (README:342)
dataset = StreamingDataset('s3://my-bucket/my-data', storage_options={
    "endpoint_url": "...", "aws_access_key_id": "...", "aws_secret_access_key": "..."})

# Custom cache dir + cache cap (README:360; dataset.py:63 default "100GB")
dataset = StreamingDataset('s3://my-bucket/my-data', cache_dir="/path/to/cache",
                           max_cache_size="50GB")

# Combine datasets with weights (README:846)
from litdata import CombinedStreamingDataset
combined = CombinedStreamingDataset(datasets=[ds1, ds2], weights=(0.5, 0.5),
                                    iterate_over_all=False)

# Pause & resume (README:573) — StreamingDataLoader exposes state_dict()/load_state_dict()
state = dataloader.state_dict()          # in the main process
dataloader.load_state_dict(state)        # resume exactly where you stopped
```

README feature sections (from `grep -n '<summary>' README.md`): multi-GPU/multi-node (483), multiple providers (515), pause/resume (573), combine (846), parallel streaming (918), cycle (968), subsets (1115), parquet (1192), compression (1251), on-demand access (1284), transforms (1302), cache limits (1386).
