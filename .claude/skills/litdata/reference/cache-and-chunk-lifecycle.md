# Cache, Writer/Reader, PrepareChunksThread & the chunk lifecycle

This is the deepest and most race-prone part of LitData. Read it before touching anything under `streaming/{cache,reader,writer,config,item_loader,downloader}.py`.

## Cache: the façade over Writer and Reader

`Cache` (`cache.py:35`) is the single object a `StreamingDataset` talks to. It is **bidirectional**:

- **Write side** — `Cache.__setitem__`/`_add_item` → `BinaryWriter` (`writer.py:50`). Used by the *caching* path (`CacheDataLoader`) and by `optimize`'s `Cache`. `Cache.done()` flushes, `Cache.merge()` concatenates per-rank `{rank}.index.json` into one `index.json`.
- **Read side** — `Cache.__getitem__` (`cache.py:151`) → `BinaryReader.read` (`reader.py:438`). Used by `StreamingDataset`.

`Cache.filled` (`cache.py:120`) decides write-vs-read: it checks for `index.json` + expected worker count. A missing `index.json` when reading is the "did you optimize?" error (`dataset.py:322`).

The chunk binary format (`[num_items:uint32][offset:uint32[N+1]][data]`, `writer.py:218`) and item-loader coupling are covered in [streaming.md](streaming.md). This file focuses on the **runtime read machinery** and the **cross-worker chunk lifecycle**, which streaming.md only summarizes.

## BinaryReader.read — the per-item hot path

`BinaryReader.read(index)` (`reader.py:438`):

1. Lazily loads `ChunksConfig` (`_try_load_config`) — downloads `index.json` on first call.
2. `config[index]` (`config.py:289`) → `(local_chunkpath, begin, filesize_bytes)`. For compressed data `local_chunkpath` is the **decompressed** `.bin` path.
3. If remote/compressed: `setup_thread_and_download_chunk` starts (once) the per-worker `PrepareChunksThread` and enqueues this worker's chunks for download.
4. `item_loader.load_item_from_chunk(...)` (`item_loader.py:172`) opens the `.bin`, seeks the offset table, reads `[begin,end)`, deserializes. **If the `.bin` isn't present yet it busy-waits** (`item_loader.py:206-223`): sleeps 0.1s, at `FORCE_DOWNLOAD_TIME` (30s) calls `force_download`, and at `MAX_WAIT_TIME` (120s) raises `FileNotFoundError: The <path> hasn't been found.`
5. On chunk transition (`index.chunk_index != self._last_chunk_index`): `_decrement_local_lock(last)` + `delete([last])` (enqueues deletion of the chunk just finished).
6. On `index.is_last_index`: close handle, decrement, delete, stop the thread, reset per-epoch state.

`rank` (`reader.py:435`) = `global_rank * worker_world_size + worker_rank` — the flat worker identity used everywhere.

## PrepareChunksThread — one daemon thread per worker

`PrepareChunksThread` (`reader.py:50`) is what makes streaming overlap with training. Each `BinaryReader` owns one. It services three queues in its `run()` loop (`reader.py:276`):

- `_force_download_queue` (checked first each iteration) → `_force_download` (`reader.py:240`): delete-then-redownload a chunk a reader is blocked on. Attached to the item loader at `reader.py:411` so `item_loader.force_download` feeds this thread.
- `_to_download_queue` → `download_chunk_from_index` for each assigned chunk, bounded by `max_pre_download` (default 2) via `_pre_download_counter`. `_END_TOKEN` stops it.
- `_to_delete_queue` → `_maybe_delete_chunks` → `_apply_delete`, gated by `max_cache_size` and `_can_delete_chunk` (`reader.py:228`).

**Eviction is only active when `max_cache_size` is set** (or `MAX_CACHE_SIZE` env). `_delete_chunks_when_processed` (`reader.py:84`) is true when a node's slice doesn't fit in the cache; then chunks are deleted aggressively as they're consumed. Otherwise deletion waits until the folder exceeds `max_cache_size` (`_get_folder_size`, `reader.py:581`, which is deliberately robust to deletion races and ignores `.cnt`/`.lock`/`.zstd.bin`).

## Distributed sampling — how workers get disjoint (mostly) chunk slices

Covered in depth in [streaming.md](streaming.md) ("Shuffling & sharding"). The essentials that matter for the lifecycle:

- `Shuffle.get_chunks_and_intervals_per_workers` returns `workers_chunks` / `workers_intervals` — **flat lists indexed by `rank * num_workers + worker`**, length `world_size * num_workers`.
- `_associate_chunks_and_intervals_to_workers` (`utilities/shuffle.py:65`) gives each worker an item budget and **splits a chunk's interval across worker boundaries** when it straddles them. Consequence: **a single chunk can be assigned to several workers**, each reading a different sub-interval. This is the source of all the deletion complexity below.
- `dataset.__iter__` slices out this worker's `worker_chunks`/`worker_intervals` (`dataset.py:378`) and computes the skip-deletion mapping for the node.

## The shared-chunk deletion problem (and the three mechanisms guarding it)

Because a chunk can be shared by multiple workers on a node (all pointing at the same cache file), **one worker deleting a chunk after it finishes can pull the file out from under another worker that still needs it** → `FileNotFoundError: chunk-N-M.bin hasn't been found` raised at `item_loader.py:223` (PyTree) / `:515` (Tokens). Three mechanisms exist to prevent this:

### 1. Static skip-deletion list (`skip_chunk_indexes_deletion` / `can_delete`)

Computed in `dataset.__iter__` via `_find_chunks_per_workers_on_which_to_skip_deletion` (`utilities/shuffle.py:147`): for each shared chunk it determines the single worker that reads it **last in consumption order**, and tells every *other* sharing worker to skip deleting it. Stored on `ChunksConfig.skip_chunk_indexes_deletion`; queried via `config.can_delete(chunk_index)` (`config.py:117`). This is the primary, race-free guard (it is derived from the deterministic shuffle assignment, not from wall-clock state). **It must be consulted in `reader._apply_delete`** — historically it was computed but ignored (see the bug note below).

### 2. Cross-worker reference counting (`.cnt` + `.cnt.lock` files)

`downloader._increment_local_lock` (`downloader.py:55`) bumps `<chunk>.bin.cnt` under a `FileLock` when a worker will read a chunk; `reader._decrement_local_lock` (`reader.py:111`) decrements when a worker finishes it and removes the file at zero. `_apply_delete` refuses to delete while `_remaining_locks > 0` (`reader.py:180`). This is the dynamic guard. **Its correctness depends on every sharing worker incrementing before any sharer decrements to zero** — see the increment-lag hazard below.

### 3. Priority force-redownload (`_force_download_queue`)

When a reader blocks on a missing `.bin` for `FORCE_DOWNLOAD_TIME`, `item_loader.force_download` enqueues the chunk onto its own thread's `_force_download_queue`; `_force_download` (`reader.py:240`) deletes any stale copy and re-downloads under `FileLock(..., timeout=0)`. This is a last-resort recovery, not prevention.

### Failure modes / invariants to preserve

- **`can_delete` must gate `_apply_delete`.** If its result is computed but unused, mechanism 1 is dead and only the racy refcount protects shared chunks.
- **Increment-lag:** increments happen lazily in the prefetch thread (bounded by `max_pre_download`), while decrement+delete happen when a worker finishes a chunk. If a fast worker finishes a shared chunk before a slow co-worker has prefetched (incremented) it, the count is 0 and the chunk is deleted prematurely. Mechanism 1 (static list) is what actually closes this hole; the refcount alone does not.
- **The skip list must be set on resume too.** It is computed in `dataset.__iter__`; if it is only set on the fresh-epoch branch, resumed runs have no mechanism-1 protection.
- **Node slicing:** `workers_chunks` is indexed by `rank * num_workers + worker`, so a node's workers start at `first_rank_this_node * num_workers`. Using `* (node_size * num_workers)` overshoots for multi-node.
- **`skip_lock=True`** (force-redownload path) intentionally bypasses both the skip list and the refcount — a worker re-fetching its *own* needed chunk must be allowed through.
- **s3transfer temp files:** boto3 downloads to `chunk-*.zstd.bin.<random>` then renames. Leftover `<random>`-suffixed files in the cache are interrupted/duplicated downloads — a symptom of delete/redownload churn, not the root cause. `_get_folder_size` counts `.bin`-containing temp files but ignores `.zstd.bin`.

### Debugging this class of failure

1. Reproduce with `num_workers` > 1 and `max_cache_size` set small enough to force eviction (that's when deletion runs).
2. `enable_tracer()` — the `increment_lock_*` / `decrement_lock_*` / `delete_chunk_*` events show the refcount timeline per chunk; a `delete_chunk_X` with a later `read_chunk_X` on another worker is the race.
3. `DEBUG_LITDATA=1` writes `<chunk>.tmb` tombstones recording which rank deleted a chunk and its `can_delete` verdict.
4. Inspect the cache dir: leftover `.zstd.bin.<random>` + missing `.bin` for a low-numbered chunk = premature deletion + failed redownload.
