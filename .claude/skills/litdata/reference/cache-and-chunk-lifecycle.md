# Cache, prefetch, and chunk lifecycle

How remote chunks land on disk, how much space they use, and how workers share/delete them. User knobs first; contributor invariants below. Chunk binary layout → [streaming.md](streaming.md).

## User model

```
remote URL / local path  →  download into cache_dir  →  deserialize sample  →  (optional) delete chunk when done
```

`StreamingDataset` builds a `Cache` (`streaming/cache.py`) that owns a `BinaryReader` (and, on the write path, a `BinaryWriter`). Users almost never construct `Cache` directly.

### Knobs

| Knob               | Default                                                                      | Effect                                                   |
| ------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------- |
| `cache_dir`        | `LITDATA_CACHE_DIR` or `~/.lightning/chunks` (Studio: often `/cache/chunks`) | Where `.bin` chunks are stored                           |
| `max_cache_size`   | `"100GB"`                                                                    | Evict consumed chunks when the cache folder exceeds this |
| `max_pre_download` | `2` (async remote often floors to ≥4)                                        | Chunks each worker may prefetch ahead                    |

**Peak disk ≈ `num_workers × max_pre_download × mean_chunk_size`.**
Example: 8 workers × prefetch 4 × 64 MB ≈ 2 GB in flight before eviction. Keep `max_cache_size` comfortably above that (code warns below ~25 GB for multi-worker training).

```python
dataset = StreamingDataset(
    "s3://bucket/data",
    cache_dir="/data/cache",
    max_cache_size="50GB",
    max_pre_download=4,
)
```

- `LITDATA_CACHE_DIR=/path` — global default without passing `cache_dir`.
- CLI: `litdata cache path` · `litdata cache clear`.
- `Dir(path=local_cache, url=remote)` when cache location and remote URL differ.
- Random access (`dataset[i]`) may fetch byte ranges (`on_demand_bytes`); iteration prefers full-chunk cache.

### Prefetch env (optional)

| Env                                | Role                                                                          |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| `LITDATA_ASYNC_CHUNK_PREFETCH=0/1` | Force off/on async download overlap (default on for remote)                   |
| `LITDATA_ASYNC_MIN_PRE_DOWNLOAD`   | Floor for `max_pre_download` when async is on (default 4; `0` disables floor) |

## Contributor: read hot path

`BinaryReader.read` (`reader.py`):

1. Load `ChunksConfig` / `index.json` (download once if remote).
2. Resolve `(chunk_path, begin, size)` for the index.
3. Ensure `PrepareChunksThread` is running and the chunk is queued.
4. Item loader waits until the `.bin` exists (busy-wait; force-redownload after ~30s; fail ~120s).
5. On chunk change: decrement lock + enqueue delete of the previous chunk.
6. On last index: stop the prepare thread.

`PrepareChunksThread` (one daemon per worker) services download / force-download / delete queues, gated by `max_pre_download` and `max_cache_size`.

### Prefetch / eviction invariants

- When delete-when-processed is active, budget capping may shrink `max_pre_download` so `workers × max_pre × chunk` fits near `max_cache_size`.
- **Never leave live `max_pre_download == 1` via that cap** — download and delete can deadlock (reader waits forever for the next chunk). Floor capped values at **2**.
- Prepare-thread teardown must not block on stuck `asyncio` default-executor workers (`shutdown(wait=False)`).

## Contributor: shared-chunk deletion

Shuffle may **split one chunk across several workers** on a node. Three guards prevent one worker deleting a file another still needs:

1. **Static skip list** — only the worker that finishes a shared chunk last may delete it (`can_delete` / `skip_chunk_indexes_deletion`). Must also apply on resume.
2. **`.cnt` + file lock** — refcount per chunk; delete only at zero.
3. **Force redownload** — last resort if a reader blocks on a missing `.bin`.

Debugging: multi-worker + small `max_cache_size` to force eviction; `enable_tracer()` for lock/delete timeline; `DEBUG_LITDATA=1` for tombstones. Hangs in “chunk not found” waits under tiny budgets → check prefetch floor (≥2), not only deletion.

Shuffle assignment details → [streaming.md](streaming.md). Fair throughput measurement → [benchmarking.md](benchmarking.md).
