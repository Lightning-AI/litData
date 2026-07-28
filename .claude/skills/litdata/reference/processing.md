# The processing (write) pipeline — `optimize` / `map`

All paths under `src/litdata/`. This pipeline fans work across workers (and machines) to transform raw data, and for `optimize` writes it into the litdata chunk format that the streaming pipeline reads.

**Load these when the task touches I/O or scale:**

| Topic                                                                                | Doc                                                   |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| Downloaders / uploaders / removers, FUSE→URL, cache dirs, FsProvider vs `Downloader` | **[data-movement.md](data-movement.md)** (exhaustive) |
| Multi-node Studio jobs, sharding, index merge, checkpoints, pitfalls                 | **[multi-node.md](multi-node.md)** (exhaustive)       |
| Chunk / `index.json` / `BinaryWriter` / `FsProvider`                                 | [storage-format.md](storage-format.md)                |
| Path URI tables                                                                      | [resolver.md](resolver.md)                            |

## Public API (`processing/functions.py`)

User-facing arg tables → [using-litdata.md](using-litdata.md) §9 and README `#optimize-kwargs` / `#map` / `#walk`.

- **`optimize(...)`** — `functions.py` `optimize`. Runs `fn` per input; flatten via pytree → `chunk-*.bin` + `index.json`. **Exactly one of `chunk_size` / `chunk_bytes`.** Notable: `queue`+`ALL_DONE`, `align_chunking`, `use_checkpoint`, `mode="append"|"overwrite"`, `keep_data_ordered=False` (shared queue), `encryption`, `item_loader=TokensLoader()`, `weights`/`input_dir`, `num_nodes`/`machine`, `num_downloaders`/`num_uploaders`, `broadcast_paths=False` (auto-on for `{%strftime}` paths — see [multi-node.md](multi-node.md) §3.4). → `LambdaDataChunkRecipe` / `QueueDataChunkRecipe` → `DataProcessor.run`.
- **`map(...)`** — `functions.py` `map`. `fn(input, output_dir) -> None` (side effects only). Same worker/scale knobs + `error_when_not_empty` + `broadcast_paths`. → `LambdaMapRecipe`.
- **`merge_datasets(input_dirs, output_dir, max_workers=..., storage_options={})`** — Copy chunks + concat `index.json`; matching `data_format`/compression required.
- **`walk(folder, max_workers=...)`** — Threaded cloud `os.walk` (Studio-optimized); yield order is **not** depth-first.

## Multi-node & data movement — start here

- **`num_nodes` is not local multiprocessing** and is **not** torch.distributed/SLURM. Studio-only job launch via `_execute` (`resolver.py`). Full launch gate, env vars, sharding, who uploads chunks vs who merges `index.json`, checkpoints/append, and hang modes → **[multi-node.md](multi-node.md)**.
- **Per-worker I/O children** (`_download_data_target`, `_upload_fn`, `_remove_target`), resolver FUSE→`s3://`/`gs://`/`r2://`, cache folders, and streaming `Downloader` distinction → **[data-movement.md](data-movement.md)**.
- Quick rule: pass `/teamspace/s3_connections/…` or cloud URLs into LitData so resolve+FsProvider bypass FUSE; prefer durable remote `output_dir` on multi-node.

## Orchestration (`processing/data_processor.py`)

`DataProcessor.run(recipe)` (`data_processor.py:1226`) is the single entrypoint both `optimize` and `map` call. Lifecycle:

1. **`recipe.prepare_structure(input_dir)`** (`:1244`) returns the work list — must be a `list`, a `StreamingDataLoader`, or a `multiprocessing.Queue` (else raises, `:1247`).
2. **Item→worker assignment** (`:1258-1287`) → `workers_user_items: list[list]` (one sublist per local worker), via:
   - `_map_items_to_workers_weighted` (`:377`) — default when `reorder_files` + `input_dir` exist, or when `weights` given. Bin-packs by file size (`_pack_greedily`) across `world_size = num_nodes * num_workers`, then permutes.
   - `_map_items_to_workers_sequentially` (`:303`) — contiguous slices; `align_chunking` packs full chunks.
   - Queue mode — no static assignment; `shared_queue` is set.
3. **Multi-node slicing** via `_get_node_rank()`/`_get_num_nodes()` — [multi-node.md](multi-node.md).
4. **Checkpointing** (`:1297`) trims each worker's list to resume from `checkpoint_next_index`; `fast_dev_run` trims to N items.
5. **`_create_process_workers`** (`:1462`) spawns one `DataWorkerProcess` per worker.
6. **Progress loop** (`:1376`) polls `error_queue` (re-raises via `_exit_on_error`, which `terminate()`s all workers) and `progress_queue` (tqdm). Exits when the counter equals `num_items` or all workers die.
7. **`recipe._done(...)`** merges caches / writes the index; index merge runs only on the last node.

## Key classes

- **`DataProcessor`** (`:1114`) — the main-process orchestrator. Owns `error_queue`, `msg_queue`, `progress_queue`, `stop_queues`, `shared_queue`.
- **`BaseWorker`** (`:481`) — the processing unit. `run()` (`:570`) = `_setup()` → `_loop()` → `_terminate()`, wrapping all exceptions into `error_queue`.
- **`DataWorkerProcess`** (`:927`) — `BaseWorker` + `multiprocessing.Process`; what actually runs in a child process.
- **`DataRecipe`** (`:948`) abstract: `prepare_structure` + `prepare_item` + `_done`.
  - **`DataChunkRecipe`** (`:981`) — for `optimize`; chunk_size/chunk_bytes (defaults to 64 MB, `:995`), compression, encryption; `_done` merges + uploads index.
  - **`MapRecipe`** (`:1100`) — for `map`; `prepare_item` writes to output and must return `None`.
- **`FakeQueue`** (`:461`) — drop-in for `Queue` when no downloading is needed (in-process, avoids serialization).

## Producer/consumer model (inside each worker)

Each `BaseWorker` runs a local pipeline of child processes (spawned in `_setup`):

| Child       | Start                | Target                  | Default count              | Role                                                                                              |
| ----------- | -------------------- | ----------------------- | -------------------------- | ------------------------------------------------------------------------------------------------- |
| Downloaders | `_start_downloaders` | `_download_data_target` | `num_downloaders or 2`     | Prefetch inputs into `DATA_OPTIMIZER_DATA_CACHE_FOLDER` via **FsProvider** (not `Downloader` ABC) |
| Uploaders   | `_start_uploaders`   | `_upload_fn`            | `num_uploaders or 1`       | Push chunks / map outputs to `output_dir`                                                         |
| Remover     | `_start_remover`     | `_remove_target`        | 1 if `delete_cached_files` | Delete local cached inputs + uploaded chunk files                                                 |

Worker main loop (`_loop`): `ready_to_process_queue.get()` → `_handle_data_chunk_recipe` or `_handle_data_transform_recipe`.

**`no_downloaders`** when `input_dir.path is None` **or** a `reader` is set — including pure `s3://` `Dir(path=None, url=…)` (downloaders need a FUSE/local `path` to rewrite). Studio connections set both `path` and `url`.

**`remove` flag** = `DataProcessor.delete_cached_files` (default True); not exposed on public `optimize()`/`map()`.

Exhaustive I/O (path rewrite, disk wait 25 GB, index upload vs chunk uploaders, error modes) → **[data-movement.md](data-movement.md)**.

**Ordered vs shared-queue** (`keep_data_ordered`): `True` (default) → each worker consumes its static slice in order. `False` → all workers share one `Queue`; termination uses `ALL_DONE` (re-inserted so peers stop). Multi-node: shared queue is **per node**, not global — [multi-node.md](multi-node.md).

## Cross-process queues

| Queue                                                      | Direction         | Purpose                                              |
| ---------------------------------------------------------- | ----------------- | ---------------------------------------------------- |
| `error_queue`                                              | worker→main       | tracebacks; `_exit_on_error` `terminate()`s siblings |
| `progress_queue`                                           | worker→main       | `(index, counter)` for tqdm                          |
| `msg_queue`                                                | worker→main       | log lines routed around tqdm                         |
| `stop_queues`                                              | main→worker       | SIGINT graceful stop                                 |
| `ready_to_process_queue` / `shared_queue`                  | downloader→worker | core work items                                      |
| `to_download_queues` / `to_upload_queues` / `remove_queue` | worker→child      | I/O offload                                          |

## `raw/` — `StreamingRawDataset` (first-class; no optimize)

User cookbook → [using-litdata.md](using-litdata.md) §10. README → `#stream-raw`. Adaptive stages → repo `benchmarks/ADAPTIVE_CONCURRENCY.md`.

`StreamingRawDataset` (`raw/dataset.py`) is a **map-style** `torch.utils.data.Dataset` that streams **original files** (JPEG, audio, …) from local or cloud paths. It does **not** use LitData chunks, `BinaryReader`, or `StreamingDataLoader`.

```
input_dir → FileIndexer (index.json.zstd) → setup(files) → items
         → CacheManager + Downloader (fully async) → raw bytes [/ transform]
```

| Piece                                            | Role                                                                                            |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `FileIndexer` / `BaseIndexer` (`raw/indexer.py`) | Discover files; cache `index.json.zstd` locally + upload beside remote data                     |
| `CacheManager`                                   | Optional on-disk file cache (`cache_files=True`); always holds index cache dir                  |
| `_LoopRunner`                                    | Per-process dedicated asyncio thread (optional uvloop); recreate after fork                     |
| `setup(files)`                                   | Default identity; override to filter/group → `list[FileMetadata]` or `list[list[FileMetadata]]` |
| `__getitem__` / `__getitems__`                   | **Fully async** download; batches use `asyncio.gather` over `adownload_fileobj`                 |
| Cloud clients                                    | **Built-in retries** (e.g. S3 adaptive `max_attempts`) for transient failures                   |
| Default item                                     | **`bytes`** (or `list[bytes]` if grouped) — caller decodes however they want                    |
| `transform`                                      | Optional post-download; signature matches item shape (`bytes` vs `list[bytes]`)                 |

**Do not conflate indexes:** raw = `index.json.zstd` (file list). Optimized = `index.json` (chunk metadata).

### Operational invariants (edit with care)

- **Division of labor:** clients own **rate** (boto/obstore retries). Litdata owns **concurrency** (`max_concurrent_downloads`) and **look-ahead** (`max_prefetch`). Do not nest a litdata rate loop that fights client retries. Stage 1 = static size-aware budget when `max_concurrent_downloads=None`; Stages 2+ (prefetch hit-rate, AIMD) are deferred — see design note.
- **Fork / spawn safety:** `register_at_fork` shuts down the runner; pid-guarded caches recreate downloader / permits / range executor when pid or event loop changes. `__getstate__` is an **allowlist** of constructor knobs (runtime handles reset on unpickle).
- **Atomic publishes:** downloaded cache files **and** `index.json.zstd` use tmp + `os.replace` (tmp includes pid). Partial writes must not become visible readers.
- **Batch timeout:** `download_timeout` wraps the batch gather once; per-item GETs stay on the fast path when `hedge_delay=0`. Timeout must cancel `_inflight` entries or retries hang on the poisoned task.
- **Indexer schemes:** `urlparse("C:\\Users\\...")` yields `scheme='c'`. Single-letter schemes are Windows drive letters — local paths, not unsupported remotes (`_is_windows_drive_scheme`).
- **Tests:** `tests/raw/test_fork_safety.py` covers fork reinit, allowlist pickle, atomic publish, batch-timeout hang recovery, and fast-path coexistence with default `download_timeout=120`.

**Agent guidance:** lead with `StreamingRawDataset` when the user has an existing file tree and has not asked for max throughput / resume. Prefer cloud URL over FUSE. Stress: raw bytes + async batched downloads + retries; upgrade path (shuffle inputs →) `optimize` + `StreamingDataset`. Same path resolver as streaming (`/teamspace/s3_connections/…`, `s3://`, …).

## Gotchas (read before editing the engine)

01. `prepare_structure` must return `list | StreamingDataLoader | multiprocessing.Queue` (`:1247`).
02. `map` recipes' `prepare_item` must return `None` (`:913`); `optimize` recipes' return value is serialized.
03. Start method forced globally: `set_start_method(..., force=True)` (`:1177`), `fork` in notebooks else `spawn`. Under `spawn`, worker args must be picklable.
04. Distribution is env-var driven (`DATA_OPTIMIZER_*`); mappers slice by `world_size = num_nodes * num_workers`.
05. `keep_data_ordered=False` switches to a shared queue + `ALL_DONE` sentinel; early-exit paths must keep re-inserting `ALL_DONE` or peers hang. Timeout is lower (200s vs 300s).
06. `FakeQueue` is not a real queue — no inter-process/blocking semantics.
07. Path detection is heuristic (`_is_path`/`_to_path`, `functions.py:439`); an item with zero detected file paths raises (`:775`).
08. `align_chunking` requires `chunk_size` (not bytes) (`:497`, `:1275`).
09. Index merge only on the last node (`num_nodes == node_rank + 1`).
10. Errors are swallowed into `error_queue` then re-raised in the main loop; a worker exception without a traceback string can hang the loop. `_exit_on_error` hard-`terminate()`s siblings — no graceful flush. **To surface a hidden error, run with `num_workers=1` or `fast_dev_run=True`.**
11. Checkpointing unsupported for Queue inputs (`:1298`) or generator `fn`s (`:1303`).

## Runnable examples (from README)

```python
# Optimize a dataset into chunks (README:144)
import litdata as ld
def fn(index):
    return {"index": index, "image": ..., "class": ...}
if __name__ == "__main__":
    ld.optimize(fn=fn, inputs=list(range(1000)), output_dir="fast_data",
                num_workers=4, chunk_bytes="64MB")

# Map: transform files in parallel, write to output_dir (README:205)
import os, litdata as ld
from PIL import Image
def resize_image(image_path, output_dir):
    out = os.path.join(output_dir, os.path.basename(image_path))
    Image.open(image_path).resize((224, 224)).save(out)
ld.map(fn=resize_image, inputs=inputs, output_dir="output_dir")

# Encrypt at sample level (README:1491)
from litdata.utilities.encryption import FernetEncryption
enc = FernetEncryption(password="secret", level="sample")
ld.optimize(fn=fn, inputs=..., output_dir="enc_data", chunk_bytes="64MB", encryption=enc)
```

README feature sections: LLM pre-training / tokenization (721), filter illegal data (784), shared queue for optimize (607), queue as input (664), merge (995), distributed optimize (1436), encryption (1478), Lightning data connections (1615).
