# The processing (write) pipeline — `optimize` / `map`

All paths under `src/litdata/`. This pipeline fans work across workers (and machines) to transform raw data, and for `optimize` writes it into the litdata chunk format that the streaming pipeline reads. Chunk / `index.json` / `BinaryWriter` / `FsProvider` details → [storage-format.md](storage-format.md).

## Public API (`processing/functions.py`)

User-facing arg tables → [using-litdata.md](using-litdata.md) §9 and README `#optimize-kwargs` / `#map` / `#walk`.

- **`optimize(...)`** — `functions.py:387`. Runs `fn` per input; flatten via pytree → `chunk-*.bin` + `index.json`. **Exactly one of `chunk_size` / `chunk_bytes`.** Notable: `queue`+`ALL_DONE`, `align_chunking`, `use_checkpoint`, `mode="append"|"overwrite"`, `keep_data_ordered=False` (shared queue), `encryption`, `item_loader=TokensLoader()`, `weights`/`input_dir`, `num_nodes`/`machine`. → `LambdaDataChunkRecipe` / `QueueDataChunkRecipe` → `DataProcessor.run`.
- **`map(...)`** — `functions.py:242`. `fn(input, output_dir) -> None` (side effects only). Same worker/scale knobs + `error_when_not_empty`. → `LambdaMapRecipe`.
- **`merge_datasets(input_dirs, output_dir, max_workers=..., storage_options={})`** — `functions.py:675`. Copy chunks + concat `index.json`; matching `data_format`/compression required.
- **`walk(folder, max_workers=...)`** — `functions.py:621`. Threaded cloud `os.walk` (Studio-optimized); yield order is **not** depth-first.

## Multi-node launch (`num_nodes` / `machine`) — read this first

`num_nodes` is **not** local multiprocessing. It only works inside **Lightning Studio** (`_IS_IN_STUDIO`; else `ValueError` in `functions.py`). Dual path for both `map` and `optimize`:

```
if num_nodes is None OR DATA_OPTIMIZER_NUM_NODES > 0:
    → run DataProcessor on this machine (single-node OR a job worker)
else:
    → _execute(...)   # resolver.py:461 — create Studio data-prep job, block until done
```

1. User calls `optimize(..., num_nodes=N, machine=Machine.DATA_PREP)` on a Studio.
2. `_execute` starts a multi-instance job that re-runs `python {' '.join(sys.argv)}` on **N** machines (`resolver.py:483–492`). `machine=None` → current Studio machine. (`interruptible` exists on `_execute` but **optimize/map never pass it** — always `False`; do not document as a public knob.)
3. Platform injects `DATA_OPTIMIZER_NUM_NODES`, `DATA_OPTIMIZER_NODE_RANK`, etc. on each instance.
4. The same script hits the **local** branch (gate sees `DATA_OPTIMIZER_NUM_NODES > 0`) and each node processes only its shard.
5. Caller blocks until the job completes or fails (`FAILED` → `RuntimeError`). Job URL is printed to the Studio Runs UI.

**Prefer durable `output_dir`:** `/teamspace/s3_connections/...`, `/teamspace/datasets/...`, or `s3://...`.
If optimize’s `output_dir` is under `/teamspace/studios/this_studio` **and** workers are multi-node, LitData rewrites it to the job artifacts bucket via `_get_work_dir()` (`functions.py:515–524` → `utilities.py:196–205` → `s3://{LIGHTNING_BUCKET_NAME}/projects/.../artifacts/{work_id}/content/...`). **`map` does not apply this remap.** Paths like `/teamspace/jobs/...` are the Studio job mount UI — LitData does not construct that string itself. Rejects outputs whose URL contains `cloudspaces` (use connections/datasets instead).

### Env vars (multi-node / workers)

| Var                                                                                                        | Role                                                                          |
| ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `DATA_OPTIMIZER_NUM_NODES`                                                                                 | Launch gate + world size; `>0` means “I’m a job worker / already distributed” |
| `DATA_OPTIMIZER_NODE_RANK`                                                                                 | This node’s rank `[0, num_nodes)`                                             |
| `DATA_OPTIMIZER_GLOBAL_RANK` / `DATA_OPTIMIZER_NUM_WORKERS`                                                | Set inside workers for chunk filenames / writer rank                          |
| `DATA_OPTIMIZER_CACHE_FOLDER` / `DATA_CACHE_FOLDER`                                                        | Cache roots (Studio often `/cache/...`)                                       |
| `DATA_OPTIMIZER_TIMEOUT`                                                                                   | Queue get timeout (default 300s; shared-queue ~200s)                          |
| `DATA_OPTIMIZER_FAST_DEV_RUN`                                                                              | Related to `fast_dev_run` defaults                                            |
| `LIGHTNING_SKIP_INSTALL` / `LIGHTNING_BRANCH`                                                              | Injected into remote job command                                              |
| `LIGHTNING_BUCKET_NAME`, `LIGHTNING_CLOUD_PROJECT_ID`, `LIGHTNING_CLOUD_APP_ID`, `LIGHTNING_CLOUD_WORK_ID` | `_get_work_dir()` artifacts URL                                               |
| `ENABLE_STATUS_REPORT`                                                                                     | Extra progress reporting                                                      |

### Sharding & index merge (`data_processor.py`)

- `world_size = num_nodes * num_workers`. Items packed across **all** ranks, then each node keeps only its worker slice. **No cross-node RPC** — pure env coordination.
- Each node writes per-rank chunk files + `{rank}-index.json` (node-local).
- **Last node** (`num_nodes == node_rank + 1`) waits for peer index files, merges into final `index.json`, and uploads. Peer wait can **hang** if a node never writes its index.
- Every node needs credentials for inputs/outputs (connections → temp creds; raw `s3://` → keys on all instances).

User cookbook: [using-litdata.md](using-litdata.md) §9. Studio UX: [lightning-studio.md](lightning-studio.md).

## Orchestration (`processing/data_processor.py`)

`DataProcessor.run(recipe)` (`data_processor.py:1226`) is the single entrypoint both `optimize` and `map` call. Lifecycle:

1. **`recipe.prepare_structure(input_dir)`** (`:1244`) returns the work list — must be a `list`, a `StreamingDataLoader`, or a `multiprocessing.Queue` (else raises, `:1247`).
2. **Item→worker assignment** (`:1258-1287`) → `workers_user_items: list[list]` (one sublist per local worker), via:
   - `_map_items_to_workers_weighted` (`:377`) — default when `reorder_files` + `input_dir` exist, or when `weights` given. Bin-packs by file size (`_pack_greedily`) across `world_size = num_nodes * num_workers`, then permutes.
   - `_map_items_to_workers_sequentially` (`:303`) — contiguous slices; `align_chunking` packs full chunks.
   - Queue mode — no static assignment; `shared_queue` is set.
3. **Multi-node slicing** via `_get_node_rank()`/`_get_num_nodes()` — see section above.
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

Each `BaseWorker` runs a local pipeline of child processes (spawned in `_setup`, `:580`):

- **Downloaders** (`_start_downloaders`, `:797`; target `_download_data_target`, `:128`): `num_downloaders` procs pull `(index, item, paths)` off `to_download_queues`, fetch remote files into cache, push ready tuples onto `ready_to_process_queue`.
- **Worker main loop** (`_loop`, `:601`) — the **consumer**: `ready_to_process_queue.get()` → `_handle_data_chunk_recipe` (optimize) or `_handle_data_transform_recipe` (map). Reports progress ~1/s.
- **Uploaders** (`_start_uploaders`, `:839`; target `_upload_fn`, `:232`): push finished chunks/files to `output_dir`.
- **Remover** (`_start_remover`, `:825`; target `_remove_target`, `:190`): deletes processed source files when `remove=True`.

When `no_downloaders` (no `input_dir`, or a `reader` is set), `ready_to_process_queue` is a `FakeQueue` and `_collect_paths` (`:743`) pushes items directly.

**Ordered vs shared-queue** (`keep_data_ordered`): `True` (default) → each worker consumes its static slice in order. `False` → all workers share one `Queue` for dynamic load balancing; termination uses the `ALL_DONE` sentinel (`:64`), which each worker re-inserts so peers also stop (`:621`).

## Cross-process queues

| Queue                                                      | Direction         | Purpose                           |
| ---------------------------------------------------------- | ----------------- | --------------------------------- |
| `error_queue`                                              | worker→main       | tracebacks; triggers global abort |
| `progress_queue`                                           | worker→main       | `(index, counter)` for tqdm       |
| `msg_queue`                                                | worker→main       | log lines routed around tqdm      |
| `stop_queues`                                              | main→worker       | SIGINT graceful stop              |
| `ready_to_process_queue` / `shared_queue`                  | downloader→worker | core work items                   |
| `to_download_queues` / `to_upload_queues` / `remove_queue` | worker→child      | I/O offload                       |

## `raw/` — `StreamingRawDataset` and the indexer

`StreamingRawDataset` (`raw/dataset.py:95`) is a plain `torch.utils.data.Dataset` streaming **original files** (no optimize step). Item structure is user-defined via `setup(files)` (`:151`) returning `list[FileMetadata]` or `list[list[FileMetadata]]` (grouped items). Uses async batched downloads (`__getitems__`/`_download_batch`, `asyncio.gather`). `CacheManager` (`:32`) mirrors remote structure into a local cache (opt-in). Pass `recompute_index=True` to force a rebuild (`indexer.py:58`).

The **indexer** (`raw/indexer.py`) replaces the optimize pass: `BaseIndexer.build_or_load_index` (`:53`) tries a local cached index, then a remote one (`input_dir/index.json.zstd`), rebuilding via `discover_files` only if neither exists or `recompute_index=True`. `FileIndexer` (`:217`) recursively lists files (fsspec `fs.find` / `Path.rglob`). Note the index filename `index.json.zstd` differs from the optimized `index.json` — don't conflate them.

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
