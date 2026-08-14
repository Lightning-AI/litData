# Data movement: downloaders, uploaders, removers, path resolution

Operational reference for how LitData moves bytes between local disk and remote storage during **`optimize` / `map`** and how that relates to the **streaming / raw read** path. Grep landmarks use `src/litdata/` paths.

**Related:** [processing.md](processing.md) (orchestrator) · [multi-node.md](multi-node.md) · [resolver.md](resolver.md) · [storage-format.md](storage-format.md) (`FsProvider` vs `Downloader`) · [lightning-studio.md](lightning-studio.md)

______________________________________________________________________

## 0. Two different “downloaders” — do not conflate

| Path                                                             | What agents mean by “downloader”                                           | Module / symbols                                                                                                                                                                 | Transport                                                                                           |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Write / processing** (`optimize`, `map`)                       | Child **processes** per worker that prefetch input files into a data cache | `processing/data_processor.py`: `_download_data_target`, `_start_downloaders`                                                                                                    | **`FsProvider`** (`streaming/fs_provider.py`) for `s3`/`gs`/`r2`; local `shutil.copyfile` otherwise |
| **Read / streaming** (`StreamingDataset`, `StreamingRawDataset`) | **`Downloader` ABC** subclasses selected by URL prefix                     | `streaming/downloader.py`: `Downloader`, `S3Downloader`, `GCPDownloader`, `R2Downloader`, `AzureDownloader`, `HFDownloader`, `LocalDownloader`, `get_downloader`, `_DOWNLOADERS` | Cloud SDKs / obstore / boto3 per subclass                                                           |

There are **no** classes named `Uploader` or `Remover`. Processing upload/remove are process targets `_upload_fn` and `_remove_target` in `data_processor.py`.

**Schemes:**

- Processing I/O via FsProvider: `_SUPPORTED_PROVIDERS = ("s3", "gs", "r2")` in `constants.py`.
- Streaming downloaders also support `azure://`, `hf://`, `local:` (see `_DOWNLOADERS`).

______________________________________________________________________

## 1. End-to-end data movement (`optimize` / `map`)

```
User paths (inputs / input_dir / output_dir)
        │
        ▼
_resolve_dir  (streaming/resolver.py)
        │  Dir(path=…, url=…, data_connection_id=?)
        ▼
DataProcessor  (processing/data_processor.py)
        │  broadcast_object input/output Dir (only if broadcast_paths / `{%strftime}`)
        │  shard items → DataWorkerProcess × num_workers
        ▼
Per worker (BaseWorker._setup):
  _collect_paths → rewrite item paths to cache_data_dir when remote/FUSE
  _start_downloaders  → Process(_download_data_target) × num_downloaders
  _start_uploaders    → Process(_upload_fn) × num_uploaders
  _start_remover      → Process(_remove_target) if delete_cached_files
        │
        ▼
ready_to_process_queue → user fn (optimize→Cache/BinaryWriter | map→temp out dir)
        │
        ▼
_try_upload → to_upload_queues → remote/local output_dir
        │
        ▼
remove_queue → delete local intermediates (inputs from data cache; uploaded chunks)
        │
        ▼
DataChunkRecipe._done → merge per-worker indexes → upload index.json
  (multi-node: {node_rank}-index.json then last-node merge — see multi-node.md)
```

Public knobs (`processing/functions.py` → `DataProcessor`):

| Knob                  | Default                                       | Meaning                                                                                                                                                           |
| --------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_downloaders`     | `2` (`DataProcessor`: `num_downloaders or 2`) | Downloader processes **per worker**                                                                                                                               |
| `num_uploaders`       | `1`                                           | Uploader processes **per worker**                                                                                                                                 |
| `delete_cached_files` | `True` on `DataProcessor`                     | Passed to worker as `remove`; starts remover. **Not** exposed on public `optimize()` / `map()` — stays default True unless you construct `DataProcessor` yourself |
| `input_dir`           | Auto via `_get_input_dir(inputs)` or explicit | Resolved `Dir`; drives download + path rewrite                                                                                                                    |
| `output_dir`          | Required                                      | Resolved `Dir`; drives upload                                                                                                                                     |
| `storage_options`     | `{}`                                          | Merged with `data_connection_id` via `construct_storage_options` (`processing/utilities.py`)                                                                      |

Cache roots (`data_processor.py`):

| Helper                                    | Env override                       | Default                                          |
| ----------------------------------------- | ---------------------------------- | ------------------------------------------------ |
| `_get_cache_dir` (chunks)                 | `DATA_OPTIMIZER_CACHE_FOLDER`      | Studio: `/cache/chunks`; else `{tempdir}/chunks` |
| `_get_cache_data_dir` (downloaded inputs) | `DATA_OPTIMIZER_DATA_CACHE_FOLDER` | Studio: `/cache/data`; else `{tempdir}/data`     |

`DataProcessor._cleanup_cache` **rmtrees both** at the start of each `run()` so prior runs cannot poison the job.

______________________________________________________________________

## 2. Path resolution → when remote I/O kicks in

Canonical tables: [resolver.md](resolver.md). Processing always goes through `_resolve_dir` in `DataProcessor.__init__` and in `optimize`/`map` before constructing the processor.

### 2.1 `Dir` fields that control movement

```python
@dataclass
class Dir:
    path: str | None   # local / FUSE mount path (identity, cache rewrite base)
    url: str | None    # cloud URL used for FsProvider download/upload
    data_connection_id: str | None  # temp creds for some Studio connections
```

| Situation                                                                                                        | `path`         | `url`                               | Processing download behavior                                                                 |
| ---------------------------------------------------------------------------------------------------------------- | -------------- | ----------------------------------- | -------------------------------------------------------------------------------------------- |
| Plain local dir                                                                                                  | abs path       | `None`                              | No cloud download; may `shutil.copyfile` into data cache if path is outside `this_studio`    |
| Direct `s3://` / `gs://` / `r2://`                                                                               | `None`         | cloud URL                           | **Downloader procs skip** (`no_downloaders` when `input_dir.path is None`) — see §3.1 caveat |
| Studio FUSE: `/teamspace/s3_connections/…`, `s3_folders`, `gcs_*`, `lightning_storage`, `datasets`, other studio | FUSE path      | backing `s3://` / `gs://` / `r2://` | Downloaders rewrite FUSE→URL and `FsProvider.download_file`                                  |
| `/teamspace/studios/this_studio/…`                                                                               | workspace path | `None`                              | Local; LitData does not invent a bucket URL                                                  |

**Agent rule (same as raw streaming):** pass `/teamspace/s3_connections/…` or `s3://…` into LitData. Do **not** train or bulk-copy through FUSE with bare `open()` / `cp`. Resolver + FsProvider talk to the object store directly.

### 2.2 Studio mount → URL (resolver functions)

| Mount prefix                            | Resolver                     | Typical `url`                                       |
| --------------------------------------- | ---------------------------- | --------------------------------------------------- |
| `/teamspace/s3_connections/<name>/…`    | `_resolve_s3_connections`    | customer S3 (`data_connection.aws.source` + suffix) |
| `/teamspace/s3_folders/<name>/…`        | `_resolve_s3_folders`        | S3 folder connection source + suffix                |
| `/teamspace/gcs_connections/<name>/…`   | `_resolve_gcs_connections`   | `gs://…`                                            |
| `/teamspace/gcs_folders/<name>/…`       | `_resolve_gcs_folders`       | `gs://…`                                            |
| `/teamspace/lightning_storage/<name>/…` | `_resolve_lightning_storage` | `r2://…` + **always** `data_connection_id`          |
| `/teamspace/datasets/…`                 | `_resolve_datasets`          | cluster datasets S3                                 |
| `/teamspace/studios/<other>/…`          | `_resolve_studio`            | studio content `s3://` or `gs://`                   |

Connection name = path segment `[3]`. Credentials: ambient cloud keys, or temp project-role creds when `data_connection_id` is set (`streaming/client.py`).

### 2.3 How `input_dir` is inferred

`_get_input_dir(inputs)` (`functions.py`):

1. Flatten first (or second) input; find filepath-like strings (`_get_indexed_paths` / `_is_remote_file`).
2. Remote scheme → `os.path.dirname(path)` (e.g. `s3://bucket/prefix`).
3. Studio / `/teamspace…` → keep first **four** path segments as root (e.g. `/teamspace/s3_connections/my-conn`).
4. Else → `None` (no shared input root; often `no_downloaders`).

Workers detect per-item paths with `_is_path(input_dir.path, element)` / `_to_path` — heuristic; items with **zero** paths raise in `_collect_paths`.

______________________________________________________________________

## 3. Processing downloaders (`_download_data_target`)

### 3.1 When they start

`BaseWorker.no_downloaders = (input_dir.path is None) or (reader is not None)`.

- **`no_downloaders` True** → `_start_downloaders` returns immediately; items go straight to `ready_to_process_queue` (or `FakeQueue` when ordered + no downloaders).
- **Important:** pure `s3://…` input resolves to `Dir(path=None, url=…)`, so **`path is None` ⇒ downloaders do not run**. Background download assumes a local/FUSE `path` to rewrite into `cache_data_dir`. For Studio connections, `path` is set (FUSE) **and** `url` is set — that is the path where downloaders matter.
- Custom `reader` (e.g. `StreamingDataLoaderReader`) also disables downloaders; the reader supplies bytes.

Defaults: `num_downloaders or 2` processes per worker. Each is `multiprocessing.Process(target=_download_data_target, args=(input_dir, cache_data_dir, to_download_queue, ready_to_process_queue, storage_options))`.

### 3.2 Queue protocol

1. `_collect_paths` builds `self.paths` and rewrites flattened filepath leaves under `input_dir.path` → `cache_data_dir` (unless path starts with `/teamspace/studios/this_studio`).
2. `_start_downloaders` enqueues `(index, item, paths)` round-robin across `to_download_queues`, then sends `None` sentinel per downloader.
3. Downloader loop: `queue_in.get()` → download/copy → `queue_out.put((index, item, paths))`; on `None`, put `None` and exit.
4. Worker `_loop` consumes `ready_to_process_queue` and runs the recipe.

### 3.3 Per-path download logic (`_download_data_target`, ~`:128`)

For each path in the item:

1. If all paths already exist under the cache rewrite → skip download, forward tuple.
2. If `input_dir.url` is set → `_wait_for_disk_usage_higher_than_threshold("/", 25)` (wait until **>25 GB free** on `/`) so removers can catch up under pressure.
3. Local cache target: `path.replace(input_dir.path, cache_dir)`.
4. If `url` and `path` and the FUSE/local file is **missing**: rewrite path with `path.replace(input_dir.path, input_dir.url)` → cloud URL.
5. If `urlparse(path).scheme in _SUPPORTED_PROVIDERS` (`s3`/`gs`/`r2`):
   - `construct_storage_options(storage_options, input_dir)` (injects `data_connection_id`)
   - `_get_fs_provider(input_dir.url, …).download_file(remote, local_path)`
6. Elif `os.path.isfile(path)` and not under `this_studio`: `shutil.copyfile` into cache.
7. Else: `ValueError` unsupported URL.

**Local vs remote summary:**

| Input                                 | Action                                                                        |
| ------------------------------------- | ----------------------------------------------------------------------------- |
| FUSE connection + missing local file  | Resolve to `url`, FsProvider download into `DATA_OPTIMIZER_DATA_CACHE_FOLDER` |
| Already cached under `cache_data_dir` | No-op, pass through                                                           |
| Real local file outside `this_studio` | Copy into data cache                                                          |
| `this_studio` local                   | Leave path as-is (no copy into cache for that prefix)                         |
| Unsupported scheme                    | Raise                                                                         |

### 3.4 Interaction with user `fn`

After download, item tree leaves point at **cache_data_dir** paths (rewritten in `_collect_paths`). `prepare_item` / user `fn` should open those local paths — not the original FUSE or `s3://` strings — when downloaders ran.

______________________________________________________________________

## 4. Processing uploaders (`_upload_fn`)

### 4.1 When they start

`_start_uploaders` runs unless **both** `output_dir.path` and `output_dir.url` are `None`.

- Remote `output_dir.url` with scheme in `_SUPPORTED_PROVIDERS` → `FsProvider.upload_file`.
- Local `output_dir.path` → `shutil.copy` into destination (makedirs as needed).
- Else → `ValueError`.

Default `num_uploaders or 1` per worker.

### 4.2 Who enqueues uploads

| Recipe                             | What gets uploaded                                                                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`optimize` / `DataChunkRecipe`** | Each closed chunk filepath from `Cache._add_item` / `cache.done()`; optional checkpoint JSON under `.checkpoints` when `use_checkpoint`                             |
| **`map` / `MapRecipe`**            | Every file under a per-item `tempfile.mkdtemp()` after `prepare_item` (user writes into that dir); uploaded as `(tmpdir, filepath)` so relative layout is preserved |

`_try_upload` no-ops if output_dir has neither path nor url, or data is empty/missing on disk. Round-robins across `to_upload_queues`.

On worker shutdown (downloaders finished / timeout / `ALL_DONE`): send `None` to each uploader and `join`.

### 4.3 Upload path construction (`_upload_fn`, ~`:232`)

1. Ensure `local_filepath` is under `cache_chunks_dir` (join if relative).
2. Remote destination:
   - Base = `output_dir.url`
   - If path contains `.checkpoints` → nest under `…/.checkpoints`
   - Basename-only upload for optimize chunks; map preserves relative path under tmpdir
   - `remove_uuid_from_filename` strips UUID from checkpoint names → `checkpoint-<rank>.json`
3. After successful upload (or local copy): if `remove_queue` and file exists → `remove_queue.put([local_filepath])` so the **remover deletes the local chunk** after upload.

### 4.4 Index upload (not the uploader pool)

Chunk uploaders do **not** own the final index. After all workers finish, `DataChunkRecipe._done` → `_merge_no_wait` → `_upload_index`:

- Single-node: upload `index.json` from cache to `output_dir`.
- Multi-node: each node uploads `{node_rank}-index.json`; **last node** downloads peers’ indexes, merges, uploads final `index.json` — [multi-node.md](multi-node.md).

`MapRecipe._done` does not merge LitData chunk indexes (map is side-effect files only).

______________________________________________________________________

## 5. Processing removers (`_remove_target`)

### 5.1 When they start

`_start_remover` only if `self.remove` is True. That flag is `DataProcessor.delete_cached_files` (default **True**), passed positionally into `DataWorkerProcess` / `BaseWorker`.

### 5.2 What gets deleted

Two producers feed `remove_queue`:

1. **After each item** (if `remove` and `input_dir.path` and no `reader`): worker puts the item’s **source paths** (original path list) so cached downloads under `cache_data_dir` can be freed.
2. **After each upload**: uploader puts the **local chunk/file** path.

`_remove_target` (~`:190`):

- Rewrite paths from `input_dir.path` → `cache_dir` when needed.
- `os.remove` if exists.
- If `input_dir` is falsy: only delete if `keep_path(path)` is True — **refuses** to delete paths containing Studio mount tokens: `s3_connections`, `s3_folders`, `gcs_connections`, `efs_*`, `lightning_storage`, `snowflake_connections` (safety against wiping FUSE mounts).

On shutdown with `remove`: put `None` sentinel and join remover.

### 5.3 Post-run check

`DataChunkRecipe._done`: if `delete_cached_files` and **local** `output_dir.path` is set and any `.bin` still remain in chunk cache → `RuntimeError` (“All the chunks should have been deleted”). Remote outputs rely on uploaders + remover; local outputs expect chunks to have been copied away and removed.

### 5.4 What removers do **not** do

- They do **not** delete remote objects.
- They do **not** delete the durable dataset under `output_dir` (except separate overwrite/checkpoint cleanup via FsProvider in `_cleanup_checkpoints` / resolver immutability helpers).
- `DataProcessor._cleanup_cache` wipes entire cache dirs at **start** of the next run, independent of the remover process.

______________________________________________________________________

## 6. Streaming / raw `Downloader` (read path)

Used when **reading** optimized chunks or raw files — not the optimize worker pool.

| Piece                                                                                  | Role                                                                                                        |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `get_downloader(remote_dir, cache_dir, chunks, storage_options, session_options)`      | Prefix match on `_DOWNLOADERS`                                                                              |
| `Downloader.download_file` / `download_bytes` / `adownload_file` / `adownload_fileobj` | Sync + async APIs                                                                                           |
| Atomic publish                                                                         | `_temp_download_path` + `_atomic_replace` (tmp includes pid)                                                |
| `register_downloader` / `unregister_downloader`                                        | Extension points                                                                                            |
| `StreamingRawDataset.downloader`                                                       | Uses same registry; prefer cloud URL / connection path over FUSE ([using-litdata.md](using-litdata.md) §10) |
| `async_prefetch.py`                                                                    | Prefers `adownload_file` when overridden                                                                    |

**S3 / R2 fork-safety and Studio connections** (`downloader.py`):

- `index.json` GETs **never** go through obstore (`_use_obstore_for_s3_key`) so the DataLoader parent does not start tokio before fork.
- Chunk GETs lazy-init a process-local `S3Store` (`_cached_obstore_store`). If this PID forked after the parent initialized obstore, `obstore_usable()` is false → boto3 fallback.
- `_build_obstore_s3_store(bucket, s3_client)` builds the store from the **already configured** `S3Client`/`R2Client` (endpoint, region, credential provider). Do **not** pass resolver `storage_options` (`data_connection_id`, `endpoint_url`) into `boto3.Session`.
- `__getstate__` drops `_store` / `_store_pid` so spawn/pickle cannot ship a live tokio store.

Symptom if either rule is broken: `FileNotFoundError` after ~120s with `num_workers>0` (`num_workers=0` works). Details: [debugging.md](debugging.md), [streaming.md](streaming.md).

**FsProvider vs Downloader** (also [storage-format.md](storage-format.md) §5):

| | FsProvider | Downloader |
| | \---------- | ---------- |
| Optimize input download / chunk upload / index / merge / empty checks | ✅ | ❌ |
| StreamingDataset chunk prefetch / StreamingRawDataset | ❌ | ✅ |
| Schemes | s3, gs, r2 | + azure, hf, local |

______________________________________________________________________

## 7. Local vs remote — decision table for agents

| Goal                                   | Prefer                                                                                                                                                                 | What LitData does                                                                                                     |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Optimize files on Studio S3 connection | `input_dir` / paths under `/teamspace/s3_connections/…`                                                                                                                | Resolve → downloaders + FsProvider GET into `/cache/data`                                                             |
| Optimize from laptop with AWS creds    | `s3://bucket/…` in inputs; may need design that doesn’t rely on `path`-based downloaders — verify whether your inputs are local copies or you read via SDK inside `fn` | Pure `s3://` `Dir` has `path=None` → **no** `_download_data_target` pool                                              |
| Write durable chunks                   | `output_dir=/teamspace/s3_connections/…/vN` or `s3://…`                                                                                                                | Uploaders + `_upload_index` via FsProvider                                                                            |
| Scratch only                           | local / `this_studio` (small)                                                                                                                                          | Local copy uploaders; multi-node remaps `this_studio` optimize outs to job artifacts ([multi-node.md](multi-node.md)) |
| Raw training I/O                       | `StreamingRawDataset("s3://…")` or connection path                                                                                                                     | `Downloader` async; **not** processing downloaders                                                                    |

______________________________________________________________________

## 8. Error modes & agent checklists

| Symptom                                                             | Likely cause                                                                                                     | What to check                                                                      |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `ValueError: The provided … isn't supported` in downloader/uploader | Scheme outside `_SUPPORTED_PROVIDERS` for processing                                                             | Use s3/gs/r2 for optimize I/O; azure/hf are streaming-Downloader-only              |
| Auth / 403 on download or upload                                    | Missing keys; RO bucket; connection without write; missing `data_connection_id` for R2                           | `storage_options`, Studio connection attach, IAM                                   |
| Hang with remote inputs                                             | Disk wait (`_wait_for_disk_usage_higher_than_threshold` 25 GB); remover stuck; uploader exception only `print`ed | Free space on `/`; `num_workers=1`; watch uploader `print(e)`                      |
| `The provided item … didn't contain any filepaths`                  | `_collect_paths` / `_is_path` failed                                                                             | Pass real paths under `input_dir.path`; set `input_dir` explicitly                 |
| Chunks left / RuntimeError in `_done`                               | Uploader failed or `delete_cached_files` + local output mismatch                                                 | Inspect cache dirs; uploader errors                                                |
| Index never appears (multi-node)                                    | Last node waiting on peer `{rank}-index.json`                                                                    | [multi-node.md](multi-node.md) peer wait                                           |
| FUSE “works” in `ls` but training/optimize is slow or crashes       | Reading mount directly                                                                                           | Pass path into LitData; confirm `Dir.url` is set                                   |
| `FileNotFoundError` chunk-\*.bin after ~120s, only `num_workers>0`  | Obstore tokio started in parent before fork, **or** prefetch thread crashed on `data_connection_id`              | [debugging.md](debugging.md); stderr `PrepareChunksThread CRASHED`; tracer `crash` |
| Partial / corrupt local file                                        | Crash mid-download (FsProvider path is not always atomic the way `Downloader._atomic_replace` is)                | Wipe `DATA_OPTIMIZER_DATA_CACHE_FOLDER` / re-run; prefer connection+resolver path  |
| `cloudspaces` in output URL                                         | Rejected in `optimize`/`map`                                                                                     | Use connections / datasets, not studio content URLs                                |

**Debug tip:** `num_workers=1`, `fast_dev_run=True`, and inspect `/cache/data` + `/cache/chunks` (or temp equivalents). Worker exceptions land in `error_queue` → main `RuntimeError` + `terminate()` siblings.

______________________________________________________________________

## 9. Grep landmarks

```
processing/data_processor.py
  _download_data_target   _upload_fn   _remove_target   keep_path
  BaseWorker._collect_paths  _start_downloaders  _start_uploaders  _start_remover
  _try_upload  no_downloaders  delete_cached_files
  DataChunkRecipe._done  _upload_index

processing/functions.py
  optimize  map  _get_input_dir  _resolve_dir(...)

processing/utilities.py
  construct_storage_options  remove_uuid_from_filename  _get_work_dir
  read_index_file_content

streaming/resolver.py
  _resolve_dir  _resolve_s3_connections  _resolve_s3_folders
  _resolve_gcs_*  _resolve_lightning_storage  _resolve_datasets  _execute

streaming/fs_provider.py
  FsProvider  _get_fs_provider

streaming/downloader.py
  Downloader  get_downloader  _DOWNLOADERS  register_downloader
  obstore_usable  _use_obstore_for_s3_key  _build_obstore_s3_store  _cached_obstore_store

constants.py
  _SUPPORTED_PROVIDERS
```
