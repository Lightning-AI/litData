# Environment variables

All LitData / related env knobs agents and users commonly need. Platform-injected Studio vars that are not meant to be set by hand are marked **(Studio)**.

## Streaming / cache (most useful)

| Env                                    | Default                                                        | Effect                                                                                                                                                                                                                          |
| -------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LITDATA_POSIX_FAST`                   | auto (on for local paths)                                      | In-place mmap reads. Default **on** when `input_dir` is a local/POSIX path (Vast, NFS, NVMe, Studio filestore). **Off** for `s3://` / `gs://` / `r2://`. `0` disables, `1` forces.                                              |
| `LITDATA_POSIX_SHUFFLE_WINDOW`         | `16`                                                           | `WindowShuffle` only on Vast/NFS/Lustre/GPFS or `LITDATA_POSIX_FAST=1`: sequential chunk stripes, then a window permute per worker **and** inside each chunk. `1` = sequential. Local disks and object URLs keep `FullShuffle`. |
| `LITDATA_POSIX_PAGE_BYTES`             | `262144` (256KiB)                                              | POSIX-fast: copy this many payload bytes from the mapped chunk in one `mmap` slice, then split items from that buffer. `0` falls back to one slice per sample.                                                                  |
| `LITDATA_POSIX_MAX_WORKERS`            | auto from `MemAvailable`                                       | Cap on DataLoader workers for POSIX-fast. `0` disables the cap; `N` forces it.                                                                                                                                                  |
| `LITDATA_POSIX_WILLNEED`               | auto                                                           | `0` skips `POSIX_FADV_WILLNEED` / `MADV_WILLNEED` (skipped automatically when workers × keep × chunk would exceed ~half `MemAvailable`).                                                                                        |
| `LITDATA_POSIX_RAM_FRACTION`           | `0.5`                                                          | Fraction of `MemAvailable` used as the WILLNEED / worker-cap budget.                                                                                                                                                            |
| `LITDATA_POSIX_WORKER_RSS`             | `268435456` (256 MiB)                                          | Per-worker RSS assumed by `posix_max_data_workers` when auto-capping `num_workers`.                                                                                                                                             |
| `MAX_CACHE_SIZE`                       | ctor `max_cache_size` (default: 75% of free disk, leave ≥50GB) | Override: `"50GB"` / `"100G"` or a fraction (`0.90`)                                                                                                                                                                            |
| `LITDATA_ASYNC_CHUNK_PREFETCH`         | unset → **on if remote**, off if local-only                    | `1` force on · `0` force off. Overlaps **chunk downloads** via `asyncio.gather` inside `PrepareChunksThread` — **not** an async DataLoader                                                                                      |
| `LITDATA_ASYNC_MIN_PRE_DOWNLOAD`       | `4`                                                            | When async is on, raise `max_pre_download` to at least this. `0` disables the floor                                                                                                                                             |
| `LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB` | `8`                                                            | obstore S3 stream chunk size (MiB)                                                                                                                                                                                              |
| `MAX_WAIT_TIME`                        | `120`                                                          | Seconds to wait for a chunk file before failing                                                                                                                                                                                 |
| `FORCE_DOWNLOAD_TIME`                  | `30`                                                           | Seconds blocked on missing chunk before force re-download                                                                                                                                                                       |
| `LITDATA_DISABLE_VERSION_CHECK`        | `0`                                                            | `1` skips upgrade prompt                                                                                                                                                                                                        |
| `LITDATA_TIMING`                       | off                                                            | `1`/`true` enables `StreamingTimingStats`                                                                                                                                                                                       |

### Async chunk prefetch (detail)

```
Training loop stays sync:  for batch in StreamingDataLoader(...): ...
Async only overlaps remote GETs inside each worker's PrepareChunksThread.
```

- Default **on** for remote `input_dir` (`s3://`, `gs://`, …); **off** for local-only.
- Needs gather width: with async on, `max_pre_download` is floored to `LITDATA_ASYNC_MIN_PRE_DOWNLOAD` (default **4**) unless you set the floor to `0`.
- Peak disk still ≈ `num_workers × max_pre_download × chunk_size` — raise cache budget accordingly.
- Force compare sync vs async fairly: same `max_pre_download` on both arms ([benchmarking.md](benchmarking.md)).

```bash
# Force off (e.g. debugging download races)
export LITDATA_ASYNC_CHUNK_PREFETCH=0

# Force on even for local paths
export LITDATA_ASYNC_CHUNK_PREFETCH=1

# Keep user max_pre_download=2 while async is on
export LITDATA_ASYNC_MIN_PRE_DOWNLOAD=0
```

Code: `streaming/async_prefetch.py`. Cache interaction: [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md).

## Debug / tracing

| Env                             | Default             | Effect                                                |
| ------------------------------- | ------------------- | ----------------------------------------------------- |
| `DEBUG_LITDATA`                 | `0`                 | Internal debug behavior                               |
| `PRINT_DEBUG_LOGS`              | `0`                 | Print debug logs to stdout                            |
| `LITDATA_LOG_FILE`              | `litdata_debug.log` | Trace log path (`enable_tracer`). Handler **appends** |
| `LITDATA_LOG_LEVEL`             | `DEBUG`             | Python log level for the tracer logger                |
| `LITDATA_TRACE_LEVEL`           | unset               | `batch` / `chunk` / `sample` / `debug` / `off`        |
| `LITDATA_TRACE_CATEGORIES`      | from level          | Comma-separated cats; wins over level if set          |
| `LITDATA_LOG_ITERATING_DATASET` | from level          | Legacy include flag for epoch events                  |
| `LITDATA_LOG_GETITEM`           | from level          | Legacy include flag for sample events                 |
| `LITDATA_LOG_ITEM_LOADER`       | from level          | Legacy include flag for sample events                 |
| `ENABLE_STATUS_REPORT`          | `0`                 | Extra Studio progress JSON during optimize            |

`enable_tracer(level=..., log_file=..., categories=...)` writes these env vars. Workers inherit the category set. Cats: `download`, `read`, `delete`, `decompress`, `batch`, `sample`, `epoch`, `lock`, `crash`. Delete `LITDATA_LOG_FILE` before a re-trace. Full spec: [debugging.md](debugging.md). Converter: [Lightning-AI/litracer](https://github.com/Lightning-AI/litracer).

## `optimize` / `map` multi-node **(usually set by the platform)**

| Env                                | Role                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `DATA_OPTIMIZER_NUM_NODES`         | World size / launch gate (`>0` ⇒ already a job worker)                                                  |
| `DATA_OPTIMIZER_NODE_RANK`         | This node’s rank                                                                                        |
| `DATA_OPTIMIZER_GLOBAL_RANK`       | Flat rank for chunk filenames                                                                           |
| `DATA_OPTIMIZER_NUM_WORKERS`       | Workers per node                                                                                        |
| `DATA_OPTIMIZER_CACHE_FOLDER`      | Chunk/work cache root                                                                                   |
| `DATA_OPTIMIZER_DATA_CACHE_FOLDER` | Downloaded input cache                                                                                  |
| `DATA_OPTIMIZER_TIMEOUT`           | Queue get timeout (≈300s; shared-queue ≈200s)                                                           |
| `DATA_OPTIMIZER_FAST_DEV_RUN`      | `DataProcessor` treats missing/`None` as **on** (`"1"`). Public `optimize(fast_dev_run=False)` is safe. |

### `optimize` / `map` I/O (user-settable)

| Env                                     | Default | Effect                                                                                   |
| --------------------------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `LITDATA_PREFETCH_BYTES`                | `512MB` | Shared-queue prefetch budget (slots × item size)                                         |
| `LITDATA_OPTIMIZE_DOWNLOAD_BATCH`       | `16`    | How many remote inputs to gather per download flush                                      |
| `LITDATA_OPTIMIZE_UPLOAD_BATCH`         | `16`    | How many local chunks to `aupload_file` per flush                                        |
| `LITDATA_OPTIMIZE_DOWNLOAD_CONCURRENCY` | `16`    | Semaphore for `adownload_file`; can scale with workers/CPU and back off when disk is low |
| `LITDATA_OPTIMIZE_SPLIT_WRITERS`        | `0`     | `1` splits transform vs write on unordered optimize (1–2 chunk writers)                  |

Launch flow: [processing.md](processing.md). I/O details: [data-movement.md](data-movement.md).

## Hugging Face / cloud credentials (user)

| Env                         | Role                                                                      |
| --------------------------- | ------------------------------------------------------------------------- |
| `HF_TOKEN`                  | Gated Hugging Face datasets                                               |
| `HF_HUB_ENABLE_HF_TRANSFER` | Faster HF downloads when `hf_transfer` is installed                       |
| `AWS_*` / profile vars      | Standard AWS credentials (also via `storage_options` / `session_options`) |

## Lightning Studio **(injected — do not invent)**

| Env                                                                            | Role                                                                     |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `LIGHTNING_CLOUD_PROJECT_ID` + `LIGHTNING_CLUSTER_ID`                          | Marks `_IS_IN_STUDIO`; required to resolve `/teamspace/...`              |
| `LIGHTNING_CLOUD_SPACE_ID`                                                     | Studio id (`/teamspace/datasets`)                                        |
| `LIGHTNING_CLOUD_PROVIDER`                                                     | `aws` / GCP scheme for other-studio content                              |
| `LIGHTNING_CLOUD_URL`                                                          | Control plane (default `https://lightning.ai`)                           |
| `LIGHTNING_API_KEY` / `LIGHTNING_USERNAME`                                     | Auth for some client paths                                               |
| `LIGHTNING_BUCKET_NAME` / `LIGHTNING_CLOUD_APP_ID` / `LIGHTNING_CLOUD_WORK_ID` | Job artifacts URL (`_get_work_dir`)                                      |
| `LIGHTNING_SKIP_INSTALL` / `LIGHTNING_BRANCH`                                  | Injected into remote optimize/map job command                            |
| `LIGHTNING_APP_EXTERNAL_URL` / `LIGHTNING_APP_STATE_URL`                       | Broadcast / multi-process coordination helpers                           |
| `VSCODE_PROXY_URI`                                                             | Used heuristically when resolving Lightning cloud URL in some IDE setups |

Details: [lightning-studio.md](lightning-studio.md), [resolver.md](resolver.md).

## Distributed training (PyTorch)

| Env                                                    | Role                                          |
| ------------------------------------------------------ | --------------------------------------------- |
| `WORLD_SIZE` / `GLOBAL_RANK` / `LOCAL_RANK` / `NNODES` | Detected by `_DistributedEnv` (e.g. torchrun) |
