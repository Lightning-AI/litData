# Debugging & profiling LitData

## StreamingDataLoader profiling (viztracer)

User-facing DataLoader worker CPU trace — complementary to `enable_tracer()` below.

```python
from litdata import StreamingDataset, StreamingDataLoader

loader = StreamingDataLoader(
    StreamingDataset("s3://bucket/data"),
    batch_size=64,
    num_workers=4,              # required
    profile_batches=20,         # or True for full iterator
    profile_skip_batches=5,     # skip cold batches
    profile_dir="./profiles",   # → profiles/result.json
)
```

- Needs `pip install viztracer`. Raises if `num_workers == 0`.
- Only **global rank 0** + **worker 0** are instrumented (`dataloader.py` `_ProfileWorkerLoop`).
- `int` → stop after `profile_skip_batches + profile_batches` `fetcher.fetch` calls; `True` → until worker loop ends.
- Overwrites existing `result.json`. View in `chrome://tracing` or Perfetto.
- Full user doc: README `#profile-loading`; cookbook: [using-litdata.md](using-litdata.md) §7.

## Breakpoints inside worker processes

Normal `breakpoint()` doesn't work in DataLoader or `optimize`/`map` worker subprocesses (no stdin). Use LitData's multiprocessing-safe pdb, which reopens `sys.stdin` under a lock (`utilities/breakpoint.py:33`, exported as `litdata.breakpoint`):

```python
from litdata.utilities.breakpoint import breakpoint   # or: import litdata; litdata.breakpoint()
breakpoint()   # works inside a worker
```

Run with `num_workers=0` first when you can — it keeps everything in the main process so a plain debugger works.

## Structured tracing + Litracer/Perfetto (`debugger.py`)

`enable_tracer()` (`debugger.py`) logs the streaming pipeline as **one semicolon-separated line per event**. [Litracer](https://github.com/Lightning-AI/litracer) converts that file to Chrome / Perfetto JSON.

Call **once per process, before** creating the DataLoader. The file handler **appends** — delete `litdata_debug.log` before a re-trace.

```python
import litdata as ld
from litdata.debugger import enable_tracer

enable_tracer(level="chunk", log_file="litdata_debug.log")
# enable_tracer(level="batch")
# enable_tracer(level="sample")
# enable_tracer(level="debug")
# enable_tracer(categories=["download", "read", "delete"])

if __name__ == "__main__":
    dataset = ld.StreamingDataset("s3://my-bucket/my-data", shuffle=True)
    for batch in ld.StreamingDataLoader(dataset, batch_size=64, num_workers=8):
        ...
```

```bash
python train.py
# clone https://github.com/Lightning-AI/litracer && go build -o litracer .
litracer --quiet --validate -o litdata_trace.json.gz litdata_debug.log
litracer --quiet --cat download,read,delete -o io.json.gz litdata_debug.log
# open in https://ui.perfetto.dev
```

`go install github.com/deependujha/litracer@latest` still works (Go module path). `go install github.com/Lightning-AI/litracer@latest` does not until `go.mod` is renamed.

Default Litracer output is **gzip Chrome JSON** (`.json.gz`). Both Perfetto and `chrome://tracing` open it (gzip magic). That is the Chrome-compatible compact format — typically far smaller than raw JSON. Perfetto protobuf (`.pftrace`) is smaller still but **not** accepted by `chrome://tracing`. Pass `-o file.json` for uncompressed JSON.

### Levels vs categories

| Level             | Categories                                   |
| ----------------- | -------------------------------------------- |
| `off`             | none                                         |
| `batch`           | `epoch`, `batch`, `crash`                    |
| `chunk` (default) | + `download`, `read`, `delete`, `decompress` |
| `sample`          | + `sample`                                   |
| `debug`           | + `lock` (all of `ALL_CATEGORIES`)           |

`categories=["download", "read", "delete"]` replaces the level set. Legacy kwargs `item_loader=False` / `iterating_dataset=False` / `getitem_dataset_for_chunk_index=False` drop `sample` / `epoch`.

`is_tracing(cat)` / `emit_trace` / `trace_span` are **no-ops** when the category is off — cheap enough for hot paths.

### Stable event names

Indexes belong in **args**, not in `name`, so Perfetto groups all downloads together.

| `name`                    | `cat`        | Site                                                          |
| ------------------------- | ------------ | ------------------------------------------------------------- |
| `download`                | `download`   | `downloader.py` chunk GET                                     |
| `prefetch`                | `download`   | `reader.py` async prefetch gather                             |
| `read`                    | `read`       | `reader.py` mmap/decode; last span is closed on worker finish |
| `delete`                  | `delete`     | `item_loader.py` eviction                                     |
| `decompress`              | `decompress` | `compression.py`                                              |
| `batch`                   | `batch`      | `dataloader.py`                                               |
| `dataloader` / `combined` | `epoch`      | loader / CombinedStreamingDataset                             |
| `sample`                  | `sample`     | `dataset.py` `__getitem__`                                    |
| `lock`                    | `lock`       | `.cnt` increment (`downloader.py`) / decrement (`config.py`)  |
| `crash`                   | `crash`      | `PrepareChunksThread._report_crash` — `ph: I`                 |

`env_info()` injects `dist_*` / `worker_*` on every event. Colors: `_CAT_CNAME` in `debugger.py`.

### Log format (load-bearing for Litracer)

- One line per event: `ts:%(asctime)s;PID:%(process)d; TID:%(thread)d; name: download;ph: B;cat: download;…`
- `ts` is Chrome microseconds (`record.created * 1e6`) via `_OneLineTraceFormatter`.
- Values are sanitized: newlines/CRs → space, `;` → `,` (`_sanitize_log_value`). **Never** `logger.exception` into this logger — a traceback splits the file into unparsable lines. Crashes print the traceback to **stderr** and emit a one-line `crash` instant.
- `TimedFlushFileHandler` flushes every `flush_interval` seconds (default 5 via `enable_tracer`).

When adding a new span: pick a stable `name` + `cat`, put indexes in kwargs, use `trace_span` / `emit_trace`, and keep values one-line.

## Environment variables

Categorized catalog (streaming, async prefetch, debug, `DATA_OPTIMIZER_*`, Studio, HF): **[env-vars.md](env-vars.md)**.

Quick hits:

| Env                                     | Default               | Effect                                             |
| --------------------------------------- | --------------------- | -------------------------------------------------- |
| `LITDATA_CACHE_DIR`                     | `~/.lightning/chunks` | Default chunk cache                                |
| `LITDATA_ASYNC_CHUNK_PREFETCH`          | on for remote         | `0`/`1` force async chunk download overlap         |
| `LITDATA_ASYNC_MIN_PRE_DOWNLOAD`        | `4`                   | Floor `max_pre_download` when async on (`0` = off) |
| `MAX_WAIT_TIME` / `FORCE_DOWNLOAD_TIME` | `120` / `30`          | Chunk wait / force re-download                     |
| `DEBUG_LITDATA` / `PRINT_DEBUG_LOGS`    | `0`                   | Internal debug / stdout                            |
| `LITDATA_LOG_FILE`                      | `litdata_debug.log`   | Tracer output path                                 |
| `LITDATA_TRACE_LEVEL`                   | unset                 | `batch` / `chunk` / `sample` / `debug` / `off`     |
| `LITDATA_TRACE_CATEGORIES`              | from level            | Comma-separated cats; wins over level if set       |

Cache CLI: `litdata cache path` · `litdata cache clear`.

## Common failure modes

**Streaming (read) — see streaming.md**

- **"did you optimize?" / `FileNotFoundError`** → no `index.json` at the path (`dataset.py`); the data wasn't optimized, or you pointed at raw files (use `StreamingRawDataset`).
- **Hang/timeout on chunk download** → raise `MAX_WAIT_TIME`; check credentials/`storage_options`; confirm the URL prefix matches a registered `Downloader`.
- **`FileNotFoundError: The …/chunk-*.bin hasn't been found` after ~120s, `num_workers>0`, `num_workers=0` never fails:**
  - **Plain `s3://`:** obstore's tokio runtime is **not fork-safe**. If the DataLoader parent starts obstore (historically by fetching `index.json` via obstore), forked workers inherit a dead runtime, GETs hang, `FileLock(..., timeout=0)` skips co-workers, waiters time out. **Current code:** `index.json` always uses boto3; workers lazy-init obstore on the first chunk GET; if `_OBSTORE_INIT_PID` is the parent, workers fall back to boto3 (`obstore_usable()`). Re-creating `S3Store` after fork is **not** enough — tokio is process-global. Tests: `test_s3_index_download_does_not_start_obstore`, `test_obstore_usable_false_after_parent_init` in `tests/streaming/test_downloader.py`.
  - **Studio R2 / `lightning_storage` / connections:** prefetch thread died because `data_connection_id` / `endpoint_url` were unpacked into `boto3.Session` (`TypeError`). **Current code:** `_build_obstore_s3_store` builds the store from the existing `S3Client`/`R2Client` (credential provider + endpoint from the boto client). The crash path prints `[litdata] PrepareChunksThread CRASHED (rank=…, worker=…)` to stderr, emits `crash` (`ph: I`), and waiters raise `RuntimeError: Chunk prefetch thread crashed…` immediately instead of waiting 120s. Test: `test_build_obstore_s3_store_does_not_pass_data_connection_id_to_session`.
- **Hang under tiny `max_cache_size` (CI 120s)** → often `max_pre_download` capped to 1 + delete-when-processed deadlock, or prepare-thread stuck in `shutdown_default_executor`. See [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md) § Prefetch & eviction. Look for log `capping max_pre_download … → 1`.
- **Cache grows without bound** → set `max_cache_size` (default `"100GB"`); eviction is gated by `.cnt` refcount + `FileLock`. Stale `.lock`/`.cnt` files from a crash can block deletion — `litdata cache clear`. Peak disk ≈ `num_workers × max_pre_download × chunk_size` (async floor often makes `max_pre` 4).
- **Deadlock: parquet + workers** → `ParquetLoader` under `fork` with `num_workers>0` raises by design (`dataloader.py`); use `spawn`/`forkserver`.
- **Windows `PermissionError` opening `.bin`** → decompress `os.replace` race; should retry via `_open_chunk_file`. Still close handles before delete.
- **Nondeterministic / wrong order after resume** → shuffling is seeded by `seed`+`epoch`+`num_chunks`+`chunk_index`; `_validate_state_dict` hard-fails if `shuffle`/`num_workers`/`seed`/`input_dir`/`item_loader`/`drop_last` changed between save and resume (override with `force_override_state_dict`).
- **`state_dict()` raises** → must be called in the main process, not a worker.
- **Uneven batches across ranks** → `drop_last` defaults to `True` in distributed for a reason; don't force `False` there.

For Studio cold-epoch ImageNet methodology and fair obstore vs boto3 comparisons, see [benchmarking.md](benchmarking.md).

**Processing (write) — see processing.md**

- **`optimize` hangs, no error** → a worker exception is swallowed into `error_queue` then re-raised by the main loop; if it can't build a traceback string the loop waits until all workers die. Run with `num_workers=1` (or `fast_dev_run=True`) to surface the real exception.
- **Peers hang with `keep_data_ordered=False`** → termination relies on the `ALL_DONE` sentinel being re-inserted by each worker; a custom early-exit that drops it hangs the pool.
- **`ValueError` on `prepare_structure`** → must return `list | StreamingDataLoader | multiprocessing.Queue`.
- **`map` fn silently does nothing** → `map` recipes' `prepare_item` must return `None` and must write outputs to `output_dir`.
- **Pickling error under spawn** → worker args must be picklable; avoid closures/weakrefs captured on `self`.

## Fast triage checklist

1. Reproduce with `num_workers=0` + a small local dataset → isolates worker/multiprocessing issues from logic bugs. If **only** `num_workers>0` on `s3://` fails, suspect obstore-after-fork (above).
2. `fast_dev_run=True` (optimize/map) for a ~10-item smoke run.
3. `enable_tracer(level="chunk")` + Litracer `--quiet --validate --cat download,read,delete` to see where time goes. Check stderr for `PrepareChunksThread CRASHED`.
4. `litdata cache clear` to rule out stale cache/lock state.
5. `PRINT_DEBUG_LOGS=1` for stdout logs without the trace pipeline.
