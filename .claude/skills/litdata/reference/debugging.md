# Debugging & profiling LitData

## Breakpoints inside worker processes

Normal `breakpoint()` doesn't work in DataLoader or `optimize`/`map` worker subprocesses (no stdin). Use LitData's multiprocessing-safe pdb, which reopens `sys.stdin` under a lock (`utilities/breakpoint.py:33`, exported as `litdata.breakpoint`):

```python
from litdata.utilities.breakpoint import breakpoint   # or: import litdata; litdata.breakpoint()
breakpoint()   # works inside a worker
```

Run with `num_workers=0` first when you can — it keeps everything in the main process so a plain debugger works.

## Structured tracing + Litracer/Perfetto (`debugger.py`)

`enable_tracer()` (`debugger.py:124`) logs the streaming pipeline to a file, which you convert into a Chrome/Perfetto trace.

```python
import litdata as ld
from litdata.debugger import enable_tracer

enable_tracer()   # WARNING: delete an existing litdata_debug.log first, or traces mix

if __name__ == "__main__":
    dataset = ld.StreamingDataset("s3://my-bucket/my-data", shuffle=True)
    for batch in ld.StreamingDataLoader(dataset, batch_size=64):
        ...
```

Then:

```bash
python main.py                                        # writes litdata_debug.log
go install github.com/deependujha/litracer@latest     # or download a release binary
litracer litdata_debug.log -o litdata_trace.json -w 100
# open litdata_trace.json in ui.perfetto.dev (preferred) or chrome://tracing
```

`enable_tracer(flush_interval=5, item_loader=True, iterating_dataset=True, getitem_dataset_for_chunk_index=True)` sets the `LITDATA_LOG_*` env vars and returns a singleton `LitDataLogger`. `TimedFlushFileHandler` flushes every N seconds from a daemon thread. `env_info()` auto-injects distributed + worker rank/world-size into every event. `ChromeTraceColors` holds trace phase colors.

## Environment-variable knobs (`constants.py` + `debugger.py`)

| Env var                                | Effect                                                         | Default               |
| -------------------------------------- | -------------------------------------------------------------- | --------------------- |
| `LITDATA_CACHE_DIR`                    | Override the chunk cache directory                             | `~/.lightning/chunks` |
| `DEBUG_LITDATA`                        | Enable internal debug behavior (`_DEBUG`)                      | `0`                   |
| `PRINT_DEBUG_LOGS`                     | Print debug logs to stdout                                     | `0`                   |
| `MAX_WAIT_TIME`                        | Max seconds to wait for a chunk download                       | `120`                 |
| `FORCE_DOWNLOAD_TIME`                  | Force re-download threshold (s)                                | `30`                  |
| `LITDATA_ASYNC_CHUNK_PREFETCH`         | `1`/`0` force async chunk gather on/off; unset = on for remote | unset                 |
| `LITDATA_ASYNC_MIN_PRE_DOWNLOAD`       | Floor for `max_pre_download` when async on (`0` = no floor)    | `4`                   |
| `LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB` | obstore `stream(min_chunk_size=…)` in MiB                      | `8`                   |
| `LITDATA_TIMING`                       | Enable `StreamingTimingStats`                                  | unset                 |
| `LITDATA_LOG_FILE`                     | Trace log file path                                            | `litdata_debug.log`   |
| `LITDATA_LOG_LEVEL`                    | Trace log level                                                | `DEBUG`               |
| `LITDATA_LOG_ITERATING_DATASET`        | Include `iterating_dataset` events                             | `True`                |
| `LITDATA_LOG_GETITEM`                  | Include `getitem_dataset_for_chunk_index` events               | `True`                |
| `LITDATA_LOG_ITEM_LOADER`              | Include `item_loader` events                                   | `True`                |
| `LITDATA_DISABLE_VERSION_CHECK`        | Skip the upgrade prompt                                        | `0`                   |
| `ENABLE_STATUS_REPORT`                 | Emit status reports                                            | `0`                   |

`optimize`/`map` cross-node coordination: `DATA_OPTIMIZER_NODE_RANK`, `DATA_OPTIMIZER_NUM_NODES`, `DATA_OPTIMIZER_GLOBAL_RANK`, `DATA_OPTIMIZER_NUM_WORKERS`.

Cache CLI: `litdata cache path` (print dir) · `litdata cache clear` (rmtree it).

## Common failure modes

**Streaming (read) — see streaming.md**

- **"did you optimize?" / `FileNotFoundError`** → no `index.json` at the path (`dataset.py:322`); the data wasn't optimized, or you pointed at raw files (use `StreamingRawDataset`).
- **Hang/timeout on chunk download** → raise `MAX_WAIT_TIME`; check credentials/`storage_options`; confirm the URL prefix matches a registered `Downloader`.
- **Hang under tiny `max_cache_size` (CI 120s)** → often `max_pre_download` capped to 1 + delete-when-processed deadlock, or prepare-thread stuck in `shutdown_default_executor`. See [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md) § Prefetch & eviction. Look for log `capping max_pre_download … → 1`.
- **Cache grows without bound** → set `max_cache_size` (default `"100GB"`); eviction is gated by `.cnt` refcount + `FileLock`. Stale `.lock`/`.cnt` files from a crash can block deletion — `litdata cache clear`. Peak disk ≈ `num_workers × max_pre_download × chunk_size` (async floor often makes `max_pre` 4).
- **Deadlock: parquet + workers** → `ParquetLoader` under `fork` with `num_workers>0` raises by design (`dataloader.py:629`); use `spawn`/`forkserver`.
- **Windows `PermissionError` opening `.bin`** → decompress `os.replace` race; should retry via `_open_chunk_file`. Still close handles before delete.
- **Nondeterministic / wrong order after resume** → shuffling is seeded by `seed`+`epoch`+`num_chunks`+`chunk_index`; `_validate_state_dict` (`dataset.py:607`) hard-fails if `shuffle`/`num_workers`/`seed`/`input_dir`/`item_loader`/`drop_last` changed between save and resume (override with `force_override_state_dict`).
- **`state_dict()` raises** → must be called in the main process, not a worker (`dataset.py:574`).
- **Uneven batches across ranks** → `drop_last` defaults to `True` in distributed for a reason; don't force `False` there.

For Studio cold-epoch ImageNet methodology and fair obstore vs boto3 comparisons, see [benchmarking.md](benchmarking.md).

**Processing (write) — see processing.md**

- **`optimize` hangs, no error** → a worker exception is swallowed into `error_queue` then re-raised by the main loop; if it can't build a traceback string the loop waits until all workers die. Run with `num_workers=1` (or `fast_dev_run=True`) to surface the real exception.
- **Peers hang with `keep_data_ordered=False`** → termination relies on the `ALL_DONE` sentinel being re-inserted by each worker; a custom early-exit that drops it hangs the pool.
- **`ValueError` on `prepare_structure`** → must return `list | StreamingDataLoader | multiprocessing.Queue` (`:1247`).
- **`map` fn silently does nothing** → `map` recipes' `prepare_item` must return `None` (`:913`) and must write outputs to `output_dir`.
- **Pickling error under spawn** → worker args must be picklable; avoid closures/weakrefs captured on `self`.

## Fast triage checklist

1. Reproduce with `num_workers=0` + a small local dataset → isolates worker/multiprocessing issues from logic bugs.
2. `fast_dev_run=True` (optimize/map) for a ~10-item smoke run.
3. `enable_tracer()` + Litracer/Perfetto to see where time goes (download vs decode vs collate).
4. `litdata cache clear` to rule out stale cache/lock state.
5. `PRINT_DEBUG_LOGS=1` for stdout logs without the trace pipeline.
