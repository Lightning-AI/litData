---
name: litdata
description: >-
  Expert use of the LitData library and work on its codebase. Use when writing or
  reviewing code that calls litdata (StreamingDataset, StreamingDataLoader,
  StreamingRawDataset, optimize, map, CombinedStreamingDataset,
  ParallelStreamingDataset, TokensLoader, serializers, train_test_split,
  merge_datasets, index_parquet_dataset, index_hf_dataset), answering how-to
  questions, choosing raw vs optimize vs parquet/HF/MDS, keyed lookup
  (`key_fn`, `build_keys_index`, `dataset_update`, `get_by_key`), tuning
  cache/prefetch/shuffle/seed, resolving paths (s3/gs/r2/azure/hf/local:/teamspace
  via resolver.py), documenting or debugging optimize/map downloaders/uploaders/
  removers, FsProvider vs Downloader, FUSE s3_connections/s3_folders, or
  multi-node DATA_OPTIMIZER_* / num_nodes jobs, or when navigating/editing
  src/litdata, tests, CI, or debugging streaming / optimize / map. Also use for
  enable_tracer / Litracer / Perfetto pipeline traces, and multi-worker s3://
  FileNotFoundError or obstore-after-fork / Studio R2 data_connection_id Session
  TypeError.
---

# LitData

LitData (`import litdata`) preprocesses and streams datasets for PyTorch training:

- **Write** (`optimize` / `map`) → chunked `chunk-*.bin` + `index.json` → `src/litdata/processing/`
- **Read** (`StreamingDataset` + `StreamingDataLoader`) → cache → decode → batch → `src/litdata/streaming/`
- **Raw** (`StreamingRawDataset`) → stream original files without optimize → `src/litdata/raw/`

**To use the library expertly:** always load [reference/using-litdata.md](reference/using-litdata.md) first. Narrative source: repo `README.md`.

## Install this skill

From any project (Cursor, Claude Code, and other agents supported by the [skills CLI](https://github.com/vercel-labs/skills)):

```bash
npx skills add Lightning-AI/litData
```

Useful options: `-g` (user-global), `-a cursor` (Cursor only), `-y` (non-interactive). In this repo the skill already lives at `.claude/skills/litdata/`.

## Expert usage (load using-litdata.md)

Before writing examples or answering how-tos, read the cookbook. Highlights:

| Topic            | Remember                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Raw files**    | `StreamingRawDataset` + torch `DataLoader` — `#stream-raw` / §10. Prefer cloud URL / connection path over FUSE. Defaults: `max_concurrent_downloads=None` (adaptive Stage 1), `max_prefetch=16` (worker-aware ~64 aggregate), `hedge_delay=0`, `download_timeout=120` (**batch-level**), `range_parallel_threshold=0`. Explicit `int` concurrency = exact permits.          |
| Images           | Return **JPEG** (`JpegImageFile` / quality ≈95). Plain `PIL.Image` / `fromarray` → huge PIL RAW                                                                                                                                                                                                                                                                             |
| Train stream     | Optimized: `StreamingDataLoader` + `shuffle=True, drop_last=True, seed=…`                                                                                                                                                                                                                                                                                                   |
| Optimize         | `if __name__ == "__main__"`; exactly one of `chunk_bytes` \| `chunk_size`. Default **64MB**; multi‑MB samples → consider **256–512MB**. **Shuffle the sample list before `optimize()`** when source order matters — README `#faq-chunk-shuffle`                                                                                                                             |
| Ordered data     | Chunk/item shuffle ≠ file-level shuffle. Shuffle before `optimize`, or use `StreamingRawDataset` + `DataLoader(shuffle=True)`. LitData does distributed + within-chunk bucket sampling automatically                                                                                                                                                                        |
| Cache            | Peak disk ≈ `num_workers × max_pre_download × chunk_size`; default `max_cache_size="100GB"`                                                                                                                                                                                                                                                                                 |
| **POSIX-fast**   | Local/Vast/NFS: mmap in place, no cache copy. `WindowShuffle` only on parallel FS (or `LITDATA_POSIX_FAST=1`). Cap workers / skip WILLNEED from `MemAvailable`. `LITDATA_POSIX_FAST=0` disables.                                                                                                                                                                            |
| **Paths**        | Studio `/teamspace/s3_connections` & co are **FUSE** (convenience only — slow, can crash under load). LitData resolves them and talks **directly** to S3/GCS/**R2**. Never read the mount by hand. `reference/resolver.md` + `reference/data-movement.md`                                                                                                                   |
| **Optimize I/O** | Processing downloaders/uploaders/removers are **processes** in `data_processor.py` using **FsProvider** — not the streaming `Downloader` ABC. FUSE → `Dir.url` → `/cache/data`. Load `reference/data-movement.md`.                                                                                                                                                          |
| **Multi-node**   | `num_nodes=` = Lightning Studio job (`_execute`), not torchrun/SLURM. Shard by `DATA_OPTIMIZER_*`; all ranks upload chunks; **last node** merges `{node}-index.json` → `index.json`. Load `reference/multi-node.md`.                                                                                                                                                        |
| Throughput       | Rough ImageNet Studio order-of-magnitude (not guarantees): FUSE ~**600**/s · Raw (right tuning) ~**6–7k**/s · Optimized 64MB chunks ~**11k**/s — `using-litdata.md` FAQ. Raw benches: medians + provenance SHAs; never cite short-window n=1 against Stage 0 medians.                                                                                                       |
| **Tracing**      | `from litdata.debugger import enable_tracer` then `enable_tracer(level="chunk")` (or `categories=["download","read","delete"]`). Convert with [Lightning-AI/litracer](https://github.com/Lightning-AI/litracer): `litracer --quiet --validate -o trace.json.gz litdata_debug.log`. One line per event; crashes go to stderr + `ph: I`. Full spec: `reference/debugging.md`. |
| **S3 workers**   | `index.json` is boto3 (no parent tokio). Workers lazy-init obstore; boto3 fallback if parent already started it. Do **not** pass `data_connection_id` / `endpoint_url` into `boto3.Session` — build obstore from `S3Client`/`R2Client`. `FileNotFoundError` after 120s with `num_workers>0` only → [debugging.md](reference/debugging.md) failure modes.                    |
| **Keyed lookup** | `optimize(..., key_fn=...)` writes `keys/`. Read `ds["id"]` / `get_by_key`. Patch locally with `dataset_update`. Needs `polars`. [keyed-lookup.md](reference/keyed-lookup.md).                                                                                                                                                                                              |
| Parquet / HF     | Index + `ParquetLoader` (HF auto); `spawn` with workers; `using-litdata.md` §10                                                                                                                                                                                                                                                                                             |

## Reference map

| Task                                                                          | Read                                                |
| ----------------------------------------------------------------------------- | --------------------------------------------------- |
| **Use the library** (raw, optimize/stream, parquet/HF, serializers, shuffle)  | `reference/using-litdata.md`                        |
| **Paths / URLs / Studio mounts / `Dir` / time templates**                     | `reference/resolver.md` (+ README `#resolve-paths`) |
| **Downloaders / uploaders / removers / FUSE→cloud / optimize I/O**            | `reference/data-movement.md` (+ `processing.md`)    |
| **Multi-node optimize/map** (`num_nodes`, `DATA_OPTIMIZER_*`, index merge)    | `reference/multi-node.md` (+ `processing.md`)       |
| Read path, shuffle math, item loaders, Combined/Parallel                      | `reference/streaming.md`                            |
| **Cache / BinaryWriter / BinaryReader / `index.json` / FsProvider / sampler** | `reference/storage-format.md`                       |
| Cache / prefetch / eviction / shared-chunk deletion                           | `reference/cache-and-chunk-lifecycle.md`            |
| **Env vars** (async prefetch, cache, debug, `DATA_OPTIMIZER_*`, Studio)       | `reference/env-vars.md`                             |
| Fair streaming benchmarks (`benchmarks/` suite)                               | `reference/benchmarking.md`                         |
| **Raw adaptive concurrency / look-ahead stages** (clients own rate)           | repo `benchmarks/ADAPTIVE_CONCURRENCY.md`           |
| Lightning Studio env, credentials, free-threading                             | `reference/lightning-studio.md`                     |
| Write path orchestration; raw internals; pointers to I/O + multi-node         | `reference/processing.md`                           |
| Dev env, PR/CI style                                                          | `reference/contributing.md`                         |
| Tests & fixtures                                                              | `reference/testing.md`                              |
| Tracing (`enable_tracer`, Litracer, Perfetto), breakpoints, env knobs         | `reference/debugging.md`                            |
| **Keyed lookup / `dataset_update` / `key_fn`**                                | `reference/keyed-lookup.md`                         |

## Public API (`src/litdata/__init__.py`)

| Symbol                                                  | Purpose                                 |
| ------------------------------------------------------- | --------------------------------------- |
| `StreamingDataset` / `StreamingDataLoader`              | Optimized stream + resumable loader     |
| `CombinedStreamingDataset` / `ParallelStreamingDataset` | Mix or zip streams                      |
| `StreamingRawDataset`                                   | Raw file stream                         |
| `TokensLoader`                                          | Token windows for LLMs                  |
| `optimize` / `map` / `merge_datasets` / `walk`          | Write / transform / merge / list        |
| `dataset_update` / `build_keys_index`                   | Keyed in-place patch / backfill sidecar |
| `train_test_split`                                      | Split by chunk ROIs                     |
| `index_parquet_dataset` / `index_hf_dataset`            | Index for streaming                     |
| `breakpoint`                                            | Multiprocessing-safe pdb                |
| `enable_tracer` (`litdata.debugger`)                    | Pipeline log → Litracer / Perfetto      |

Defined under `streaming/`, `processing/`, `raw/`, `utilities/` — see cookbook §6–9 for constructor args.

## Package map

- `streaming/` — read · `processing/` — write · `raw/` — raw stream · `cli/` — `litdata cache path|clear`
- `utilities/` — env, encryption, subsample, split, parquet, HF
- `constants.py` — optional-dep flags, env knobs, default chunk 64 MB
- Registries: downloaders, fs providers, serializers, compressors; `resolver.py` → `Dir`

## Shared concepts

- Chunk: `[num_items][offsets][data]`; `index.json` holds chunks + config (`data_format`, `item_loader`, …) — [storage-format.md](reference/storage-format.md).
- Item loaders own layout + intervals (`PyTreeLoader`, `TokensLoader`, `ParquetLoader`).
- Write/management I/O = `FsProvider` (s3/gs/r2); training downloads = `Downloader`. Optimize worker pools use FsProvider via `_download_data_target` / `_upload_fn` — see `data-movement.md`. Sampler `ChunkedIndex` is read-path; `CacheBatchSampler` is `CacheDataLoader` only.
- Ranks from env (`_DistributedEnv` / `DATA_OPTIMIZER_*`), not a custom network.
- Shuffle deterministic from `seed`+epoch+chunk → resumable (`shuffle.py`, not `sampler.py`). Local POSIX-fast keeps `FullShuffle` unless the mount is Vast/NFS/Lustre/GPFS (or `LITDATA_POSIX_FAST=1` → `WindowShuffle`).
- Design: one less thing to remember; pure PyTorch; backward compatible; test-driven.

## Traps (do not “fix” by guessing)

- **`FileNotFoundError` / `ChunkWaitTimeoutError` after ~120s** on chunks with `num_workers>0` is usually prefetch-thread death or wait timeout — not a missing object. Check stderr + tracer `crash`.
- **`mode="append"` vs `use_checkpoint=True`**: append continues **chunk indices**; checkpoint resumes **input slices**. Combining them is unsafe; checkpoint filenames / `done_till_index` are known-fragile (`writer.save_checkpoint` vs `data_processor` load).
- **Azure/HF** resolve for **read**; optimize **write** providers are `s3`/`gs`/`r2` only (`_SUPPORTED_PROVIDERS`).
- **`dataset_update`**: local dirs only; `ds[int]` is positional.

## Quick commands

```bash
make setup
pre-commit run --all-files
mypy
pytest tests/path/test_x.py::test_name -v --capture=no
litdata cache path
litdata cache clear
```

Examples: `examples/`. Version: `src/litdata/__about__.py`.
