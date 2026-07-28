---
name: litdata
description: >-
  Expert use of the LitData library and work on its codebase. Use when writing or
  reviewing code that calls litdata (StreamingDataset, StreamingDataLoader,
  StreamingRawDataset, optimize, map, CombinedStreamingDataset,
  ParallelStreamingDataset, TokensLoader, serializers, train_test_split,
  merge_datasets, index_parquet_dataset, index_hf_dataset), answering how-to
  questions, choosing raw vs optimize vs parquet/HF/MDS, tuning cache/prefetch/
  shuffle/seed, resolving paths (s3/gs/r2/azure/hf/local:/teamspace via
  resolver.py), or when navigating/editing src/litdata, tests, CI, or debugging
  streaming / optimize / map.
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

| Topic          | Remember                                                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Raw files**  | `StreamingRawDataset`: raw `bytes`, fully async + batched downloads, retries; torch `DataLoader` — `#stream-raw` / `using-litdata.md` §10 |
| Images         | Return **JPEG** (`JpegImageFile` / quality ≈95). Plain `PIL.Image` / `fromarray` → huge PIL RAW                                           |
| Train stream   | Optimized: `StreamingDataLoader` + `shuffle=True, drop_last=True, seed=…`                                                                 |
| Optimize       | `if __name__ == "__main__"`; exactly one of `chunk_bytes` \| `chunk_size`                                                                 |
| Cache          | Peak disk ≈ `num_workers × max_pre_download × chunk_size`; default `max_cache_size="100GB"`                                               |
| Async prefetch | Remote downloads overlapped by default; `LITDATA_ASYNC_CHUNK_PREFETCH=0/1`; floor `max_pre` to 4 — `reference/env-vars.md`                |
| **Paths**      | Studio `/teamspace/s3_connections` & co are **FUSE** — LitData hits S3/GCS/**R2** (`lightning_storage`) directly. `reference/resolver.md` |
| Parquet / HF   | Index + `ParquetLoader` (HF auto); `spawn` with workers; `using-litdata.md` §10                                                           |

## Reference map

| Task                                                                          | Read                                                |
| ----------------------------------------------------------------------------- | --------------------------------------------------- |
| **Use the library** (raw, optimize/stream, parquet/HF, serializers, shuffle)  | `reference/using-litdata.md`                        |
| **Paths / URLs / Studio mounts / `Dir` / time templates**                     | `reference/resolver.md` (+ README `#resolve-paths`) |
| Read path, shuffle math, item loaders, Combined/Parallel                      | `reference/streaming.md`                            |
| **Cache / BinaryWriter / BinaryReader / `index.json` / FsProvider / sampler** | `reference/storage-format.md`                       |
| Cache / prefetch / eviction / shared-chunk deletion                           | `reference/cache-and-chunk-lifecycle.md`            |
| **Env vars** (async prefetch, cache, debug, `DATA_OPTIMIZER_*`, Studio)       | `reference/env-vars.md`                             |
| Fair streaming benchmarks (`benchmarks/` suite)                               | `reference/benchmarking.md`                         |
| Lightning Studio env, credentials, free-threading                             | `reference/lightning-studio.md`                     |
| Write path / **multi-node** `num_nodes` job launch                            | `reference/processing.md`                           |
| Dev env, PR/CI style                                                          | `reference/contributing.md`                         |
| Tests & fixtures                                                              | `reference/testing.md`                              |
| Tracing, breakpoints, env knobs                                               | `reference/debugging.md`                            |

## Public API (`src/litdata/__init__.py`)

| Symbol                                                  | Purpose                             |
| ------------------------------------------------------- | ----------------------------------- |
| `StreamingDataset` / `StreamingDataLoader`              | Optimized stream + resumable loader |
| `CombinedStreamingDataset` / `ParallelStreamingDataset` | Mix or zip streams                  |
| `StreamingRawDataset`                                   | Raw file stream                     |
| `TokensLoader`                                          | Token windows for LLMs              |
| `optimize` / `map` / `merge_datasets` / `walk`          | Write / transform / merge / list    |
| `train_test_split`                                      | Split by chunk ROIs                 |
| `index_parquet_dataset` / `index_hf_dataset`            | Index for streaming                 |
| `breakpoint`                                            | Multiprocessing-safe pdb            |

Defined under `streaming/`, `processing/`, `raw/`, `utilities/` — see cookbook §6–9 for constructor args.

## Package map

- `streaming/` — read · `processing/` — write · `raw/` — raw stream · `cli/` — `litdata cache path|clear`
- `utilities/` — env, encryption, subsample, split, parquet, HF
- `constants.py` — optional-dep flags, env knobs, default chunk 64 MB
- Registries: downloaders, fs providers, serializers, compressors; `resolver.py` → `Dir`

## Shared concepts

- Chunk: `[num_items][offsets][data]`; `index.json` holds chunks + config (`data_format`, `item_loader`, …) — [storage-format.md](reference/storage-format.md).
- Item loaders own layout + intervals (`PyTreeLoader`, `TokensLoader`, `ParquetLoader`).
- Write/management I/O = `FsProvider` (s3/gs/r2); training downloads = `Downloader`. Sampler `ChunkedIndex` is read-path; `CacheBatchSampler` is `CacheDataLoader` only.
- Ranks from env (`_DistributedEnv` / `DATA_OPTIMIZER_*`), not a custom network.
- Shuffle deterministic from `seed`+epoch+chunk → resumable (`shuffle.py`, not `sampler.py`).
- Design: one less thing to remember; pure PyTorch; backward compatible; test-driven.

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
