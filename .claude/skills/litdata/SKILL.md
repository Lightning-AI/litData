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

## Expert usage (load using-litdata.md)

Before writing examples or answering how-tos, read the cookbook. Highlights:

| Topic           | Remember                                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Images          | Return **JPEG** (`JpegImageFile` / quality ≈95). Plain `PIL.Image` / `fromarray` → huge PIL RAW                                |
| Train stream    | `StreamingDataLoader` + `shuffle=True, drop_last=True, seed=…`                                                                 |
| Optimize        | `if __name__ == "__main__"`; exactly one of `chunk_bytes` \| `chunk_size`                                                      |
| Cache           | Peak disk ≈ `num_workers × max_pre_download × chunk_size`; default `max_cache_size="100GB"`                                    |
| **Paths**       | Always use LitData resolution — `s3/gs/r2/azure/hf/local:` + `/teamspace/...` (direct bucket I/O). See `reference/resolver.md` |
| Parquet workers | `multiprocessing_context="spawn"` on Linux                                                                                     |

## Reference map

| Task                                                                               | Read                                                |
| ---------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Use the library** (APIs, recipes, optimize/map/walk knobs, serializers, shuffle) | `reference/using-litdata.md`                        |
| **Paths / URLs / Studio mounts / `Dir` / time templates**                          | `reference/resolver.md` (+ README `#resolve-paths`) |
| Read path, chunk format, shuffle math, item loaders                                | `reference/streaming.md`                            |
| Cache / prefetch / eviction / shared-chunk deletion                                | `reference/cache-and-chunk-lifecycle.md`            |
| Fair streaming benchmarks (`benchmarks/` suite)                                    | `reference/benchmarking.md`                         |
| Lightning Studio env, credentials, free-threading                                  | `reference/lightning-studio.md`                     |
| Write path (`optimize`/`map`)                                                      | `reference/processing.md`                           |
| Dev env, PR/CI style                                                               | `reference/contributing.md`                         |
| Tests & fixtures                                                                   | `reference/testing.md`                              |
| Tracing, breakpoints, env knobs                                                    | `reference/debugging.md`                            |

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

- Chunk: `[num_items][offsets][data]`; `index.json` holds chunks + config (`data_format`, `item_loader`, …).
- Item loaders own layout + intervals (`PyTreeLoader`, `TokensLoader`, `ParquetLoader`).
- Ranks from env (`_DistributedEnv` / `DATA_OPTIMIZER_*`), not a custom network.
- Shuffle deterministic from `seed`+epoch+chunk → resumable.
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
