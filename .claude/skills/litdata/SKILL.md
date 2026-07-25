---
name: litdata
description: Work on the LitData codebase — understand its architecture, contribute code, write/run tests, and debug/profile runtime issues. Use when navigating or editing src/litdata, answering "how does X work / where does X live", getting a change merge-ready, adding or running tests, diagnosing a slow/hanging/nondeterministic StreamingDataset or optimize/map run, or working inside Lightning Studio (/teamspace paths, data connections, free-threaded benches). Covers streaming (read) and processing (write) pipelines, cloud backends, Studio resolution, chunk format, shuffling, CLI, tests, CI, and tracing.
---

# LitData

LitData (`import litdata`) preprocesses and streams datasets for PyTorch training. Two pipelines that mirror each other:

- **Write** (`optimize` / `map`): distributed workers turn raw data into a chunked binary format (`chunk-*.bin` + `index.json`) and upload it. → `src/litdata/processing/`
- **Read** (`StreamingDataset` + `StreamingDataLoader`): stream those chunks back cloud → local cache → decoded items → batches, with deterministic shuffling and resumable state. → `src/litdata/streaming/`

A lighter path, `StreamingRawDataset` (`src/litdata/raw/`), streams *un-optimized* original files, skipping the write step.

## Read this first, then open the matching reference file

Keep this SKILL.md as the map; load a `reference/` file only for the task at hand.

| Your task                                                               | Read                                                                                      |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Understand the read path, chunk format, shuffling, resume, item loaders | `reference/streaming.md`                                                                  |
| Cache↔Writer/Reader, PrepareChunksThread, shared-chunk deletion races   | `reference/cache-and-chunk-lifecycle.md`                                                  |
| Prefetch/`max_pre_download`, eviction deadlocks, async+obstore knobs    | `reference/cache-and-chunk-lifecycle.md` (Prefetch & eviction) + `reference/debugging.md` |
| Studio ImageNet cold-epoch benches, fair async vs boto3 comparisons     | `reference/benchmarking.md`                                                               |
| Lightning Studio: `/teamspace` paths, data connections, credentials     | `reference/lightning-studio.md`                                                           |
| Understand the write path (`optimize`/`map`), worker model, raw indexer | `reference/processing.md`                                                                 |
| Set up dev env, coding style, branch/PR flow, lint/type/CI gates        | `reference/contributing.md`                                                               |
| Write or run tests, fixtures, mocking cloud, gating                     | `reference/testing.md`                                                                    |
| Trace/profile, set worker breakpoints, env knobs, diagnose failures     | `reference/debugging.md`                                                                  |

## Public API (`src/litdata/__init__.py`)

| Symbol                                       | Purpose                                       | Defined in                                              |
| -------------------------------------------- | --------------------------------------------- | ------------------------------------------------------- |
| `StreamingDataset`                           | Read optimized chunks (`IterableDataset`)     | `streaming/dataset.py:51`                               |
| `StreamingDataLoader`                        | `DataLoader` subclass with resumable state    | `streaming/dataloader.py:559`                           |
| `CombinedStreamingDataset`                   | Sample one of N datasets per step (by weight) | `streaming/combined.py:40`                              |
| `ParallelStreamingDataset`                   | Pull one sample from every dataset per step   | `streaming/parallel.py:44`                              |
| `StreamingRawDataset`                        | Stream raw files, no optimize step            | `raw/dataset.py:95`                                     |
| `TokensLoader`                               | Item loader for tokenized/NLP data            | `streaming/item_loader.py:402`                          |
| `optimize`                                   | Turn a dataset into litdata chunks            | `processing/functions.py:387`                           |
| `map`                                        | Run a fn over inputs for side effects         | `processing/functions.py:242`                           |
| `merge_datasets`                             | Merge several optimized datasets              | `processing/functions.py:675`                           |
| `walk`                                       | Cloud-optimized `os.walk`                     | `processing/functions.py:621`                           |
| `train_test_split`                           | Split a `StreamingDataset` by chunk ROIs      | `utilities/train_test_split.py:14`                      |
| `index_parquet_dataset` / `index_hf_dataset` | Index parquet / HF for streaming              | `streaming/writer.py:578`, `utilities/hf_dataset.py:13` |
| `breakpoint`                                 | Multiprocessing-safe pdb (works in workers)   | `utilities/breakpoint.py:33`                            |

## Package map

- `streaming/` — read pipeline · `processing/` — write pipeline (`optimize`/`map`) · `raw/` — raw streaming + file indexer
- `cli/` — the `litdata` command; dispatch is `__main__.py:app` → `parser.parse_args` iterates `COMMAND_REGISTRY` (`cli/commands.py:68`). Commands: `cache clear`, `cache path`, `optimize` (stub). Add one by appending a registrar.
- `utilities/` — `env.py` (rank detection), `encryption.py`, `subsample.py`, `train_test_split.py`, `parquet.py`, `hf_dataset.py`, `dataset_utilities.py` (`get_default_cache_dir`), `_pytree.py` (vendored; excluded from lint/type)
- `constants.py` — `_*_AVAILABLE` optional-dep flags, env-var knobs, dtype maps, default chunk size (`1<<26` = 64 MB) · `debugger.py` — structured tracing

## Concepts shared across both pipelines

- **Chunk format** (`writer.py:218`): `[num_items:uint32][offset_array:uint32[N+1]][item_data:bytes]`. Per-worker `{rank}.index.json` files are merged into one `index.json` holding a `chunks` list + `config` (compression, `data_format`, `data_spec` treespec, `item_loader` class).
- **Item loaders own byte layout AND interval math** (`item_loader.py`): `PyTreeLoader` (default), `TokensLoader` (mmap token windows), `ParquetLoader`. The class in `index.json` must match the reader's (`config.py:361`).
- **Distribution is env-var driven, not networked** — rank read from `_DistributedEnv`/`_WorkerEnv` (read) or `DATA_OPTIMIZER_*` (write).
- **Determinism**: read-path shuffling is seeded by `seed`+`epoch`+`num_chunks`+`chunk_index`; same inputs ⇒ same order ⇒ resumable.
- **Pluggable registries**: `Downloader` (read, by URL prefix, `downloader.py`), `FsProvider` (write/management, `fs_provider.py`), serializers (`serializers.py`), compressors (`compression.py`). `resolver.py` turns a path/URL/teamspace path into a `Dir`.

## Lightning Studio (short)

Much of LitData development and S3 benchmarking happens in **Lightning Studio**. Paths like `/teamspace/s3_connections/<name>/…` are resolved by `streaming/resolver.py` into a `Dir(path, url, data_connection_id)` — downloads use the cloud URL + temp credentials from the Lightning API, while `/teamspace/studios/this_studio` stays local workspace disk. Cache for benches is often `/cache/chunks`. Details: [reference/lightning-studio.md](reference/lightning-studio.md).

## Design principles (CONTRIBUTING.md) — honor when editing

"One less thing to remember." No abstractions on top of pure PyTorch. Simple, readable internal code (many users aren't engineers). Backward-compatible APIs with deprecation warnings. Test-driven: reproduce a bug as a failing test, then fix.

## Quick commands

```bash
make setup                                           # dev env (uv install + pre-commit)
pre-commit run --all-files                           # lint + format + hooks
mypy                                                 # type check (files=["src"])
pytest tests/path/test_x.py::test_name -v --capture=no   # one test
litdata cache path        # print cache dir      · litdata cache clear   # wipe it
```

Runnable end-to-end examples live in `examples/`; the README's `<details>` feature blocks are the narrative docs (line refs are cited in the reference files). Repo version: see `src/litdata/__about__.py`.
