# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

---

## [unreleased] - YYYY-MM-DD

### Fixed

- Temporary S3/R2 credential refresh no longer kills a run on a single control-plane blip. The `/v1/auth/login` POST is now retried (urllib3 excludes POST from its default `allowed_methods`, so it never was), a failed refresh keeps serving the current credentials and retries on a timer instead of re-requesting on every read, refreshes are jittered per process so forked DataLoader workers stop stampeding, and the default refresh interval moved from 55 to 45 minutes to leave a usable grace period inside the 1 hour credential TTL. Creating the first client now waits out a control-plane outage instead of failing the job, while missing configuration, rejected credentials and local errors such as a bad `storage_options` key still fail immediately. Also fixes `_CustomRetryAdapter` never applying its default request timeout, because `requests` always passes `timeout` explicitly as `None`, and declares the `urllib3 >=1.26` floor that `streaming/client.py` has always needed for its direct `Retry` import.
- Multi-node `FullShuffle` epoch ≥ 2 keeps each node's unique chunk set when that shard fits in `max_cache_size` (in-chunk permute is seeded by the global chunk id). If it does not fit, chunks are re-scheduled across nodes.

### Changed

- `StreamingDataset` / `Cache` `max_cache_size`: `None` uses 75% of free disk (leave ≥50GB when possible). `"100G"` / `"50GB"` pins bytes; `0.90` (or `MAX_CACHE_SIZE=0.90`) uses that fraction of currently free space.

## [0.2.71] - 2026-08-15

### Added

- `StreamingDataLoader(..., profile_cprofile=True)` writes stdlib cProfile stats for the main process and worker 0 (`cprofile_main.prof` / `cprofile_worker0.prof` plus `.txt` summaries). No extra dependency. Cannot be combined with `profile_batches` (viztracer). ([#885](https://github.com/Lightning-AI/litData/pull/885))

### Fixed

- Prefetch accepts `numpy.int64` chunk indexes from shuffle (they were dropped by `isinstance(..., int)`), so force-download stays a last resort after `_FORCE_DOWNLOAD_TIME`. Concurrent `os.replace` of the same chunk is a no-op when the destination already exists.

### Changed

- Faster pytree flatten on the writer hot path: skip typing-generic `isinstance` and PIL JPEG probes on non-list/tuple nodes, skip namedtuple/JPEG probes on scalar leaves, and call `_get_node_type` once per node. After the first sample, `BinaryWriter` walks `tree_leaves` (non-generator collect) instead of rebuilding a `TreeSpec`, caches per-leaf byte sizes, and packs the size header with `struct` instead of NumPy. When every leaf has a fixed size (int/float/bool), later samples reuse a cached size header and write into one buffer. `BooleanSerializer` advertises `size = 1`. Reader offset pairs and size headers use `struct` instead of NumPy.
- Remote→local streaming: cap in-flight chunk GETs with `LITDATA_ASYNC_DOWNLOAD_CONCURRENCY` (default 8), drain the prefetch queue up to gather width when slots are free, signal file-ready with `Event.set` after the downloader's atomic `os.replace` (decompress publishes the readable `.bin`), and `posix_fadvise(WILLNEED)` downloaded cache files from the prefetch thread (mmap stays on the reader thread).

## [0.2.70] - 2026-08-15

### Added

- Introduced `CHANGELOG.md` to track changes across releases ([#733](https://github.com/lightning-ai/litdata/pull/733))
- Elastic mid-epoch resume for `StreamingDataset` ([#878](https://github.com/Lightning-AI/litData/pull/878)): same-topology pause/resume keeps the existing shuffler assignment. Changing `num_workers` or `world_size` restripes the remaining `sample_in_epoch` suffix (no duplicates). Combined/Parallel datasets resume only with the same topology. `force_override_state_dict` does not override worker count.
- Add environment variable `LITDATA_DISABLE_VERSION_CHECK` to disable PyPI version check ([#737](https://github.com/Lightning-AI/litData/pull/737)). The upgrade tip is **off by default**; set `LITDATA_CHECK_UPDATES=1` to enable it.
- In-place POSIX reads (mmap + `posix_fadvise`) are **automatic** for local / Vast / NFS datasets. Object URLs (`s3://`) still GET compacted chunks. Disable with `LITDATA_POSIX_FAST=0`. `WindowShuffle` is used on parallel filesystems (or `LITDATA_POSIX_FAST=1`); local disks keep `FullShuffle`. `WILLNEED` and `num_workers` are capped from `MemAvailable` (`LITDATA_POSIX_MAX_WORKERS`, `LITDATA_POSIX_WILLNEED`). ([#876](https://github.com/Lightning-AI/litData/pull/876))
- Leveled pipeline tracing: `enable_tracer(level="batch"|"chunk"|"sample"|"debug")` or `categories=["download", "read", "delete"]`. Events use stable names (`download`, `read`, `delete`, `batch`, `sample`) with indexes in args so Perfetto groups download vs read vs delete. Convert with [Lightning-AI/litracer](https://github.com/Lightning-AI/litracer) (`--quiet --validate --cat`).
- `ChunkWaitTimeoutError` when a chunk does not appear within `MAX_WAIT_TIME` (still a `FileNotFoundError` subclass).
- Media serializers for video (torchcodec), audio (torchcodec ``AudioDecoder``), mesh, PDF, and NIfTI. Video/audio default to lazy decoders. Use `decode="all"` / `decode="samples"` to materialize tensors. `VideoSerializer(device="cuda")` is forced to CPU in DataLoader and optimize workers.
- `list_media_folder` / `iter_webdataset_tar` for class-folder trees (`root/class/file`) and WebDataset tars.
- `Graph` / `graph` serializer: pack PyG `Data.to_dict()` / `HeteroData.to_dict()` tensors (plus a small store tree). Reconstruct with `Data.from_dict` / `HeteroData.from_dict` when torch-geometric is installed. `StreamingDataLoader` defaults to `litdata_collate` (graphs → `Batch.from_data_list`, other samples → `default_collate`). NetworkX uses `graph:pickle`.
- `complete_dataset` / `is_complete_dataset` merge `{rank}.index.json` shards. `StreamingDataset` also tries this (and auto-indexes a parquet folder) when `index.json` is missing.
- `StreamingDataset.subset(indices)` for an index/slice view (same ROI logic as `train_test_split`).
- Typed media wrappers matching serializers (`Text`, `Audio`, `Video`, `Image`, `Jpeg`, `JpegArray`, `Pil`, `Tiff`, `File`, `Mesh`, `Pdf`, `Nifti`, `Tensor`, `Graph`) so `optimize` can tell a filepath from a caption. `Tensor(array=)` is a pytree leaf for `TensorSerializer` / `NoHeaderTensorSerializer`; 1-D tensors expose `.shape` for `TokensLoader.encode_data`. Native payloads: `Audio(array=, sampling_rate=)`, `Image(array=, quality=95, format="jpeg")`, `Video(array=, fps=)`, `Mesh(mesh=)`, `Pdf(pdf=)`, `Nifti(array=, affine=)`. Audio/Video also re-encode torchcodec decoders via `_hf_encoded` / `_litdata_encoded`. Image encode downcasts arrays and keeps the PIL native format. Decode is **bytes → tensor** via torchvision (no PIL). EXIF orientation uses PIL only when a JPEG APP1 Exif marker is present. Bare `*.jpg` / `*.png` paths are claimed by `ImageSerializer` so `optimize` can return tensors. Audio decoders support `audio["array"]` / `audio["sampling_rate"]`.
- `ParquetReader` / `ParquetLoader` column projection. `ParquetReader` also takes PyArrow `filters` and reshard by row group instead of loading each file. Low-memory `ParquetLoader` returns a row from the Arrow table (no Polars copy of the row group).

### Changed

- Faster `BinaryWriter` / `BinaryReader` hot path: shallow-copy serializers, cache per-leaf serializers after the first sample, assemble chunks into a pre-sized buffer, and deserialize mmap slices without an extra `bytes()` copy per leaf.
- `optimize`/`map` default to `keep_data_ordered=False` (shared per-node queue) ([#880](https://github.com/Lightning-AI/litData/pull/880)). `use_checkpoint` and `align_chunking` still force ordered slices. The ready queue carries item indexes (workers already have the item list). Prefetch is capped by slots and `LITDATA_PREFETCH_BYTES` (default 512MB). Node upload queues are bounded. The parent polls worker queues once per second. Start method stays `spawn`. Remote input downloads use the streaming `Downloader` (`adownload_file` + `asyncio.gather` / obstore when usable) instead of sync `fs_provider.download_file`. Tune with `LITDATA_OPTIMIZE_DOWNLOAD_BATCH` and `LITDATA_OPTIMIZE_DOWNLOAD_CONCURRENCY`. Remote uploads use the same async downloader path (`aupload_file` / `LITDATA_OPTIMIZE_UPLOAD_BATCH`); local `output_dir` still copies or write-through. Download concurrency scales with workers/CPU and backs off when disk is low. Local `output_dir` writes chunks in place (no cache→output copy). Unordered `optimize` can split transform vs write (`LITDATA_OPTIMIZE_SPLIT_WRITERS=1`, 1–2 chunk writers).
- `optimize`/`map` with `keep_data_ordered=False` shards work **per node**, then all workers on that node pull from one bounded queue. Download / upload / remove run as **node-level threads** (writers stay processes). Same-process I/O uses ``queue.Queue``; ``multiprocessing.Queue`` is only used at the writer process boundary. Direct `s3://` / `gs://` / `r2://` item paths and Studio `lightning_storage` URLs are downloaded instead of skipped. Studio FUSE mounts are not ``stat``/``listdir``'d for path detection or file-size packing.
- Tracer log timestamps are Chrome/Perfetto microseconds (`created * 1e6`). `enable_tracer(log_file=...)` selects the Litracer input file. Close the last `read` span when a worker finishes so Litracer B/E pairs match. Tracer calls are no-ops when tracing is off. Litracer writes gzip Chrome JSON (`.json.gz`) by default; Perfetto and `chrome://tracing` both open it.
- Optimize checkpoints write `checkpoint-{rank}.json` with `inputs_done`, `samples_written`, and `next_chunk_index`.
- Stream zstd decompress and obstore `adownload_file` to disk for S3/R2/Azure.

### Removed

### Fixed

- Fix async S3/R2 chunk prefetch crashing the prepare thread when `storage_options` contains LitData metadata (`data_connection_id`) or client-only keys (`endpoint_url`). The wait loop now surfaces that crash immediately instead of timing out as `FileNotFoundError`. Prepare-thread deaths print a traceback to stderr and emit a one-line Litracer instant event (`ph: I`) instead of a multi-line `logger.exception` that breaks `litdata_debug.log`.
- Fix multi-worker `StreamingDataLoader` `FileNotFoundError` on `s3://` datasets: obstore's tokio runtime is not fork-safe, so `index.json` is fetched with boto3 in the parent and workers lazy-init a fresh obstore store. If the parent already started obstore, workers fall back to boto3 instead of hanging until the chunk wait times out.
- Cloud `optimize`/`map` uploads re-raise instead of printing and continuing (which could delete the local file after a failed upload).
- Reject `azure://` / other non-s3-gs-r2 URLs as `optimize`/`map` output with a clear error (streaming those schemes for **read** is unchanged).
- Checkpoint resume no longer adds sample counts onto the chunk file index.
- TokensLoader copies each token block off the memmap before DataLoader IPC so unmapping the previous chunk cannot SIGSEGV workers ([#876](https://github.com/Lightning-AI/litData/pull/876)).

## [0.2.58] - 2025-10-07

## [0.2.57] - 2025-10-06

## [0.2.56] - 2025-09-23

## [0.2.55] - 2025-09-19

## [0.2.54] - 2025-09-10

## [0.2.53] - 2025-09-09

## [0.2.52] - 2025-08-12

## [0.2.51] - 2025-07-29

## [0.2.50] - 2025-06-27

## [0.2.49] - 2025-06-04

## [0.2.48] - 2025-05-24

## [0.2.47] - 2025-05-13

## [0.2.46] - 2025-05-03

## [0.2.45] - 2025-04-14

## [0.2.44] - 2025-03-26

## [0.2.43] - 2025-03-25

## [0.2.42] (yanked) - 2025-03-11

## [0.2.41] - 2025-03-07

## [0.2.40] - 2025-03-04

## [0.2.39] - 2025-02-14

## [0.2.38] - 2025-02-06

## [0.2.37] - 2025-01-22

## [0.2.36] - 2025-01-14

## [0.2.35] - 2025-01-14

## [0.2.34] - 2024-12-04

## [0.2.33] - 2024-11-29

## [0.2.32] - 2024-11-27

## [0.2.31] - 2024-11-21

## [0.2.30] - 2024-11-05

## [0.2.29] - 2024-09-26

## [0.2.28] - 2024-09-19

## [0.2.27] (yanked) - 2024-09-19

## [0.2.26] - 2024-09-03

## [0.2.25] - 2024-08-28

## [0.2.24] - 2024-08-14

## [0.2.23] - 2024-08-07

## [0.2.22] - 2024-08-05

## [0.2.21] - 2024-08-01

## [0.2.20] - 2024-08-01

## [0.2.19] - 2024-07-30

## [0.2.18] - 2024-07-24

## [0.2.17] - 2024-07-22

## [0.2.16] - 2024-07-11

## [0.2.15] - 2024-07-05

## [0.2.14] - 2024-06-27

## [0.2.13] - 2024-06-27

## [0.2.12] - 2024-06-17

## [0.2.11] - 2024-06-14

## [0.2.10] - 2024-06-13

## [0.2.9] - 2024-06-12

## [0.2.8] - 2024-06-03

## [0.2.7] - 2024-05-24

## [0.2.6] - 2024-05-07

## [0.2.5] - 2024-04-24

## [0.2.4] - 2024-04-24

## [0.2.3] - 2024-04-03

## [0.2.2] - 2024-03-08

## [0.2.1] - 2024-03-05

## [0.2.0] - 2024-02-26

## [0.2.0rc2] (pre-release) - 2024-02-26

## [0.2.0rc1] (pre-release) - 2024-02-24

## [0.2.0rc0] (pre-release) - 2024-02-23
