# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

---

## [unreleased] - YYYY-MM-DD

### Added

- ``Jpeg`` / ``JpegArray`` default to ``max_quality=95`` with ``quality=None``. ``max_quality`` is a cap: existing JPEG bytes whose estimated quality (IJG luminance q-table) is already ≤ the cap are kept as-is (Hub tiny-imagenet q=75 stays q=75). Higher-quality JPEGs, PNG, or raw pixels encode at the cap. ``quality=`` still force-encodes at that JPEG quality. ``quality`` and ``max_quality`` are mutually exclusive. ``Image`` leaves both unset so path/bytes stay pass-through unless you set one.
- Pytree ``optimize(..., compression="zstd")`` accepts ``compression_level="chunk"|"batch"|"sample"`` (omitted zstd defaults to ``batch``: framed ``.bin`` with ``compression_batch_size=256``, matching Arrow IPC / decode windows). ``chunk`` is whole-file ``.zstd.bin``. ``sample`` zstd-compresses each item payload between offsets. Numeric zstd levels stay ``compression="zstd:N"``. Nested Arrow IPC is unchanged.

### Fixed

- ``optimize_hf(..., overwrite=False)`` reuses a remote ``index.json`` (``r2://``, ``s3://``, lightning storage) the same way ``optimize`` does, not only a local file.
- ``StreamingDataset.load_state_dict`` treats a missing ``item_shuffle_window`` as the pre-PR full in-chunk shuffle (``0``), not the new 256 default.
- Nested Arrow ``chunk_bytes`` accounting no longer assumes 3× zstd before the first flush. First shards of Hub JPEG/WAV were 140–188MB on a 64MB target. The writer now counts sample payload bytes, then uses measured on-disk bytes/item.
- Unordered ``optimize()`` workers load items from a unique temp pickle instead of a shared ``node-{rank}-items.pkl`` under the chunk cache, so overlapping runs (pytest-xdist) no longer ``IndexError`` or mix inputs.
- Process-level cache of Lightning Cloud temp-bucket credentials **and** the boto3 R2 client, keyed by ``data_connection_id`` (TTL aligned to the 2700s refetch interval). A new ``StreamingDataset`` / ``R2Client`` in the same process no longer re-logins (~1s) or rebuilds boto3 (~50ms) on the first GET. Scheduled refresh still mints new credentials. R2 clients skip optional SDK checksums. Tiny indexed chunks (<8MB) use ``get_object`` instead of TransferManager/obstore. ``clear_temp_bucket_credentials_cache()`` isolates tests.
- `optimize` to lightning_storage (FUSE path + R2 url) is treated as remote: leftover ``.bin`` in a shared chunk cache no longer abort index merge. Removers now run even when ``input_dir`` is empty (HF optimize). ``R2Client`` copies ``storage_options`` so ``data_connection_id`` cannot be popped off a shared dict before client create. ``optimize`` / ``DataProcessor`` / ``_read_updated_at`` merge ``data_connection_id`` from the resolved Dir into ``storage_options``. A truncated FUSE ``index.json`` falls back to the object-store copy instead of raising ``JSONDecodeError``.

### Changed

- `optimize_hf` default ``chunk_bytes`` is **256MB** (was 64MB) so remote scans issue fewer R2 GETs. Pass ``chunk_bytes="64MB"`` for the old layout.
- Arrow IPC zstd is skipped when the row is already-compressed binary (Hub JPEG/WAV ``{bytes, path}``). Text/JSON still uses IPC zstd.
- Remote prefetch raises toward unique chunk count within a ~3GB RAM budget (cap 32). Obstore reuses a 32-connection idle pool and issues parallel Range GETs for objects ≥16MB.
- `optimize_hf` keeps Hub parquet ``image`` / ``img`` / ``audio`` / ``video`` ``{bytes, path}`` as Arrow binary/string structs (same as parquet ``to_pylist``). It no longer wraps them as ``Image`` / ``Audio`` / ``Video`` (that decoded to tensors). Explicit ``PIL.Image`` / ``litdata.Image`` still use pytree serializers.
- `R2Downloader.download_file` prefers obstore for chunk GETs (same as S3); ``index.json`` stays on boto3 so the DataLoader parent does not start tokio before fork. Single-chunk lightning_storage streams no longer wait on serial boto3.
- `ParquetLoader` (low-memory) converts each row group with Arrow ``to_pylist()`` instead of per-cell ``as_py()``. Sequential IMDB parquet matches Hugging Face `datasets` streaming dicts and is ~2× faster than the old cell path.
- `StreamingDataset("hf://...")` still prefetches whole parquet files (PrepareChunksThread + ``hf_hub_download``), then converts row groups with ``to_pylist()``. Hub UltraChat: ~56k rows/s vs HF streaming ~7k once cached. Nested parquet trees are listed recursively; ``storage_options`` / ``HF_TOKEN`` are forwarded to indexing. Removed unused ``ParquetLoader(range_read=...)`` (Hub range-reads on getitem were slower than HF streaming).
- ``index_hf_dataset`` persists ``index.json`` at ``{cache_dir}/hf-index/<url_hash>/`` (no timestamp folder) and reuses it on later calls. A ready ``{cache_dir}/index.json`` is also reused.
- ``HFDownloader`` parses ``hf://datasets/org/name@refs/convert/parquet/...`` (revision + path) instead of treating ``name@refs`` as the repo id. Wildcard ``*.parquet`` paths also match nested chunk filenames.
- ``optimize_hf(name, output_dir=..., revision=..., split=..., chunk_size=...)`` indexes first, persists parquet under ``hf-parquet/<hash>/`` **outside** the chunk cache (so ``optimize`` cannot delete it), then writes **256MB** chunks from ``hf://`` URLs (workers re-download if a persist file is missing). Variable-length nested lists/dicts are one compact binary leaf (list[str]/list[int]/records; legacy JSON still reads, via ``orjson`` when installed) so SQuAD / UltraChat / OASST no longer break the index merge. Nested chunks append an Arrow IPC **file** (256-row zstd batches when compressed; skipped for Hub JPEG/WAV); the reader ``get_batch`` + ``to_pylist()``s one batch (same C++ inflate as parquet row groups). Reuses ``output_dir`` when ``index.json`` already exists.
- ``StreamingDataset(batch_decode="auto")`` (default) picks a decode window from the data format and mean sample size: 256 for text/nested, down to 1 for multi-MB images/video. Pass ``0`` / ``N`` / ``"all"`` to pin it. ``LITDATA_BATCH_DECODE`` / ``LITDATA_BATCH_ROWS`` apply only when ``batch_decode`` is ``"auto"``. Decode windows are **aligned** (``[kW, (k+1)W)``). ``StreamingDataset(item_shuffle_window=...)`` (default 256 / ``"auto"``) shuffles those blocks then the items inside each, so ``shuffle=True`` still reuses the cache. ``0`` / ``"full"`` restores a full in-chunk permutation. ``LITDATA_ITEM_SHUFFLE_WINDOW`` applies only when the argument is omitted.
- Nested HF chunks write **only** the Arrow IPC **file** footer (no duplicate JSON pytree body): ``ARROW1`` + 256-row record batches (Arrow IPC zstd when ``optimize(compression="zstd")``) + slim LitData prefix + ``LDARW01`` trailer. The same path now covers **flat JSON structs** (cnn_dailymail ``article``/``highlights``/``id``, IMDB, sst2), not only nested lists/maps. ``chunk_bytes`` is that on-disk size. LitData whole-file ``.zstd.bin`` is skipped for these chunks. The reader ``open_file``s the footer, ``get_batch(i).to_pylist()``s one batch, and caches that window like parquet row groups. Legacy IPC **stream** footers still ``read_all()`` + slice.
- Framed pytree zstd (``compression_level="batch"``) inflates each ``LDFZ01`` frame with PyArrow's C++ ``Codec`` (``decompressed_size``, mmap ``memoryview``, one reused codec per loader). Falls back to the ``zstd`` package if pyarrow is missing.
- Omitted ``compression_level`` with ``compression="zstd"`` / ``"zstd:N"`` now defaults to ``"batch"`` (framed ``.bin``, ``compression_batch_size=256``) instead of whole-file ``.zstd.bin``. Pass ``compression_level="chunk"`` for the previous wrap. Nested Arrow IPC still skips pytree file compression.

## [0.2.73] - 2026-08-31

### Changed

- `optimize` / `map` no longer pickle the full input list into every spawned worker. ([#890](https://github.com/Lightning-AI/litData/pull/890))

### Fixed

- Multi-node `optimize(mode="append")` no longer repeats existing chunks once per node in the merged `index.json`. ([#866](https://github.com/Lightning-AI/litData/pull/866))
- Restore PyTorch's process-global DataLoader worker loop after VizTracer/cProfile setup, so a later loader in the same process does not inherit LitData's profiling worker. ([#893](https://github.com/Lightning-AI/litData/pull/893))

## [0.2.72] - 2026-08-24

### Fixed

- Temporary S3/R2 credential refresh no longer kills a run on a single control-plane blip. The `/v1/auth/login` POST is now retried (urllib3 excludes POST from its default `allowed_methods`, so it never was), a failed refresh keeps serving the current credentials and retries on a timer instead of re-requesting on every read, refreshes are jittered per process so forked DataLoader workers stop stampeding, and the default refresh interval moved from 55 to 45 minutes to leave a usable grace period inside the 1 hour credential TTL. Creating the first client now waits out a control-plane outage instead of failing the job, while missing configuration, rejected credentials and local errors such as a bad `storage_options` key still fail immediately. Also fixes `_CustomRetryAdapter` never applying its default request timeout, because `requests` always passes `timeout` explicitly as `None`, and declares the `urllib3 >=1.26` floor that `streaming/client.py` has always needed for its direct `Retry` import. ([#891](https://github.com/Lightning-AI/litData/pull/891))
- `NumpySerializer` / `NoHeaderNumpySerializer` copy on deserialize, so arrays are writable (`np.frombuffer` returns a read-only view, which made `torch.from_numpy` warn on every collate) and no longer alias an mmap the cache can unmap underneath them. ([#887](https://github.com/Lightning-AI/litData/pull/887))

## [0.2.71] - 2026-08-15

### Added

- `StreamingDataLoader(..., profile_cprofile=True)` writes stdlib cProfile stats for the main process and worker 0 (`cprofile_main.prof` / `cprofile_worker0.prof` plus `.txt` summaries). No extra dependency. Cannot be combined with `profile_batches` (viztracer). ([#885](https://github.com/Lightning-AI/litData/pull/885))

### Fixed

- Prefetch accepts `numpy.int64` chunk indexes from shuffle (they were dropped by `isinstance(..., int)`), so force-download stays a last resort after `_FORCE_DOWNLOAD_TIME`. Concurrent `os.replace` of the same chunk is a no-op when the destination already exists.
- Multi-node `FullShuffle` epoch ≥ 2 keeps each node's unique chunk set when that shard fits in `max_cache_size` (in-chunk permute is seeded by the global chunk id). If it does not fit, chunks are re-scheduled across nodes. ([#886](https://github.com/Lightning-AI/litData/pull/886))

### Changed

- Faster pytree flatten on the writer hot path: skip typing-generic `isinstance` and PIL JPEG probes on non-list/tuple nodes, skip namedtuple/JPEG probes on scalar leaves, and call `_get_node_type` once per node. After the first sample, `BinaryWriter` walks `tree_leaves` (non-generator collect) instead of rebuilding a `TreeSpec`, caches per-leaf byte sizes, and packs the size header with `struct` instead of NumPy. When every leaf has a fixed size (int/float/bool), later samples reuse a cached size header and write into one buffer. `BooleanSerializer` advertises `size = 1`. Reader offset pairs and size headers use `struct` instead of NumPy.
- Remote→local streaming: cap in-flight chunk GETs with `LITDATA_ASYNC_DOWNLOAD_CONCURRENCY` (default 8), drain the prefetch queue up to gather width when slots are free, signal file-ready with `Event.set` after the downloader's atomic `os.replace` (decompress publishes the readable `.bin`), and `posix_fadvise(WILLNEED)` downloaded cache files from the prefetch thread (mmap stays on the reader thread).
- `StreamingDataset` / `Cache` `max_cache_size`: `None` uses 75% of free disk (leave ≥50GB when possible). `"100G"` / `"50GB"` pins bytes; `0.90` (or `MAX_CACHE_SIZE=0.90`) uses that fraction of currently free space. ([#886](https://github.com/Lightning-AI/litData/pull/886))

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
