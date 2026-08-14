# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

---

## [unreleased] - YYYY-MM-DD

### Added

- Introduced `CHANGELOG.md` to track changes across releases ([#733](https://github.com/lightning-ai/litdata/pull/733))
- Add environment variable `LITDATA_DISABLE_VERSION_CHECK` to disable PyPI version check ([#737](https://github.com/Lightning-AI/litData/pull/737))
- Leveled pipeline tracing: `enable_tracer(level="batch"|"chunk"|"sample"|"debug")` or `categories=["download", "read", "delete"]`. Events use stable names (`download`, `read`, `delete`, `batch`, `sample`) with indexes in args so Perfetto groups download vs read vs delete. Convert with [Lightning-AI/litracer](https://github.com/Lightning-AI/litracer) (`--quiet --validate --cat`).
- `ChunkWaitTimeoutError` when a chunk does not appear within `MAX_WAIT_TIME` (still a `FileNotFoundError` subclass).

### Changed

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
