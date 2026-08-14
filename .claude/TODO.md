# LitData follow-ups

From the Aug 2026 codebase review. Check items off as they land. Skills live in `skills/litdata/`.

## P1 — Reliability

- [x] Typed wait timeout instead of bare `FileNotFoundError` (`ChunkWaitTimeoutError`)
- [x] Re-raise cloud upload errors in `_upload_fn` (do not `print` and continue / delete)
- [x] Unify `use_checkpoint`: `checkpoint-{rank}.json`, separate `inputs_done` / `samples_written` / `next_chunk_index`, fail if checkpoint upload fails
- [x] Document that `mode="append"` is not checkpoint resume (README + optimize docstring)

## P2 — Performance

- [x] Stream zstd decompress (no full `f.read()` of compressed chunk)
- [x] Stream `adownload` to disk (obstore stream for S3/R2/GCS/Azure)
- [ ] Broadcast optimize file-size weights across nodes (`data_processor.py` TODO)
- [ ] `ParquetReader`: scan batches instead of loading a full table per file

## P2 — Ease of use / API

- [x] Explicit error when `optimize`/`map` write URL is not s3/gs/r2 (e.g. `azure://`)
- [x] Honest `walk()` docstring (threaded `os.listdir`, Studio-oriented — not cloud `os.walk`)
- [x] README keyed lookup (`key_fn`, `build_keys_index`, `dataset_update`, `get_by_key`)
- [ ] Gate Studio upsell `print` on `verbose` / env
- [ ] Elastic resume (`num_workers` / ranks change) or explicit error

## P3 — Tests & coverage

- [x] Multi-node `key_fn` merge / `concatenate_key_files`
- [ ] `StreamingDataset` + fork + S3 (obstore parent vs worker)
- [ ] Windows `PermissionError` retries (`_open_chunk_file`, keys `os.replace`)
- [ ] `dataset_update` remote (or keep `NotImplementedError` + test)
- [ ] Do not swallow `read_index_file_content` cloud errors (404 → `None`, 403 → raise)

## Skills (done 2026-08-14)

- [x] `reference/keyed-lookup.md` + SKILL / using / processing / testing / debugging / env-vars
