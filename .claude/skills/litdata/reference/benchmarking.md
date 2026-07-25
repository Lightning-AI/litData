# Benchmarking streaming (Studio / real S3)

Use this when comparing download backends, `max_pre_download`, async prefetch, or cache sizes on a real ImageNet-scale dataset. Microbenches lie; cold epoch-0 with a wiped cache is the truth for S3.

Studio path / credential background: [lightning-studio.md](lightning-studio.md).

## Canonical Studio dataset & runner

| Item          | Value                                                                                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dataset       | `/teamspace/s3_connections/optimized-imagenet-1m/lightning_data_search` (~64 MB chunks)                                                                                              |
| Cache         | `/cache/chunks` — **wipe before every cold ep0** (`rm -rf /cache/chunks/*`)                                                                                                          |
| Runner        | `scripts/bench/bench_s3_full_epochs.py`                                                                                                                                              |
| Python        | Studio default is **GIL on**. Nogil needs a free-threading conda/Studio env first — see [lightning-studio.md](lightning-studio.md). `PYTHON_GIL=0` / `-Xgil=0` alone are not enough. |
| Typical knobs | `--workers 48 --batch-size 256 --max-pre-download 4 --max-cache-size 200GB --epochs 1`                                                                                               |

```bash
cd /path/to/litData
rm -rf /cache/chunks/*

# Default Studio / laptop (GIL build):
PYTHONPATH=src python scripts/bench/bench_s3_full_epochs.py \
  --label my-run --epochs 1 --workers 48 --batch-size 256 \
  --max-pre-download 4 --max-cache-size 200GB --async-prefetch

# Only after switching to a free-threading Python (conda / Studio image), not the default:
# python -c 'import sys; print(sys.version, getattr(sys, "_is_gil_enabled", lambda: None)())'
# PYTHON_GIL=0 PYTHONPATH=src python -Xgil=0 scripts/bench/bench_s3_full_epochs.py ...
```

Report both:

- **`images_per_s`** (full epoch throughput)
- **`t_first_batch_s`** (time to first batch — cold-start / bandwidth fairness)

Rough Studio baselines (noisy; re-measure after big changes):

- Warm / decode-bound epoch ≈ **12k** img/s
- Cold ep0 async+obstore @ `max_pre=4` ≈ **10–11k** img/s, `t_first` ≈ **7–9s**
- Single-run deltas under ~5% are usually S3 noise — repeat before claiming a win

## Fair comparisons (mandatory)

1. **Same `max_pre_download` (and same async floor)** on every arm. Comparing async (floor→4) vs sync (user `max_pre=2`) is not fair.
2. **Wipe cache** between cold runs.
3. **Force boto3** when measuring the boto3 path: setting env in the parent is not enough if workers re-import. Prefer a `sitecustomize.py` on `PYTHONPATH` that patches `litdata.streaming.downloader._OBSTORE_AVAILABLE = False` (or equivalent) so **all** DataLoader workers see it. `PYTHONSTARTUP` alone does **not** run in worker processes.
4. **obstore is already a hard dependency** (`requirements.txt`). Do not add it again in release PRs; code still falls back via `RequirementCache("obstore")` if missing at runtime.

Related scripts under `scripts/bench/`:

- `bench_obstore_vs_boto3.py` — micro download compare
- `bench_obstore_chunksize_grid.py` / `bench_obstore_imagenet_grid.py` — `LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB` grids (default **8 MiB** was fine on Studio ImageNet)
- `bench_s3_cache_size.py` — `max_cache_size` sweep

## What not to ship without evidence

- **Two-phase cold `max_pre` (start at 1, promote to target):** intended for multi-worker first-chunk fairness. Studio ep0 showed mixed results (possible slight ips gain, worse `t_first` at cold=1). Keep **off** unless multi-run benches beat the steady `max_pre=4` baseline on both metrics. Details: [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md).
- **Raising async gather width past the budget cap:** peak disk ≈ `num_workers × max_pre × chunk_size`. Cap exists so tiny `max_cache_size` tests and small caches do not thrash; never allow the cap to land on `max_pre=1` (deadlock).

## CI vs Studio

Unit tests use tiny `max_cache_size` and `local:` remotes to exercise delete-when-processed. They will **not** tell you if obstore or async helps ImageNet. If CI hangs waiting for chunks under tiny budgets, fix download/delete gating first (see cache lifecycle doc) before trusting Studio numbers.
