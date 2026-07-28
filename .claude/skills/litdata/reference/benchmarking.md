# Benchmarking streaming performance

Use this when measuring or comparing LitData streaming or optimization. Prefer real datasets and cold caches over microbenchmarks.

**Primary suite:** repo `benchmarks/` — ready-to-run CLI scripts (LitData + FFCV). Read the READMEs there before inventing a new harness.

Studio paths / free-threading → [lightning-studio.md](lightning-studio.md). Prefetch/cache math → [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md). Image JPEG vs PIL → [using-litdata.md](using-litdata.md).

## Repo layout (`benchmarks/`)

| Path                                      | Role                                                        |
| ----------------------------------------- | ----------------------------------------------------------- |
| `benchmarks/README.md`                    | Index: LitData vs FFCV folders                              |
| `benchmarks/litdata/`                     | Optimize + stream ImageNet with LitData                     |
| `benchmarks/litdata/README.md`            | CLI flags (`--write_mode jpeg`, `--quality`, `--resize`, …) |
| `benchmarks/litdata/optimize_imagenet.py` | `optimize` ImageNet (JPEG/PIL write modes)                  |
| `benchmarks/litdata/stream_imagenet.py`   | Epoch throughput for an optimized dataset                   |
| `benchmarks/stream_raw_imagenet.py`       | `StreamingRawDataset` baseline (no optimize)                |
| `benchmarks/bench_raw_before_vs_after.py` | Stage 0 A/B harness for raw cloud download changes          |
| `benchmarks/ADAPTIVE_CONCURRENCY.md`      | Adaptive concurrency / look-ahead design + Stage 1          |
| `benchmarks/ffcv/`                        | Convert / write / stream with FFCV for format comparison    |
| `benchmarks/ffcv/README.md`               | FFCV install + write/stream steps                           |

Start from `benchmarks/litdata/README.md` for LitData-only runs; use `benchmarks/ffcv/` when comparing formats. All scripts are CLI-based (`--help`).

## Raw streaming Stage 0 protocol

When measuring or claiming `StreamingRawDataset` cloud download wins (`bench_raw_before_vs_after.py` and friends):

1. **Window = `max(N batches, T seconds)`** — require **both** floors (default ≥300 batches **and** ≥30s). Not either/or.
2. **Warm** `max(1, num_workers × prefetch_factor)` batches before timing.
3. **Repeats + medians** — prefer interleaved A/B, `n≥5` for grids, report median + spread. Single-run digs are exploratory only.
4. **Append-only artifacts** — write `*.{sha}.{unix_ts}.json` (+ JSONL); never overwrite prior result files.
5. **Provenance** — record `before_sha` / `after_sha` from `git rev-parse` on each PYTHONPATH tree (not only the runner SHA in the filename). Refuse to publish without both.
6. **Trust hierarchy:** provenance-verified confirm cell (known `before_sha`/`after_sha`, protocol floors, n≥3) ≫ full-grid medians with null tree SHAs ≫ short-window or n=1 digs. Never cite a short-window n=1 against Stage 0 medians.

Design note / Stage 1 formula: `benchmarks/ADAPTIVE_CONCURRENCY.md`.

### Typical LitData flow

```bash
# 1) Optimize (prefer JPEG — see using-litdata.md)
python benchmarks/litdata/optimize_imagenet.py \
  --input_dir /path/to/raw/imagenet/train \
  --output_dir /path/to/optimized/imagenet \
  --resize --resize_size 256 \
  --write_mode jpeg \
  --quality 90 \
  --num_workers 32

# 2) Stream (wipe cache first for a cold epoch)
litdata cache clear
python benchmarks/litdata/stream_imagenet.py \
  --input_dir /path/to/optimized/imagenet \
  --batch_size 256 \
  --epochs 2
# Use --use_pil if you optimized with raw PIL images
```

Raw (unoptimized) baseline:

```bash
python benchmarks/stream_raw_imagenet.py --help
```

## What to measure

| Metric                                                 | Why                                |
| ------------------------------------------------------ | ---------------------------------- |
| Throughput (samples or images / sec) over a full epoch | Steady-state training rate         |
| Time to first batch (if available)                     | Cold-start / first-chunk fairness  |
| Epoch 2+                                               | Warm-cache / decode-bound behavior |

Always state: dataset, chunk/write mode, `num_workers`, `batch_size`, `max_pre_download`, `max_cache_size`, local vs remote.

## Fair comparison checklist

1. **Wipe the chunk cache** before every cold run (`litdata cache clear` or delete the cache dir).
2. **Same knobs on every arm** — workers, batch size, image size/quality, prefetch/cache when applicable. For remote streams, async prefetch is **on by default** and floors `max_pre_download` to 4 — pin `LITDATA_ASYNC_CHUNK_PREFETCH` and `max_pre_download` identically when comparing sync vs async or LitData vs another loader ([env-vars.md](env-vars.md)).
3. **Same machine, network, and dataset revision.**
4. **JPEG vs PIL** — LitData `--write_mode jpeg` vs PIL RAW are different formats; don’t mix when comparing loaders. FFCV has its own write modes (`benchmarks/ffcv/`).
5. **Repeat noisy cloud runs** before claiming small wins.
6. **CI unit tests ≠ production throughput.**

## Peak disk reminder

```
peak ≈ num_workers × max_pre_download × mean_chunk_size
```

Raise prefetch only if disk/`max_cache_size` can hold the peak.

## Interpreting results

- **Cold epoch** → remote/download bound?
- **Warm epoch** → decode / transform bound?
- Compare LitData optimized vs raw (`stream_raw_imagenet.py`) vs FFCV on the same source when possible.
- Root `README.md` Benchmarks section is the published narrative; re-run `benchmarks/` to reproduce on your hardware.

Dev-only sweeps (prefetch grids, download backends, …) may also live under `scripts/bench/` — still apply the fair-comparison rules above.
