# Adaptive concurrency / look-ahead (design note)

Status: **Stage 0 + Stage 1 shipped** on `feature/raw-streaming-perf`. Stages 2–4 deferred.

## Division of labor

- **Clients own rate.** Botocore adaptive retries (and obstore’s retry layer) already token-bucket on 503/SlowDown. Litdata must not nest a second rate loop that fights them.
- **Litdata owns concurrency and look-ahead.** Permit counts and prefetch depth are the actuators we control; throttle *events* mostly never surface as exceptions (they look like latency).

A litdata controller that keys only on raised 429/503 will be nearly blind until downloaders expose throttle-retry counts (Downloader contract + conformance suite).

## Objective

**Throttle-avoiding max throughput** — maximize samples/s subject to not inducing prefix/NIC congestion. Easier to build and defend than pure max-throughput on a shared bucket.

## Stages

| Stage | What                                                                                                                  | Status                         |
| ----: | --------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
|     0 | Bench protocol: `max(N batches, T seconds)`, ≥5 interleaved repeats, median+spread, append-only artifacts             | Done (this work)               |
|     1 | Static worker-aware concurrency: `max(floor, budget // num_workers)` from median file size (bandwidth + Little’s-law) | Done (this work)               |
|     2 | Prefetch hit-rate controller (hit \<30% → halve, floor 0; hysteresis)                                                 | Pending                        |
|     3 | AIMD on concurrency (needs downloader throttle counts)                                                                | After contract                 |
|     4 | Full throughput-gradient control                                                                                      | Only if Stage 3 shows headroom |

## Stage 1 justification (honest)

The original Stage 1 premise — that **w=24 with 64 permits/worker (1536 aggregate) stampeded** and needed a hard clamp — was **falsified** by the batch-timeout dig. Post-timeout, w=24 with 64 permits was within/near run-to-run noise of lower concurrency.

What remains as justification is thinner and **unmeasured until the Stage 1 A/B**:

- S3 prefix / neighbor pressure and request cost at pathological aggregates (N×64 → 1536+)
- Prefer a size-aware default over asking users to tune `max_concurrent_downloads` per `num_workers`

The A/B (pre-Stage-1 HEAD vs Stage-1 HEAD) is what decides whether the clamp earns its keep or needs loosening.

## Stage 1 formula (shipped)

`max_concurrent_downloads=None` (default) → adaptive:

```
target_bytes = ASSUMED_AGGREGATE_BANDWIDTH_BPS × CONCURRENCY_PIPELINE_SECONDS
             = 100 MiB/s × 0.5 s ≈ 50 MiB
bandwidth_model = target_bytes // median_file_bytes
latency_model   = ASSUMED_REQUEST_RATE × ASSUMED_REQUEST_LATENCY_S
                ≈ 6000 req/s × 0.040 s ≈ 240
aggregate_budget = clamp(max(bandwidth_model, latency_model), 32, 512)
effective_concurrency = max(8, aggregate_budget // num_workers)   # n>1
                      = aggregate_budget                            # n≤1
```

**Important:** per-worker floor of 8 means realized aggregate is
`max(budget, 8 × num_workers)` — high worker counts can exceed the budget via the floor.

ImageNet ~100 KiB → bandwidth ≈ 524 → capped at 512 → at w=8 permits=64 (aggregate 512), not the old 128-cap path that compressed mid-w cells.

Explicit `max_concurrent_downloads=int` → **exactly** that many permits (no silent clamp). Tuned users who pass `64` keep `64`.

Defaults when size unknown: median = 256 KiB. Semaphore uses this permit count (computed once per process; cleared on fork/spawn like other runtime clients).

## Acceptance (future adaptive)

Beats **default** static everywhere; never loses by more than run-to-run noise; removes the w×p tuning matrix from the user’s cognitive load. “Beats tuned static” is the wrong bar — tuned static ties it at best per configuration.
