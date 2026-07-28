# Adaptive concurrency / look-ahead (design note)

Status: **Stage 0 + Stage 1 shipped** on `feature/raw-streaming-perf`. Stages 2–4 deferred.

## Division of labor

- **Clients own rate.** Botocore adaptive retries (and obstore’s retry layer) already token-bucket on 503/SlowDown. Litdata must not nest a second rate loop that fights them.
- **Litdata owns concurrency and look-ahead.** Permit counts and prefetch depth are the actuators we control; throttle *events* mostly never surface as exceptions (they look like latency).

A litdata controller that keys only on raised 429/503 will be nearly blind until downloaders expose throttle-retry counts (Downloader contract + conformance suite).

## Objective

**Throttle-avoiding max throughput** — maximize samples/s subject to not inducing prefix/NIC congestion. Easier to build and defend than pure max-throughput on a shared bucket.

## Stages

| Stage | What                                                                                                      | Status                         |
| ----: | --------------------------------------------------------------------------------------------------------- | ------------------------------ |
|     0 | Bench protocol: `max(N batches, T seconds)`, ≥5 interleaved repeats, median+spread, append-only artifacts | Done (this work)               |
|     1 | Static worker-aware concurrency: adaptive clamp **gated to `num_workers ≥ 16`**                           | Done (this work)               |
|     2 | Prefetch hit-rate controller (hit \<30% → halve, floor 0; hysteresis)                                     | Pending                        |
|     3 | AIMD on concurrency (needs downloader throttle counts)                                                    | After contract                 |
|     4 | Full throughput-gradient control                                                                          | Only if Stage 3 shows headroom |

## Stage 1 justification (honest)

The original Stage 1 premise — that **w=24 with 64 permits/worker (1536 aggregate) stampeded** and needed a hard clamp — was **falsified** by the batch-timeout dig. Post-timeout, w=24 with 64 permits was within/near run-to-run noise of lower concurrency.

What remains as justification is thinner:

- S3 prefix / neighbor pressure and request cost at pathological aggregates (N×64 → 1536+)
- Prefer a size-aware default over asking users to tune `max_concurrent_downloads` per `num_workers`

## Stage 1 A/B decision (pre-Stage-1 `52dba61` vs Stage 1 HEAD, n=5)

Protocol: `max(≥300 batches, ≥30s)`, interleaved, ImageNet val ~100 KiB, append-only artifacts under `benchmarks/results/`.

| w   | p   | before | after (always-on clamp) | Δ%            |
| --- | --- | -----: | ----------------------: | ------------- |
| 2   | 0   |   1053 |                    1224 | +16%          |
| 2   | 16  |   1146 |                    1154 | +1%           |
| 4   | 0   |   2029 |                    2075 | +2%           |
| 4   | 16  |   2215 |                    2161 | −2%           |
| 8   | 0   |   4592 |                    3545 | **−23% FAIL** |
| 8   | 16  |   4142 |                    4044 | −2%           |
| 16  | 0   |   3637 |                    5269 | +45%          |
| 16  | 16  |   3848 |                    5336 | +39%          |
| 24  | 0   |   3747 |                    5742 | +53%          |
| 24  | 16  |   3595 |                    5619 | +56%          |

Acceptance: no winning cell loses by more than measured spread. **w=8 p0 failed.** High-w wins look real (tight after spreads at w24).

**Fix shipped:** gate adaptive clamp to `num_workers >= 16`. Below the gate, `None` → historical 64 permits/worker (same as pre-Stage-1). High-w keep the size-aware split that delivered +40–55%.

## Stage 1 formula (shipped)

`max_concurrent_downloads=None` (default):

```
if num_workers < 16:
    effective = 64                          # historical default; no mid-w clamp
else:
    target_bytes = 100 MiB/s × 0.5 s ≈ 50 MiB
    bandwidth_model = target_bytes // median_file_bytes
    latency_model   ≈ 6000 req/s × 0.040 s ≈ 240
    aggregate_budget = clamp(max(bandwidth_model, latency_model), 32, 512)
    effective = max(8, aggregate_budget // num_workers)
```

**Important:** per-worker floor of 8 means realized aggregate is
`max(budget, 8 × num_workers)` when the clamp is active.

Explicit `max_concurrent_downloads=int` → **exactly** that many permits (no silent clamp). Tuned users who pass `64` keep `64`.

Defaults when size unknown: median = 256 KiB. Semaphore permit count is computed once per process; cleared on fork/spawn.

## Acceptance (future adaptive)

Beats **default** static everywhere; never loses by more than run-to-run noise; removes the w×p tuning matrix from the user’s cognitive load. “Beats tuned static” is the wrong bar — tuned static ties it at best per configuration.
