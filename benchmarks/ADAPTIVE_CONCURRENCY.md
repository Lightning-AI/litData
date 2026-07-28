# Adaptive concurrency / look-ahead (design note)

Status: **Stage 0 + Stage 1 shipped** on `feature/raw-streaming-perf`. Stages 2–4 deferred.

## Division of labor

- **Clients own rate.** Botocore adaptive retries (and obstore’s retry layer) already token-bucket on 503/SlowDown. Litdata must not nest a second rate loop that fights them.
- **Litdata owns concurrency and look-ahead.** Permit counts and prefetch depth are the actuators we control; throttle *events* mostly never surface as exceptions (they look like latency).

A litdata controller that keys only on raised 429/503 will be nearly blind until downloaders expose throttle-retry counts (Downloader contract + conformance suite).

## Objective

**Throttle-avoiding max throughput** — maximize samples/s subject to not inducing prefix/NIC congestion. Easier to build and defend than pure max-throughput on a shared bucket.

## Stages

| Stage | What | Status |
|------:|------|--------|
| 0 | Bench protocol: `max(N batches, T seconds)`, ≥5 interleaved repeats, median+spread, append-only artifacts | Done (this work) |
| 1 | Static worker-aware concurrency: `clamp(budget // num_workers, floor, max)` from median file size + bandwidth | Done (this work) |
| 2 | Prefetch hit-rate controller (hit &lt;30% → halve, floor 0; hysteresis) | Pending |
| 3 | AIMD on concurrency (needs downloader throttle counts) | After contract |
| 4 | Full throughput-gradient control | Only if Stage 3 shows headroom |

## Stage 1 formula (shipped)

```
target_bytes = ASSUMED_AGGREGATE_BANDWIDTH_BPS × CONCURRENCY_PIPELINE_SECONDS
             = 100 MiB/s × 0.5 s ≈ 50 MiB
aggregate_budget = clamp(target_bytes // median_file_bytes, 32, 128)
effective_concurrency = min(max_concurrent_downloads,
                            max(8, aggregate_budget // num_workers))
```

Defaults when size unknown: median = 256 KiB. Semaphore uses this permit count (loop-keyed; cleared on fork/spawn like other runtime clients).

## Acceptance (future adaptive)

Beats **default** static everywhere; never loses by more than run-to-run noise; removes the w×p tuning matrix from the user’s cognitive load. “Beats tuned static” is the wrong bar — tuned static ties it at best per configuration.
