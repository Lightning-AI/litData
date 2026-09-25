# Adaptive concurrency / look-ahead (design note)

Status: **Stage 0 + Stage 1 shipped** on `feature/raw-streaming-perf`. Stages 2–4 deferred.

## Division of labor

- **Clients own rate.** Botocore adaptive retries (and obstore’s retry layer) already token-bucket on 503/SlowDown. Litdata must not nest a second rate loop that fights them.
- **Litdata owns concurrency and look-ahead.** Permit counts and prefetch depth are the actuators we control; throttle *events* mostly never surface as exceptions (they look like latency).

A litdata controller that keys only on raised 429/503 will be nearly blind until downloaders expose throttle-retry counts (Downloader contract + conformance suite).

## Objective

**Throttle-avoiding max throughput** — maximize samples/s subject to not inducing prefix/NIC congestion. Easier to build and defend than pure max-throughput on a shared bucket.

## Stages

| Stage | What                                                                                                          | Status                         |
| ----: | ------------------------------------------------------------------------------------------------------------- | ------------------------------ |
|     0 | Bench protocol: `max(N batches, T seconds)`, ≥5 interleaved repeats, median+spread, append-only artifacts     | Done (this work)               |
|     1 | Static worker-aware concurrency: size-aware budget (bandwidth + size-gated Little’s-law) split across workers | Done (this work)               |
|     2 | Prefetch hit-rate controller (hit \<30% → halve, floor 0; hysteresis)                                         | Pending                        |
|     3 | AIMD on concurrency (needs downloader throttle counts)                                                        | After contract                 |
|     4 | Full throughput-gradient control                                                                              | Only if Stage 3 shows headroom |

## Stage 1 justification (honest)

The original Stage 1 premise — that **w=24 with 64 permits/worker (1536 aggregate) stampeded** and needed a hard clamp — was **falsified** by the batch-timeout dig. Post-timeout, w=24 with 64 permits was within/near run-to-run noise of lower concurrency.

What remains:

- Prefer a size-aware default over asking users to tune `max_concurrent_downloads` per `num_workers`
- Bound pathological aggregates at high `num_workers` without crushing mid-w cells

A prior full-grid A/B (pre-Stage-1 vs always-on clamp) showed high-w gains and a w=8 p0 −23% cell. That mid-w drop is treated as **likely A/A noise** after Little’s-law widening (both sides ~60–64 permits at w=8), **not** as proof to gate the clamp to `w≥16`. High-w “+53%” headlines require a provenance-verified confirmation cell (see below).

## Stage 1 formula (shipped)

`max_concurrent_downloads=None` (default) → adaptive:

```
target_bytes = 100 MiB/s × 0.5 s ≈ 50 MiB
bandwidth_model = target_bytes // median_file_bytes
latency_model   = 6000 × 0.040 ≈ 240   if median < 1 MiB else 0
aggregate_budget = clamp(max(bandwidth_model, latency_model), 32, 512)
effective = min(budget, 128)                         # num_workers ≤ 1
          = max(8, aggregate_budget // num_workers)  # num_workers > 1
```

Large medians (1/10/100 MiB) stay **bandwidth-bounded** — Little’s-law must not pin the budget at 240 (multi-GB in flight).

Explicit `max_concurrent_downloads=int` → **exactly** that many permits (no silent clamp).

Defaults when size unknown: median = 256 KiB. Permit count computed once per process; cleared on fork/spawn.

## Confirmation cell (provenance) — done

Bench harness records `before_sha` / `after_sha` from `git rev-parse` on each PYTHONPATH tree (not only the runner SHA in filenames).

Confirm @ `ba9da13`: interleaved n=3, `max(≥300 batches, ≥30s)`, **w=24 p=0**.

|         field | value                                             |
| ------------: | ------------------------------------------------- |
|    before_sha | `52dba61` (post-`f70f785`, pre Stage 1; fixed 64) |
|     after_sha | `ba9da13`                                         |
| before median | **3816** ips (spread 30%)                         |
|  after median | **6049** ips (spread 21%)                         |

**Verdict: (a)** — before ≈3.7k (not ~5.5k wrong-tree). Frame high-w as robustness; do **not** headline unverifiable +53%. Artifact: `benchmarks/results/raw_before_vs_after.ba9da13.1785268543.json`.

## Stateful Recovery State Machine & Deadband Control

The adaptive controller incorporates a 3-state feedback controller to prevent budget jitter:

| State         | Latency Condition                                                                        | Action                                                                                                 |
| :------------ | :--------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| **Healthy**   | (L\_{\\text{obs}} \\le L\_{\\text{target}}) (40 ms)                                      | Recover backoff factor gradually: (f \\leftarrow \\min(1.0, f + \\alpha\_{\\text{recovery}}(1.0 - f))) |
| **Deadband**  | (L\_{\\text{target}} < L\_{\\text{obs}} \\le 1.1 \\times L\_{\\text{target}}) (40–44 ms) | Hold current backoff factor constant (no recovery, no extra backoff)                                   |
| **Congested** | (L\_{\\text{obs}} > 1.1 \\times L\_{\\text{target}}) (>44 ms)                            | Immediate backoff: (f \\leftarrow \\min(f, L\_{\\text{target}} / L\_{\\text{obs}}))                    |

## Environment Variable Configuration Reference

All controller bounds and operational knobs are configurable via validated environment variables:

| Environment Variable                         | Default                | Range / Constraints                     | Description                                                 |
| :------------------------------------------- | :--------------------- | :-------------------------------------- | :---------------------------------------------------------- |
| `LITDATA_ASSUMED_BANDWIDTH_BPS`              | `104857600` (100 MB/s) | (> 0)                                   | Default aggregate network bandwidth fallback                |
| `LITDATA_ASSUMED_REQUEST_LATENCY_S`          | `0.040` (40 ms)        | (> 0.0)                                 | Target baseline GET request latency                         |
| `LITDATA_ASSUMED_REQUEST_RATE`               | `6000.0` req/s         | (\\ge 0.0)                              | Baseline request rate for Little's law model                |
| `LITDATA_SINGLE_PROCESS_CONCURRENCY_CAP`     | `128`                  | (\\ge 1)                                | Maximum single-process permit cap                           |
| `LITDATA_AGGREGATE_CONCURRENCY_BUDGET_CAP`   | `512`                  | (\\ge 1)                                | Maximum aggregate multi-worker permit cap                   |
| `LITDATA_AGGREGATE_CONCURRENCY_BUDGET_FLOOR` | `32`                   | (1 \\le \\text{floor} \\le \\text{cap}) | Adaptive dynamic concurrency budget floor                   |
| `LITDATA_BACKOFF_RECOVERY_ALPHA`             | `0.1`                  | (0.0 < \\alpha \\le 1.0)                | Recovery EMA factor for healthy state                       |
| `LITDATA_MIN_EMPIRICAL_SAMPLES`              | `5`                    | (\\ge 1)                                | Minimum observations required before applying empirical EMA |
| `LITDATA_PERMIT_REFRESH_INTERVAL`            | `10`                   | (\\ge 1)                                | Iteration interval between dynamic permit recalculations    |

## Adaptive Concurrency Stress Benchmark

To run the stress benchmark comparing aggregate permits, per-worker permits, BPS EMA, latency EMA, and throughput across worker counts:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_raw_adaptive_concurrency.py
```

## Acceptance (future adaptive)

Beats **default** static everywhere; never loses by more than run-to-run noise; removes the w×p tuning matrix from the user’s cognitive load. “Beats tuned static” is the wrong bar — tuned static ties it at best per configuration.
