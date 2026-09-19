# Indexed array windows: an adaptable reference

This example stores complete records once and reads a runtime selection of fields and contiguous frames. It is intentionally a reference reader, **not a public StreamingDataset class**. Use it as a starting point for video features, audio samples, trajectories, sensor arrays or other fixed-width time series.

```bash
PYTHONPATH=src python examples/temporal_arrays/reader.py \
  --write ./array-demo --input ./array-demo --cache ./array-cache
```

The writer requires a new directory. To read remotely, upload that directory into a new immutable prefix, publishing `index.json` last, then use `--input s3://bucket/prefix` or a Studio connection path. Supply ordinary cloud credentials; never put credentials into an index. Lightning R2 connections use LitData's resolver and managed credentials. Disable SDK HTTP debugging when handling authenticated requests.

## What is generic

- Record count, field names, numeric/bool dtypes, trailing shapes and frame counts are data-driven.
- A request is `(record, start, frames, fields)`. The reader has no customer-specific schema or random-number generator.
- Index metadata describes object names, byte offsets, shapes and dtypes. Shard packing can change while the request/decoder interface stays the same.
- `Downloader.adownload_bytes` owns transport. S3/R2 use native ranged reads when available and fork-safe. Local files are sliced directly. Other backends may download/cache a complete file through their synchronous fallback.
- Each returned array owns writable memory. Empty field selection performs no payload I/O; unknown fields and invalid windows fail before fetching.

```python
import asyncio
from reader import ArrayWindowReader

async def read_sample():
    reader = await ArrayWindowReader.open("./array-demo", "./array-cache")
    return await reader.read(record=1, start=7, frames=96, fields=["features"])

sample = asyncio.run(read_sample())
```

Create and use a reader within one worker process and one event loop. Do not construct a native object-store runtime in a parent before forking workers. The concurrency limit belongs to one reader; aggregate load multiplies across ranks and workers. Cancelling a native request propagates cancellation, but bytes already transferred cannot be undone. Cancelling a thread/SDK fallback cannot stop its running I/O. Do not hedge that fallback or delete/reuse its cache files while it may still be writing.

## Adaptation guide for developers and AI agents

1. **Replace `write_records` inputs.** Yield dictionaries of arrays. Every field must have the same leading frame axis within a record. Field names need not be filenames. Preserve a stable mapping from application entity IDs to record positions.
2. **Choose field grouping and shard layout.** This writer stores each complete field contiguously, so a request reads exactly the selected window bytes. Co-accessed small fields can be grouped to reduce requests, but then projection overfetch must be measured. A field larger than the shard target remains intact and may produce an oversized shard.
3. **Replace index storage at scale.** The JSON index is deliberately small and held in memory. Use a versioned, partitioned index for millions of records; preserve the planner's offset/shape/dtype contract. The raw shards are not standard optimized LitData chunks.
4. **Add a sampler explicitly.** Choose records/windows according to the training policy. Uniform records, uniform frames and chunk-local sampling are different distributions. Allocate deterministic global request IDs across ranks/workers before claiming reproducible distributed sampling.
5. **Add temporal joins explicitly.** Equal record IDs do not imply equal frame numbers. Validate snapshot versions, entity alignment, timestamps/frame rates and missing-interval policy before combining tables. This is a future integration point for the proposed TablesStreamingDataset, not an arbitrary SQL join implementation.
6. **Add resume only with a consumed-request contract.** Checkpoint consumed request positions, snapshot identity and sampling configuration. Prefetched requests are not necessarily consumed. This example does not implement checkpoint/resume or elastic world-size changes.
7. **Extend codecs independently.** Compressed video, variable-length strings, ragged values and whole-object compression/encryption need offsets or independently decodable blocks. A frame-byte stride is insufficient for those formats.

Indexes and object prefixes are immutable for the lifetime of a reader/cache. The example does not implement atomic multi-writer publication, checksums, cache eviction, recovery of partially written directories, or production metadata validation. Do not reuse a failed output directory or overwrite an active prefix.

## Validation and performance

`tests/streaming/test_temporal_array_example.py` verifies values/dtypes, boundaries, multiple shapes/records, projections and exact byte budgets, writable ownership, bounded native-style concurrency, failure draining and incomplete-write visibility. Downloader tests cover native cancellation, SDK fallback, fork guards, truncated responses and response cleanup.

No throughput claim is made for this example. It illustrates the reusable pieces from a customer investigation, with a different generic layout. Benchmark your actual field/window distribution and a real model before adopting it; distinguish cold remote delivery from warm local/cache measurements.
