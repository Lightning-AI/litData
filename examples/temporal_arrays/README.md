# Built-in temporal array windows

Use `optimize` to store complete records once and `StreamingDataset.read_window` to select fields and frame windows at training time. The storage, indexing, grouping, range reads and decoding live in LitData. This example is only application code.

```python
from litdata import optimize, StreamingDataset, TemporalArrayLoader

# load_record returns a dictionary of NumPy arrays or tensors, each shaped (frames, ...).
# Frame counts can vary between records. Names, types and trailing shapes stay fixed.
optimize(
    load_record,
    inputs=record_paths,
    output_dir="./tracks",
    chunk_bytes="128MB",
    num_workers=4,
    item_loader=TemporalArrayLoader(field_groups=[["clock", "valid", "pose"]]),
)

tracks = StreamingDataset("./tracks", item_loader=TemporalArrayLoader())
sample = tracks.read_window(12, start=37, frames=96, fields=["features", "pose"])
lengths = tracks.frame_counts
```

Put `optimize` inside an `if __name__ == "__main__":` guard in runnable scripts. Cloud URLs and Studio connection paths use the normal LitData resolver and storage options. Ordinary `tracks[12]` and dataset iteration return complete records. `StreamingDataLoader` retains its full-record iteration and checkpoint behavior.

## Small customer adapter

Keep the customer's choice of records/windows in their code. Their storage implementation becomes a call to LitData:

```python
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler

class TrainingWindows(Dataset):
    def __init__(self, uri, frames=64):
        self.records = StreamingDataset(uri, item_loader=TemporalArrayLoader())
        self.lengths = self.records.frame_counts
        self.frames = frames
        self.eligible = [i for i, n in enumerate(self.lengths) if n >= frames]

    def __len__(self):
        return len(self.eligible)

    def __getitem__(self, index):
        record = self.eligible[index]
        start = int(np.random.randint(self.lengths[record] - self.frames + 1))
        return self.records.read_window(record, start, self.frames)

windows = TrainingWindows("./tracks")
loader = DataLoader(
    windows, batch_size=16, num_workers=8, pin_memory=True,
    sampler=RandomSampler(windows, replacement=True, num_samples=16000),
)
```

This example samples eligible records uniformly with replacement. If the application needs weighted tracks, uniform frames, multiple tables or other policies, keep its sampler and call `read_window` with the chosen request. Initialize each dataset in its worker process before issuing concurrent reads. `await tracks.aread_window(...)` supports concurrent async callers; synchronous POSIX temporal reads execute in the caller's thread, while other synchronous readers use LitData's process-local loop. No private imports or separate storage clients are needed.

## Request count and tradeoffs

`TemporalArrayLoader` writes ordinary chunk framing with its own versioned item layout. Compact frame counts and item offsets live in each chunk's index entry, so even the first window reads only its selected field groups: **one range per group, no per-window metadata GETs**. Unlisted fields each get a separate group. Selecting one field in a group fetches that group's window, so narrow projections can overfetch. Returned NumPy arrays and tensors preserve type/dtype and own writable storage.

Choose packing from field sizes and the fields usually read together, rather than total dataset size alone. All these layouts use the same read API:

```python
# Full-field windows: all fields for each frame are adjacent -> one range.
all_fields = ["features", "clock", "valid", "pose"]
all_together = TemporalArrayLoader(field_groups=[all_fields])

# Mixed projections: large features separate; small co-accessed fields together.
grouped = TemporalArrayLoader(field_groups=[["clock", "valid", "pose"]])

# Narrow projections: each field separate -> one range per requested field.
separate = TemporalArrayLoader()
```

For separate groups, a track stores `[all feature frames][all clock/valid/pose frames]`, so a full-field window needs two disjoint ranges. A single group instead stores `[frame 0: all fields][frame 1: all fields]...`; the same window is one contiguous range. The full-field payload is approximately unchanged, apart from alignment padding. Fewer requests do not guarantee a proportional speedup. Selecting only `clock` from the single group still fetches every field in that window. Changing packing requires rewriting records; the persisted schema lets readers discover the layout automatically.

`chunk_bytes` controls how complete records are packed into objects, independently of field grouping and window length. Direct range reads do not fetch the whole chunk, so larger chunks do not inherently reduce bytes per window. Dataset size also affects index memory and cache behavior. This temporal format currently requires uncompressed data; setting chunk compression is not a supported way to compress partial windows.

Existing uncompressed default `optimize` output also supports `read_window` on flat dictionaries of numeric/bool array fields. This path reads small chunk/array headers on demand, retains metadata for at most 256 records, and uses one range per selected field. Use `TemporalArrayLoader` when request count matters.

S3/R2 use native range reads when available and fork-safe, with an SDK fallback. Local reads use byte slices; other providers may download whole chunks. `max_concurrent_reads=8` bounds field reads per call; the application bounds concurrent windows across workers/ranks. Cancellation drains native tasks; a running SDK/thread fallback can finish I/O afterward.

## Contract and limits

- Positive frame counts and valid contiguous axis-0 windows; no padding or silent truncation.
- Named, fixed-width numeric/bool arrays with a common frame axis. Strings, ragged data and encoded video need other loaders.
- Uncompressed, unencrypted chunks. Custom serializers are not supported for window reads.
- Explicit reads do not apply dataset transforms or advance iteration/checkpoint positions.
- Window sampling, table alignment and consumed-request checkpoint state remain application responsibilities. This API does not claim exact random-window resume or automatic temporal joins.
- Keep dataset prefixes immutable while reading. Metadata is stored in the existing index; very large record counts need a future partitioned index.

## Run and test

```bash
PYTHONPATH=src python examples/temporal_arrays/reader.py \
  --write ./array-demo --input ./array-demo --cache ./array-cache
PYTHONPATH=src pytest tests/streaming/test_temporal_array_example.py
```

Tests exercise the real optimize/StreamingDataset workflow, full-record compatibility, frame/field correctness, cold and warm byte/request counts, schema validation, split mapping, concurrency, cancellation and failures. Performance must be measured on the application's own field/window distribution; the generic API does not inherit a previous prototype's throughput results.

## Local and parallel filesystems

Temporal window reads reuse the POSIX-fast path on local disks and mounted parallel filesystems such as NFS/VAST. LitData maps immutable chunks in place, retains a bounded set of mappings, and copies only selected field windows into writable outputs. A mapping does not load the whole chunk or require the dataset to fit in RAM; the OS manages resident pages. Source chunks must remain unchanged while mapped.

Where supported and permitted by the existing POSIX memory policy, selected page ranges receive a prefetch hint. Windows do not request whole-chunk prefetch. Concurrent calls hold mapping leases until decoding completes, so eviction or async cancellation cannot unmap an active read. Async reads offload page faults and decoding to a thread. `LITDATA_POSIX_FAST=0` retains the buffered fallback. Request sampling and rank/worker partitioning remain the application's responsibility.
