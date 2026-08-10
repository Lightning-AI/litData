# MultiJoinStreamingDataset

## Product and implementation plan

Status: proposed design
Audience: LitData users, ML infrastructure engineers, LitData maintainers
Scope: independently versioned tables that describe the same training entity and must be joined while streaming

## Executive summary

`MultiJoinStreamingDataset` is intended for datasets in which:

- Several tables or modalities describe the same logical entity.
- One or more large tables are stable and expensive to rebuild.
- Smaller tables change more often, including schema-level changes.
- Training still requires the throughput, distributed sampling, shuffle, prefetch, cache locality, and checkpoint-resume behavior of an optimized LitData dataset.

The core design is **co-partitioned column families with an atomic version manifest**.

The central constraint cannot be removed: if tables are not joined before training, they must remain aligned by key in the storage layout. Otherwise, every sampled key requires unrelated random reads from every table, turning prefetch into a distributed query-planning problem.

The work is split into two deliberately different phases:

- **Phase V1 — strict aligned LitData chunks.** Every table is a normal optimized LitData dataset with exactly the same logical chunk boundaries, item counts, and key order. `MultiJoinStreamingDataset` validates the layout and reuses `ParallelStreamingDataset`. This minimizes new read-path code and preserves existing LitData behavior.
- **Phase V2 — logical partitions with independently compacted column families.** Logical sampling buckets remain aligned, but physical files no longer need a one-to-one correspondence. Small tables can pack many logical partitions into larger Parquet or LitData objects, while large tables keep appropriately sized binary chunks. V2 introduces one shared sampler and a coordinated multi-table reader.

The public read API is designed once and remains stable across both phases. V2 is primarily an internal storage and reader improvement.

```mermaid
flowchart LR
  canonicalKeys[Canonical ordered entity keys]
  canonicalKeys --> logicalBuckets[Shared logical sampling buckets]
  logicalBuckets --> t0[table_0 version]
  logicalBuckets --> t1[table_1 version]
  logicalBuckets --> t2[table_2 version]
  activeManifest[Atomic join manifest] --> t0
  activeManifest --> t1
  activeManifest --> t2
  t0 --> joinedReader[MultiJoinStreamingDataset]
  t1 --> joinedReader
  t2 --> joinedReader
  joinedReader --> training[Training batches]
```

## 1. Problem

The current high-throughput LitData workflow materializes the complete training sample before or during `optimize()`. This is efficient at training time because one sample is stored in one streaming layout, but it couples the lifecycle of every source table:

1. A small table changes.
2. The joined sample schema changes.
3. The complete optimized dataset is rebuilt.
4. Large unchanged tables are read, serialized, and uploaded again.

`dataset_update` is complementary but does not solve the schema-change case. It is useful when a bounded set of existing samples can be replaced under a compatible optimized schema. If every entity in a table gains or loses columns, that table must be re-optimized.

The target behavior is:

1. Optimize each logical table independently.
2. Re-optimize only the table whose values or schema changed.
3. Atomically publish a new combination of table versions.
4. Stream the selected versions as one training sample.
5. Preserve deterministic shuffle, DDP sharding, DataLoader workers, prefetch, caching, and exact resume.

## 2. Why alignment is required

LitData chunks are both:

- Physical I/O units downloaded and cached by `BinaryReader`.
- Sampling buckets assigned to ranks and workers.

The current sampling path has two stages:

1. Chunks are assigned and shuffled across ranks and DataLoader workers.
2. Item positions are shuffled within each selected chunk.

If separately optimized tables have different chunk boundaries or key order, the same seed does not make them align.

Without alignment, one sampled key needs a location in every table:

```text
entity-123:
  table_0 -> chunk 42, offset 731
  table_1 -> chunk 3,  offset 18
  table_2 -> chunk 91, offset 204
```

Successive shuffled keys are likely to reference unrelated object combinations. The system then needs to:

1. Resolve every key in every table.
2. Plan downloads across unrelated chunks.
3. Deduplicate and prioritize those downloads.
4. Coordinate cache admission and eviction.
5. Preserve distributed sampling and exact resume.
6. Join the decoded values.

That is closer to implementing a distributed feature store or query engine than extending a streaming dataset.

With alignment, one location is valid for every table:

```text
entity-123 -> logical bucket 42, position 731
```

The reader can fetch bucket group `42`, apply one item permutation, and read position `731` from every table.

## 3. Terminology

- **Entity key:** Stable identifier used to correlate tables, such as `entity_id`.
- **Canonical key order:** Immutable ordered sequence of entity keys for one layout.
- **Logical item:** One training entity in the canonical order.
- **Table sample:** The value contributed by one table for one logical item.
- **Logical bucket:** Contiguous range of canonical item positions used as one sampling bucket.
- **Physical chunk or object:** File downloaded from local or object storage.
- **Table version:** Immutable physical representation of one table under one canonical layout.
- **Snapshot:** Immutable manifest selecting exactly one version of every active table.
- **Active manifest:** Small `join.json` document pointing to the active snapshot.
- **Layout ID:** Content-derived identifier for the canonical key order and logical bucket boundaries.

## 4. Cardinality model

Phase V1 uses a strict **one logical table sample per entity key** contract.

This does not require every source table to contain one physical row per key. A table sample may contain:

- One scalar or dictionary.
- A tensor.
- A list of rows.
- A variable-length NumPy array.
- An Arrow-like or serialized tabular bundle.
- An explicit empty collection when that table has no source rows for the entity.

For example, a nested events table may contribute a variable-length bundle containing every event row for one parent entity. The logical join remains one-to-one even though the source table is one-to-many.

V1 rejects:

- A missing logical table sample.
- Duplicate logical entity keys.
- Filtering that removes an entity.
- A generator that emits zero or multiple logical samples for one input key.

An empty table contribution must be represented as an explicit empty value. This keeps every table positionally aligned.

V2 can add a native offset-based representation for zero-to-many rows per key, but the canonical entity key and logical bucket alignment remain mandatory.

## 5. Design goals

### Correctness

- Never silently combine values from different entity keys.
- Validate alignment before a table version can become active.
- Pin every training run to an immutable snapshot.
- Resume only against the same snapshot and sampling configuration.
- Fail closed when metadata is missing, incompatible, or ambiguous.

### Performance

- Keep the iterative training path sequential and prefetchable.
- Avoid per-sample key lookups during normal training.
- Reuse existing LitData downloader, cache, serializers, item loaders, and shuffle behavior in V1.
- Avoid reading or rewriting unchanged table versions.
- Preserve direct object-store access for S3, GCS, R2, and Lightning Storage paths.

### Operability

- Publish a table version immutably.
- Publish a new snapshot atomically only after validation.
- Allow instant rollback to a previous snapshot.
- Keep failed or incomplete versions unreachable.
- Expose clear diagnostics for alignment and manifest failures.

### API quality

- Write path feels like `optimize`: independent `joint_optimize(...)` calls, including multi-node jobs.
- No long-lived writer context manager that must wrap every table build.
- No required `key_name` / schema-field argument; keys come from `key_fn` like existing `optimize`.
- A simple root-path read API.
- Named table outputs rather than positional tuples.
- One shared set of sampling options.
- No need for users to configure `align_chunking`, `reorder_files`, child seeds, or child cache directories.
- The same user-facing read API in V1 and V2.

## 6. Non-goals

The following are not goals of V1:

- Arbitrary SQL joins at training time.
- Joining independently optimized legacy datasets without a one-time aligned rebuild.
- Different key populations per required table.
- Independent per-table shuffle.
- Byte-based chunking for aligned table versions.
- Dynamic filtering in one table.
- Many-to-many joins.
- Replacing `dataset_update`.
- Replacing a feature store or query engine.
- Automatically changing the canonical key population without rebuilding all table versions.

The full baked dataset remains the recommended default when all tables normally change together or when maximum simplicity and minimum object count are more important than independent table versioning.

## 7. Stable public API

The API below is the target public contract. Some advanced storage arguments become effective only in V2, but normal training code does not change.

The write path mirrors existing `optimize`: each table is an independent function call that can run as a local or multi-node job. There is no `MultiJoinWriter` context manager and no `key_name` argument.

### 7.1 Build tables with `joint_optimize`

```python
from litdata import joint_optimize

# Independent jobs — each call can use num_workers / num_nodes like optimize().
# The first successful call into an empty join root creates the shared layout
# from the ordered inputs + key_fn. Later calls must align to that layout.

joint_optimize(
    fn=build_table_0,
    inputs=inputs_0,
    output_dir="s3://bucket/multi-join-dataset",
    table="table_0",
    chunk_size=2048,
    key_fn=lambda sample: sample["id"],
    num_workers=32,
    num_nodes=8,
    compression="zstd",
)

joint_optimize(
    fn=build_table_1,
    inputs=inputs_1,
    output_dir="s3://bucket/multi-join-dataset",
    table="table_1",
    chunk_size=2048,
    key_fn=lambda sample: sample["id"],
    num_workers=8,
)
```

`joint_optimize` accepts the same core knobs as `optimize` (`fn`, `inputs`, `output_dir`, `chunk_size`, `key_fn`, `num_workers`, `num_nodes`, `compression`, `encryption`, `storage_options`, …) plus join-specific arguments:

- `table`: logical table name under the join root (for example `"table_0"`).
- `version`: optional immutable table-version name; defaults to an auto-generated unique version.
- `snapshot`: optional human-readable snapshot name to publish on success.
- `expected_snapshot`: optional concurrency guard against the currently active snapshot.

Semantics:

- Each call is a complete, distributed-capable job. Jobs do not share a process-local writer session.
- `inputs` order is the stream / layout order. Shuffle or otherwise curate inputs before calling when source order has structure.
- `key_fn` works like existing `optimize(..., key_fn=...)`: it extracts an opaque key from each sample for the alignment sidecar. There is no separate `key_name` schema field.
- `chunk_size` is required (item count). `chunk_bytes` is rejected.
- The first table that successfully initializes an empty join root writes the shared layout (ordered keys, bucket boundaries, digests).
- Later `joint_optimize` calls for other tables must reproduce the same length, bucket sizes, and ordered-key digests.
- On success, the job validates alignment and atomically publishes a new snapshot that includes this table version, keeping previously published sibling table versions from the prior active snapshot.
- Failed jobs leave the previous active snapshot unchanged.
- Rebuilding one table later is just another `joint_optimize` into a new `version=` under the same `table=`.

### 7.2 Same ordered inputs across tables

Every table must be optimized from inputs that follow the same entity order. Typical patterns:

```python
# inputs_0 / inputs_1 are already aligned to the same entity order
joint_optimize(fn=build_table_0, inputs=inputs_0, output_dir=root, table="table_0", chunk_size=2048, key_fn=get_id)
joint_optimize(fn=build_table_1, inputs=inputs_1, output_dir=root, table="table_1", chunk_size=2048, key_fn=get_id)
```

Contract:

- Do not silently reorder inputs inside `joint_optimize`.
- A key mismatch reports table, global index, expected key, and actual key.
- Users may prepare inputs by joining against the published layout key store and sorting by `global_index`.
- Parallel first-time builds of multiple tables into an empty root are unsafe without a shared layout. Create the layout with one `joint_optimize` (or a small driver table) first, then launch sibling table jobs.

### 7.3 Re-optimize one table after a schema change

```python
from litdata import joint_optimize

joint_optimize(
    fn=build_table_1_v2,
    inputs=inputs_1,
    output_dir="s3://bucket/multi-join-dataset",
    table="table_1",
    version="v2",
    chunk_size=2048,
    key_fn=lambda sample: sample["id"],
    num_workers=8,
    expected_snapshot="snap-baseline",
    snapshot="snap-table1-v2",
)
```

Only the new `table_1` version prefix and a new snapshot document are written. Other tables keep their previously published versions.

`expected_snapshot` / `snapshot` are **human-readable names** (for example `"snap-baseline"`), not hashes of chunk bytes. `expected_snapshot` is an optimistic concurrency guard: if another publisher advances the active snapshot first, this job fails rather than overwriting that change. LitData does not compute content hashes of chunk payloads to form snapshot IDs.

### 7.4 Stream the active snapshot

```python
from litdata import MultiJoinStreamingDataset, StreamingDataLoader

dataset = MultiJoinStreamingDataset(
    "s3://bucket/multi-join-dataset",
    tables=("table_0", "table_1"),
    shuffle=True,
    seed=42,
    drop_last=True,
    transform=lambda parts: {
        **parts["table_0"],
        **parts["table_1"],
    },
    max_cache_size="200GB",
    max_pre_download=4,
)

loader = StreamingDataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
)
```

Default output without `transform`:

```python
{
    "table_0": <table_0 sample>,
    "table_1": <table_1 sample>,
}
```

The table namespace is preserved by default. LitData does not implicitly merge dictionaries because duplicate field names would otherwise be ambiguous.

### 7.5 Pin an exact snapshot

```python
dataset = MultiJoinStreamingDataset(
    "s3://bucket/multi-join-dataset",
    snapshot="snap-baseline",
    tables=("table_0", "table_1"),
    shuffle=True,
    seed=42,
)
```

Useful properties:

```python
dataset.snapshot_id
dataset.layout_id
dataset.table_versions
dataset.tables
```

### 7.6 Select only required tables

```python
dataset = MultiJoinStreamingDataset(
    root,
    tables=("table_0", "table_1"),
)
```

Selection changes which column families are downloaded, but it does not alter canonical sampling order.

Unknown, duplicated, or inactive table names raise a clear error during construction.

### 7.7 Per-table decoding options

Sampling options cannot vary by table. Decoding-specific options may:

```python
dataset = MultiJoinStreamingDataset(
    root,
    tables=("table_0", "table_1"),
    table_options={
        "table_0": {"encryption": table_0_key},
        "table_1": {"serializers": custom_serializers},
    },
)
```

The following are always shared and cannot appear in `table_options`:

- `shuffle`
- `seed`
- `drop_last`
- `subsample`
- epoch
- number of workers
- batch size
- distributed rank and world size

### 7.8 Validate without training

```python
from litdata import validate_multi_join

report = validate_multi_join(
    "s3://bucket/multi-join-dataset",
    snapshot="snap-baseline",
    deep=True,
)

report.raise_for_errors()
```

Validation levels:

- Construction always performs mandatory constant-size metadata validation.
- `deep=False` verifies manifests, table indexes, counts, layout IDs, and alignment roots.
- `deep=True` streams partition alignment metadata and verifies every ordered-key digest.
- Writer / `joint_optimize` success always performs the validation required to prove a newly published table matches the canonical layout.

### 7.9 Keyed debugging access

```python
sample = dataset.get_by_key("entity-123")
```

Keyed lookup is intended for inspection, debugging, and bounded retrieval. It is not used by the iterative training path.

For integer entity keys, `get_by_key()` remains explicit so integer positional indexing is unambiguous.

## 8. Common storage and snapshot model

The root is a versioned store:

```text
multi-join-dataset/
  join.json
  snapshots/
    snap-baseline.json
    snap-table1-v2.json
  layouts/
    layout-entity-v1/
      index.json
      keys/
        shard-00000.parquet
        shard-00001.parquet
      partitions.parquet
  tables/
    table_0/
      v1/
        index.json
        alignment.parquet
        chunk-0-0.bin
        ...
    table_1/
      v1/
        index.json
        alignment.parquet
        chunk-0-0.bin
        ...
      v2/
        index.json
        alignment.parquet
        chunk-0-0.bin
        ...
```

### 8.1 Active manifest

`join.json` is small and published last:

```json
{
  "format": "litdata-multi-join",
  "format_version": 1,
  "active_snapshot": "snap-table1-v2",
  "updated_at": "2026-08-17T10:42:11Z"
}
```

### 8.2 Immutable snapshot

```json
{
  "format": "litdata-multi-join-snapshot",
  "format_version": 1,
  "snapshot_id": "snap-table1-v2",
  "parent_snapshot_id": "snap-baseline",
  "created_at": "2026-08-17T10:42:10Z",
  "layout": {
    "id": "layout-entity-v1",
    "path": "layouts/layout-entity-v1",
    "key_type": "string",
    "length": 1000000000,
    "chunk_size": 2048,
    "num_chunks": 488282,
    "alignment_root": "blake2b-256:..."
  },
  "tables": {
    "table_0": {
      "version": "v1",
      "path": "tables/table_0/v1",
      "format": "litdata",
      "layout_id": "layout-entity-v1",
      "alignment_root": "blake2b-256:...",
      "schema_fingerprint": "sha256:..."
    },
    "table_1": {
      "version": "v2",
      "path": "tables/table_1/v2",
      "format": "litdata",
      "layout_id": "layout-entity-v1",
      "alignment_root": "blake2b-256:...",
      "schema_fingerprint": "sha256:..."
    }
  }
}
```

All paths are relative to the root in V1. Manifest parsing rejects:

- Absolute paths.
- Parent traversal.
- A different URI scheme or bucket.
- Duplicate normalized table names.
- Unknown format versions.

This keeps one snapshot within one trust and credential boundary.

### 8.3 Canonical layout

The layout is immutable and contains:

- Canonical entity key type and order.
- Global item count.
- Logical bucket size.
- Tail bucket size.
- Per-bucket global start and stop positions.
- Per-bucket ordered-key digest.
- A digest root covering the complete ordered layout.
- A key index for debugging lookup and table re-optimization.

The key store must be sharded and streamed. Creating or validating a billion-row layout must not require a Python dictionary containing every key.

The canonical key order, rather than lexical key order, defines training positions. Key-index shards may be physically sorted or hash-partitioned for lookup as long as the stored `global_index`, `chunk_index`, and `chunk_offset` preserve the canonical order.

### 8.4 Identifiers vs digests

Three different identifiers appear in the format. They must not be confused:

- **Snapshot ID** (for example `snap-baseline`): a human-readable name for one published combination of table versions. Assigned when a `joint_optimize` job publishes successfully. It is **not** a hash of chunk bytes.
- **Table version** (for example `v1`, `v2`): a human-readable name for one immutable build of a single table.
- **Ordered-key digest / alignment root**: a compact fingerprint of the **canonical entity-key order** inside each logical bucket. Used only to prove tables are aligned. LitData does **not** compute content hashes of chunk payloads for this purpose.

### 8.5 Digest encoding

Digest computation (for ordered keys only) must be deterministic across Python versions and machines:

1. Normalize the key to the supported integer or UTF-8 string representation.
2. Prefix each key with a type tag.
3. Prefix variable-length bytes with an explicit fixed-width length.
4. Hash keys in canonical order.
5. Include bucket index, start position, and item count in the bucket digest.
6. Hash the ordered bucket metadata into the layout alignment root.

The initial algorithm is `BLAKE2b-256`. The manifest stores the algorithm and canonical encoding version so a future algorithm can coexist without ambiguity.

Using `str(key)` concatenation without type and length framing is not acceptable because it can create ambiguous encodings.

### 8.6 Table alignment metadata

Every table version contains an `alignment.parquet` sidecar with:

- `chunk_index`
- `global_start`
- `num_items`
- `ordered_key_digest`

The table `index.json` gains a backward-compatible `multi_join` section:

```json
{
  "multi_join": {
    "format_version": 1,
    "table": "table_1",
    "table_version": "v2",
    "layout_id": "layout-entity-v1",
    "length": 1000000000,
    "num_chunks": 488282,
    "alignment_root": "blake2b-256:..."
  }
}
```

Per-chunk digests live in compact Parquet rather than expanding an already large `index.json`. Normal dataset construction compares constant-size roots. Deep validation reads the Parquet sidecars.

## 9. Atomic publication and reader isolation

### 9.1 Publication protocol

Each successful `joint_optimize(...)` publishes as part of the same job (no separate writer `commit()`):

01. Read and retain the expected active snapshot when `expected_snapshot` is set.
02. Write new table data to a unique immutable version path.
03. Upload all chunk objects (existing multi-worker / multi-node optimize upload path).
04. Upload table alignment metadata.
05. Upload the table `index.json` and completion marker last.
06. If this is the first table in an empty join root, publish the shared layout from the ordered `key_fn` stream.
07. Validate the table against the canonical layout.
08. Write a new immutable snapshot document that merges this table version with sibling versions from the previous active snapshot.
09. Recheck the active snapshot or storage generation.
10. Atomically replace `join.json` with the new active snapshot.

If any operation fails before step 10, the active snapshot remains unchanged. Unreferenced objects are safe to garbage-collect later.

Because each `joint_optimize` is an independent distributed job, publication must be safe across machines and processes. There is no process-local `MultiJoinWriter` session holding uncommitted state.

### 9.2 Backend behavior

- Local files use a temporary file, `fsync` where appropriate, and `os.replace`.
- S3 and R2 use immutable version objects and a conditional active-manifest write where supported.
- GCS uses generation-match preconditions.
- A backend without safe compare-and-swap must use a single-publisher lease or fail closed for concurrent publication.

The storage abstraction must expose the precondition required for `expected_snapshot`; a read-then-unconditional-write sequence is not sufficient to prevent lost updates.

### 9.3 Reader behavior

`MultiJoinStreamingDataset` resolves the active snapshot exactly once during construction. It never polls `join.json` during iteration.

Therefore:

- Existing training jobs continue reading the old immutable table paths.
- New training jobs see the newly active snapshot.
- No job sees a mixture of old and new table versions.
- Checkpoint state records the snapshot ID.
- Resume against a different snapshot fails unless the caller explicitly opts to start a new data epoch.

## 10. Phase V1 — strict aligned LitData chunks

### 10.1 V1 objective

Deliver a safe, production-testable implementation with the smallest possible change to LitData’s proven training read path.

In V1:

- Every table version is a standard optimized LitData dataset.
- One physical LitData chunk is one logical sampling bucket.
- Corresponding table chunks contain the same keys in the same order.
- Chunk byte sizes, schemas, compression, serializers, and payload types may differ.
- `MultiJoinStreamingDataset` is a validated named wrapper around `ParallelStreamingDataset`.

### 10.2 V1 hard invariants

For every active table:

01. `layout_id` matches the snapshot layout.
02. Total logical length matches.
03. Number of chunks matches.
04. Chunk `i` has the canonical item count.
05. Chunk `i` has the canonical ordered-key digest.
06. The final partial chunk appears in the same position.
07. Every canonical key appears exactly once.
08. No additional key appears.
09. One input produces one logical table sample.
10. Table paths and versions are immutable.

For the read configuration:

1. All children receive the same `shuffle`.
2. All children receive the same `seed`.
3. All children receive the same epoch.
4. All children receive the same `drop_last`.
5. All children receive the same `subsample`.
6. All children see the same distributed environment.
7. All children receive the same DataLoader worker count and batch size.
8. A loaded state dict references the same snapshot and ordered table list.

### 10.3 V1 write path

`joint_optimize(...)` wraps the existing optimize pipeline with required safe settings:

- `chunk_size` is required and becomes the shared logical bucket size.
- `align_chunking=True`.
- `reorder_files=False`.
- `keep_data_ordered=True`.
- Static ordered inputs.
- One output sample per input.
- `key_fn` required so alignment digests can be computed (same mechanism as `optimize(..., key_fn=...)`).
- Output lands under `tables/{table}/{version}/` inside the join root rather than a standalone dataset root.

The following are hidden or rejected:

- `chunk_bytes`
- `weights`
- shared dynamic work queue
- `keep_data_ordered=False`
- `reorder_files=True`
- filtering with `None`
- variable-yield generators
- append into an existing table version
- overwrite of a published table version
- a `key_name` / schema-field argument

`align_chunking=True` is important because it makes logical chunk boundaries independent of the number of optimize workers or nodes. Workers receive complete item-count chunks, and rank-index merge order reconstructs the canonical sequence.

A small table may therefore be rebuilt with 8 workers while a large table was originally built with 256 workers, provided both consume the same canonical ordered inputs and logical `chunk_size`. Because each call is a normal optimize-style job, `num_nodes` / Studio multi-node execution works the same way as `optimize`.

### 10.4 V1 read path

Construction:

1. Resolve the root through LitData’s normal path resolver.
2. Load and pin a snapshot.
3. Validate the snapshot and selected table metadata.
4. Create one `StreamingDataset` per selected table version.
5. Give every child identical sampling options.
6. Allocate a unique cache namespace per snapshot, table, and version.
7. Pass the children to `ParallelStreamingDataset`.
8. Adapt positional tuples to a named dictionary.

Iteration:

1. Existing `FullShuffle` computes the same chunk ordering for every child.
2. Existing worker assignment computes the same chunk intervals because chunk counts and item counts match.
3. Existing in-chunk shuffle computes the same item permutation.
4. Each child `BinaryReader` prefetches its corresponding chunks.
5. `ParallelStreamingDataset` pulls one aligned value from each child.
6. `MultiJoinStreamingDataset` creates the named table mapping.
7. The optional transform creates the final training sample.

No key lookup occurs on this path.

### 10.5 Why existing shuffle remains aligned in V1

Corresponding children have identical:

- Chunk interval arrays.
- Number of chunks.
- Seed.
- Epoch.
- Chunk index.
- Worker and distributed topology.
- Batch and `drop_last` configuration.

As a result, both chunk-to-worker assignment and within-chunk permutation are deterministic and equal across children.

This must be proven with integration tests rather than assumed from unit-level seed equality.

### 10.6 V1 resume behavior

The state dict includes:

- Snapshot ID.
- Layout ID.
- Ordered table names and versions.
- Existing child `StreamingDataset` states.
- Current epoch.
- Per-worker yielded counts.
- Transform RNG state inherited from `ParallelStreamingDataset`.

On load:

- A different snapshot is rejected.
- A different selected table set or order is rejected.
- Existing LitData checks still reject incompatible seed, shuffle, worker count, batch size, item loader, or distributed world size.
- `force_override_state_dict` remains an advanced escape hatch and emits a strong warning because repeated or skipped samples are possible.

### 10.7 V1 cache and prefetch behavior

Each child keeps its existing:

- `BinaryReader`
- `PrepareChunksThread`
- Downloader
- Asynchronous remote prefetch
- Refcount and deletion logic

The wrapper does not introduce another download engine.

An explicit join cache root is namespaced:

```text
cache/
  <snapshot-id>/
    table_0/
      <version>/
    table_1/
      <version>/
```

This prevents same-named LitData chunk files from different tables from colliding.

`max_cache_size` is defined as the aggregate user budget. V1 approximates this by assigning per-table budgets in proportion to average chunk bytes, with a documented minimum and an optional per-table override.

Peak in-flight storage is approximately:

```text
num_workers × max_pre_download × sum(corresponding table chunk bytes)
```

The slowest or largest member of a chunk group determines when that group is fully available.

### 10.8 V1 limitations

V1 intentionally accepts the following trade-offs:

- Every table has the same number of physical chunk objects.
- Tiny tables may produce many small objects.
- Object GET count is approximately multiplied by the number of selected tables.
- A single `chunk_size` must balance feature chunk bytes, sampling diversity, and small-table object count.
- The wrapper relies on several independently operating child prefetchers.
- Aggregate cache enforcement is approximate because children retain independent cache managers.
- Required tables must have the same logical key population.
- Changing the canonical key population requires a new layout and rebuilding all tables.
- Native per-key zero-to-many row ranges are not yet represented; V1 bundles them into one table sample.

These limitations are the primary motivation for V2.

### 10.9 V1 error model

Introduce specific public exceptions:

- `MultiJoinError`
- `MultiJoinManifestError`
- `MultiJoinAlignmentError`
- `MultiJoinSnapshotMismatchError`
- `MultiJoinCommitConflictError`

An alignment error includes:

- Table name and version.
- Layout ID.
- Chunk or global index.
- Expected item count or key digest.
- Actual item count or key digest.
- Recommended remediation.

No warning-only mode is provided for alignment failures.

### 10.10 V1 implementation work

#### Manifest and layout utilities

Add `src/litdata/utilities/multi_join.py`:

- Typed dataclasses for active manifest, snapshot, layout, table version, and validation report.
- Strict JSON parsing and format-version checks.
- Relative-path validation.
- Canonical key encoding and digest computation.
- Streaming layout creation.
- Snapshot loading and publication helpers.
- Sharded key iteration without a full Python dictionary.

Do not add a heavy schema dependency solely for these small metadata models.

#### Write API

Add `joint_optimize` in `src/litdata/processing/functions.py` (or a thin wrapper module re-exported from `__init__.py`):

- Same distributed execution path as `optimize` (`DataProcessor`, `num_nodes`, uploaders, index merge).
- Extra args: `table`, optional `version`, `snapshot`, `expected_snapshot`.
- Force aligned write settings listed above.
- After merge, validate against the join layout and publish/update the snapshot atomically.
- No `MultiJoinWriter` context manager and no process-local uncommitted session.

Extend the optimize internals where necessary to:

- Stream ordered `(global_index, key)` metadata from `key_fn`.
- Compute per-chunk key digests.
- Avoid materializing all keys during alignment generation.
- Attach the backward-compatible `multi_join` section to `index.json`.
- Create the shared layout on first publish into an empty join root.

Likely integration points:

- `src/litdata/processing/functions.py`
- `src/litdata/processing/data_processor.py`
- `src/litdata/streaming/writer.py`
- `src/litdata/utilities/keys_index.py`
- `src/litdata/utilities/multi_join.py`

#### Read API

Add `src/litdata/streaming/multi_join.py`:

- Manifest resolution.
- Mandatory alignment validation.
- Named child construction.
- Cache namespacing.
- Tuple-to-mapping transform adapter.
- Snapshot-aware state dict.
- Optional keyed debugging access.

Reuse:

- `src/litdata/streaming/parallel.py`
- `src/litdata/utilities/base.py`
- `src/litdata/streaming/dataset.py`
- `src/litdata/streaming/dataloader.py`
- `src/litdata/streaming/shuffle.py`
- `src/litdata/streaming/reader.py`

#### Storage publication

Extend `src/litdata/streaming/fs_provider.py` or a narrowly scoped manifest-storage abstraction with:

- Read object metadata or generation.
- Conditional active-manifest write.
- Immutable JSON upload.
- Local atomic replacement.

Training downloads continue to use `Downloader`; `FsProvider` must not be introduced into `PrepareChunksThread`.

#### Public exports

Update:

- `src/litdata/__init__.py`
- Public API documentation.
- The LitData skill and reference docs.

### 10.11 V1 correctness test matrix

Add focused tests under:

- `tests/processing/test_multi_join.py`
- `tests/streaming/test_multi_join.py`
- `tests/utilities/test_multi_join.py`

Cover:

- One, two, and many tables.
- Empty dataset rejection.
- Full and partial final chunks.
- Different table payload byte sizes.
- Different schemas and serializers.
- Different compression settings.
- Different optimize worker counts.
- Different optimize node counts.
- Stable boundaries with `align_chunking=True`.
- Key order mismatch.
- Missing key.
- Duplicate key.
- Extra key.
- `fn` returning `None`.
- Variable-yield generator rejection.
- Corrupt layout ID.
- Corrupt per-chunk digest.
- Missing table `index.json`.
- Incomplete table version.
- Manifest path traversal.
- Unsupported manifest format version.

Read-path matrix:

- `shuffle=False` and `shuffle=True`.
- Multiple epochs.
- Seeds.
- DataLoader workers `0`, `1`, `2`, and a higher stress count.
- DDP world sizes and ranks.
- Chunk intervals split across workers.
- `drop_last=True` and `False`.
- Different batch sizes.
- Common subsampling.
- Persistent workers.
- Early break.
- Complete and partial checkpoint resume.
- Transform RNG resume.
- Cache pressure and eviction.
- Asynchronous remote prefetch on and off.

Snapshot tests:

- A reader opened before commit continues on the old snapshot.
- A reader opened after commit sees the new snapshot.
- `table_0` objects are not rewritten by a `table_1`-only update.
- Rollback activates the previous immutable snapshot.
- Concurrent stale `joint_optimize` publish fails.
- Failure before active-manifest publication leaves the old snapshot active.

Remote tests:

- S3-compatible mocked backend.
- R2 path resolution.
- Lightning Storage connection path resolution.
- GCS manifest generation behavior where available.
- Direct object access rather than hand-reading a FUSE mount.

### 10.12 V1 performance validation

Benchmark against a baked single LitData dataset using the same:

- Entity population and canonical order.
- Batch size.
- DataLoader workers.
- DDP topology.
- Cache disk.
- `max_pre_download`.
- Compression.
- Cold-cache and warm-cache conditions.

Measure:

- Time to first batch.
- Median and tail batch wait.
- Samples per second.
- GPU data-wait percentage.
- Object GET count.
- Bytes downloaded.
- Peak cache usage.
- CPU decode time.
- Cost and elapsed time to rebuild only the changed table.

No production throughput claim should be made before this benchmark runs on a representative multi-table workload. Correctness and independent table replacement are hard release gates; throughput is a measured release gate agreed against the baseline.

### 10.13 V1 completion criteria

V1 is complete when:

1. A two-table and a many-table dataset can be created through independent `joint_optimize` jobs.
2. Re-optimizing one table does not write under unchanged table-version prefixes.
3. A new table snapshot is atomically activated.
4. Old readers continue without observing mixed versions.
5. Every tested shuffle, worker, DDP, and resume configuration preserves key alignment.
6. Intentional misalignment fails before iteration.
7. The Lightning Storage end-to-end test is green.
8. Representative cold-cache benchmarks are documented.
9. API documentation includes creation, update, rollback, training, and troubleshooting.

## 11. Phase V2 — logical partitions and independent physical compaction

### 11.1 V2 objective

Remove V1’s requirement that every logical sampling bucket be a separate physical object in every table, while retaining canonical key alignment and deterministic bucket sampling.

V2 does **not** remove alignment. It separates:

- Logical sampling layout.
- Table-specific physical storage layout.

This permits small or schema-volatile tables to use an efficient physical representation without changing the training entity order.

### 11.2 Why V2 is needed

Assume the canonical bucket contains 2,048 entities:

- A large `table_0` bucket may be 64–256 MB.
- A small `table_1` bucket may be only tens or hundreds of KB.

V1 writes one object for each in both tables. At very large scale, a small table may have hundreds of thousands of tiny objects. This increases:

- Object listing and metadata cost.
- GET request count.
- Time to first batch.
- Index size.
- Cache bookkeeping.
- Publication and garbage-collection overhead.

V2 allows many small-table buckets to be packed into one appropriately sized object while preserving the original logical bucket boundaries for sampling.

### 11.3 V2 storage model

```text
multi-join-dataset/
  join.json
  snapshots/
  layouts/
    <layout-id>/
      index.json
      keys/
      partitions.parquet
  tables/
    table_0/
      <version>/
        table.json
        mapping.parquet
        objects/
          pack-00000.bin
          pack-00001.bin
    table_1/
      <version>/
        table.json
        mapping.parquet
        objects/
          pack-00000.parquet
          pack-00001.parquet
```

Each table’s `mapping.parquet` maps canonical logical buckets to physical storage:

- Logical bucket ID.
- Canonical global start and item count.
- Physical object path.
- Encoding or item-loader type.
- Byte, row-group, or row interval.
- Ordered-key digest.
- Optional per-key offset-vector location.

One physical object may contain several consecutive logical buckets.

V2 initially avoids splitting one logical bucket across many physical objects unless a single bucket exceeds the configured maximum. Supporting one object span per table per logical bucket keeps prefetch planning bounded.

### 11.4 Heterogeneous column families

V2 can support different physical formats behind one logical contract:

- LitData binary for large tensors, images, audio, or nested Python structures.
- Parquet for schema-volatile tabular data.
- A compact offset-based binary representation for variable-length rows.

Every format adapter must provide:

- Metadata loading.
- Physical dependency resolution for a logical bucket.
- Prefetch request generation.
- Loading a logical item by position.
- Resource release.
- Format-specific schema fingerprint.

The join sampler must not contain format-specific decode logic.

### 11.5 V2 read engine

V2 replaces the V1 `ParallelStreamingDataset` composition internally with a shared engine:

1. Load the canonical logical partition intervals once.
2. Run chunk-to-rank and chunk-to-worker assignment once.
3. Generate one within-bucket item permutation.
4. Resolve each selected table’s physical dependencies for upcoming buckets.
5. Deduplicate dependencies when several logical buckets share a physical object.
6. Prefetch the complete dependency group.
7. Apply the same item position to every table loader.
8. Return the named table mapping.

Potential internal components:

- `MultiJoinChunksConfig`
- `MultiJoinShuffle`
- `MultiJoinReader`
- `PrepareJoinPartitionsThread`
- `ColumnFamilyLoader`
- `LitDataColumnFamilyLoader`
- `ParquetColumnFamilyLoader`

Existing `Downloader` implementations, async remote fetch, serializers, and cache locking should be reused wherever possible.

V2 must not use `FsProvider` on the training read path.

### 11.6 V2 cache behavior

The shared reader owns one aggregate cache budget:

- Cache keys include snapshot, table, version, and physical object.
- One downloaded packed object can satisfy several upcoming logical buckets.
- Admission and eviction use the shared future-use schedule.
- Refcounts cover all workers and all logical buckets referencing an object.
- Prefetch depth is measured in logical bucket groups, not independently per table.

This eliminates V1’s approximate per-child budget split.

### 11.7 V2 `joint_optimize` storage extension

`joint_optimize` gains optional per-table storage configuration:

```python
from litdata import joint_optimize, TableStorage

joint_optimize(
    fn=build_table_1_v3,
    inputs=inputs_1,
    output_dir=root,
    table="table_1",
    chunk_size=2048,
    key_fn=get_id,
    storage=TableStorage(
        format="parquet",
        target_chunk_bytes="128MB",
        pack_logical_chunks=True,
    ),
)
```

Large table (`table_0`):

```python
joint_optimize(
    fn=build_table_0,
    inputs=inputs_0,
    output_dir=root,
    table="table_0",
    chunk_size=2048,
    key_fn=get_id,
    storage=TableStorage(
        format="litdata",
        target_chunk_bytes="256MB",
        pack_logical_chunks=False,
    ),
)
```

The normal `MultiJoinStreamingDataset` construction remains unchanged.

### 11.8 V2 variable-cardinality column families

After the shared reader and packed-object model are stable, V2 can represent source-table cardinality natively:

- Each logical key position maps to a start and stop row offset.
- An empty range represents no rows for that table and key.
- A non-empty range returns one or many rows as the table contribution.
- Duplicate source rows are valid within that range.
- Duplicate logical entity keys remain invalid.

This is similar to a compressed sparse row layout:

```text
logical keys:   [k0, k1, k2, k3]
row offsets:    [0, 3, 3, 8, 9]
rows for k0:    [0:3]
rows for k1:    [3:3]  -> empty
rows for k2:    [3:8]
rows for k3:    [8:9]
```

The entity sampler still chooses `k0..k3`; the table loader resolves the associated row slice without a hash join.

This feature should be staged after V2’s one-envelope-per-key implementation because it changes collation, memory bounds, and schema semantics.

### 11.9 V2 state and resume

V2 stores one canonical sampling state rather than one child state per table:

- Snapshot and layout ID.
- Epoch.
- Canonical bucket permutation.
- Worker assignment inputs.
- Current logical bucket.
- Consumed positions in the bucket.
- Shared transform RNG state.

Table versions are data dependencies of the snapshot, not independent samplers.

This reduces the possibility of child state divergence and makes the alignment contract explicit in code.

### 11.10 V2 failure behavior

- A missing physical object reports the table, version, logical bucket, and object.
- A corrupt mapping entry fails before returning a sample.
- A loader returning a different item count from the mapping fails the bucket.
- Repeated download failure follows existing LitData retry and timeout behavior.
- Partial table publication remains unreachable because snapshots reference only complete versions.
- Unsupported table formats fail during construction.

### 11.11 V2 implementation work

Add or evolve:

- `src/litdata/streaming/multi_join.py`
- `src/litdata/streaming/multi_join_reader.py`
- `src/litdata/streaming/multi_join_config.py`
- `src/litdata/streaming/multi_join_shuffle.py`
- `src/litdata/processing/multi_join.py`
- Format-specific column-family loaders.

Refactor the V1 implementation behind internal protocols so the public class and state schema can evolve without breaking user code.

### 11.12 V2 test matrix

In addition to all V1 correctness tests:

- Different physical object counts per table.
- Several logical buckets in one small-table object.
- Large object spanning upcoming worker buckets.
- Dependency deduplication.
- LitData binary plus Parquet in one snapshot.
- Row-group and byte-interval mapping.
- Shared aggregate cache eviction.
- Multiple workers referencing the same packed object.
- Resume in the middle of a packed object.
- Missing and corrupt mapping entries.
- Optional empty per-key row ranges.
- Variable-cardinality collation.
- Very large partition manifests.
- Bounded metadata memory usage.

### 11.13 V2 performance validation

Compare:

1. Baked single dataset.
2. V1 strict aligned physical chunks.
3. V2 independently compacted column families.

The expected V2 improvement is primarily:

- Fewer object GETs for small tables.
- Smaller index and file-management overhead.
- Better cache reuse across adjacent logical buckets.
- Better control over each table’s physical object size.

V2 must preserve V1’s correctness and independent-version benefits.

### 11.14 V2 completion criteria

V2 is complete when:

1. V1 read code runs unchanged against the public API.
2. Different table object counts are supported under one canonical layout.
3. One shared sampler controls all tables.
4. Packed small-table objects reduce object GET count on the representative workload.
5. Aggregate cache usage is bounded by the configured budget.
6. Resume reproduces exactly the same logical entity sequence.
7. Mixed LitData and Parquet column families pass local, cloud, worker, and DDP tests.
8. V2 meets the agreed throughput target relative to the baked baseline.

## 12. V1 and V2 boundary

The boundary is intentionally simple:

### V1 aligns physical chunks

```text
logical bucket 0 -> table_0 chunk 0 -> table_1 chunk 0
logical bucket 1 -> table_0 chunk 1 -> table_1 chunk 1
logical bucket 2 -> table_0 chunk 2 -> table_1 chunk 2
```

The existing LitData sampler and independent readers can be reused.

### V2 aligns logical buckets

```text
logical buckets 0..9:
  table_0 -> 10 objects
  table_1 -> 1 packed object
```

A new shared sampler and mapping-aware reader are required.

### What V2 still does not do

V2 does not accept arbitrary unrelated table sharding and repair it with per-sample random joins. Logical key order and bucket membership remain shared. This is the requirement that preserves predictable training I/O.

## 13. Migration workflow

### One-time migration

01. Choose the entity key extraction (`key_fn`).
02. Produce unique ordered inputs for every table (same entity order).
03. Intentionally choose that training order before the first `joint_optimize`.
04. Choose the logical bucket item count based primarily on the largest table's bytes and desired bucket sampling.
05. Run `joint_optimize` for the first / driver table to create the join root and layout.
06. Run independent `joint_optimize` jobs for the remaining tables (local or multi-node).
07. Validate the published snapshot.
08. Benchmark against the current baked dataset.

### Small-table schema update

01. Keep existing sibling table versions as-is.
02. Run `joint_optimize(..., table=..., version=..., expected_snapshot=...)` for the changed table only.
03. Start new training jobs on the newly published snapshot.
04. Leave existing jobs pinned to their original snapshot.

### Rollback

1. Select a known-good immutable snapshot.
2. Atomically update `join.json`.
3. Start new jobs.

No table data is copied during rollback.

### Canonical key population change

Adding or removing entity keys changes:

- Global indexes.
- Logical bucket membership.
- Chunk counts or tail.
- Ordered-key digests.

V1 therefore creates a new layout and rebuilds every table. This is a deliberate correctness boundary.

A future append-only layout extension could preserve complete existing buckets and add new buckets, but it requires separate sampling, snapshot, and resume semantics and is not included in the initial plan.

## 14. Operational guidance

### Version naming

- User-provided readable versions are accepted.
- Internally generated immutable IDs prevent collisions.
- Reusing an existing table version path is rejected.

### Retention and garbage collection

Initial behavior:

- Successful `joint_optimize` jobs never delete previous table versions.
- Rollback remains possible while snapshots and versions are retained.
- Failed staging outputs are recorded as unreachable.

Later garbage collection:

1. Enumerate retained snapshots.
2. Mark referenced layouts and table versions.
3. Apply a minimum age safety window.
4. Delete only unreferenced immutable prefixes.
5. Support dry-run output before deletion.

Garbage collection is never part of `joint_optimize` publication.

### Observability

Expose structured diagnostics for:

- Snapshot resolution.
- Selected table versions.
- Alignment validation duration.
- Per-table download bytes and latency.
- Prefetch queue depth.
- Cache hit and eviction counts.
- Bucket wait on the slowest table.
- Keyed debug lookup latency.

Tracing should identify table and version without including entity keys by default.

## 15. Risks and mitigations

### Silent positional mis-join

Risk: the most serious failure mode.

Mitigation:

- Canonical key registry.
- Per-bucket ordered-key digests.
- Alignment-root comparison.
- Validation before publication.
- Mandatory metadata validation on open.
- End-to-end tests that include the key in test payloads.

### Tiny-object explosion in V1

Risk: schema-small tables produce many tiny chunks.

Mitigation:

- Measure V1 object count and GET overhead.
- Keep V1 as the minimal safe implementation.
- Implement V2 physical packing if the measured cost is material.

### Cache multiplication

Risk: independent V1 child readers each reserve and prefetch data.

Mitigation:

- Namespace caches.
- Interpret the public cache budget as aggregate.
- Allocate proportional child budgets.
- Move to one shared cache coordinator in V2.

### Concurrent publisher race

Risk: two updates overwrite each other’s active snapshot.

Mitigation:

- `expected_snapshot`.
- Conditional object-store publication.
- Immutable snapshots.
- Explicit commit-conflict error.

### Metadata size at billion-row scale

Risk: keys and per-chunk digests cannot be held in Python memory or expanded into large JSON documents.

Mitigation:

- Sharded Parquet key storage.
- Compact Parquet partition metadata.
- Streaming digest computation.
- Constant-size roots in normal open validation.
- Deep validation as a sequential metadata scan.

### Optimize worker-count dependence

Risk: independent table runs use different worker or node counts.

Mitigation:

- Force item-count chunking.
- Force ordered static inputs.
- Use `align_chunking=True`.
- Verify merged chunk counts and digests.
- Test all worker-count combinations.

### Manifest or path injection

Risk: a modified manifest points outside the dataset root.

Mitigation:

- Strict relative paths in V1.
- Reject traversal and cross-storage references.
- Validate format and field types.
- Store no credentials in manifests.

## 16. Decision summary

01. Alignment is required; arbitrary random keyed joins are not the product direction.
02. The entity key and canonical order are immutable within one layout.
03. One logical table contribution per entity is the V1 contract.
04. V1 uses standard LitData datasets with exactly aligned physical chunks.
05. V1 reuses `ParallelStreamingDataset` and existing read/prefetch infrastructure.
06. V2 retains logical alignment but permits independent physical compaction and formats.
07. V2 introduces one shared sampler and coordinated reader.
08. Table versions and snapshots are immutable.
09. `join.json` is published last and readers pin one snapshot.
10. Each `joint_optimize` job validates and publishes atomically; no writer context / deferred `commit()`.
11. Training iteration performs no per-sample key lookup.
12. The public `MultiJoinStreamingDataset` API remains stable across V1 and V2.
13. Keys come from `key_fn`, not from a `key_name` schema-field argument.

## 17. Recommended delivery order

### Phase V1

01. Freeze manifest, layout, digest, and API specifications.
02. Implement metadata models and validation.
03. Implement scalable canonical layout creation.
04. Implement `joint_optimize` with strict aligned optimize settings and multi-node support.
05. Implement atomic immutable snapshots.
06. Implement the validated `ParallelStreamingDataset` wrapper.
07. Add snapshot-aware state and resume.
08. Add local, remote, DDP, worker, cache, and failure tests.
09. Validate with Lightning Storage.
10. Run the representative multi-table benchmark.
11. Publish documentation and migration tooling.

### Phase V2

01. Freeze logical-to-physical mapping format.
02. Define the column-family loader protocol.
03. Implement one canonical sampler.
04. Implement coordinated dependency prefetch.
05. Implement shared aggregate cache ownership.
06. Add LitData packed-object support.
07. Add Parquet column-family support.
08. Add optional offset-based variable-cardinality tables.
09. Run comparative V1, V2, and baked benchmarks.
10. Migrate internals while keeping the public read API unchanged.
