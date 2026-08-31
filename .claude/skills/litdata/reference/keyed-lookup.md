# Keyed lookup and in-place updates

Public API: `optimize(..., key_fn=...)`, `build_keys_index`, `dataset_update`, `StreamingDataset.get_by_key` / `ds["id"]`.

Requires **`polars>1.0`**. Sidecar: `keys/shard-*.parquet` next to `index.json`. Tests: `tests/streaming/test_dataset_update.py`.

## When to use

- Patch a few samples without a full re-`optimize`.
- Debug / join by entity id (`dataset["entity-id"]`).
- Integer **entity** keys must go through `get_by_key(3)` — `ds[3]` stays **positional**.

## Write the sidecar

```python
ld.optimize(
    fn=fn,
    inputs=inputs,
    output_dir="fast_data",
    chunk_bytes="64MB",
    key_fn=lambda sample: sample["id"],  # str or int
)
```

Last optimize node merges per-rank key files (`data_processor._merge_and_upload_keys`). `mode="append"` concatenates onto existing keys.

Backfill a dataset that was optimized without `key_fn`:

```python
from litdata import build_keys_index
build_keys_index("fast_data", key_fn=lambda sample: sample["id"], overwrite=False)
```

`build_keys_index` needs a **local** dataset directory. Close mmaps / `StreamingDataset` handles first on Windows (`os.replace` cannot overwrite an open `index.json`).

## Read by key

```python
ds = ld.StreamingDataset("s3://bucket/fast_data")  # or local
sample = ds["entity-id"]           # str key
sample = ds.get_by_key(42)         # int entity key
```

Remote: `KeyIndex` uses Polars `scan_parquet` + predicate pushdown (does not download the whole dataset). Missing sidecar → `KeyError` mentioning `keys/`.

## In-place update (local only)

```python
from litdata import dataset_update

with dataset_update("fast_data") as update:
    update["entity-id"] = {"id": "entity-id", "x": 1}
    update.commit()  # omit commit → discard
```

v1: **local directory** (or Studio `lightning_storage` FUSE path). Pure `s3://` update is `NotImplementedError`. Writes `chunk-*-uN.bin` plus sidecar refresh.

## Agent pitfalls

- `mode="append"` is **chunk numbering**, not `use_checkpoint` input resume — do not treat them as the same.
- Multi-node `key_fn` merge has **no** processing tests yet; preserve `_merge_and_upload_keys` / `concatenate_key_files`.
- Do not document remote `dataset_update` as supported.
