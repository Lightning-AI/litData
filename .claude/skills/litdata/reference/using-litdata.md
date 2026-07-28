# Using LitData — expert cookbook

Become productive immediately. Narrative docs: repo `README.md`. Internals: [streaming.md](streaming.md), [processing.md](processing.md), [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md), [storage-format.md](storage-format.md).

Install: `pip install litdata` · extras: `pip install 'litdata[extras]'` (README also uses `litdata[extra]` + `s3fs` / `gcsfs` / `huggingface_hub` for specific features).

______________________________________________________________________

## 1. Choose a workflow

| Goal                                          | API                                                     |
| --------------------------------------------- | ------------------------------------------------------- |
| Stream files as-is (no preprocess)            | `StreamingRawDataset` + torch `DataLoader`              |
| Fastest training I/O                          | `optimize` → `StreamingDataset` + `StreamingDataLoader` |
| Parallel side effects (resize, scrape, embed) | `map`                                                   |
| Weighted mix                                  | `CombinedStreamingDataset`                              |
| One sample from each dataset / cycle length   | `ParallelStreamingDataset`                              |
| Existing MDS / Parquet / HF parquet           | `StreamingDataset` (+ `ParquetLoader` when needed)      |
| LLM token windows                             | `TokensLoader` on optimize **and** stream               |

**Rule:** raw = instant native files. Optimized = chunk once, then stream fastest.

______________________________________________________________________

## 2. Canonical optimize → stream

```python
import io
import litdata as ld
from PIL import Image

def fn(path):
    # Prefer JPEG (JpegImageFile) — see §3. Plain Image.fromarray → huge PIL RAW.
    img = Image.open(path)  # keep .jpg as JpegImageFile; or re-encode at quality≈95
    if not path.lower().endswith((".jpg", ".jpeg")):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        buf.seek(0)
        img = Image.open(buf)
    return {"image": img, "path": path}  # stable keys/types across samples

if __name__ == "__main__":  # required for multiprocessing
    ld.optimize(
        fn=fn,
        inputs=list_of_paths,
        output_dir="fast_data",       # local, s3://, gs://, r2://, azure://, /teamspace/...
        num_workers=4,
        chunk_bytes="64MB",           # exactly one of chunk_bytes | chunk_size
    )

train = ld.StreamingDataset("s3://bucket/fast_data", shuffle=True, drop_last=True, seed=42)
loader = ld.StreamingDataLoader(train, batch_size=64, num_workers=8)
```

Train: `shuffle=True, drop_last=True`. Val: usually `shuffle=False, drop_last=False`. Prefer `StreamingDataLoader` (resume via `state_dict`).

______________________________________________________________________

## 3. Images & serializers

Built-in serializers (`streaming/serializers.py`), tried in registry order:
`str`, `bool`, `int`, `float`, `video`, `tifffile`, `pil`, `jpeg`, `jpeg_array`, `bytes`, `numpy`/`tensor` (+ no-header variants), `pickle`.

| Return type                           | Serializer   | Result                                       |
| ------------------------------------- | ------------ | -------------------------------------------- |
| `PIL.JpegImageFile` (opened `.jpg`)   | `jpeg`       | Stores compressed JPEG bytes — **preferred** |
| Plain `PIL.Image` / `Image.fromarray` | `pil`        | Uncompressed pixels — **large**              |
| List of JPEGs                         | `jpeg_array` | Packed JPEGs                                 |

**Best practice:** optimize images as JPEG at **quality ≈ 95** (or keep existing JPEGs). Resize when helpful. README benches: PIL RAW ~168 GB vs JPEG 90% ~12 GB at similar stream speed.

Custom / override:

```python
dataset = StreamingDataset(..., serializers={"my_type": MySerializer()})
```

Subclass `Serializer` with `serialize` / `deserialize` / `can_serialize`. Keys you pass are merged ahead of built-ins (win over `pickle`). `optimize()` picks built-ins from the types your `fn` returns — prefer JPEG / numpy / tensor leaves.

______________________________________________________________________

## 4. Paths & resolver (load [resolver.md](resolver.md))

**Always resolve paths through LitData.** In Studio, `/teamspace/s3_connections` & co are **FUSE** over S3/GCS/**R2** (`lightning_storage`); LitData resolves them to the backing URL and talks to the store directly — faster and more reliable than opening the mount by hand.

`streaming/resolver.py` → `Dir(path, url, data_connection_id)` for every `input_dir` / `output_dir` / `cache_dir`.

| Input                                                                     | Result                                                                                                                                                             |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `s3://` `gs://` `r2://` `azure://` `hf://`                                | Remote `url`                                                                                                                                                       |
| `local:/mnt/nfs/...`                                                      | Network drive; chunks still cached locally                                                                                                                         |
| Absolute / relative path                                                  | Local `path`                                                                                                                                                       |
| `/teamspace/studios/this_studio/...`                                      | Studio home: LitData sees local disk (`url=None`), but home is **persisted remotely** — don’t dump huge raw data here ([lightning-studio.md](lightning-studio.md)) |
| `/teamspace/studios/<other>/...`                                          | FUSE → other studio content bucket; LitData uses bucket URL                                                                                                        |
| `/teamspace/s3_connections\|gcs_connections\|s3_folders\|gcs_folders/...` | FUSE → customer S3/GCS; LitData **direct** object I/O                                                                                                              |
| `/teamspace/lightning_storage/...`                                        | FUSE → Lightning **R2**; LitData **direct** R2 + temp creds                                                                                                        |
| `/teamspace/datasets/...`                                                 | FUSE → cluster datasets S3; LitData **direct**                                                                                                                     |
| `{%Y-%m-%d}` in path                                                      | `strftime` expanded to now                                                                                                                                         |
| `Dir(path=..., url=...)`                                                  | Cache location ≠ remote URL                                                                                                                                        |

```python
from litdata.streaming.resolver import Dir
StreamingDataset("s3://bucket/data")
StreamingDataset("/teamspace/s3_connections/my-data/optimized")  # Studio: direct S3 (not FUSE)
StreamingDataset("/teamspace/lightning_storage/team-store/shards")  # Studio: direct R2
StreamingDataset(Dir(path="/fast-ssd/cache", url="s3://bucket/data"))
optimize(..., output_dir="s3://bucket/out/run_{%Y-%m-%d}")
```

Exhaustive FUSE vs backing-store map: [resolver.md](resolver.md). Studio home persistence + dataset prep: [lightning-studio.md](lightning-studio.md).

______________________________________________________________________

## 5. Shuffle, seed, drop_last, resume

- `shuffle=True` → deterministic **chunk assignment then in-chunk item order** from `seed` + epoch (+ chunk index).
- Default `seed=42`. Keep it fixed across ranks and when resuming.
- `drop_last=None` → **True under DDP**, else False. Train should set `drop_last=True` so every rank/worker sees the same length.
- `StreamingDataLoader(shuffle=..., drop_last=...)` **overrides** the dataset.
- Resume: `torch.save(loader.state_dict(), ...)`; `loader.load_state_dict(...)`. Matching `seed` / shuffle / `num_workers` required unless `force_override_state_dict=True`.

______________________________________________________________________

## 6. StreamingDataset arguments

| Arg                                   | Default                     | Notes                                                     |
| ------------------------------------- | --------------------------- | --------------------------------------------------------- |
| `input_dir`                           | required                    | Path, URL, `Dir`, or parquet path with basename wildcards |
| `cache_dir`                           | env / `~/.lightning/chunks` | Local chunk store                                         |
| `item_loader`                         | from index / `PyTreeLoader` | `TokensLoader`, `ParquetLoader`, …                        |
| `shuffle`                             | `False`                     | See §5                                                    |
| `drop_last`                           | DDP-aware                   | See §5                                                    |
| `seed`                                | `42`                        | Shuffle + subsample RNG                                   |
| `serializers`                         | built-ins                   | See §3                                                    |
| `max_cache_size`                      | `"100GB"`                   | Eviction budget                                           |
| `max_pre_download`                    | `2`                         | Prefetch depth; peak disk ≈ workers × this × chunk        |
| `subsample`                           | `1.0`                       | Fraction or >1 to upsample                                |
| `encryption`                          | `None`                      | `FernetEncryption` / `RSAEncryption` / custom             |
| `storage_options` / `session_options` | `{}`                        | Cloud creds / boto3 session                               |
| `index_path`                          | `None`                      | Parquet/HF index file or directory                        |
| `force_override_state_dict`           | `False`                     | Local args win over checkpoint                            |
| `transform`                           | `None`                      | Callable or list applied per sample                       |

______________________________________________________________________

## 7. StreamingDataLoader arguments

Torch `DataLoader` args plus:

| Arg                       | Notes                                                                                                           |
| ------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `shuffle` / `drop_last`   | Forwarded onto the streaming dataset                                                                            |
| `multiprocessing_context` | **Required `'spawn'` (or forkserver)** with `ParquetLoader` + `num_workers>0` on Linux (Polars + fork deadlock) |

Use `StreamingDataLoader` (not plain `DataLoader`) for optimized / combined / parallel datasets so batch metadata and `state_dict` work.

### Profiling (`profile_batches`) — viztracer

```python
StreamingDataLoader(
    dataset,
    batch_size=64,
    num_workers=4,                 # required (>=1)
    profile_batches=20,            # int = N batches; True = whole epoch; False = off
    profile_skip_batches=5,        # warm-up batches before recording
    profile_dir="./profiles",      # writes result.json (default cwd; overwrites)
)
```

| Requirement | Detail                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------ |
| Dep         | `pip install viztracer`                                                                                |
| Workers     | `num_workers >= 1` or raises                                                                           |
| Rank        | Only global rank 0 patches the worker loop                                                             |
| Scope       | Worker **0** only                                                                                      |
| Output      | `{profile_dir}/result.json` — open in `chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev) |

`int` wraps `fetcher.fetch` and stops after skip+N fetches; `True` traces until the worker loop ends. Complementary to `enable_tracer()` + Litracer (pipeline events) — see [debugging.md](debugging.md). README: `#profile-loading`.

______________________________________________________________________

## 8. Cache, async prefetch & environment variables

See [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md) and full catalog [env-vars.md](env-vars.md).

```python
StreamingDataset(..., cache_dir="/data/cache", max_cache_size="50GB", max_pre_download=4)
```

- `LITDATA_CACHE_DIR` — default cache root · CLI: `litdata cache path` / `litdata cache clear`
- **Async chunk prefetch** (remote downloads only; training loop stays sync):
  - Default **on** for remote, **off** for local
  - `LITDATA_ASYNC_CHUNK_PREFETCH=0/1` force off/on
  - When on, floors `max_pre_download` to `LITDATA_ASYNC_MIN_PRE_DOWNLOAD` (default **4**; `0` disables floor)
- Other common: `MAX_WAIT_TIME`, `FORCE_DOWNLOAD_TIME`, `LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB`, `HF_TOKEN`, `LITDATA_DISABLE_VERSION_CHECK`
- Multi-node optimize: `DATA_OPTIMIZER_*` (platform-set) — [processing.md](processing.md)

README: `#async-prefetch-env`.

______________________________________________________________________

## 9. optimize / map / walk (full knobs)

**Always:** `if __name__ == "__main__"` · optimize needs **exactly one** of `chunk_bytes` | `chunk_size`.

### `optimize`

| Arg                                 | Default         | Use                                                                                             |
| ----------------------------------- | --------------- | ----------------------------------------------------------------------------------------------- |
| `fn` / `inputs` / `output_dir`      | —               | Core recipe                                                                                     |
| `queue`                             | `None`          | Live inputs; one `ALL_DONE` sentinel (`from litdata.processing.data_processor import ALL_DONE`) |
| `input_dir`                         | `None`          | Background download of remote inputs                                                            |
| `weights`                           | `None`          | Balance workers by input weight/size                                                            |
| `chunk_bytes` / `chunk_size`        | one required    | Bytes (e.g. `"64MB"`) **or** item/token count                                                   |
| `align_chunking`                    | `False`         | Single-worker chunk boundaries (needs `chunk_size`; uneven load)                                |
| `compression`                       | `None`          | `"zstd"`                                                                                        |
| `encryption`                        | `None`          | Fernet / RSA / custom; `level="sample"` or `"chunk"`                                            |
| `num_workers`                       | CPUs            | Local parallelism                                                                               |
| `fast_dev_run`                      | `False`         | Smoke subset                                                                                    |
| `num_nodes` / `machine`             | `None`          | **Studio-only multi-node job** (see below) — not local MP                                       |
| `num_downloaders` / `num_uploaders` | auto            | I/O concurrency                                                                                 |
| `reorder_files`                     | `True`          | Size packing; `False` preserves order                                                           |
| `reader` / `batch_size`             | —               | Custom reader; group inputs                                                                     |
| `mode`                              | `None`          | `"append"` \| `"overwrite"` (else immutable)                                                    |
| `use_checkpoint`                    | `False`         | Resume interrupted job                                                                          |
| `item_loader`                       | `None`          | e.g. `TokensLoader()`                                                                           |
| `start_method` / `optimize_dns`     | spawn† / `None` | MP start; DNS tweak                                                                             |
| `storage_options`                   | `{}`            | Cloud creds                                                                                     |
| `keep_data_ordered`                 | `True`          | `False` = shared work queue                                                                     |
| `verbose`                           | `True`          | Progress                                                                                        |

### Multi-node (`num_nodes` / `machine`) — Lightning Studios

```python
from litdata import optimize, Machine

if __name__ == "__main__":
    optimize(
        fn=fn,
        inputs=inputs,
        output_dir="/teamspace/s3_connections/my-data/v1",  # prefer connection / s3:// / datasets
        chunk_bytes="64MB",
        num_workers=8,
        num_nodes=32,
        machine=Machine.DATA_PREP,  # or omit to inherit the Studio machine; enum from lightning_sdk
    )
```

| Fact         | Detail                                                                                                                                                                                                                |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Where        | Lightning Studio only (`lightning_sdk` required)                                                                                                                                                                      |
| What happens | Spawns a data-prep **job** that re-runs your script on N machines; platform sets `DATA_OPTIMIZER_*`                                                                                                                   |
| Sharding     | `world_size = num_nodes × num_workers`; each node processes its slice; **last node merges** `index.json`                                                                                                              |
| Outputs      | Prefer `/teamspace/s3_connections/...` or cloud URL. Local / `this_studio` optimize outputs remap to job **artifacts** S3 (`map` does **not** remap the same way). Studio may also show `/teamspace/jobs/...` mounts. |
| Creds        | Every node must reach inputs + outputs                                                                                                                                                                                |

Internals / env table → [processing.md](processing.md) (Multi-node launch). Studio UX → [lightning-studio.md](lightning-studio.md).

### `map`

`fn(input, output_dir) -> None` (must write files). Same family of knobs as optimize **plus** `error_when_not_empty`. No `chunk_*` / `encryption` / `item_loader` / `mode` / `use_checkpoint` / `verbose`. Same `num_nodes`/`machine` job launch as optimize.

### `walk`

```python
from litdata import walk
for root, dirs, files in walk("/teamspace/s3_connections/data/raw", max_workers=32):
    ...
```

Threaded cloud listing (Studio-optimized; warns elsewhere). Order ≠ depth-first. Use to build `inputs=` for optimize/map.

### Other

- Filter: `yield` only keepers (or catch and skip).
- Merge: `merge_datasets(input_dirs, output_dir, max_workers=..., storage_options=...)` — same `data_format`/compression.
- README tables: `#optimize-kwargs`, `#map`, `#walk`.

______________________________________________________________________

## 10. Format-specific streaming

**Raw** — `StreamingRawDataset(url, recompute_index=True?, cache_files=?, transform=?, indexer=?)`; override `setup` to group files.

**MDS** — point at Mosaic shards + `index.json`; auto `format=mds`. No encryption.

### Parquet & Hugging Face (exhaustive)

Deps: `pip install 'litdata[extras]'` (+ `s3fs` / `gcsfs` / `huggingface_hub`). Samples from `ParquetLoader` are **`dict`s**. Import: `from litdata.streaming.item_loader import ParquetLoader`.

| Goal                                | API                                                              |
| ----------------------------------- | ---------------------------------------------------------------- |
| Stream parquet as-is                | `index_*` → `StreamingDataset(..., item_loader=ParquetLoader())` |
| Convert / tokenize                  | `optimize` + `yield` from parquet (`README` `#reduce-memory`)    |
| Reshard huge files for map/optimize | `reader=ParquetReader(cache_folder, num_rows=65536)`             |

**Index**

```python
ld.index_parquet_dataset(uri, cache_dir=None, storage_options={}, num_workers=4)
cache = ld.index_hf_dataset("hf://datasets/org/name/data")  # returns local cache dir
```

| Scheme            | Index written to                               |
| ----------------- | ---------------------------------------------- |
| local             | beside files or `cache_dir`                    |
| `s3://` / `gs://` | uploaded to `{url}/index.json` (write access)  |
| `hf://`           | local cache (`index_hf_dataset` / `cache_dir`) |

Top-level `.parquet` only; uniform schema; index schemes today: local / s3 / gs / hf (**not** r2/azure).

**Stream**

| Source                    | Auto-index?                             | Auto `ParquetLoader`?       |
| ------------------------- | --------------------------------------- | --------------------------- |
| `hf://...`                | Yes (if no `index_path`)                | Yes                         |
| local / `s3://` / `gs://` | No — call `index_parquet_dataset` first | No — pass `ParquetLoader()` |

```python
# HF
ds = StreamingDataset("hf://datasets/org/name/data")

# S3 / local — explicit loader
ds = StreamingDataset("s3://bucket/pq", item_loader=ParquetLoader(low_memory=True))
# Wildcards if path ends with .parquet:
ds = StreamingDataset("s3://bucket/data/train-*.parquet", item_loader=ParquetLoader())

StreamingDataLoader(ds, num_workers=4, multiprocessing_context="spawn")  # required on Linux
```

| `ParquetLoader` arg | Default | Notes                                                         |
| ------------------- | ------- | ------------------------------------------------------------- |
| `low_memory`        | `True`  | Row-group path (pyarrow + polars). `False` = full file in RAM |
| `pre_load_chunk`    | `False` | Only effective when `low_memory=False`                        |

**`ParquetReader`** (`litdata.processing.readers`) for `map`/`optimize` `reader=` — splits oversized parquet inputs by `num_rows` into a cache folder; `fn` receives a `ParquetFile`.

README: `#stream-parquet`, `#stream-hf`. Internals: [streaming.md](streaming.md).

______________________________________________________________________

## 11. Combine / parallel / cycle

**`CombinedStreamingDataset`**

| Mode                              | Rule                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------- |
| `iterate_over_all=True` (default) | Exhaust **all** datasets. **Do not** pass `weights` (derived from lengths; `ValueError` if both). |
| `iterate_over_all=False`          | Stop when **any** dataset ends. Pass explicit `weights` for mixtures (TinyLlama-style).           |

```python
CombinedStreamingDataset(
    [ds_a, ds_b], seed=42, weights=(0.7, 0.3),
    iterate_over_all=False,
    batching_method="stratified",     # or "per_stream" (one source per batch)
    force_override_state_dict=False,
)
```

**`ParallelStreamingDataset`** — one sample from **each** dataset per step; optional `transform(samples)` or `transform(samples, rngs)` with `rngs["random"|"numpy"|"torch"]`.

```python
ParallelStreamingDataset([ds1, ds2], transform=fn, length=100)  # or float("inf") to cycle
# resume= / reset_rngs= control epoch restart + transform RNG
```

______________________________________________________________________

## 12. Encryption

```python
from litdata.utilities.encryption import FernetEncryption, RSAEncryption

enc = FernetEncryption(password="...", level="sample")  # or "chunk"
# enc = RSAEncryption(password="...", level="chunk")
optimize(..., encryption=enc)
enc.save("key.pem")
enc = FernetEncryption.load("key.pem", password="...")  # classmethod
StreamingDataset(..., encryption=enc)
```

- Levels: `"sample"` (per item) · `"chunk"` (whole chunk).
- Custom: subclass `Encryption` (`encrypt`/`decrypt`/`save`/`load`/`state_dict`/`algorithm`).
- **Not supported for MDS.** Needs `cryptography`.

______________________________________________________________________

## 13. Debug & profile

**DataLoader worker CPU** — `StreamingDataLoader(..., profile_batches=20, num_workers=4)` → viztracer `result.json` (§7 / README `#profile-loading`).

**LitData pipeline events** — `enable_tracer()` → `litdata_debug.log` → Litracer → Perfetto:

```python
from litdata.debugger import enable_tracer
enable_tracer()  # delete existing litdata_debug.log before re-trace
```

`litdata.breakpoint` — safe in optimize / DataLoader workers. Full knobs → [debugging.md](debugging.md).

______________________________________________________________________

## 14. Answering “how do I…?”

1. Pick §1 workflow.
2. Minimal recipe; add only needed knobs from §6–9.
3. Images → §3 (JPEG ~95, not PIL RAW).
4. Train → `StreamingDataLoader` + `shuffle`/`drop_last`/`seed`.
5. Disk/slow stream → §8 + cache doc.
6. Paths/Studio → §4 + [lightning-studio.md](lightning-studio.md).
7. Internals / races / benches → sibling reference files.
