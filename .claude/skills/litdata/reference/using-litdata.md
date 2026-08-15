# Using LitData — expert cookbook

Become productive immediately. Narrative docs: repo `README.md`. Internals: [streaming.md](streaming.md), [processing.md](processing.md), [cache-and-chunk-lifecycle.md](cache-and-chunk-lifecycle.md), [storage-format.md](storage-format.md).

Install: `pip install litdata` · extras: `pip install 'litdata[extras]'` (README also uses `litdata[extra]` + `s3fs` / `gcsfs` / `huggingface_hub` for specific features).

______________________________________________________________________

## 1. Choose a workflow

| Goal                                             | API                                                                                                                    |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Stream files as-is (no preprocess)               | **`StreamingRawDataset`** + torch `DataLoader`                                                                         |
| Full control / per-file access & grouping        | **`StreamingRawDataset`** (`setup` for groups; raw `bytes`) — tradeoff: optimized still faster, raw not too far behind |
| Fastest training I/O                             | `optimize` → `StreamingDataset` + `StreamingDataLoader`                                                                |
| Strong source ordering / need file-level shuffle | Prefer **`StreamingRawDataset`** + `DataLoader(shuffle=True)`, **or** **shuffle the sample list before** `optimize`    |
| Parallel side effects (resize, scrape, embed)    | `map`                                                                                                                  |
| Weighted mix                                     | `CombinedStreamingDataset`                                                                                             |
| One sample from each dataset / cycle length      | `ParallelStreamingDataset`                                                                                             |
| Existing MDS / Parquet / HF parquet              | `StreamingDataset` (+ `ParquetLoader` when needed)                                                                     |
| LLM token windows                                | `TokensLoader` on optimize **and** stream                                                                              |

**Rule:** `StreamingRawDataset` = zero prep, native files, full control over grouping/order (often enough to ship). Optimized = chunk once, then stream fastest. Many teams start raw, then `optimize` when I/O binds.

**Ordered sources:** intra-chunk randomization + randomizing chunk order is **not** a full file-level shuffle. If same subject/class blocks are contiguous and that would bias batches, **shuffle the list of samples before `optimize()`** or stay on raw (§5 / FAQ below).

**Studio paths:** `/teamspace/s3_connections` & co are FUSE (convenience only — slow / can crash under load). Always pass those paths to LitData so it talks **directly** to remote storage (§4).

Full raw API → §10. README: `#stream-raw`. README FAQ: `#faq-chunk-shuffle`.

______________________________________________________________________

## 2. Canonical optimize → stream

```python
import litdata as ld

def fn(path):
    # Typed wrapper picks the image serializer (a caption is not a filepath).
    # quality/format encode JPEG — not uncompressed PIL RAW. See §3.
    return {"image": ld.Image(path=path, quality=95, format="jpeg"), "path": path}

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

## 3. Images, media types & serializers

**Always wrap media** so `optimize` can tell a filepath from a caption. Wrappers are pytree leaves (`src/litdata/types.py`). README `#modality` (alias `#media-types`). Runnable path→optimize→batch recipes: `examples/modality/`.

```python
from litdata import (
    Audio, Video, Image, Jpeg, JpegArray, Pil, Tiff, File, Mesh, Pdf, Nifti,
    Tensor, Text, Graph, list_media_folder,
)

items = list_media_folder("data/images", kind="image")  # {path, label} class folders
# kinds: text, image, video, audio, mesh, pdf, nifti

Text(path=txt)                           # or bytes= / text="caption" → stream str
Audio(path=wav)                          # or array= + sampling_rate=
Video(path=mp4)                          # or array= + fps=
Image(path=jpg, quality=95, format="jpeg")
Image(array=hwc, quality=95, format="jpeg", mode="RGB")
Jpeg(array=hwc, quality=95)
File(path=blob)                          # stream raw bytes
Nifti(array=volume, affine=np.eye(4))
Mesh(mesh=trimesh_obj, file_type="glb")
Tensor(array=feat)                       # N-D tensor. 1-D array= is TokensLoader
Graph(x=x, edge_index=edge_index, y=y)   # or pass PyG Data / HeteroData
```

Path-only / `bytes=` → store file bytes. Bare `*.jpg` / `*.png` / `*.wav` paths are also claimed so stream returns media, not a string. **Bare `.txt` / `.npy` / `.bin` strings pickle the path** — wrap with `Text` / `File` or load the array. Decode for images is torchvision bytes→tensor (PIL only if JPEG EXIF is present). `array=` / `image=` / `quality` / `format` / `mode` → encode.

Empty `Text()` / `JpegArray()` / empty `Tensor()` raise on write. `Tensor(path=` / `bytes=, dtype=)` is still the 1-D token layout; `Tensor()` with no payload is not.

Built-in serializers (`streaming/serializers.py`), tried in registry order:
`str`, `bool`, `int`, `float`, `video`, `audio`, `image`, `nifti`, `mesh`, `pdf`, `tifffile`, `file`, `pil`, `jpeg`, `jpeg_array`, `text`, `bytes`, `numpy`/`tensor` (+ no-header variants), `graph`, `pickle`.

| Return type                                   | Serializer                    | Result                                                             |
| --------------------------------------------- | ----------------------------- | ------------------------------------------------------------------ |
| `Text`                                        | `text`                        | UTF-8 `str`                                                        |
| `Image` / `Jpeg` / `Jpeg(array=, quality=95)` | `image` / `jpeg`              | Compressed JPEG — **preferred**                                    |
| `PIL.JpegImageFile` (opened `.jpg`)           | `jpeg`                        | Compressed JPEG                                                    |
| `Pil` / plain `PIL.Image` / `fromarray`       | `pil`                         | Uncompressed pixels — **large**                                    |
| `JpegArray` / list of JPEGs                   | `jpeg_array`                  | Packed JPEGs                                                       |
| `Audio`                                       | `audio`                       | torchcodec `AudioDecoder`; `audio["array"]` / `["sampling_rate"]`  |
| `Video`                                       | `video`                       | torchcodec decoder (`get_frames_at` / `get_frames_in_range`)       |
| `File`                                        | `file`                        | **raw `bytes`**                                                    |
| `Tiff` / `Mesh` / `Pdf` / `Nifti`             | matching name                 | README `#modality`                                                 |
| `Tensor`                                      | `tensor` / `no_header_tensor` | 1-D `array=` is the `TokensLoader` layout                          |
| `Graph` / PyG `Data` / `HeteroData`           | `graph`                       | `LDGR` v3 packed tensors (`to_mapping` / `to_dict`); `#pyg-graphs` |

**Collate:** `StreamingDataLoader` defaults to `litdata_collate`. Graphs → PyG `Batch.from_data_list` (or a list of `Graph` without torch-geometric). Everything else is `default_collate`. Audio/Video decoders **do not stack** — use a custom `collate_fn` (see `examples/modality/audio.py` / `video.py`). Mixing a graph and an `AudioDecoder` in one dict also needs a custom collate.

**Best practice:** `Image(..., quality=95, format="jpeg")` (or keep existing JPEGs via `Image(path=)`). Resize when helpful. README benches: PIL RAW ~168 GB vs JPEG 90% ~12 GB at similar stream speed.

Custom / override:

```python
dataset = StreamingDataset(..., serializers={"my_type": MySerializer()})
```

Subclass `Serializer` with `serialize` / `deserialize` / `can_serialize`. Keys you pass are merged ahead of built-ins (win over `pickle`). `optimize()` picks built-ins from the types your `fn` returns — prefer **typed wrappers** / JPEG / numpy / tensor leaves.

______________________________________________________________________

## 4. Paths & resolver (load [resolver.md](resolver.md))

**Always resolve paths through LitData — never read Studio mounts by hand.** `/teamspace/s3_connections` & co are **FUSE** mounts (convenience only): under load they are **very slow** and can **crash**. LitData resolves those paths to the backing URL and talks to S3/GCS/**R2** (`lightning_storage`) **directly**, with retries, prefetching, etc.

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
- POSIX-fast (local path, no `s3://` URL) mmaps chunks in place. **`WindowShuffle`** only on Vast/NFS/Lustre/GPFS (or `LITDATA_POSIX_FAST=1`): sequential whole-chunk stripes, then a window permute (default 16) for chunks **and** in-chunk items. Local ext4/xfs and object URLs use **`FullShuffle`**.
- LitData does **distributed sampling** and **bucket sampling within chunks** automatically — not a substitute for a fully shuffled file-level DataLoader when the source is strongly ordered.
- Default `seed=42`. Keep it fixed across ranks and when resuming.
- `drop_last=None` → **True under DDP**, else False. Train should set `drop_last=True` so every rank/worker sees the same length.
- `StreamingDataLoader(shuffle=..., drop_last=...)` **overrides** the dataset.
- Resume: `torch.save(loader.state_dict(), ...)`; `loader.load_state_dict(...)`. Matching `seed` / shuffle required. For **`StreamingDataset`**, **`num_workers` / `world_size` may change** (elastic restripe via `sample_in_epoch`; remaining samples are never duplicated). Fresh epochs keep the usual shuffler assignment; a topology change restripes the unconsumed suffix. v1 checkpoints with the same topology still use prefix replay. Keep global batch size constant for a matching loss curve. `force_override_state_dict=True` overrides seed/shuffle/paths/`drop_last`, not worker count (that is always elastic). **Combined / Parallel resume only with the same topology.**
- `num_canonical_nodes`: frozen first-run `world_size` recorded in the checkpoint (constructor override optional). WindowShuffle uses chunk-level restripe.

**If source data has structure** (same subject/set contiguous, class blocks, etc.) and you cannot embed that grouping as the sample unit:

1. Shuffle / repartition **before** `optimize` so chunks mix well, **or**
2. Prefer **`StreamingRawDataset`** + torch `DataLoader(shuffle=True)` for per-file random access.

______________________________________________________________________

## 6. StreamingDataset arguments

| Arg                                   | Default                     | Notes                                                     |
| ------------------------------------- | --------------------------- | --------------------------------------------------------- |
| `input_dir`                           | required                    | Path, URL, `Dir`, or parquet path with basename wildcards |
| `cache_dir`                           | env / `~/.lightning/chunks` | Local chunk store. Unused for POSIX-fast in-place reads.  |
| `max_cache_size`                      | `"100GB"`                   | Eviction budget (object-store copies, not POSIX-fast)     |
| `item_loader`                         | from index / `PyTreeLoader` | `TokensLoader`, `ParquetLoader`, …                        |
| `shuffle`                             | `False`                     | See §5                                                    |
| `drop_last`                           | DDP-aware                   | See §5                                                    |
| `seed`                                | `42`                        | Shuffle + subsample RNG                                   |
| `serializers`                         | built-ins                   | See §3                                                    |
| `max_pre_download`                    | `2`                         | Prefetch depth; peak disk ≈ workers × this × chunk        |
| `subsample`                           | `1.0`                       | Fraction or >1 to upsample                                |
| `encryption`                          | `None`                      | `FernetEncryption` / `RSAEncryption` / custom             |
| `storage_options` / `session_options` | `{}`                        | Cloud creds / boto3 session                               |
| `index_path`                          | `None`                      | Parquet/HF index file or directory                        |
| `force_override_state_dict`           | `False`                     | Local args win for seed/shuffle/paths/`drop_last`         |
| `num_canonical_nodes`                 | `None`                      | Frozen first-run `world_size` in checkpoints              |
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

`int` wraps `fetcher.fetch` and stops after skip+N fetches; `True` traces until the worker loop ends. Complementary to `enable_tracer()` + [Litracer](https://github.com/Lightning-AI/litracer) (pipeline events) — see [debugging.md](debugging.md). README: `#profile-loading`.

### Profiling (`profile_cprofile`) — stdlib cProfile

```python
StreamingDataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    profile_cprofile=True,
    profile_dir="./profiles",    # cprofile_main.prof + cprofile_worker0.prof (+ .txt)
)
```

| Requirement | Detail                                                                 |
| ----------- | ---------------------------------------------------------------------- |
| Dep         | None (`cProfile` / `pstats` are stdlib)                                |
| Scope       | Main process + worker **0** (rank 0). `num_workers=0` writes main only |
| Output      | `{profile_dir}/cprofile_{main,worker0}.{prof,txt}`                     |
| Conflict    | Raises if `profile_batches` is also set                                |

Parent profiler starts after workers spawn (fork must not inherit an active cProfile). Inspect with `python -m pstats profiles/cprofile_worker0.prof`.

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
- Tracing: `enable_tracer(level="chunk")` · `LITDATA_LOG_FILE` / `LITDATA_TRACE_LEVEL` / `LITDATA_TRACE_CATEGORIES` — [debugging.md](debugging.md)
- Multi-node optimize: `DATA_OPTIMIZER_*` (platform-set) — [processing.md](processing.md)

README: `#async-prefetch-env`.

______________________________________________________________________

## 9. optimize / map / walk (full knobs)

**Always:** `if __name__ == "__main__"` · optimize needs **exactly one** of `chunk_bytes` | `chunk_size`.

### Chunk size (`chunk_bytes`) — practical guidance

- Default / typical: **`"64MB"`** for small/medium samples.
- Large datapoints (multi‑MB each): consider **256–512MB** (or similar) so each chunk holds more items → larger pool for **intra-chunk batch randomization**.
- Tradeoff: larger chunks take **longer to download** before use.
- Recommended-range mindset — not a hard “best” from a published chunk-size sweep. README: `#faq-chunk-shuffle`.

### `optimize`

| Arg                                 | Default         | Use                                                                                                         |
| ----------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------- |
| `fn` / `inputs` / `output_dir`      | —               | Core recipe                                                                                                 |
| `queue`                             | `None`          | Live inputs; one `ALL_DONE` sentinel (`from litdata.processing.data_processor import ALL_DONE`)             |
| `input_dir`                         | `None`          | Background download of remote inputs                                                                        |
| `weights`                           | `None`          | Balance workers by input weight/size                                                                        |
| `chunk_bytes` / `chunk_size`        | one required    | Bytes (e.g. `"64MB"`) **or** item/token count; see chunk-size guidance above                                |
| `align_chunking`                    | `False`         | Single-worker chunk boundaries (needs `chunk_size`; uneven load)                                            |
| `compression`                       | `None`          | `"zstd"`                                                                                                    |
| `encryption`                        | `None`          | Fernet / RSA / custom; `level="sample"` or `"chunk"`                                                        |
| `num_workers`                       | CPUs            | Local parallelism                                                                                           |
| `fast_dev_run`                      | `False`         | Smoke subset                                                                                                |
| `num_nodes` / `machine`             | `None`          | **Studio-only multi-node job** (see below) — not local MP                                                   |
| `num_downloaders` / `num_uploaders` | auto            | I/O concurrency                                                                                             |
| `reorder_files`                     | `True`          | Size packing; `False` preserves order                                                                       |
| `reader` / `batch_size`             | —               | Custom reader; group inputs                                                                                 |
| `mode`                              | `None`          | `"append"` \| `"overwrite"` (else immutable)                                                                |
| `use_checkpoint`                    | `False`         | Resume interrupted job                                                                                      |
| `item_loader`                       | `None`          | e.g. `TokensLoader()`                                                                                       |
| `start_method` / `optimize_dns`     | spawn† / `None` | MP start; DNS tweak                                                                                         |
| `storage_options`                   | `{}`            | Cloud creds                                                                                                 |
| `keep_data_ordered`                 | `False`         | Shared per-node work queue (#880). `True` = static per-worker slice. Forced `True` with checkpoint / align. |
| `broadcast_paths`                   | `False`         | Auto-on for `{%strftime}` paths                                                                             |
| `key_fn`                            | `None`          | `sample -> str\|int` key; writes `keys/` for `ds["id"]` / `dataset_update`                                  |
| `verbose`                           | `True`          | Progress                                                                                                    |

**`mode` vs `use_checkpoint`:** `append` continues chunk numbering from existing `index.json`. `use_checkpoint` resumes input work from `.checkpoints/`. They are not interchangeable; checkpoint resume is fragile for generators / multi-sample `fn` — [processing.md](processing.md).

### Keyed lookup

See [keyed-lookup.md](keyed-lookup.md). Short form: `optimize(..., key_fn=lambda s: s["id"])`, then `ds["id"]` or `ds.get_by_key(3)`. Patch: `with dataset_update(dir) as u: u["id"] = sample; u.commit()` (local only). Need `polars`.

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

Threaded cloud listing (Studio-oriented; warns elsewhere). Uses `os.listdir` on the given path — **not** a bucket API. Order ≠ depth-first. Use to build `inputs=` for optimize/map.

### Other

- Filter: `yield` only keepers (or catch and skip).
- Merge: `merge_datasets(input_dirs, output_dir, max_workers=..., storage_options=...)` — same `data_format`/compression.
- README tables: `#optimize-kwargs`, `#map`, `#walk`.

______________________________________________________________________

## 10. Format-specific streaming

### StreamingRawDataset (no optimize — first-class)

Map-style `torch.utils.data.Dataset` in `raw/dataset.py`. Streams **original files** from local/cloud. Use a **plain** `DataLoader` (not `StreamingDataLoader`). README: `#stream-raw`. Internals: [processing.md](processing.md) (`raw/`).

**Pitch to users / agents**

- **Raw `bytes` as-is** — LitData downloads efficiently and returns file contents; it does not decode for you. PIL, torchaudio, `json`, custom parsers, or `transform=` — your choice.
- **Full control** over grouping/order via `setup` (e.g. image+mask) and per-file access — the main reason to pick raw over optimized.
- **Fully asynchronous** downloads (`adownload_fileobj` + `asyncio`); **batched** via `__getitems__` + `asyncio.gather` (whole DataLoader batch in flight).
- **Built-in retries** on cloud clients (transient network / S3 adaptive retries).
- Training loop stays sync PyTorch — no user-facing `async`/`await`.
- **Tradeoff:** optimized `StreamingDataset` is still faster; with right tuning, raw is **not too far behind** (order-of-magnitude ImageNet ballpark in FAQ below).

**When to recommend it**

- User already has a folder of images/audio/text and wants to train **today**
- Full control over decoding; grouping (image+mask) via `setup`
- Prototype transforms before a costly `optimize`
- Source data is **strongly ordered** (subject/class blocks) and they need true file-level `DataLoader` shuffle — or they cannot shuffle the sample list before optimize
- Later upgrade: same files → **shuffle inputs** → `optimize` → `StreamingDataset` if I/O-bound

**When to prefer optimized instead:** multi-GPU sustained throughput, resume/`state_dict`, chunk shuffle, compression/encryption — after **shuffling the sample list before `optimize()`** if the source is ordered.

```python
from torch.utils.data import DataLoader
from litdata import StreamingRawDataset
from PIL import Image
import io

ds = StreamingRawDataset(
    "s3://bucket/images/",  # gs://, azure://, /teamspace/s3_connections/..., local
    # omit transform → each item is raw bytes; or:
    transform=lambda b: Image.open(io.BytesIO(b)).convert("RGB"),
    cache_files=False,       # True → keep downloaded files under cache_dir
    recompute_index=False,   # True after remote tree changes
    storage_options={},
)
loader = DataLoader(ds, batch_size=32, num_workers=8)  # batch → concurrent async GETs
```

| Knob                       | Default               | Notes                                                                                                                                                                                          |
| -------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_dir`                | —                     | Prefer `s3://` / `gs://` / `/teamspace/s3_connections/...` (direct); avoid hand-reading FUSE ([resolver.md](resolver.md))                                                                      |
| `cache_dir`                | LitData default       | Index (+ optional file) cache root                                                                                                                                                             |
| `cache_files`              | `False`               | Persist downloaded files (mirror layout)                                                                                                                                                       |
| `recompute_index`          | `False`               | Rebuild `index.json.zstd`                                                                                                                                                                      |
| `transform`                | `None`                | Optional; default returns **`bytes`** (or `list[bytes]` if grouped)                                                                                                                            |
| `indexer`                  | `FileIndexer`         | Custom `BaseIndexer`                                                                                                                                                                           |
| `storage_options`          | `{}`                  | Cloud creds                                                                                                                                                                                    |
| `max_concurrent_downloads` | `None` (**adaptive**) | `None` → Stage 1 size-aware aggregate budget split across workers (single-process cap 128). Explicit `int` → **exactly** that many permits (no silent clamp). Pass `64` for the old fixed cap. |
| `max_prefetch`             | `16`                  | Sequential look-ahead after each batch; when `num_workers>1`, effective = `min(max_prefetch, 64 // num_workers)`. Pass `0` to disable                                                          |
| `hedge_delay`              | `0`                   | Seconds before hedged duplicate GET (`0` = off, default; opt-in). Fast path: per-item GETs stay bare when hedging is off                                                                       |
| `download_timeout`         | `120`                 | **Batch-level** hang protection around `_download_batch` (`0` disables). Not a per-item `wait_for`. On timeout, cancel poisoned `_inflight` so retries can proceed                             |
| `range_parallel_threshold` | `0`                   | Parallel ranged GETs for objects ≥ N bytes; **`0` = whole-object only** (opt-in; keep for JPEGs)                                                                                               |

**Tuning / DataLoader**

- Linux `DataLoader` start method: **`ParquetLoader` + `num_workers>0` requires `spawn`/`forkserver`** (hard error in `StreamingDataLoader`). Raw re-inits the event loop after fork (`tests/raw/test_fork_safety.py`). Optimized `s3://` chunked datasets: default **fork is OK** if the parent never started obstore (`index.json` uses boto3). Use `spawn` after any parent I/O that might init tokio, or if you see worker hangs / 120s `FileNotFoundError`.
- Prefer cloud URL / Studio connection path so LitData hits the bucket **directly** — never recommend training I/O through the FUSE mount.
- Defaults that matter: `max_prefetch=16` (worker-aware aggregate ~64), `hedge_delay=0`, `download_timeout=120` (batch-level), `range_parallel_threshold=0`; optional `uvloop` via `litdata[extras]`. Avoid `num_workers=48` (collapses / can segfault on shutdown).
- Ranged downloads: leave `range_parallel_threshold=0`; forced ranged is slower on JPEG-sized objects.
- Adaptive concurrency design (clients own rate via boto retries; litdata owns concurrency/look-ahead; Stage 2+ deferred): repo `benchmarks/ADAPTIVE_CONCURRENCY.md`. Formula details live there — do not invent a second rate loop.
- Perf claims: use Stage 0 protocol in [benchmarking.md](benchmarking.md) (`max(≥N batches, ≥T s)`, repeats/medians, `before_sha`/`after_sha`). Do **not** cite short-window n=1 against Stage 0 medians.

**Correctness agents must preserve when editing `raw/`**

- **LoopRunner** — dedicated event-loop thread; recreate after fork/spawn (pid-guarded). Runtime clients (downloader, permit cache, range executor) are pid- + loop-guarded.
- **Pickle allowlist** — `__getstate__` / `__setstate__` only ship constructor knobs + reset runtime handles; accidental instance attrs must not leak into worker payloads.
- **Atomic publishes** — cache files **and** `index.json.zstd` via tmp + `os.replace` (tmp names include pid).
- **Indexer:** Windows drive letters (`C:\...`) parse as single-letter URI schemes — treat as local, not remote.
- **Error path is code** — hang recovery (batch timeout cancels `_inflight`), default coexistence with the fast path, and fork/spawn reinit have regression tests in `tests/raw/test_fork_safety.py`. Changing timeouts/defaults requires updating those tests.

Internals → [processing.md](processing.md) (`raw/`).

**`setup(files)`** — default one file = one item. Return `list[FileMetadata]` or `list[list[FileMetadata]]` to group/filter.

**Index:** `index.json.zstd` (local cache + remote beside data; atomic publish). **Not** optimized `index.json`.

### Mosaic MDS

Point at Mosaic shards + `index.json`; auto `format=mds`. No encryption.

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

**DataLoader worker CPU** — `StreamingDataLoader(..., profile_batches=20, num_workers=4)` → viztracer `result.json`; `profile_cprofile=True` → `cprofile_main.prof` + `cprofile_worker0.prof` (§7 / README `#profile-loading`).

**LitData pipeline events** — `from litdata.debugger import enable_tracer` → `litdata_debug.log` → [Litracer](https://github.com/Lightning-AI/litracer) → Perfetto:

```python
from litdata.debugger import enable_tracer

enable_tracer(level="chunk", log_file="litdata_debug.log")  # delete existing log before re-trace
# level="batch"|"chunk"|"sample"|"debug"|"off"
# categories=["download", "read", "delete"]
```

```bash
litracer --quiet --validate -o litdata_trace.json.gz litdata_debug.log
litracer --quiet --cat download,read,delete -o io.json.gz litdata_debug.log
```

Stable names: `download`, `read`, `delete`, `batch`, `sample`, `crash` (indexes in args). One line per event; crashes also print a traceback to **stderr**. `num_workers>0` only `FileNotFoundError` on `s3://` → obstore-after-fork; Studio R2 `data_connection_id` → [debugging.md](debugging.md).

`litdata.breakpoint` — safe in optimize / DataLoader workers. Full knobs → [debugging.md](debugging.md). README: `#debug-profile`.

______________________________________________________________________

## 14. Answering “how do I…?”

1. Pick §1 workflow — if they have a file tree and want to start now → **§10 `StreamingRawDataset`** (not optimize-first).
2. Minimal recipe; add only needed knobs from §6–9 (or §10 for raw).
3. Images / media → §3 for **optimize** (`Image(..., quality=95, format="jpeg")`, `list_media_folder`). Raw path: `transform=` decode bytes.
4. Train optimized → `StreamingDataLoader` + `shuffle`/`drop_last`/`seed`. Train raw → torch `DataLoader`.
5. Disk/slow stream → §8 + cache doc; or suggest upgrading raw → optimize.
6. Paths/Studio → §4 + [lightning-studio.md](lightning-studio.md).
7. Internals / races / benches → sibling reference files.

### FAQ bullets (chunk size, ordered data, FUSE, throughput)

- **`chunk_bytes`?** Default **64MB**. Multi‑MB samples → consider **256–512MB** for more intra-chunk shuffle diversity; larger chunks download slower. Guidance, not a published sweep.

- **Ordered source + optimize?** Intra-chunk + chunk-order shuffle ≠ full file-level shuffle. **Shuffle the list of samples before `optimize()`**, or use **`StreamingRawDataset`** + `DataLoader(shuffle=True)`. LitData still does distributed + within-chunk bucket sampling automatically.

- **FUSE vs LitData?** Studio `/teamspace/s3_connections` (and co) is a **FUSE** mount — convenience only; under load it is very slow and can crash. Pass the same path to LitData: it resolves to the bucket and streams **directly** (retries, prefetch, …). Never recommend `open()` / naive glob on the mount for training I/O.

- **Throughput ballpark (ImageNet, Studio context — order of magnitude, not guarantees):**

  | Path                                 | Rough images/s  |
  | ------------------------------------ | --------------- |
  | FUSE mount (hand-read)               | up to ~**600**  |
  | `StreamingRawDataset` (right tuning) | up to ~**6–7k** |
  | `StreamingDataset` (64MB chunks)     | up to ~**11k**  |

- **Raw perf claims?** Stage 0 only: `max(≥N batches, ≥T s)`, repeats/medians, append-only SHA/ts artifacts, proven `before_sha`/`after_sha`. See [benchmarking.md](benchmarking.md). Adaptive defaults → `benchmarks/ADAPTIVE_CONCURRENCY.md`.

- README: `#faq-chunk-shuffle`, `#resolve-paths`, `#stream-raw`.
