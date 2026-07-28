# Path & URL resolver (`streaming/resolver.py`)

LitData does **not** require callers to know whether a dataset is local, S3, GCS, R2, Azure, Hugging Face, a Studio FUSE mount, or a networked drive. Pass a string (or `Dir`) into `StreamingDataset`, `StreamingRawDataset`, `optimize`, `map`, `merge_datasets`, etc. — **`_resolve_dir`** normalizes it to:

```python
@dataclass
class Dir:
    path: str | None = None              # local filesystem path (cache identity / local read)
    url: str | None = None               # remote scheme URL used for download/upload
    data_connection_id: str | None = None  # Lightning temp credentials for some connections
```

Import: `from litdata.streaming.resolver import Dir` (also re-exported via `streaming.cache`).

**Why customers care:** the same code works on a laptop (`./data`), a bucket (`s3://…`), or Lightning Studio (`/teamspace/s3_connections/…`) without rewriting I/O. Studio connection paths skip slow FUSE reads and hit the bucket directly.

Studio env / credentials deep-dive → [lightning-studio.md](lightning-studio.md). User cookbook → [using-litdata.md](using-litdata.md).

______________________________________________________________________

## Resolution order (`_resolve_dir`)

Given `dir_path: str | Path | Dir | None`:

| #   | Input                                      | Result                                                                                |
| --- | ------------------------------------------ | ------------------------------------------------------------------------------------- |
| 1   | Already a `Dir`                            | Copied (stringified path/url)                                                         |
| 2   | `None`                                     | Empty `Dir()`                                                                         |
| 3   | `s3://` `gs://` `r2://` `azure://` `hf://` | `Dir(path=None, url=<as-is>)`                                                         |
| 4   | `local:…`                                  | `Dir(path=None, url="local:…")` — networked / shared drive mode                       |
| 5   | String with `{strftime}`                   | Time template expanded (§ below), then continue                                       |
| 6   | `/teamspace/studios/this_studio…`          | **Local** `Dir(path=abs, url=None)`                                                   |
| 7   | `/teamspace/studios/<other>…`              | Studio content bucket URL (AWS `s3://` or GCP `gs://` via `LIGHTNING_CLOUD_PROVIDER`) |
| 8   | `/teamspace/s3_connections/<name>/…`       | `Dir(path, url=aws.source+suffix, data_connection_id?)`                               |
| 9   | `/teamspace/gcs_connections/<name>/…`      | `Dir(path, url=gcp.source+suffix)`                                                    |
| 10  | `/teamspace/s3_folders/<name>/…`           | `Dir(path, url=s3_folder.source+suffix)`                                              |
| 11  | `/teamspace/gcs_folders/<name>/…`          | `Dir(path, url=gcs_folder.source+suffix)`                                             |
| 12  | `/teamspace/lightning_storage/<name>/…`    | `Dir(path, url=r2.source+suffix, data_connection_id=id)`                              |
| 13  | `/teamspace/datasets/…`                    | Cluster datasets bucket under `s3://…/projects/{id}/datasets/…`                       |
| 14  | Anything else                              | Absolute local path `Dir(path=abs, url=None)`                                         |

Relative paths are resolved with `Path(…).absolute().resolve()`.

Connection / studio **name** is path segment index `[3]`
(e.g. `/teamspace/s3_connections/my-bucket/train` → connection `my-bucket`, suffix `train`).

______________________________________________________________________

## Cloud URI schemes (anywhere)

| Scheme     | Example                       | Notes                                                                                            |
| ---------- | ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `s3://`    | `s3://my-bucket/data`         | AWS S3 (obstore/boto3). `storage_options` / `session_options` supported                          |
| `gs://`    | `gs://my-bucket/data`         | Google Cloud Storage                                                                             |
| `r2://`    | `r2://my-bucket/data`         | Cloudflare R2                                                                                    |
| `azure://` | `azure://container/data`      | Azure Blob                                                                                       |
| `hf://`    | `hf://datasets/org/name/data` | Hugging Face (parquet index/stream)                                                              |
| `local:`   | `local:/mnt/nfs/dataset`      | Treat as remote-style source but on a mount; LitData still caches chunks locally to cut NAS load |

```python
from litdata import StreamingDataset, optimize

StreamingDataset("s3://bucket/optimized")
StreamingDataset("gs://bucket/optimized", storage_options={"project": "..."})
StreamingDataset("r2://bucket/optimized", storage_options={...})
StreamingDataset("azure://container/optimized", storage_options={...})
StreamingDataset("hf://datasets/org/name/data")
StreamingDataset("local:/data/shared-drive/some-data")
```

Same schemes work for `optimize(..., output_dir=...)`, `map(..., output_dir=...)`, `merge_datasets`, `StreamingRawDataset`, etc.

______________________________________________________________________

## Explicit `Dir` (cache ≠ remote)

When the local cache directory and the remote dataset URL must differ:

```python
from litdata.streaming.resolver import Dir
from litdata import StreamingDataset

dataset = StreamingDataset(
    Dir(path="/fast-ssd/cache/my-run", url="s3://bucket/optimized-dataset")
)
# Or equivalently:
StreamingDataset("s3://bucket/optimized-dataset", cache_dir="/fast-ssd/cache/my-run")
```

- **`path`** — local folder for cached `.bin` / index materialization.
- **`url`** — where bytes are downloaded/uploaded.
- **`data_connection_id`** — set automatically for some Studio connections (R2 / lightning_storage / cross-cloud S3); enables temporary project-role credentials. Users rarely set this by hand.

______________________________________________________________________

## Lightning Studio `/teamspace/…` paths

Inside [Lightning Studios](https://lightning.ai/), teamspace folders look like a normal filesystem but LitData **resolves them to the backing bucket** and streams/uploads directly. That is much faster than reading through FUSE for every chunk.

### Full prefix table

| Prefix                                  | Resolved to                                                                         | Credentials                                                    |
| --------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `/teamspace/studios/this_studio/…`      | Local workspace disk only (`url=None`)                                              | N/A                                                            |
| `/teamspace/studios/<studio_name>/…`    | That studio’s content bucket (`s3://` or `gs://` …/cloudspaces/{id}/code/content/…) | Studio / cluster env                                           |
| `/teamspace/s3_connections/<conn>/…`    | `data_connection.aws.source` + suffix                                               | Ambient AWS, or temp creds if `available_in_non_aws_providers` |
| `/teamspace/gcs_connections/<conn>/…`   | `data_connection.gcp.source` + suffix                                               | GCP connection                                                 |
| `/teamspace/s3_folders/<conn>/…`        | `data_connection.s3_folder.source` + suffix                                         | Folder connection                                              |
| `/teamspace/gcs_folders/<conn>/…`       | `data_connection.gcs_folder.source` + suffix                                        | Folder connection                                              |
| `/teamspace/lightning_storage/<conn>/…` | `data_connection.r2.source` + suffix                                                | Always `data_connection_id` (temp creds)                       |
| `/teamspace/datasets/…`                 | `s3://{cluster_bucket}/projects/{project_id}/datasets/…`                            | Studio env (`LIGHTNING_CLOUD_SPACE_ID`, …)                     |

Also referenced in platform docs / jobs (not all go through the same `_resolve_dir` branches): `/teamspace/jobs/…` for multi-node optimize outputs.

### Customer examples

```python
from litdata import StreamingDataset, StreamingRawDataset, optimize, map

# Stream an optimized dataset attached as an S3 connection (direct bucket I/O)
ds = StreamingDataset("/teamspace/s3_connections/imagenet/optimized")

# Raw files on a connection
raw = StreamingRawDataset("/teamspace/s3_connections/my-bucket/raw-images")

# Optimize *into* a connection (upload goes to the bucket, not only FUSE)
optimize(fn=..., inputs=..., output_dir="/teamspace/s3_connections/my-data/v1", chunk_bytes="64MB")

# GCS / Lightning storage / datasets mount
StreamingDataset("/teamspace/gcs_connections/my-gcs/data")
StreamingDataset("/teamspace/lightning_storage/team-store/shards")
optimize(..., output_dir="/teamspace/datasets/my-llm-tokens")

# This studio workspace = local disk (great for scratch; not a cloud URL)
optimize(..., output_dir="/teamspace/studios/this_studio/artifacts/run-01")
```

### Required Studio environment (auto-injected in Studios)

| Env                          | Used for                                       |
| ---------------------------- | ---------------------------------------------- |
| `LIGHTNING_CLOUD_PROJECT_ID` | List connections / studios                     |
| `LIGHTNING_CLUSTER_ID`       | Studio + datasets resolve                      |
| `LIGHTNING_CLOUD_SPACE_ID`   | `/teamspace/datasets`                          |
| `LIGHTNING_CLOUD_PROVIDER`   | `aws` or GCP scheme for other-studio content   |
| `LIGHTNING_CLOUD_URL`        | Control plane (default `https://lightning.ai`) |

Outside Studio, prefer `s3://` / `gs://` + normal cloud credentials. Resolving `/teamspace/s3_connections/…` without those env vars raises `RuntimeError` / `ValueError`.

______________________________________________________________________

## Date/time path templates

Any path string containing `{…strftime…}` is expanded with **now** before further resolution:

```python
# On 2025-05-05 → ".../log_2025-05-05" or output_dir with that date
optimize(..., output_dir="/teamspace/s3_connections/my-data/run_{%Y-%m-%d}")
StreamingDataset("./snapshots/ds_{%Y%m%d_%H%M}")
```

Pattern is the substring inside `{` `}` passed to `datetime.strftime`.

______________________________________________________________________

## Immutability checks (write path)

When `output_dir` resolves to a **remote** URL, optimize/map assert emptiness / index rules via the resolver helpers:

- Existing data without `mode=` → error (“datasets are meant to be immutable”); hint to version the suffix or use `mode='append'|'overwrite'`.
- `data_connection_id` is merged into `storage_options` for FsProvider deletes/listings (R2 / lightning_storage).

Agents and users: always version remote output prefixes (`…/v2`, `…/run_{%Y-%m-%d}`) unless intentionally appending.

______________________________________________________________________

## Where resolution runs

Any API that takes an input/output directory goes through `_resolve_dir`, including:

- `StreamingDataset` / `StreamingRawDataset` (`input_dir`, `cache_dir`)
- `optimize` / `map` / `merge_datasets` (`output_dir`, `input_dir`)
- Index helpers that accept cloud URIs
- Cache identity / downloaders / uploaders that consume `Dir`

______________________________________________________________________

## Agent / expert checklist

1. Prefer **connection paths** in Studio (`/teamspace/s3_connections/…`) over inventing buckets when the data is already attached.
2. Prefer **`s3://` + creds** outside Studio; don’t require Lightning env vars.
3. Use **`Dir(path, url)`** or `cache_dir=` when cache disk ≠ dataset location.
4. Use **`local:`** for NFS/shared drives so chunk caching reduces network thrash.
5. Use **`{%Y-%m-%d}`** (or similar) to version outputs.
6. Never tell users to “just read the FUSE path with open()” for training I/O — LitData’s resolver + downloader is the fast path.
7. Document **all** schemes (`s3/gs/r2/azure/hf/local`) and **all** teamspace prefixes when answering path questions — customers often only know S3.
