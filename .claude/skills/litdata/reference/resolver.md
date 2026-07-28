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

**Why this exists:** the same code works on a laptop (`./data`), a bucket (`s3://…`), or Lightning Studio (`/teamspace/s3_connections/…`) without rewriting I/O.

### Studio mounts are FUSE — LitData bypasses them

In Lightning Studio, paths under `/teamspace/s3_connections`, `gcs_connections`, `s3_folders`, `gcs_folders`, `lightning_storage`, `datasets`, and other-studio content are **FUSE mounts**: they look like a normal filesystem (`ls`, `open`) but every read/write goes through a userspace filesystem into object storage.

That is convenient for browsing, and **a poor path for training I/O**:

| Path                                                                                           | What happens                                                                                                                                    | Reliability / speed                              |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `open("/teamspace/s3_connections/…/chunk.bin")` or plain `DataLoader` on the mount             | Bytes travel **Studio → FUSE → object store**                                                                                                   | Slower, more brittle under many parallel workers |
| `StreamingDataset("/teamspace/s3_connections/…")` / `optimize(..., output_dir="/teamspace/…")` | Resolver fills `Dir.url` with the **backing store URL**; downloaders / `FsProvider` talk **directly** to S3 / GCS / R2 (temp creds when needed) | Faster and more reliable                         |

So: **pass the `/teamspace/...` path into LitData** — do not treat the FUSE mount as the training filesystem. LitData keeps `path` for local identity/cache hints and uses `url` for real I/O.

**Backing stores (what `url` becomes):**

| Mount prefix                            | Under the hood                                                                                                                |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `/teamspace/s3_connections/<name>/…`    | Customer **S3** bucket (`data_connection.aws.source`)                                                                         |
| `/teamspace/gcs_connections/<name>/…`   | Customer **GCS** bucket                                                                                                       |
| `/teamspace/s3_folders/<name>/…`        | S3 **folder** connection                                                                                                      |
| `/teamspace/gcs_folders/<name>/…`       | GCS **folder** connection                                                                                                     |
| `/teamspace/lightning_storage/<name>/…` | Lightning-managed storage on **Cloudflare R2** (`data_connection.r2.source` + always `data_connection_id` for temp creds)     |
| `/teamspace/datasets/…`                 | Cluster **S3** datasets bucket (`…/projects/{id}/datasets/…`)                                                                 |
| `/teamspace/studios/<other>/…`          | That Studio’s content bucket (`s3://` or `gs://` via `LIGHTNING_CLOUD_PROVIDER`)                                              |
| `/teamspace/studios/this_studio/…`      | Workspace disk; LitData `url=None`. Platform **persists home to Studio’s remote backing store** — code OK, huge raw dumps not |

Prep datasets on a connection / scratch-outside-home → `optimize` — [lightning-studio.md](lightning-studio.md). Cookbook → [using-litdata.md](using-litdata.md).

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

Inside [Lightning Studios](https://lightning.ai/), connection folders appear under `/teamspace/…` as **FUSE mounts** (local-looking paths whose bytes still live in object storage). LitData **does not stream through FUSE** for these prefixes: `_resolve_dir` looks up the data connection and sets `Dir.url` to the backing bucket, then downloaders / uploaders hit that URL directly (plus temp credentials when `data_connection_id` is set). That is both **faster and more reliable** than `open()` / naive I/O on the mount under multi-worker training.

Full FUSE vs direct table → top of this doc. Code: `_resolve_s3_connections`, `_resolve_lightning_storage`, etc. in `resolver.py`.

### Full prefix table

| Prefix                                  | Backing store (FUSE under the hood except `this_studio`)                                                                                                                                 | Credentials                                                    |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `/teamspace/studios/this_studio/…`      | Workspace disk; LitData `url=None`. **Platform still persists home to Studio remote storage** — don’t store huge raw datasets here (see [lightning-studio.md](lightning-studio.md) prep) | N/A                                                            |
| `/teamspace/studios/<studio_name>/…`    | That studio’s content bucket (`s3://` or `gs://` …/cloudspaces/{id}/code/content/…)                                                                                                      | Studio / cluster env                                           |
| `/teamspace/s3_connections/<conn>/…`    | Customer **S3** (`data_connection.aws.source` + suffix)                                                                                                                                  | Ambient AWS, or temp creds if `available_in_non_aws_providers` |
| `/teamspace/gcs_connections/<conn>/…`   | Customer **GCS** (`data_connection.gcp.source` + suffix)                                                                                                                                 | GCP connection                                                 |
| `/teamspace/s3_folders/<conn>/…`        | S3 folder (`data_connection.s3_folder.source` + suffix)                                                                                                                                  | Folder connection                                              |
| `/teamspace/gcs_folders/<conn>/…`       | GCS folder (`data_connection.gcs_folder.source` + suffix)                                                                                                                                | Folder connection                                              |
| `/teamspace/lightning_storage/<conn>/…` | Lightning storage on **R2** (`data_connection.r2.source` + suffix)                                                                                                                       | Always `data_connection_id` (temp creds)                       |
| `/teamspace/datasets/…`                 | Cluster datasets **S3** (`s3://{cluster_bucket}/projects/{project_id}/datasets/…`)                                                                                                       | Studio env (`LIGHTNING_CLOUD_SPACE_ID`, …)                     |

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

# This studio home = code/notebooks OK; don’t park huge raw datasets here (persisted remotely)
# Prefer: optimize(..., output_dir="/teamspace/s3_connections/.../v1")
optimize(..., output_dir="/teamspace/studios/this_studio/artifacts/run-01")  # small artifacts only
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
6. Never tell users to train by reading the FUSE mount with `open()` / a plain filesystem DataLoader — pass `/teamspace/s3_connections/…` (etc.) into LitData so I/O hits S3/GCS/R2 directly ([resolver.md](resolver.md) FUSE section).
7. Document **all** schemes (`s3/gs/r2/azure/hf/local`) and **all** teamspace prefixes when answering path questions — customers often only know S3. Remind: `lightning_storage` ⇒ **R2**.
