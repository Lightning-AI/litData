# LitData on Lightning Studio

Lightning Studio is the usual place LitData is developed and benchmarked. Paths under `/teamspace/…` look like a local filesystem but often resolve to cloud URLs + temporary credentials via the Lightning SDK.

## Mental model

```
User path                          What LitData actually uses
─────────────────────────────      ────────────────────────────────────────
/teamspace/studios/this_studio/…   Local Dir(path=…, url=None) — workspace disk
/teamspace/s3_connections/<name>/… Dir(path=…, url=s3://…, data_connection_id=…)
/teamspace/gcs_connections/…       Same idea for GCS
s3://bucket/prefix                 Direct URL (boto3/obstore; no Studio resolver)
```

`StreamingDataset(input_dir="/teamspace/s3_connections/…")` therefore:

1. **`_resolve_dir`** (`streaming/resolver.py`) → `Dir(path, url, data_connection_id)`
2. Downloads go to **`url`** (real S3/GCS), not by reading the FUSE path as a normal file for every chunk
3. Credentials often come from **`data_connection_id`** → Lightning control plane temp bucket credentials (`streaming/client.py`), not long-lived keys in the environment
4. Local chunk cache still lands under `LITDATA_CACHE_DIR` / `~/.lightning/chunks` or an explicit `cache_dir=` (Studio benches often use `/cache/chunks`)

**`/teamspace/studios/this_studio`** is special: always treated as **local workspace** (`url=None`). Other `/teamspace/studios/<name>/…` paths resolve to that Studio’s cloud-backed content bucket via the Lightning API.

## Teamspace path prefixes

| Prefix                                                                                    | Role                                                                                |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `/teamspace/studios/this_studio/`                                                         | This Studio’s workspace (local)                                                     |
| `/teamspace/studios/<other>/`                                                             | Another Studio’s code/content (resolved to cluster bucket URL)                      |
| `/teamspace/s3_connections/<conn>/`                                                       | Named S3 data connection → bucket URL + `data_connection_id`                        |
| `/teamspace/gcs_connections/<conn>/`                                                      | Named GCS data connection                                                           |
| `/teamspace/s3_folders/`, `/teamspace/gcs_folders/`                                       | Folder-style connections                                                            |
| `/teamspace/lightning_storage/`                                                           | Lightning-managed object storage (often R2-style creds)                             |
| `/teamspace/datasets/`                                                                    | Dataset mounts (path may be rewritten for cache identity)                           |
| `/teamspace/efs_connections/`, `/teamspace/efs_folders/`, `/teamspace/filestore_folders/` | POSIX / EFS-style stores (filestore path rewrite helpers in `dataset_utilities.py`) |

Connection name is path segment `[3]` (e.g. `/teamspace/s3_connections/optimized-imagenet-1m/...` → connection `optimized-imagenet-1m`). Lookup: `LightningClient.data_connection_service_list_data_connections(project_id)`.

## Environment variables Studio injects

Required for resolving non–`this_studio` teamspace paths / minting temp creds:

| Env                          | Purpose                                        |
| ---------------------------- | ---------------------------------------------- |
| `LIGHTNING_CLOUD_PROJECT_ID` | Project for API list/mint calls                |
| `LIGHTNING_CLUSTER_ID`       | Cluster (Studio resolve, some APIs)            |
| `LIGHTNING_CLOUD_URL`        | Control-plane base URL (temp credentials)      |
| `LIGHTNING_CLOUD_PROVIDER`   | `aws` / GCP (bucket scheme for Studio content) |

If these are missing outside Studio, `/teamspace/s3_connections/…` resolution fails with a clear `RuntimeError` / `ValueError`. Local CI and laptops usually use `s3://…` + normal AWS creds, or `local:` fakes in tests.

## Cache on Studio

- Default cache: `~/.lightning/chunks` (or `LITDATA_CACHE_DIR`).
- High-throughput benches: put cache on a fast volume, e.g. **`/cache/chunks`**, and **wipe before every cold epoch** (`rm -rf /cache/chunks/*`). Never delete the mountpoint itself.
- Peak disk ≈ `num_workers × max_pre_download × mean_chunk_size` (async floor often forces `max_pre≥4` on remote).
- Stale `.lock` / `.cnt` after crashes → `litdata cache clear` or wipe the bench cache dir.

## Free-threaded Python (not the Studio default)

**Default Studio Python is a normal GIL build.** `PYTHON_GIL=0` / `python -Xgil=0` do **nothing useful** unless the interpreter itself is a free-threading (nogil) build — those flags only control whether a freethreading binary enables the GIL at runtime.

To run without the GIL you must **switch the environment** (conda / Studio image / dedicated freethreading Python), not just set env vars. On Lightning Studio, conda is typically limited to **one** env per Studio (`cloudspace`); creating a freethreading env often means **starting a Studio configured for free-threaded Python** (or otherwise replacing that env), not `conda create` beside the default.

Verify before benchmarking:

```bash
python -c 'import sys; print(sys.version); print("gil_enabled", getattr(sys, "_is_gil_enabled", lambda: "n/a")())'
# Freethreading build + GIL off → version mentions "free-threading"; gil_enabled False
```

Only then:

```bash
PYTHON_GIL=0 python -Xgil=0 scripts/bench/…
```

GIL-disabled Studio runs are a useful high-throughput baseline for “how fast can LitData feed training,” not the default developer setup.

## How agents should work in a Studio checkout

1. **Repo location** is often `/teamspace/studios/this_studio/litData` (or similar). Editable install / `PYTHONPATH=src` as in [contributing.md](contributing.md).
2. **Data** for real S3: prefer an attached connection under `/teamspace/s3_connections/…` rather than inventing buckets. Canonical ImageNet-optimized path used in recent work: `/teamspace/s3_connections/optimized-imagenet-1m/lightning_data_search`.
3. **Do not treat FUSE listing latency as download latency.** Streaming performance is dominated by `PrepareChunksThread` + obstore/boto3 GETs into the local cache, then decode.
4. **Worker env:** DataLoader workers are separate processes. Parent-only `PYTHONSTARTUP` / one-off monkeypatches may not apply — use `sitecustomize` on `PYTHONPATH` or patch before fork/spawn when forcing boto3 vs obstore (see [benchmarking.md](benchmarking.md)).
5. **SDK optional dep:** `lightning-sdk` is in extras / Studio images; resolver and temp-cred paths need it. Pure open-source users on `s3://` + AWS credentials never hit that code.

## Optimize / write from Studio

`optimize` / `map` with `output_dir` under a data connection or Studio cloud path will resolve via the same `Dir` machinery and upload through `FsProvider`. Writing into a non-empty remote dir is refused unless `append` / `overwrite` ([streaming.md](streaming.md) gotchas). Prefer writing to a dedicated prefix.

## Quick Studio smoke

```python
from litdata import StreamingDataset, StreamingDataLoader

# Resolves to s3://… + temp creds when connection is attached
ds = StreamingDataset(
    "/teamspace/s3_connections/optimized-imagenet-1m/lightning_data_search",
    cache_dir="/cache/chunks",
    max_pre_download=4,
    max_cache_size="200GB",
)
loader = StreamingDataLoader(ds, batch_size=256, num_workers=8)
batch = next(iter(loader))
```

For full-epoch methodology, fair async/obstore comparisons, and what not to ship from noisy single runs → [benchmarking.md](benchmarking.md).
