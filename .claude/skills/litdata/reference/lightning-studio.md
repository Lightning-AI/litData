# LitData on Lightning Studio

Lightning Studio is a common place to develop and benchmark LitData.

**Critical:** `/teamspace/s3_connections`, `gcs_connections`, `s3_folders`, `gcs_folders`, `lightning_storage`, `datasets`, and other-studio paths are **FUSE mounts** over object storage. They are fine for browsing; for LitData **pass those paths into** `StreamingDataset` / `optimize` / `map` so the resolver talks **directly** to the backing store (S3 / GCS / **R2 for `lightning_storage`**). That is faster and more reliable than reading through FUSE. Full table → [resolver.md](resolver.md) (“Studio mounts are FUSE”).

Customer README overview: `#resolve-paths`. Canonical path/URI reference: [resolver.md](resolver.md).

## Mental model (Studio-specific)

```
User path                          Under the hood              LitData I/O
─────────────────────────────      ───────────────────────────  ─────────────────────────────
/teamspace/studios/this_studio/…   Workspace disk; **platform  Dir(path=…, url=None)
                                   persists it to Studio’s     (no bucket URL for LitData;
                                   remote backing store**      still don’t dump huge raw data here)
/teamspace/s3_connections/<n>/…    FUSE → customer S3          Dir(path=…, url=s3://…, …)  direct S3
/teamspace/gcs_connections/…       FUSE → customer GCS         url=gs://…                  direct GCS
/teamspace/lightning_storage/…     FUSE → Lightning R2         url=r2://… + connection id  direct R2
/teamspace/datasets/…              FUSE → cluster S3           url=s3://…/datasets/…       direct S3
s3://bucket/prefix                 No Studio mount             Your AWS creds
```

1. **`_resolve_dir`** → `Dir(path, url, data_connection_id)` — full table in [resolver.md](resolver.md)
2. Downloads/uploads use **`url`** (object store API), **not** FUSE, for connection / datasets / other-studio paths
3. **`data_connection_id`** → temp project-role credentials when needed (`streaming/client.py`) — always for `lightning_storage` (R2)
4. Chunk cache still under `LITDATA_CACHE_DIR` / `~/.lightning/chunks` / `cache_dir=` (benches often `/cache/chunks`)
5. **Prep datasets** on a connection / cloud URL (or scratch outside home → `optimize` out) — see below; never fill `this_studio` with raw dumps

Connection name is path segment `[3]`. Lookup: Lightning data-connection API for the project.

## Studio home (`this_studio`) is persisted remotely

`/teamspace/studios/this_studio` is the Studio **home / workspace**. LitData treats it as local disk (`Dir(path=…, url=None)` — no direct bucket URL for streaming I/O), but the **platform still persists that tree to the remote storage the Studio is backed on**. Large downloads, unpacks, and scratch datasets written under home therefore:

- Sync / replicate into Studio’s remote backing store (cost, quota, slow sync)
- Compete with code and notebooks for workspace space
- Are a worse place for multi-GB raw ImageNet-style dumps than a dedicated data connection

**Do not prepare big datasets by downloading into `this_studio`.** Prefer one of the patterns below.

### Preparing a dataset in Studio (recommended)

**1. Land data on a dedicated connection / bucket (preferred)**

Attach or create an S3 / GCS / `lightning_storage` (R2) / `datasets` folder, then put bytes **there** — not under home.

- Prefer **LitData’s resolver path** for copies and writes: pass `/teamspace/s3_connections/<name>/…` (or `s3://…`) into `optimize` / `map` / helpers so uploads go **directly** to the object store via `FsProvider`, instead of `cp`/`tar` through the FUSE mount.
- Browsing the mount in the file tree is fine; bulk transfer through FUSE is not.

```python
# Good: optimize writes chunks straight to the connection’s bucket
optimize(fn=..., inputs=..., output_dir="/teamspace/s3_connections/my-data/imagenet-opt-v1", chunk_bytes="64MB")

# Also fine: explicit cloud URL when you have creds
optimize(fn=..., inputs=..., output_dir="s3://my-bucket/imagenet-opt-v1", chunk_bytes="64MB")
```

**2. Download & unpack outside home, then `optimize` elsewhere**

If you must fetch a tarball / zip first:

1. Download and unpack **outside** `/teamspace/studios/this_studio` (ephemeral/fast scratch such as `/tmp`, `/cache`, or a machine-local volume — not the persisted home tree).
2. Run `optimize` / `map` with `output_dir` on a **connection or cloud URL** so the durable copy lives in object storage.
3. Delete the scratch unpack when done so it never syncs into Studio home persistence.

```bash
# Sketch — paths vary by Studio image
mkdir -p /cache/raw && cd /cache/raw
# download + unpack ImageNet (or similar) HERE, not under this_studio
python optimize_imagenet.py \
  --input_dir /cache/raw/train \
  --output_dir /teamspace/s3_connections/my-data/imagenet-opt-v1
```

| Avoid                                                       | Prefer                                                                   |
| ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| `~/…` or `/teamspace/studios/this_studio/datasets/huge-raw` | Scratch outside home → `optimize` → connection / `s3://` / `datasets`    |
| `cp` huge trees onto `/teamspace/s3_connections/…` via FUSE | `optimize`/`map`/`FsProvider` with a resolved connection or `s3://` path |
| Leaving multi-GB unpacks in home “temporarily”              | They still persist to Studio remote storage                              |

Chunk **cache** during training (`/cache/chunks`, `LITDATA_CACHE_DIR`) is separate from dataset prep — wipe it for cold benches; don’t confuse it with durable `output_dir`.

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

1. **Repo location** is often `/teamspace/studios/this_studio/litData` (or similar). Editable install / `PYTHONPATH=src` as in [contributing.md](contributing.md). Code in home is fine; **multi-GB datasets are not** — see prep section above.
2. **Data** for real S3: prefer an attached connection under `/teamspace/s3_connections/…` rather than inventing buckets or dumping under `this_studio`. Canonical ImageNet-optimized path used in recent work: `/teamspace/s3_connections/optimized-imagenet-1m/lightning_data_search`.
3. **Do not treat FUSE listing/read latency as LitData download latency.** Connection mounts are FUSE; streaming uses direct object GETs into the local cache, then decode. See [resolver.md](resolver.md).
4. **Worker env:** DataLoader workers are separate processes. Parent-only `PYTHONSTARTUP` / one-off monkeypatches may not apply — use `sitecustomize` on `PYTHONPATH` or patch before fork/spawn when forcing boto3 vs obstore (see [benchmarking.md](benchmarking.md)).
5. **SDK optional dep:** `lightning-sdk` is in extras / Studio images; resolver and temp-cred paths need it. Pure open-source users on `s3://` + AWS credentials never hit that code.

## Optimize / write from Studio

`optimize` / `map` with `output_dir` under a data connection or Studio cloud path will resolve via the same `Dir` machinery and upload through `FsProvider`. Writing into a non-empty remote dir is refused unless `append` / `overwrite`. Prefer a dedicated versioned prefix.

### Multi-node `num_nodes` / `machine`

Passing `num_nodes=N` (optionally `machine=Machine.DATA_PREP`) from a Studio:

1. LitData calls `_execute` → creates a **data-prep job** (Runs UI URL is printed).
2. Each of N instances re-runs your script with `DATA_OPTIMIZER_NUM_NODES` / `NODE_RANK` set, then runs `DataProcessor` on its shard.
3. Last node merges `{rank}-index.json` → final `index.json`.

**Output tip:** write to `/teamspace/s3_connections/...`, `/teamspace/datasets/...`, or `s3://...` so results land in a durable bucket. Optimize may remap `/teamspace/studios/this_studio/...` outputs to the job artifacts S3 URL; the Studio UI may expose them under `/teamspace/jobs/...`. Ensure **every** node can read inputs and write outputs (attached connections or cloud credentials).

Full launch/env/sharding → [processing.md](processing.md). User recipe → [using-litdata.md](using-litdata.md) §9.

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
