# Multi-node processing (`optimize` / `map`)

How LitData fans **`optimize`** and **`map`** across machines. This is **Lightning Studio data-prep jobs**, not PyTorch DDP training and not a built-in SLURM launcher.

**Related:** [processing.md](processing.md) · [data-movement.md](data-movement.md) · [resolver.md](resolver.md) · [lightning-studio.md](lightning-studio.md) · user cookbook [using-litdata.md](using-litdata.md) §9

______________________________________________________________________

## 0. What “multi-node” means here (and what it is not)

| Mechanism                                                       | Used for                                                                                                | Symbols / env                                                                |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Lightning Studio job** (`num_nodes=N`)                        | Distributed **optimize/map**                                                                            | `functions.py` gate → `resolver._execute` → platform sets `DATA_OPTIMIZER_*` |
| **`DATA_OPTIMIZER_*` env**                                      | Rank / world inside each job instance                                                                   | `_get_num_nodes`, `_get_node_rank`, worker `DATA_OPTIMIZER_GLOBAL_RANK`      |
| **`broadcast_object`**                                          | Align dirs when `broadcast_paths` is on (or auto for `{%strftime}` paths) and Lightning app URL present | `utilities/broadcast.py`                                                     |
| **Torch distributed / `WORLD_SIZE` / `GLOBAL_RANK` / `NNODES`** | **Training** stream path (`_DistributedEnv.detect`) — **not** how optimize jobs are launched            | `utilities/env.py`                                                           |
| **SLURM**                                                       | **Not** a first-class optimize launcher in this repo                                                    | Do not document SLURM as supported for `num_nodes`                           |

If `num_nodes` / `machine` are set **outside** Studio (`_IS_IN_STUDIO` false) → `ValueError` (“Only https://lightning.ai/ supports multiple nodes…”).

Training-time multi-GPU/node streaming (chunk shuffle, sampler) is a **separate** concern — see [streaming.md](streaming.md) / `_DistributedEnv`. Below is **write-path** multi-node only.

______________________________________________________________________

## 1. Launch dual-path (`optimize` / `map`)

Both APIs share the same gate (`functions.py`):

```
if num_nodes is None OR int(DATA_OPTIMIZER_NUM_NODES) > 0:
    → construct DataProcessor and run locally on THIS machine
else:
    → _execute(...)   # resolver.py — create Studio data-prep job, block until done
```

### 1.1 Caller Studio (launcher)

1. User: `optimize(..., num_nodes=N, machine=Machine.DATA_PREP)` (or `map` with same knobs).
2. Gate sees `num_nodes` set and `DATA_OPTIMIZER_NUM_NODES` unset/0 → **`_execute`**.
3. `_execute` (`resolver.py:461`):
   - Requires `lightning_sdk` (`_LIGHTNING_SDK_AVAILABLE`).
   - `Studio()._studio_api.create_data_prep_machine_job(...)` with:
     - `command`: `cd {cwd} &&[LIGHTNING_SKIP_INSTALL=…][LIGHTNING_BRANCH=…] python {' '.join(sys.argv)}` (re-runs the **same script/args**)
     - `num_instances=num_nodes`
     - `machine=machine or current Studio machine`
     - `interruptible=` exists on `_execute` but **optimize/map never pass it** → always default `False`
   - Prints job URL (`…/app?app_id=litdata&app_tab=Runs&job_name=…`).
   - Polls until `STOPPED` / `COMPLETED`, or raises on `FAILED`.

### 1.2 Each job instance (worker machine)

Platform injects env (see §2). Script starts again; gate now sees `DATA_OPTIMIZER_NUM_NODES > 0` → **local** `DataProcessor.run` on that instance’s shard only. No cross-node RPC for work items — pure env + shared object store for indexes/outputs.

______________________________________________________________________

## 2. Environment variables

### 2.1 Processing ranks (optimize/map workers)

| Variable                                  | Reader                                                                | Role                                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `DATA_OPTIMIZER_NUM_NODES`                | `_get_num_nodes()`; launch gate                                       | World of machines. `>0` means “already inside a distributed job / run DataProcessor” |
| `DATA_OPTIMIZER_NODE_RANK`                | `_get_node_rank()`                                                    | This machine’s rank in `[0, num_nodes)`                                              |
| `DATA_OPTIMIZER_GLOBAL_RANK`              | Set in `BaseWorker._set_environ_variables`                            | `node_rank * num_workers + worker_index` — used for chunk filenames / writer rank    |
| `DATA_OPTIMIZER_NUM_WORKERS`              | Set in worker; also `_DistributedEnv._instantiate_in_map_or_optimize` | Local worker count                                                                   |
| `DATA_OPTIMIZER_CACHE_FOLDER`             | `_get_cache_dir`                                                      | Chunk cache root (default Studio `/cache/chunks`)                                    |
| `DATA_OPTIMIZER_DATA_CACHE_FOLDER`        | `_get_cache_data_dir`                                                 | Downloaded input cache (default `/cache/data`)                                       |
| `DATA_OPTIMIZER_TIMEOUT`                  | Worker `_loop` queue get                                              | Default 300s; shared-queue mode often 200s                                           |
| `DATA_OPTIMIZER_FAST_DEV_RUN`             | `_get_fast_dev_run`                                                   | Related to fast_dev_run defaults                                                     |
| `ENABLE_STATUS_REPORT` / `_ENABLE_STATUS` | Progress                                                              | Node 0 may write `status.json` with coarse %                                         |

### 2.2 Job / artifacts (Studio)

| Variable                                                                                                   | Role                                                                |
| ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `LIGHTNING_SKIP_INSTALL` / `LIGHTNING_BRANCH`                                                              | Injected into remote job command string                             |
| `LIGHTNING_BUCKET_NAME`, `LIGHTNING_CLOUD_PROJECT_ID`, `LIGHTNING_CLOUD_APP_ID`, `LIGHTNING_CLOUD_WORK_ID` | `_get_work_dir()` → artifacts `s3://…/artifacts/{work_id}/content/` |
| `LIGHTNING_CLOUD_URL`                                                                                      | Job URL pretty-print; auth helpers                                  |
| `LIGHTNING_APP_EXTERNAL_URL`                                                                               | If set, `broadcast_object` uses Lightning broadcast HTTP API        |

### 2.3 Training distributed env (do not confuse)

`_DistributedEnv.detect()` (`utilities/env.py`):

- Inside map/optimize workers (`DATA_OPTIMIZER_GLOBAL_RANK` set): builds env from `DATA_OPTIMIZER_*` (`world_size = num_workers * num_nodes`).
- Else: `torch.distributed` if initialized, else `WORLD_SIZE` / `GLOBAL_RANK` / `NNODES`.

That path drives **streaming** writer/reader rank math when those modules run under optimize; it is **not** a SLURM or torchrun launcher for `num_nodes=`.

______________________________________________________________________

## 3. Work sharding

```
world_size = num_nodes * num_workers
```

Both `_map_items_to_workers_sequentially` and `_map_items_to_workers_weighted` (`data_processor.py`) pack across **all** ranks, then each node **keeps only** worker ids in:

```
[node_rank * num_workers, (node_rank + 1) * num_workers)
```

### 3.1 Sequential (`reorder_files=False` or no `input_dir.path` for size packing)

- Split `user_items` into `world_size` contiguous slices (remainder distributed from the end).
- With `align_chunking=True` + `chunk_size`: assign full chunks of size `chunk_size` per global worker; last worker gets the tail (may be uneven — intentional).

### 3.2 Weighted / by file size (default when `reorder_files` and `input_dir.path`)

- `_get_item_filesizes` (threaded `os.path.getsize` — **local/FUSE path based**; TODO in code notes broadcasting sizes from node 0).
- `_pack_greedily` into `world_size` bins; permute within each worker’s list (`np.random.permutation` — seed fixed in `DataProcessor.run`).
- Explicit `weights=` uses same packer with `file_size=False`.

### 3.3 Queue / shared-queue modes

- Input `multiprocessing.Queue` or `keep_data_ordered=False`: dynamic consumption; multi-node semantics are weaker / different — checkpointing **unsupported** for Queue inputs. Prefer static list inputs for multi-node jobs.
- `ALL_DONE` sentinel for shared-queue shutdown (`keep_data_ordered=False`).

### 3.4 Broadcast dirs (`broadcast_paths`)

`optimize` / `map` / `DataProcessor` take **`broadcast_paths: bool = False`**.

After resolve, broadcast runs **only when** `broadcast_paths` is effectively on:

```python
# Auto-on if input_dir or output_dir contains a `{%strftime}` template (detected before resolve).
self.broadcast_paths = broadcast_paths or _has_time_template(input_dir) or _has_time_template(output_dir)

if self.broadcast_paths:
    self.input_dir = broadcast_object("input_dir", self.input_dir, rank=_get_node_rank())
    self.output_dir = broadcast_object("output_dir", self.output_dir, rank=_get_node_rank())
```

| Case                                 | Behavior                                                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Default (`False`), no `{%…}` in path | **Skip** broadcast — each rank keeps its locally resolved `Dir` (fine for stable `s3://` / connection paths) |
| Path has `{%Y-%m-%d}` (etc.)         | **Auto-enable** — ranks must share one expanded timestamp                                                    |
| `broadcast_paths=True`               | Always broadcast after resolve                                                                               |

- If `LIGHTNING_APP_EXTERNAL_URL` is set and broadcast runs: HTTP broadcast until all ranks agree.
- Else `broadcast_object` returns the local `obj` unchanged.
- Multi-node implication when off: ranks must independently resolve to the **same** paths; do not rely on per-rank `datetime.now()` without a shared template + auto-broadcast.

______________________________________________________________________

## 4. Per-node cache and downloads

- Each node has **its own** `/cache/chunks` and `/cache/data` (or env overrides). **Not** a shared NFS assumption.
- `DataProcessor._cleanup_cache` wipes both at **start** of each node’s run.
- Downloaders/uploaders/removers run **per worker on that node** — see [data-movement.md](data-movement.md).
- Every node needs credentials for **inputs and outputs** (attached connections or keys on all instances). Missing creds on node K → that shard fails; last-node index merge may hang waiting for `{K}-index.json`.

**FUSE vs direct:** same as single-node — pass `/teamspace/s3_connections/…` so each node resolves to `Dir.url` and downloads via FsProvider, not through FUSE under multi-worker load.

______________________________________________________________________

## 5. Chunk write + upload coordination

### 5.1 Filenames and ranks

`BinaryWriter.get_chunk_filename` → `chunk-{rank}-{chunk_index}[.compression].bin` where `rank` is `DATA_OPTIMIZER_GLOBAL_RANK` when set (`writer.py`).

So **all workers on all nodes** write uniquely named chunks. No “only rank 0 uploads chunks.”

### 5.2 Who uploads chunks?

**Every worker’s uploader pool** uploads its own closed chunks to `output_dir` as they are produced (`_try_upload` → `_upload_fn`). There is no barrier that waits for other nodes before uploading bins.

### 5.3 Who builds vs merges the index?

After local workers finish, `DataChunkRecipe._done` (`data_processor.py`):

1. `Cache(...)._merge_no_wait(node_rank if num_nodes > 1 else None, existing_index?)`
   - Single-node: merge all per-worker `*-index.json` / rank indexes in cache → `index.json`.
   - Multi-node: merge **this node’s** worker indexes → **`{node_rank}-index.json`** (see `BinaryWriter._merge_no_wait`).
2. `_upload_index(output_dir, cache_dir, num_nodes, node_rank)`:
   - Upload that node’s index file (`index.json` or `{node_rank}-index.json`) via FsProvider or local copy.
   - **If `num_nodes == node_rank + 1` (last node):**
     1. For each peer `0 .. num_nodes-2`: `_wait_for_file_to_exist` on `{peer}-index.json` at `output_dir`, then download into local cache.
     2. `merge_cache._merge_no_wait()` → final `index.json`.
     3. Recurse `_upload_index(..., num_nodes=1, node_rank=None)` to upload the merged index.

Comment in code: under the Data Optimizer there should be a **single process per node** executing this merge section → no local race on last node. Cross-node coordination is **object-store presence** of peer index files, not a distributed lock.

### 5.4 Map jobs

`map` uses the same launch/shard/download/upload worker pools but **no** LitData chunk `index.json` merge. Outputs are whatever files `fn` wrote that uploaders copied. Last-node index logic is optimize-specific (`DataChunkRecipe`).

______________________________________________________________________

## 6. Output directory rules under multi-node

**Prefer durable remote:**

- `/teamspace/s3_connections/…`, `/teamspace/datasets/…`, or `s3://…` / `gs://…` / `r2://…`

**`this_studio` remap (optimize only):** if `output_dir` resolves to a path under `/teamspace/studios/this_studio` **and** `DATA_OPTIMIZER_NUM_NODES > 0`, optimize rewrites to:

```
_get_work_dir() + relative_path
→ s3://{LIGHTNING_BUCKET_NAME}/projects/.../artifacts/{work_id}/content/...
```

(`functions.py` ~515–524). **`map` does not apply this remap.** Studio UI may also show job mounts under `/teamspace/jobs/…` — LitData does not construct that string itself.

**Rejected:** resolved URL containing `cloudspaces` → `ValueError` with hint to use connections/datasets.

**Immutability:** `_assert_dir_has_index_file` / empty checks on remote before write; use `mode="append"|"overwrite"` or versioned prefixes.

______________________________________________________________________

## 7. Resume, checkpoint, append

### 7.1 `use_checkpoint=True` (optimize)

- Unsupported for Queue inputs and generator `fn`s (`DataProcessor.run`).
- Saves `.checkpoints/config.json` (`num_workers`, `workers_user_items`) and per-worker checkpoint JSON.
- Writer saves `checkpoint-{rank}-{uuid}.json`; upload strips UUID → `checkpoint-{rank}.json` (`remove_uuid_from_filename`).
- On resume: `_load_checkpoint_config` requires **same** `num_workers` and **identical** `workers_user_items`; trims each worker list to `done_till_index`.
- Remote: download `.checkpoints/` via `FsProvider.download_directory`.
- When **not** using checkpoints, `run` calls `_cleanup_checkpoints` (local rmtree or remote delete of `.checkpoints/`). Successful completion with checkpoints also cleans up at end.

**Multi-node caveat:** checkpoint compatibility is tied to **local** `workers_user_items` after node slicing. Changing `num_nodes` / `num_workers` / input order breaks resume. Verify behavior in code if mixing checkpoint + multi-node — treat as advanced.

### 7.2 `mode="append"`

- Reads existing `index.json` (`read_index_file_content`).
- Advances per-rank `state_dict` from existing `chunk-<rank>-<index>.bin` filenames so new chunks continue indexes.
- `existing_index` passed into recipe merge so final index concatenates old + new chunks.

### 7.3 `mode="overwrite"`

- Resolver / assert helpers delete prior index/chunks (and checkpoints depending on flags) before write.

______________________________________________________________________

## 8. Progress and status

- Main process aggregates `progress_queue` updates (per-worker counters). tqdm total is **this node’s** item count, not global (status.json tries to scale by `num_nodes` when enabled).
- Node 0 + `_ENABLE_STATUS`: writes `status.json` with `"progress": "{percent}%"`.
- Worker log lines via `msg_queue` → `flush_msg_queue` to avoid breaking tqdm.

______________________________________________________________________

## 9. Failure modes & pitfalls (agent checklist)

| Pitfall                                | What happens                                                                 | Mitigation                                                                                                                    |
| -------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Peer never writes `{k}-index.json`     | Last node **hangs** in `_wait_for_file_to_exist`                             | Ensure all nodes finish; same `output_dir`; credentials on every node; check failed job instances                             |
| Duplicate work                         | Mis-set `DATA_OPTIMIZER_NODE_RANK` / all nodes think they are 0              | Trust platform env; don’t manually override inconsistently                                                                    |
| NFS shared cache dir                   | Nodes stomp `/cache` if incorrectly shared                                   | Keep node-local caches; use object store for outputs                                                                          |
| Reading FUSE for size packing          | `_get_item_filesizes` hits mount; slow/wrong under load                      | Prefer connection paths that exist as files for size, or pass `weights=`; accept TODO that sizes aren’t broadcast from node 0 |
| Only node 0 has AWS keys               | Other nodes fail downloads/uploads                                           | Attach connection / inject creds on all instances                                                                             |
| `keep_data_ordered=False` + multi-node | Shared queue is **per node**, not global                                     | Prefer ordered static sharding for multi-node                                                                                 |
| Index config mismatch across workers   | `_merge_no_wait` raises inconsistent `config`                                | Same `fn` / serializers / compression on all workers                                                                          |
| Local `output_dir` on multi-node       | Machines don’t share disk; merge/upload logic expects reachable `output_dir` | Use remote `output_dir`                                                                                                       |
| Assuming torchrun/SLURM                | `num_nodes=` outside Studio errors                                           | Use Studio jobs or run your own process manager **and** set `DATA_OPTIMIZER_*` yourself (unsupported DIY — verify carefully)  |
| `map` + `this_studio` output           | No artifacts remap                                                           | Write to connection / `s3://`                                                                                                 |
| Uploader swallows exceptions           | `_upload_fn` `print(e)` then may still signal remover                        | Watch logs; missing remote chunks + stuck index                                                                               |
| Hard kill on worker error              | `_exit_on_error` → `terminate()` all local workers                           | Fix root error with `num_workers=1` / `fast_dev_run`                                                                          |

______________________________________________________________________

## 10. Mental model diagram

```
Studio caller                         Job instances (NODE 0 .. N-1)
-----------------                     ----------------------------
optimize(num_nodes=N)
     |
     v
 _execute  (create job, print URL)
     |                                env: DATA_OPTIMIZER_NUM_NODES / NODE_RANK
     |         for each instance:     resolve dirs -> shard items for this node
     |                                downloaders -> fn -> chunk-{global_rank}-*.bin
     |                                uploaders -> output_dir (all ranks upload)
     |                                _done -> upload {node_rank}-index.json
     |                                LAST NODE (node_rank == num_nodes-1):
     |                                  wait for peers' {k}-index.json
     |                                  merge -> index.json -> upload
     v
 block until job COMPLETE / FAILED
```

______________________________________________________________________

## 11. Grep landmarks

```
processing/functions.py     optimize/map num_nodes gate; this_studio remap
streaming/resolver.py       _execute
processing/data_processor.py
  _get_num_nodes  _get_node_rank
  _map_items_to_workers_sequentially  _map_items_to_workers_weighted
  DataChunkRecipe._done  _upload_index
  broadcast_object(...)  _cleanup_cache  _load_checkpoint_config
processing/utilities.py     _get_work_dir  extract_rank_and_index_from_filename
streaming/writer.py         chunk-{rank}-*.bin  _merge_no_wait
utilities/broadcast.py      broadcast_object
utilities/env.py            _DistributedEnv  _is_in_map_or_optimize
```

## 12. Minimal multi-node recipe (Studio)

```python
import litdata as ld

def fn(path):
    # read local cached path when downloaders ran; return sample pytree
    ...

if __name__ == "__main__":
    ld.optimize(
        fn=fn,
        inputs=list_of_paths_under_connection,  # or walk(...)
        input_dir="/teamspace/s3_connections/my-raw",
        output_dir="/teamspace/s3_connections/my-opt/v1",  # durable, shared
        chunk_bytes="64MB",
        num_workers=8,
        num_nodes=4,                 # Studio only
        # machine=Machine.DATA_PREP,
        num_downloaders=2,
        num_uploaders=1,
    )
```

Ensure the script is restartable with the same args (job re-invokes `sys.argv`). Prefer versioned `output_dir` prefixes; do not rely on FUSE for bulk I/O.
