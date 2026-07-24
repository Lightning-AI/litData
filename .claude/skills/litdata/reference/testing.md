# Testing LitData

Framework: **pytest** (`requirements/test.txt`). Config in `pyproject.toml [tool.pytest.ini_options]`.

## Running tests

```bash
# Single test (the fast dev loop)
pytest tests/streaming/test_dataset.py::test_name -v --capture=no

# A whole file / dir
pytest tests/streaming/ -v

# How CI runs it (authoritative). Two phases because processing tests must be serial:
pytest tests --ignore=tests/processing --ignore=tests/raw \
    -n 2 --dist=loadgroup --cov=litdata --durations=0 --timeout=120 --capture=no --verbose
pytest tests/processing tests/raw --cov=litdata --cov-append --durations=0 --timeout=120 --capture=no --verbose
```

**`--dist=loadgroup` is required, not optional, when running with `-n>1`.** `tests/conftest.py:165` (`pytest_collection_modifyitems`) pins every test using the `clean_pq_index_cache` fixture onto a single xdist worker via `xdist_group`, because that fixture `shutil.rmtree`s `~/.lightning/chunks` and would otherwise race across workers.

The `Makefile test` target is stale (`--flake8` flag, points at `src`) — prefer the CI commands above.

### Pytest config gotchas (`pyproject.toml`)

- `addopts` includes `--doctest-modules` → **docstrings in `src/` are collected as tests**; a malformed docstring example fails CI.
- `--strict-markers` → only registered markers allowed. Sole custom marker: `cloud` (real integration tests, not in the standard matrix).
- `filterwarnings = ["error::FutureWarning"]` → **FutureWarnings are hard errors.**
- `xfail_strict = true` → an `xfail` that unexpectedly passes fails the suite.

## Test layout

Tests mirror `src/litdata/`: `src/litdata/streaming/dataset.py` → `tests/streaming/test_dataset.py`. Dirs: `tests/streaming/`, `tests/processing/`, `tests/utilities/`, `tests/raw/`, plus top-level `tests/test_cli.py`, `test_debugger.py`, `test_helper.py`, `test_imports.py`, `test_requirements.py`. Every dir has `__init__.py`; cross-test helpers import as a package (`from tests.streaming.utils import filter_lock_files`).

Template (from `CONTRIBUTING.md`):

```python
def test_explain_what_is_being_tested(tmpdir):
    """One-line description of what/why."""
    cache_dir = os.path.join(tmpdir, "cache_dir")
    assert ...
```

## Gating tests — there is NO `RunIf` helper

`CONTRIBUTING.md` shows a `@RunIf(min_cuda_gpus=1)` template copied from PyTorch Lightning, but **no `RunIf` class exists in this repo.** Don't use it. The real pattern is `pytest.mark.skipif` + the `_*_AVAILABLE` `RequirementCache` constants in `src/litdata/constants.py`:

```python
from litdata.constants import _ZSTD_AVAILABLE, _PIL_AVAILABLE

# optional dependency, at param level
pytest.param("zstd", marks=pytest.mark.skipif(not _ZSTD_AVAILABLE, reason="Requires: ['zstd']"))

# at function level
@pytest.mark.skipif(not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")

# OS gating (very common)
@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")

# python-version gating
@pytest.mark.skipif(sys.version_info >= (3, 12), reason="Multiprocessing issues on 3.12+")
```

Available flags: `_ZSTD_AVAILABLE`, `_BOTO3_AVAILABLE`, `_FSSPEC_AVAILABLE`, `_CRYPTOGRAPHY_AVAILABLE`, `_GOOGLE_STORAGE_AVAILABLE`, `_AZURE_STORAGE_AVAILABLE`, `_POLARS_AVAILABLE`, `_PYARROW_AVAILABLE`, `_PIL_AVAILABLE`, `_TORCH_VISION_AVAILABLE`, `_AV_AVAILABLE`, `_OBSTORE_AVAILABLE`, `_HF_HUB_AVAILABLE`, `_LIGHTNING_SDK_AVAILABLE`, `_VIZ_TRACKER_AVAILABLE`.

## Mocking cloud — no moto / mock_aws

Two techniques:

1. **Fake modules injected into `sys.modules`** via `monkeypatch.setitem`, provided as `conftest.py` fixtures: `google_mock`, `fsspec_mock`, `obstore_mock`, `azure_mock`, `lightning_cloud_mock` (sets `rest_client.LightningClient = Mock()`), `lightning_sdk_mock`, `huggingface_hub_mock`, `huggingface_hub_fs_mock` (full fake `HfFileSystem` backed by local parquet), `fsspec_pq_mock`. These let import-guarded cloud code run without the real SDK.
2. **`monkeypatch.setattr` on the module's `boto3`/`botocore`** for S3 — e.g. `tests/streaming/test_client.py` patches `client.boto3`/`client.botocore` with `MagicMock()` and asserts on `.client.assert_called_with(...)`. `boto3` is a hard dep, so it's stubbed, not installed-away.

Use stdlib `unittest.mock` (`Mock`, `MagicMock`, `patch`) + the pytest `monkeypatch` fixture.

## Useful fixtures (`tests/conftest.py`)

- **Autouse/session**: `teardown_process_group` (destroys torch.distributed group), `_thread_police` (fails a test leaking a non-daemon "zombie" thread — special-cases `PrepareChunksThread`, `QueueFeederThread`, `pytest_timeout`).
- **Autouse/per-test**: `disable_signals` (no-ops `signal.signal` so worker signal handlers don't break tests).
- **Data**: `mosaic_mds_index_data`, `prepare_combined_dataset` (two on-disk 50-item `Cache` datasets), `combined_dataset` (a `CombinedStreamingDataset`), `pq_data`, `write_pq_data` (5 polars parquet files), `clean_pq_index_cache` (wipes the default cache dir — the one that triggers xdist grouping).

Local idiom: most files define their own `seed_everything(seed)`. CLI tests use a `run_cli(args_list)` helper (patches `sys.argv`, captures stdout). `tests/streaming/utils.py` gives `filter_lock_files` / `get_lock_files` to assert on chunk dirs ignoring `.lock`/`.cnt` artifacts.

## When tests hang or flake

- Processing/worker tests spawn real subprocesses — run **serially** (why CI separates `tests/processing`). Use `--timeout=120`.
- A leaked thread → `_thread_police` `AssertionError: Test left zombie thread`. Ensure background threads (`PrepareChunksThread`) are stopped.
- `spawn` requires picklable args; a test failing only under spawn usually means an unpicklable closure.
- To debug inside a worker: `from litdata.utilities.breakpoint import breakpoint; breakpoint()` (see debugging.md).
