# Contributing to LitData

Full reference: `CONTRIBUTING.md`. This is the fast path.

## Environment setup

Requires Python ≥ 3.10 and `make`. LitData uses `uv` for installs.

Optional — install the LitData agent skill in another checkout/tooling:

```bash
npx skills add Lightning-AI/litData
```

On **Lightning Studio**, the checkout often lives under `/teamspace/studios/this_studio/…`, with datasets attached as `/teamspace/s3_connections/…`. See [lightning-studio.md](lightning-studio.md) before assuming paths are plain local disks.

```bash
make setup            # install-dependencies + install-pre-commit (recommended one-shot)
```

Or manually:

```bash
uv pip install -e ".[extras]" -r requirements/test.txt
pre-commit install
```

The package is `src/`-layout (`package_dir={"": "src"}`), installed editable. CLI entry point: `litdata.__main__:app`.

## Design principles (honor when writing code)

LitData is used by researchers, not just engineers:

- **"One less thing to remember."** Simplify the API; minimize what the user must track.
- **No abstractions on top of pure PyTorch.**
- **Simple, readable internal code** over clever tricks.
- **Backward-compatible APIs** with clear deprecation warnings.
- **Thorough tests** — valued even more than features. Reproduce a bug as a failing test, then fix.

## Branch & PR conventions

- Branch off `main` (never work on `main` directly). Name: `<type>/<issue-id>_<short-name>`, type ∈ `bugfix|feature|docs|tests`.
- Features: open a GitHub issue first. Ask "is this NECESSARY?" — LitData rejects PRs that only add engineering complexity.
- Link the issue in the PR; update/add tests and docs. Use `[wip]` / `[blocked by #N]` title tags when relevant.
- Output uses f-strings, except logging which stays lazy `%`-style: `logging.info("Hello %s!", name)`.
- Changelog: `src/litdata/CHANGELOG.md` (exempt from the large-file pre-commit hook).

## Before you push — run these locally

```bash
pre-commit run --all-files      # lint + format + all hooks (auto-fixes most issues)
ruff check .                    # add --fix to auto-fix
ruff format .
mypy                            # config: files=["src"], disallow_untyped_defs=true
pytest tests/path/test_x.py::test_name -v --capture=no    # see testing.md
```

## Style rules enforced (`pyproject.toml [tool.ruff]`)

- `line-length = 120`, `target-version = "py310"`.
- Rule sets: `E, F, W`, `S` (bandit), `UP` (pyupgrade), `I` (isort), `C4`, `D` (pydocstyle, **Google convention**), `PT`, `RET`, `SIM`, `NPY201`, `RUF100`. Max mccabe complexity 10.
- `assert` allowed (`S101` ignored). Docstring rules `D1xx` relaxed in `src/**`, `tests/**`, `examples/**`.
- `src/litdata/debugger.py` and `utilities/_pytree.py` fully excluded from ruff; several files excluded from mypy (`[tool.mypy] exclude`).
- New public code needs type annotations (`disallow_untyped_defs`).

## CI a PR must pass (`.github/workflows/`)

**`ci-testing.yml`** — job `pytester` (matrix, `fail-fast: false`):

- OS: `ubuntu-22.04`, `macos-14`, `windows-2022`. Python: `3.10–3.14` (windows excludes 3.13/3.14).
- Two phases: fast tests parallel (`pytest tests --ignore=tests/processing --ignore=tests/raw -n 2 --dist=loadgroup ...`), then processing/raw sequential. `--dist=loadgroup` is **required** (see testing.md).
- `--timeout=120` per test; FFmpeg installed for video tests; `UV_TORCH_BACKEND=cpu`.
- `testing-guardian` job is the required gate — fails the PR if any leg fails.

**`ci-checks.yml`** — reusable Lightning workflows: `check-typing` (mypy), `check-schema`, `check-package` (import test), `check-docs`, `check-md-links`.

**pre-commit.ci** runs hooks on the PR and auto-fixes. Hooks: ruff + ruff-format, codespell, mdformat, prettier (json/yml/toml), pyproject-fmt, validate-pyproject, large-file guard (max 350 kb).

## Adding examples / docs

- Runnable examples → `examples/` (subfolders `getting_started/`, `multi_modal/`, ...). Keep minimal; use public or synthetic data.
- README is the primary narrative doc; feature sections are `<details>` blocks. `README.md` is exempt from `mdformat`/`trailing-whitespace` hooks — keep its formatting consistent manually.
- Sphinx docs build from `docs/source` (`make docs`).
