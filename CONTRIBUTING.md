# Contributing to LitData

We welcome all contributions, whatever your level of experience — bug fixes, features, examples, and docs improvements are all appreciated.

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). New to open source? Start with [GitHub's guide to your first contribution](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

## What we're looking for

- **Bug fixes** — especially streaming correctness, resumption, distributed sampling, and cloud storage backends.
- **Features** — optimize/map pipelines, item loaders, serializers, and new data formats.
- **Integrations and examples** — streaming a new dataset type or storage provider.
- **Docs** — clarifications, missing docstrings, better examples.
- **Performance** — throughput or memory regressions, ideally with a reproducible benchmark.

## How to contribute

1. **Open an issue** first, so we can align on scope. For bugs, include your setup, expected vs. actual behaviour, and a minimal reproduction. For features, explain the motivation with a use case.
2. **Fork the repo** and create a branch off `main`. Name it `<type>/<issue-id>_<short-name>`, for example `bugfix/237_uneven-batches`. Types we use: `bugfix`, `feature`, `docs`, `tests`.
3. **Write a test first** where you can: one that fails on `main` and passes with your change.
4. **Open a pull request** against `main`, describing what changed and why, and linking the issue.

Looking for somewhere to start? Try a [good first issue](https://github.com/Lightning-AI/litdata/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) or [help wanted](https://github.com/Lightning-AI/litdata/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22), and comment so we can assign it to you. [PR #237](https://github.com/Lightning-AI/litdata/pull/237) is a good example to model yours on. And if you can't find the fix, a PR with just a failing test is still a valuable contribution — we can finish it together.

## Development setup

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) and `make`, the same as CI.

```bash
# 1. clone your fork
git clone https://github.com/{YOUR_USERNAME}/litdata.git
cd litdata

# 2. create and activate an environment
#    any Python version in `python_requires` (see setup.py) works
uv venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. install dependencies and pre-commit hooks
make setup
```

That's it — you're ready to go! 🎉

If `make` is missing: `sudo apt-get install build-essential` on Debian-based systems, `brew install make` on macOS. On [Lightning Studio](https://lightning.ai/studios), skip step 2.

## Running tests

Tests use [pytest](https://docs.pytest.org/en/stable/) and live in `tests/`, mirroring `src/litdata/`. Add yours to the file covering that area, or create one.

```bash
pytest tests/ -v                                          # everything
pytest tests/streaming/test_dataset.py -v                 # one file
pytest tests/streaming/test_dataset.py::test_streaming_dataset  # one test
```

Style is enforced by [ruff](https://docs.astral.sh/ruff/) via pre-commit. `make setup` installs the hooks, so they run automatically on every commit. To check everything before pushing:

```bash
pre-commit run --all-files

# using uvx
uvx pre-commit run --all-files
```

## Guidelines

- Keep pull requests focused — one logical change per PR.
- Write tests for new functionality, and a regression test alongside any bug fix.
- Follow the existing code style (enforced via [ruff](https://docs.astral.sh/ruff/) and pre-commit).
- All code should be your own original work; third-party snippets must be attributed, and new dependencies kept at least as permissive as our [Apache-2.0 license](LICENSE).

## Reviews and merging

Anyone in the community is welcome to review — you don't have to be a maintainer, and an extra pair of eyes always helps.

Not ready for review but want CI to run? Open a **draft PR**, and prefix the title with **\[blocked by #<number>\]** if it depends on another PR.

Merging needs green CI and approval from a [code owner](.github/CODEOWNERS) for the paths you touched. Reviews can take a few days; if your PR goes quiet for a week, feel free to ping. See [GOVERNANCE.md](GOVERNANCE.md) for how decisions get made.

## Keeping your branch up to date

If `main` moves on while your PR is open, rebase rather than merge, so history stays linear. Point your fork at upstream once:

```bash
git remote add upstream https://github.com/Lightning-AI/litdata.git
git remote -v  # origin = your fork, upstream = Lightning-AI
```

Then, whenever you need to catch up:

```bash
git fetch --all --prune
git rebase upstream/main
# resolve any conflicts, following git's instructions
git push -f origin {BRANCH_NAME}  # -f is needed after a rebase
```

## Getting help

Stuck? You don't have to figure it out alone — ask on [Discord](https://discord.com/invite/MWAEvnC5fU) for the quickest answer, [open an issue](https://github.com/Lightning-AI/litdata/issues) to report a bug, or browse the [documentation](https://lightning.ai/docs/litdata).

Thanks for helping make LitData better! 💜
