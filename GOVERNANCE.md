# Governance

LitData is an open source project backed by [Lightning AI](https://lightning.ai/), released under the [Apache-2.0](LICENSE) license.

## Roles and responsibilities

**Maintainers** review and merge contributions, triage issues, guide technical direction, and cut releases. The current maintainers are listed under [Maintainers](README.md#maintainers) in the README; ownership is scoped by path in [.github/CODEOWNERS](.github/CODEOWNERS).

**Contributors** are anyone contributing code, tests, docs, triage, or review. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

**Emeritus maintainers** are past maintainers, recorded under [Emeritus Maintainers](README.md#emeritus-maintainers) and not requested for review.

Maintainers are invited on the strength of sustained contribution, with the agreement of the current maintainers. A maintainer may step down, or be moved to emeritus status after a long period of inactivity, by a pull request updating the README roster and [.github/CODEOWNERS](.github/CODEOWNERS).

## Decision making

Changes land through pull requests against `main`; direct pushes are not used. A pull request needs passing CI and approval from a maintainer who owns the affected paths. Larger changes — new public APIs, breaking changes, new dependencies, changes to the on-disk chunk format or index schema — should start as an issue. Maintainers work by consensus and resolve disagreements in the open.

## Communication

GitHub issues and pull requests are the source of truth for decisions. Discussion also happens on [Discord](https://discord.com/invite/MWAEvnC5fU), but anything affecting the project is recorded on GitHub.

## Releases

LitData follows `MAJOR.MINOR.PATCH` versioning, with the version in [src/litdata/\_\_about\_\_.py](src/litdata/__about__.py).

- **Cadence:** cut as needed rather than on a fixed calendar, once a meaningful set of fixes or features has landed on `main`. Regressions and security fixes ship as soon as they are ready.
- **Criteria:** `main` is green across the CI matrix, including tests, typing, packaging, documentation, simulator, and benchmark checks.
- **Process:** a maintainer bumps the version and publishes a GitHub Release, which builds and publishes the distributions to PyPI via trusted publishing.
- **Supported versions:** supported Python and dependency ranges are documented in the project configuration and tested in CI. LitData is validated against the latest two PyTorch minor releases; support for a PyTorch or Python version is dropped only in a minor release and announced in the release notes.

## Code of Conduct

All participants follow the [Code of Conduct](CODE_OF_CONDUCT.md). Reports go to community@lightning.ai.

## Changing this document

By pull request, with approval from at least two maintainers.
