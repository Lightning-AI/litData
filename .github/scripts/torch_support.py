"""Maintain the PyTorch support policy declared in `.github/torch-support.json`.

Subcommands:
    check   validate the policy against the ``torch`` lower bound in ``requirements.txt`` and the README
    export  publish ``latest`` / ``previous`` / ``minimum`` as step outputs for the CI matrix
    bump    refresh ``latest`` / ``previous`` from the newest stable release on PyPI, README included
            (``--dry-run`` only reports what would change)
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_POLICY_FILE = _ROOT / ".github" / "torch-support.json"
_REQUIREMENTS_FILE = _ROOT / "requirements.txt"
_README_FILE = _ROOT / "README.md"
# the README states the supported versions in the paragraph right after this marker, as the only two
# bold `**MAJOR.MINOR**` values in it — the wording is free to change, those two are rewritten by `bump`
_README_MARKER = "<!-- torch-support -->"
_TORCH_REQUIREMENT = re.compile(r"^torch\s*>=\s*(?P<version>[\w.]+)", re.MULTILINE)
_README_VERSION = re.compile(r"\*\*(\d+\.\d+)\*\*")
_MINOR = re.compile(r"^\d+\.\d+$")
_PATCH = re.compile(r"^\d+\.\d+\.\d+$")


def _load_policy() -> dict:
    """Return the parsed support policy."""
    return json.loads(_POLICY_FILE.read_text())


def _write_policy(policy: dict) -> None:
    """Write the support policy back, keeping the formatting prettier expects."""
    _POLICY_FILE.write_text(json.dumps(policy, indent=2) + "\n")


def _as_tuple(version: str) -> tuple:
    """Return a comparable tuple for a version string such as ``2.14`` or ``2.1.0``."""
    return tuple(int(part) for part in version.split("."))


def _declared_lower_bound() -> str:
    """Return the ``torch`` lower bound declared in ``requirements.txt``."""
    match = _TORCH_REQUIREMENT.search(_REQUIREMENTS_FILE.read_text())
    if match is None:
        raise SystemExit(f"no `torch >=...` requirement found in {_REQUIREMENTS_FILE.name}")
    return match.group("version")


def _readme_span(text: str) -> tuple:
    """Return the ``(start, end)`` offsets of the README paragraph that states the supported versions."""
    start = text.find(_README_MARKER)
    if start < 0:
        raise SystemExit(f"no `{_README_MARKER}` marker in {_README_FILE.name}")
    start += len(_README_MARKER)
    end = text.find("\n\n", start)
    return start, len(text) if end < 0 else end


def _readme_versions(text: str) -> list:
    """Return the supported versions the README announces."""
    start, end = _readme_span(text)
    return _README_VERSION.findall(text[start:end])


def _update_readme(latest: str, previous: str) -> None:
    """Rewrite the versions the README announces, leaving the wording around them alone."""
    text = _README_FILE.read_text()
    start, end = _readme_span(text)
    replacements = iter((f"**{latest}**", f"**{previous}**"))
    _README_FILE.write_text(
        text[:start] + _README_VERSION.sub(lambda _: next(replacements), text[start:end]) + text[end:]
    )


def _latest_on_pypi() -> str:
    """Return the newest stable PyTorch release published on PyPI, as ``MAJOR.MINOR``."""
    with urllib.request.urlopen("https://pypi.org/pypi/torch/json", timeout=30) as response:
        version = json.load(response)["info"]["version"]
    major, minor = version.split(".")[:2]
    return f"{major}.{minor}"


def _emit(**outputs: str) -> None:
    """Send step outputs to GitHub Actions, or to stdout when running locally."""
    lines = [f"{key}={value}" for key, value in outputs.items()]
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as fo:
            fo.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def check() -> None:
    """Fail if the policy is malformed or drifted apart from ``requirements.txt``."""
    policy = _load_policy()
    errors = []

    for key in ("latest", "previous"):
        if not _MINOR.match(policy[key]):
            errors.append(f"`{key}` must be a MAJOR.MINOR version, got {policy[key]!r}")
    # `minimum` is installed as an exact pin, so it needs the patch component
    if not _PATCH.match(policy["minimum"]):
        errors.append(f"`minimum` must be a MAJOR.MINOR.PATCH version, got {policy['minimum']!r}")
    if errors:
        raise SystemExit("\n".join(errors))

    latest, previous, minimum = policy["latest"], policy["previous"], policy["minimum"]
    if _as_tuple(previous) >= _as_tuple(latest):
        errors.append(f"`previous` ({previous}) must be older than `latest` ({latest})")
    if _as_tuple(minimum) > _as_tuple(previous):
        errors.append(f"`minimum` ({minimum}) must not be newer than `previous` ({previous})")

    declared = _declared_lower_bound()
    if declared != minimum:
        errors.append(
            f"{_REQUIREMENTS_FILE.name} pins `torch >={declared}` but the policy `minimum` is {minimum};"
            " update whichever is wrong"
        )

    announced = _readme_versions(_README_FILE.read_text())
    if announced != [latest, previous]:
        errors.append(
            f"{_README_FILE.name} announces PyTorch {announced} after the `{_README_MARKER}` marker,"
            f" expected exactly ['{latest}', '{previous}'] as bold `**MAJOR.MINOR**` values"
        )

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"supported PyTorch: {latest} and {previous}, minimum {minimum}")


def export() -> None:
    """Publish the version specs the CI matrix installs and names its jobs after."""
    policy = _load_policy()
    # version specs, installed as `torch==<spec>` and shown in the job names: the two supported
    # minors track patch releases, while the floor is exact, as that is the one version the
    # `torch >=` bound in requirements.txt actually promises
    _emit(
        latest_spec=f"{policy['latest']}.*",
        previous_spec=f"{policy['previous']}.*",
        minimum_spec=policy["minimum"],
    )


def bump(dry_run: bool = False) -> None:
    """Move the policy forward when PyTorch publishes a newer minor release."""
    policy = _load_policy()
    current, newest = policy["latest"], _latest_on_pypi()

    if _as_tuple(newest) <= _as_tuple(current):
        print(f"policy is up to date: latest PyTorch is {newest}, policy says {current}")
        _emit(changed="false", latest=current)
        return

    major, minor = (int(part) for part in newest.split("."))
    policy["latest"] = newest
    policy["previous"] = f"{major}.{minor - 1}" if minor else current
    if len(_readme_versions(_README_FILE.read_text())) != 2:
        raise SystemExit(f"{_README_FILE.name} must announce exactly two versions, refusing to bump")
    if not dry_run:
        _write_policy(policy)
        _update_readme(policy["latest"], policy["previous"])
    verb = "would bump" if dry_run else "bumped"
    print(f"{verb}: latest {current} -> {policy['latest']}, previous -> {policy['previous']}")
    _emit(changed="true", latest=policy["latest"], previous=policy["previous"])


def main() -> None:
    """Run the requested subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "export", "bump"))
    parser.add_argument("--dry-run", action="store_true", help="`bump`: report what would change, write nothing")
    args = parser.parse_args()
    if args.command == "bump":
        bump(dry_run=args.dry_run)
    else:
        {"check": check, "export": export}[args.command]()


if __name__ == "__main__":
    sys.exit(main())
