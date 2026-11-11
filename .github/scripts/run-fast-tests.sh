#!/usr/bin/env bash
set -euo pipefail

case "${RUNNER_OS:-$(uname -s)}" in
  Linux) workers=4 ;;
  *) workers=2 ;;
esac

pytest tests \
  --ignore=tests/processing \
  --ignore=tests/raw \
  -n "$workers" --dist=loadgroup --cov=litdata --durations=0 --timeout=120 --capture=no --verbose
