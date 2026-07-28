#!/usr/bin/env bash
# Append-only Stage 1 A/B resume into LITDATA_BENCH_ACCUM_OUT JSON files.
# Usage example (confirm gate fix):
#   BEFORE_ACCUM=... AFTER_ACCUM=... WORKERS=8,16,24 REPEATS_PAIRS=3 \
#     bash benchmarks/stage1_ab_resume.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SCRIPT=benchmarks/bench_raw_before_vs_after.py
BEFORE_PY=${BEFORE_PY:-/tmp/litdata-raw-pre-stage1/src}
AFTER_PY=${AFTER_PY:-src}
BEFORE_ACCUM=${BEFORE_ACCUM:?set BEFORE_ACCUM to existing before JSON}
AFTER_ACCUM=${AFTER_ACCUM:?set AFTER_ACCUM to existing after JSON}
WORKERS=${WORKERS:-2,4,8,16,24}
PREFETCH=${PREFETCH:-0,16}
CATCH_UP_AFTER=${CATCH_UP_AFTER:-0}
REPEATS_PAIRS=${REPEATS_PAIRS:-3}
GIT_SHA=${LITDATA_BENCH_GIT:-$(git rev-parse --short HEAD)}
PY=${PY:-python}

log() { echo "$(date -u +%H:%M:%S) $*"; }

run_side() {
  local side=$1 pypath=$2 accum=$3
  log "=== resume side=$side accum=$(basename "$accum") ==="
  env \
    PYTHONPATH="$pypath" \
    LITDATA_BENCH_GIT="$GIT_SHA" \
    LITDATA_BENCH_ACCUM_OUT="$(pwd)/$accum" \
    "$PY" "$SCRIPT" \
      --side "$side" \
      --workers "$WORKERS" \
      --after-prefetch "$PREFETCH" \
      --batches 300 \
      --min-seconds 30 \
      --repeats 1
}

log "Stage 1 A/B resume start pid=$$ sha=$GIT_SHA workers=$WORKERS pairs=$REPEATS_PAIRS"
if [[ "$CATCH_UP_AFTER" == "1" ]]; then
  run_side after "$AFTER_PY" "$AFTER_ACCUM"
fi
for ((i=1; i<=REPEATS_PAIRS; i++)); do
  log "=== pair $i/$REPEATS_PAIRS ==="
  run_side before "$BEFORE_PY" "$BEFORE_ACCUM"
  run_side after "$AFTER_PY" "$AFTER_ACCUM"
done
log "=== merge ==="
env PYTHONPATH=src LITDATA_BENCH_GIT="$GIT_SHA" "$PY" "$SCRIPT" --merge
log "Stage 1 A/B resume complete"
