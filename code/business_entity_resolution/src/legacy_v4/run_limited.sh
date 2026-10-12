#!/usr/bin/env bash
# Run a pipeline step inside a memory-capped systemd scope so an OOM only kills the step.
# Usage: ./run_limited.sh <log_name> <python> <script.py> [args]
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="${BER_WORK:-$HERE/../../work}/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/$1.log"; shift
cd "$HERE/src" && POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-8} \
  systemd-run --user --scope -q -p MemoryMax=${MEM_MAX:-10G} -p MemorySwapMax=0 "$@" > "$LOG" 2>&1
RC=$?
echo "exit=$RC" >> "$LOG"
exit $RC
