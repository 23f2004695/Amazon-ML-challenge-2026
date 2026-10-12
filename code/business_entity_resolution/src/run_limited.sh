#!/usr/bin/env bash
# Run a pipeline step inside a memory-capped systemd scope so an out-of-memory condition kills only
# that step. Usage (from src/): ./run_limited.sh <log_name> <python> <script.py> [args]
# MEM_MAX (default 10G) sets the cap. Without systemd, run the same command directly from src/.
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="${BER_WORK:-${BER_ROOT:-$HERE/../../..}/work}/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/$1.log"; shift
cd "$HERE" && POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-8} \
  systemd-run --user --scope -q -p MemoryMax=${MEM_MAX:-10G} -p MemorySwapMax=0 "$@" > "$LOG" 2>&1
RC=$?
echo "exit=$RC" >> "$LOG"
exit $RC
