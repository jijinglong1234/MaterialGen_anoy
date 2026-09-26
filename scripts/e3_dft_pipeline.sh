#!/bin/bash
# E-3 end to end: wait for Slurm, submit the staged tier, wait for the array to
# drain, collect.
#
# Why one script rather than three steps by hand: the Slurm controller on this
# cluster has been answering intermittently (sbatch/squeue/sinfo blocking for
# 60 s and returning "Socket timed out on send/recv operation") while already
# running jobs kept running.  Each hand-off is a chance to lose the window, and
# the batch itself takes only minutes once it starts, so the whole sequence is
# chained here.  Every stage is idempotent: --force is not passed, so a re-run
# skips structures that already carry a DONE marker, and the collector re-parses
# whatever OUTCARs exist.
#
# Usage: scripts/e3_dft_pipeline.sh [--tier N] [--cpus N] [--partition P]
#                                  [--max-wait-min N] [--max-run-min N]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY=python
TIER=1
CPUS=16
PARTITION=cpu
MAXWAIT=240     # minutes to wait for the controller to answer
MAXRUN=360      # minutes to wait for the array to drain

while [ $# -gt 0 ]; do
    case "$1" in
        --tier) TIER="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --max-wait-min) MAXWAIT="$2"; shift 2 ;;
        --max-run-min) MAXRUN="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
cd "$REPO"

# ---------------------------------------------------------------- 1. Slurm up
deadline=$(( $(date +%s) + MAXWAIT * 60 ))
echo "[$(date +%H:%M:%S)] waiting for the Slurm controller (up to ${MAXWAIT} min)"
until timeout 20 sinfo -h -o "%P" > /dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[$(date +%H:%M:%S)] controller still down after ${MAXWAIT} min; giving up" >&2
        exit 1
    fi
    sleep 60
done
echo "[$(date +%H:%M:%S)] controller is answering"

# ---------------------------------------------------------------- 2. Submit
OUT="$(bash "$HERE/e3_dft_submit.sh" --tier "$TIER" --partition "$PARTITION" --cpus "$CPUS" 2>&1)"
echo "$OUT"
JOBID="$(printf '%s\n' "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | head -1)"
if [ -z "$JOBID" ]; then
    echo "[$(date +%H:%M:%S)] no job id in the submit output; collect by hand once the array drains:" >&2
    echo "  $PY scripts/e3_dft_collect.py --tier $TIER" >&2
    exit 1
fi
echo "[$(date +%H:%M:%S)] job $JOBID submitted"

# ---------------------------------------------------------------- 3. Drain
# squeue itself can time out; a failed query must not be read as "finished", so
# only an empty *successful* query counts as drained, and 3 consecutive empty
# answers are required before believing it.
run_deadline=$(( $(date +%s) + MAXRUN * 60 ))
empty=0
while :; do
    if q="$(timeout 25 squeue -h -j "$JOBID" 2>/dev/null)"; then
        if [ -z "$q" ]; then
            empty=$((empty + 1))
            [ "$empty" -ge 3 ] && break
        else
            empty=0
        fi
    fi
    if [ "$(date +%s)" -ge "$run_deadline" ]; then
        echo "[$(date +%H:%M:%S)] job $JOBID still not drained after ${MAXRUN} min; collecting what exists" >&2
        break
    fi
    sleep 60
done
echo "[$(date +%H:%M:%S)] array drained"

# ---------------------------------------------------------------- 4. Collect
"$PY" scripts/e3_dft_collect.py --tier "$TIER"
echo "[$(date +%H:%M:%S)] pipeline done"
