#!/bin/bash
# Wait for the Slurm controller to answer, then submit the E-3 DFT batch.
#
# The controller can stop answering entirely -- sinfo and sbatch both blocked
# for 60 s and returned "Socket timed out on send/recv operation" -- while
# already-running jobs kept running and writing output.  Since the batch is
# independent single points that can start whenever, polling for the
# controller is cheaper than losing the window.
#
# Usage: scripts/e3_dft_submit_when_up.sh [--tier N] [--cpus N] [--max-wait-min N]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TIER=1
CPUS=16
MAXWAIT=180
while [ $# -gt 0 ]; do
    case "$1" in
        --tier) TIER="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --max-wait-min) MAXWAIT="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

deadline=$(( $(date +%s) + MAXWAIT * 60 ))
echo "[$(date +%H:%M:%S)] waiting for the Slurm controller (up to ${MAXWAIT} min)"

while [ "$(date +%s)" -lt "$deadline" ]; do
    if timeout 20 sinfo -h -o "%P" > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] controller is answering -- submitting"
        "$HERE/e3_dft_submit.sh" --tier "$TIER" --partition cpu --cpus "$CPUS"
        exit $?
    fi
    sleep 60
done

echo "[$(date +%H:%M:%S)] gave up after ${MAXWAIT} min; submit by hand with"
echo "  scripts/e3_dft_submit.sh --tier $TIER --partition cpu --cpus $CPUS"
exit 1
