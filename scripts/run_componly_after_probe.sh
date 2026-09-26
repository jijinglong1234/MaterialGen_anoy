#!/bin/bash
# Wait for the PF-ODE reached-sigma probe to release the local host's four
# GPUs, then start the composition-only pilot on them.
#
# Why chained rather than queued: the host has exactly four usable GPUs
# and the probe is already using all four.  Rather than poll by hand, this
# waits for the three re-run shards to write their JSON and then hands the
# same four cards to the pilot.  The probe's shard0/shard1/shard3 JSONs are
# written per task (scripts/probe_pfode_reached_sigma.py flushes after every
# trajectory), so a shard that finished is recognisable even mid-run.
#
# Usage: scripts/run_componly_after_probe.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PROBE="$REPO/results/interim/pfode_probe"
DEADLINE=$(( $(date +%s) + ${MAXWAIT_S:-3600} ))

echo "[$(date +%H:%M:%S)] waiting for the probe shards to finish"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    done=1
    for k in 0 1 2 3; do
        # A shard is finished when its log ends with the summary line; the
        # per-task JSON exists from the first trajectory onward and so cannot
        # be used as the completion signal.
        grep -q "wrote results/interim/pfode_probe/shard$k.json" "$PROBE/shard$k.log" 2>/dev/null || done=0
    done
    if [ "$done" -eq 1 ]; then
        echo "[$(date +%H:%M:%S)] all four shards done -- launching the comp-only pilot"
        exec "$HERE/run_componly_pilot.sh"
    fi
    sleep 60
done

echo "[$(date +%H:%M:%S)] timed out waiting for the probe; launch by hand with"
echo "  scripts/run_componly_pilot.sh"
exit 1
