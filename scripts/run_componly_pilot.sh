#!/bin/bash
# E-1 composition-only arm, mini scale, ALD only, on the 4 free GPUs of the
# local host.
#
# Why the local host: the fleet's whole cluster GPU allowance is held by the
# long-running Phase-1 job, so only these cards are free.  They are the
# *faster* ones -- the shared partition runs ~2.5x slower -- so this pilot is
# comparable to a fleet run of the same task count.
#
# Scope: run_phase1 --subexp comp --mini --samplers ald
#   5 sigmas {0.1,0.5,1.0,2.0,5.0} x 3 arms x 5 compositions (one per chemistry
#   class) x 2 seeds x 10 candidates = 1500 trajectories, 150 task files, each
#   holding all 10 candidates.
#
# PF-ODE is excluded deliberately, not for cost alone: its high-sigma cells are
# budget-truncated (see scripts/probe_pfode_reached_sigma.py), so a composition-
# only ODE arm could not be read against the reference ODE arm without first
# separating prior effects from truncation effects.  The ALD arm has a fixed
# step budget, so the two priors are compared on equal work.
#
# Resumable: run_phase1 skips task files that exist and carry the current
# init_cell_gen marker, so re-running after an interruption continues.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY=python
LOGDIR="$REPO/results/interim/componly_pilot"
SHARDS=${SHARDS:-4}
mkdir -p "$LOGDIR"

cd "$REPO"
for k in $(seq 0 $((SHARDS - 1))); do
    "$PY" scripts/run_phase1.py --subexp comp --nnp mace --mini \
        --samplers ald --shard "$k/$SHARDS" --device "cuda:$k" \
        > "$LOGDIR/shard$k.log" 2>&1 &
    echo "shard $k -> cuda:$k (pid $!)"
done
wait
echo "all shards done: $(date)"
grep -h "worker done" "$LOGDIR"/shard*.log
