#!/bin/bash
# waits for the mini #2 fleet (logs/mini.log) to finish,
# verifies completeness, archives the symmetric-cap ALD non-bare results as
# evidence, then relaunches the affected tasks with the asymmetric wall cap.
#
#   - bare/PF-ODE conditions are unaffected by the drift cap and are KEPT
#   - subexp1 *l1*_ald (l1 + l1l4) and subexp2 l1/l1l2/l1l2l3/l1l4 are moved
#     to results/phase1/archive/mini_symcap/ and re-run
#   - the run_phase1 worker skips existing files, so the rerun only computes
#     the moved-away tasks (100 + 40 = 140 tasks, ~13 h)
#
# If the fleet dies incompletely, a failure marker is written and NO rerun
# is started (do not clobber partial evidence).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=python
FLEET_LOG=logs/mini.log
DONE_MARK=logs/mini_DONE.txt
FAIL_MARK=logs/mini_INCOMPLETE.txt
ARCH=results/phase1/archive/mini_symcap

# wait for any run_phase1 --mini process (bracket avoids matching this script)
while pgrep -f "run_phase1[.]py --mini" > /dev/null; do sleep 300; done

done_lines=$(grep -c "worker done" "$FLEET_LOG" 2>/dev/null || true)
n1=$(find results/phase1/subexp1 -name "*.json" 2>/dev/null | wc -l)
n2=$(find results/phase1/subexp2 -name "*.json" 2>/dev/null | wc -l)

if [ "$done_lines" -ge 2 ] && [ "$n1" -eq 300 ] && [ "$n2" -eq 50 ]; then
    mkdir -p "$ARCH"
    for d in results/phase1/subexp1/*l1*_ald; do
        [ -d "$d" ] && mv "$d" "$ARCH/"
    done
    for d in l1 l1l2 l1l2l3 l1l4; do
        [ -d "results/phase1/subexp2/$d" ] && mv "results/phase1/subexp2/$d" "$ARCH/"
    done
    date "+%F %T" > "$DONE_MARK"
    echo "fleet done (worker lines=$done_lines, files=$n1/300+$n2/50); symcap ALD results archived; launching asymmetric-cap rerun" >> "$DONE_MARK"
    nohup "$PY" scripts/run_phase1.py --mini --subexp 1 --nnp mace --device cuda:0 \
        > logs/rerun_asym.log 2>&1 && \
    "$PY" scripts/run_phase1.py --mini --subexp 2 --nnp mace --device cuda:0 \
        >> logs/rerun_asym.log 2>&1 &
    echo "rerun launched (pid $!)" >> "$DONE_MARK"
else
    date "+%F %T" > "$FAIL_MARK"
    echo "fleet finished INCOMPLETE: worker_done_lines=$done_lines subexp1=$n1/300 subexp2=$n2/50 — no rerun launched" >> "$FAIL_MARK"
fi
