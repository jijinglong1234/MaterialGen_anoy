#!/bin/bash
# E-3 smoke test: run ONE staged structure under Slurm and verify VASP actually
# does a self-consistent-field cycle.
#
# Why this exists: the first real batch had already failed twice for two
# different reasons -- Intel MPI hydra proxy collisions (112 concurrent mpirun
# bootstraps), then srun refusing the step with "More processors requested than
# permitted" because srun inherits --cpus-per-task from the allocation, making
# "-n 16" mean 16 tasks x 16 CPUs.  A 112-element array is a bad place to
# discover a third.  This runs the same launcher line as the real batch, on its
# own list file so it cannot disturb .joblist_tier1.txt, and then asserts on the
# OUTCAR: a real run reaches "reached required accuracy" or at least produces a
# TOTEN, whereas the failure mode produced nothing at all.
#
# Usage: scripts/e3_dft_smoke.sh [--tier N] [--cpus N] [--max-wait-min N]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TIER=1
CPUS=16
MAXWAIT=240
while [ $# -gt 0 ]; do
    case "$1" in
        --tier) TIER="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --max-wait-min) MAXWAIT="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

ROOT="$REPO/results/interim/e3_dft/tier$TIER"
[ -d "$ROOT" ] || { echo "no staged structures under $ROOT" >&2; exit 1; }
# Pick a structure that has NOT run yet.  The two that already carry an OUTCAR
# (the Ag2S pilot structures) sort first alphabetically, and asserting on one of
# those would "pass" on stale output no matter how broken the launcher is.
D=""
for cand in $(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort); do
    if [ ! -f "$cand/OUTCAR" ]; then D="$cand"; break; fi
done
[ -n "$D" ] || { echo "every staged structure already has an OUTCAR; nothing to smoke-test" >&2; exit 1; }

deadline=$(( $(date +%s) + MAXWAIT * 60 ))
echo "[$(date +%H:%M:%S)] waiting for Slurm (up to ${MAXWAIT} min)"
until timeout 20 sinfo -h -o "%P" > /dev/null 2>&1; do
    [ "$(date +%s)" -ge "$deadline" ] && { echo "controller never came back" >&2; exit 1; }
    sleep 60
done

# Own list file + own script name so the real batch's files stay untouched.
LIST="$ROOT/.smokelist.txt"
printf '%s\n' "$D" > "$LIST"
SCRIPT="$ROOT/.smoke.slurm"
cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=e3smoke
#SBATCH --output=slurm.smoke.%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --qos=low
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=32G

source ~/intel/oneapi/setvars.sh > /dev/null 2>&1
# Point this at your local VASP build.
export PATH=third_party/vasp/bin:\${PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_PIN=1
export I_MPI_HYDRA_BOOTSTRAP=slurm
D=\$(sed -n "1p" "$LIST")
cd "\$D" || exit 1
echo "smoke: \$(basename "\$D") on \$(hostname) \$(date)"
srun --mpi=pmi2 --ntasks=$CPUS --cpus-per-task=1 vasp_std > vasp.out 2>&1
echo "srun exit=\$? \$(date)"
EOF

cd "$ROOT" || exit 1
JOBID="$(sbatch --array=0 "$SCRIPT" | grep -oE '[0-9]+')"
[ -n "$JOBID" ] || { echo "submit failed" >&2; exit 1; }
echo "[$(date +%H:%M:%S)] smoke job $JOBID on $(basename "$D")"

run_deadline=$(( $(date +%s) + 3600 ))
while :; do
    if q="$(timeout 25 squeue -h -j "$JOBID" 2>/dev/null)"; then
        [ -z "$q" ] && break
    fi
    [ "$(date +%s)" -ge "$run_deadline" ] && { echo "smoke job did not finish in 60 min" >&2; break; }
    sleep 30
done

echo "=== slurm out ==="
cat "$ROOT"/slurm.smoke."$JOBID".out 2>/dev/null | tail -6
echo "=== verdict ==="
if grep -q "reached required accuracy" "$D/OUTCAR" 2>/dev/null; then
    echo "PASS: OUTCAR reports 'reached required accuracy'"
elif grep -q "free  energy   TOTEN  =" "$D/OUTCAR" 2>/dev/null; then
    echo "PASS (partial): OUTCAR has a TOTEN but no converged SCF marker"
else
    echo "FAIL: no usable OUTCAR -- the launcher line is still wrong"
    tail -5 "$D/vasp.out" 2>/dev/null
    exit 1
fi
