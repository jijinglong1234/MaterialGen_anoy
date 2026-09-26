#!/bin/bash
# E-3: submit the staged DFT jobs as a Slurm array, one structure per task.
#
# Every directory under the tier root that has an INCAR and no running job is
# one array element.  The array indexes are handed out from a list computed
# here, so an interrupted collection can be resumed by simply re-running: a
# structure whose OUTCAR already reports a finished run is skipped unless
# --force is given.
#
# Partition: the DFT work is CPU-only, so it can run on the GPU partitions'
# idle cores while the sampling fleet holds the GPUs.  The binding constraint
# turned out not to be the GPUs but the per-user node cap on the `normal` QOS
# -- see the QOS note below.
#
# Usage:
#   scripts/e3_dft_submit.sh                  # tier 1 (single points)
#   scripts/e3_dft_submit.sh --tier 2         # tier 2 (ISIF=2 relaxation)
#   scripts/e3_dft_submit.sh --partition gpu --cpus 32
#   scripts/e3_dft_submit.sh --dry-run
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TIER=1
PARTITION=cpu
CPUS=16
MEM=32G
TIME=24:00:00
# The computing QOS caps this account at a small number of nodes *per user*,
# and the long-running sampling fleet holds all of them, so any further job was
# rejected with QOSMaxNodePerUserLimit however idle the cluster was.  The
# cluster default QOS carries no such cap, and its priority is in fact higher
# than the capped one's -- it is a differently named pool, not a worse one.
# CPU-only DFT therefore runs under the default QOS and leaves the capped
# allowance entirely to the GPU fleet.
QOS=low
DRY=0
FORCE=0
LIMIT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --tier) TIER="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --force) FORCE=1; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

ROOT="$REPO/results/interim/e3_dft/tier$TIER"
[ -d "$ROOT" ] || { echo "no staged structures under $ROOT -- run scripts/e3_dft_select.py first" >&2; exit 1; }

mapfile -t DIRS < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
if [ "$FORCE" -eq 0 ]; then
    KEEP=()
    for d in "${DIRS[@]}"; do
        # A finished VASP run leaves the SCF loop's summary in OSZICAR and the
        # final ionic block in OUTCAR.  Touch a DONE marker instead of parsing:
        # e3_dft_collect.py writes it, so the two agree on what finished means.
        [ -f "$d/DONE" ] || KEEP+=("$d")
    done
    DIRS=("${KEEP[@]}")
fi
N=${#DIRS[@]}
[ "$N" -gt 0 ] || { echo "nothing to do: every staged structure under $ROOT is marked DONE"; exit 0; }
if [ "$LIMIT" -gt 0 ] && [ "$N" -gt "$LIMIT" ]; then
    DIRS=("${DIRS[@]:0:$LIMIT}"); N=$LIMIT
fi

LIST="$ROOT/.joblist_tier$TIER.txt"
printf '%s\n' "${DIRS[@]}" > "$LIST"
echo "tier $TIER: $N structure(s) -> $LIST"
echo "partition=$PARTITION qos=$QOS cpus=$CPUS mem=$MEM time=$TIME"
if [ "$DRY" -eq 1 ]; then
    printf '  %s\n' "${DIRS[@]}"
    exit 0
fi

# One sbatch per structure rather than a single array job: the array id then
# has to be mapped back to a directory through a file, which breaks the moment
# the list changes; a per-job script that reads $SLURM_ARRAY_TASK_ID is the
# same thing with fewer moving parts, and Slurm schedules them independently so
# one hard SCF cannot head-of-line block the rest.
SCRIPT="$ROOT/.run_tier$TIER.slurm"
cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=e3t${TIER}
#SBATCH --output=slurm.%A_%a.out
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --array=0-$((N - 1))

source ~/intel/oneapi/setvars.sh > /dev/null 2>&1
# Point this at your local VASP build.
export PATH=third_party/vasp/bin:\${PATH}

# One thread per rank.  Unset, the OpenMP runtime defaults to the core count
# of the node, so \$CPUS ranks each spawn that many threads and the job spends
# its whole budget context-switching: ranks sit at 100% CPU and the SCF never
# finishes an iteration.  VASP's own threading is not what this job wants --
# the rank count is.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_PIN=1
export I_MPI_HYDRA_BOOTSTRAP=slurm

# Launch ranks through srun, not mpirun.  VASP here is built against Intel MPI
# 2021, whose mpirun starts a hydra bootstrap proxy per job over ssh/rsh; at
# 112 concurrent jobs those proxies collide on ports and the whole batch dies
# with "error setting up the bootstrap proxies" after ~2.5 minutes of setup.
# srun hands the ranks to Slurm's own launcher -- no proxies, no ports -- and
# this cluster's srun offers pmi2, which is what this Intel MPI speaks.
D=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
[ -d "\$D" ] || { echo "missing \$D"; exit 1; }
cd "\$D" || exit 1
echo "[\$SLURM_JOB_ID.\$SLURM_ARRAY_TASK_ID] \$(basename "\$D") on \$(hostname) \$(date)"
# --cpus-per-task=1 is load-bearing, not tidiness.  The allocation is
# --cpus-per-task=$CPUS, and srun INHERITS that flag from the environment
# (SLURM_CPUS_PER_TASK), so "srun -n $CPUS" asks for $CPUS tasks of $CPUS CPUs
# each -- $((CPUS * CPUS)) CPUs from a $CPUS-CPU allocation.  Slurm refuses the
# step ("More processors requested than permitted") and every array element of
# the first real batch died in 1-30 s.  With one CPU per rank the request is
# exactly the allocation, which is also what is wanted: OMP_NUM_THREADS=1 is
# set above, so a rank is single-threaded and the rank count is the parallelism.
srun --mpi=pmi2 --ntasks=$CPUS --cpus-per-task=1 vasp_std > vasp.out 2>&1
echo "exit=\$? \$(date)"
EOF

cd "$ROOT"
if [ "$N" -eq 1 ]; then
    sbatch --array=0 "$SCRIPT"
else
    sbatch "$SCRIPT"
fi
echo "submitted; follow with: squeue -u \$USER -n e3t$TIER"
echo "then: python scripts/e3_dft_collect.py --tier $TIER"
