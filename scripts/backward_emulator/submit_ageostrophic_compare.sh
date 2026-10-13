#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE wrapper -- backward-emulator ageostrophic comparison.
#
# Runs ageostrophic_compare.py (makani native vs cBottle regridded) for one
# case at 500 hPa over the midlatitude band. Heaviest piece: the cBottle
# overlay regrids u/v/z (3 channels) for all members HPX->lat/lon, so it needs
# more walltime than the single-variable spread job.
#
# Usage (from this directory):
#   qsub submit_ageostrophic_compare.sh \
#       --makani-dir /projectnb/eb-general/jlin404/scratch/backward_emulator_data/ian \
#       --case ian --out-dir ./out/ian/ageostrophic \
#       --survival-npz ./out/ian/survival_ian.npz \
#       --makani-analysis-dir /projectnb/eb-general/jlin404/weather_interpolation/makani-aicr/analysis
# ---------------------------------------------------------------------
#$ -P eb-general              # Project
#$ -N AgeoCmp                # Job name (overridden by dispatcher -N)
#$ -cwd                       # Run from submission directory
#$ -j y                       # Merge stdout/stderr (redirected below)
#$ -l h_rt=08:00:00           # Walltime (3-channel cBottle regrid over 1000 members)
#$ -pe omp 16                 # 16 CPU cores (regrid + h5/zarr decode threads)
#$ -l mem_per_core=16G        # 256 GB total (regrid batches hold u/v/z)
#$ -m bea                     # Mail: begin, end, abort
#$ -M jlin404@bu.edu

# NOTE: intentionally NO `set -euo pipefail`. conda's gcc_linux-64 activation
# references SYS_SYSROOT before defining it, so nounset (`-u`) aborts
# `conda activate e2s-custom` with "SYS_SYSROOT: unbound variable". Exit codes
# are checked explicitly instead.

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PY="${SCRIPT_DIR}/ageostrophic_compare.py"
if [ ! -f "$PY" ]; then
    echo "ERROR: cannot find $PY" >&2
    echo "       Submit from scripts/backward_emulator/: qsub $(basename "$0") ..." >&2
    exit 1
fi
LOG_DIR="${SCRIPT_DIR}/compare_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/ageo_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom || { echo "ERROR: cannot activate e2s-custom"; exit 1; }

export OMP_NUM_THREADS="${NSLOTS:-16}"
export MKL_NUM_THREADS="${NSLOTS:-16}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-16}"

echo "[ageo-job] $(date)  host=$(hostname)  slots=${NSLOTS:-?}"
echo "[ageo-job] python $PY $*"
python "$PY" "$@"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "[ageo-job] FAILED (exit $STATUS) $(date)"
    exit "$STATUS"
fi
echo "[ageo-job] done $(date)"
