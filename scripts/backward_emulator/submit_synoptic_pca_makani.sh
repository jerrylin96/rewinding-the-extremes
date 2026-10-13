#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE wrapper -- makani synoptic-PCA compute (Text S2).
#
# Runs synoptic_pca_makani.py: per-ensemble EOF on the free-end z500 field +
# leading-8-PC -> impact regression (in-sample + 10-fold CV), writing the
# makani npz the compare step reads against the cBottle npz.
#
# Ian needs the makani TC parquet (--tc-parquet) from the tc_tracks job.
# PNW needs the land mask (--land-mask) from make_land_mask.py.
#
# Usage (from this directory):
#   qsub submit_synoptic_pca_makani.sh \
#       --makani-dir /projectnb/.../backward_emulator_data/ian --case ian \
#       --out-dir ./out/ian/synoptic_pca \
#       --tc-parquet ./out/ian/tc_tracks/tc_tracks_makani_ian_end.parquet \
#       --survival-npz ./out/ian/survival_ian.npz
# ---------------------------------------------------------------------
#$ -P eb-general              # Project
#$ -N SynPCAmakani           # Job name (overridden by dispatcher -N)
#$ -cwd                       # Run from submission directory
#$ -j y                       # Merge stdout/stderr (redirected below)
#$ -l h_rt=04:00:00           # Walltime (region load + SVD + regression)
#$ -pe omp 16                 # 16 CPU cores
#$ -l mem_per_core=16G        # 256 GB total (region ensemble + optional survival scan)
#$ -m bea                     # Mail: begin, end, abort
#$ -M jlin404@bu.edu

# NOTE: intentionally NO `set -euo pipefail` (conda gcc_linux-64 + SYS_SYSROOT).

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PY="${SCRIPT_DIR}/synoptic_pca_makani.py"
if [ ! -f "$PY" ]; then
    echo "ERROR: cannot find $PY" >&2
    echo "       Submit from scripts/backward_emulator/: qsub $(basename "$0") ..." >&2
    exit 1
fi
LOG_DIR="${SCRIPT_DIR}/compare_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/synpca_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom || { echo "ERROR: cannot activate e2s-custom"; exit 1; }

export OMP_NUM_THREADS="${NSLOTS:-16}"
export MKL_NUM_THREADS="${NSLOTS:-16}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-16}"

echo "[synpca-job] $(date)  host=$(hostname)  slots=${NSLOTS:-?}"
echo "[synpca-job] python $PY $*"
python "$PY" "$@"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "[synpca-job] FAILED (exit $STATUS) $(date)"
    exit "$STATUS"
fi
echo "[synpca-job] done $(date)"
