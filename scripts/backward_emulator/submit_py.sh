#!/bin/bash -l

# ---------------------------------------------------------------------
# Generic BU SCC SGE wrapper: run ANY python script in e2s-custom as a
# batch job, so figure / compare / post-processing steps don't have to run
# interactively (and don't die if your ssh session drops).
#
# Usage (from scripts/backward_emulator/):
#   qsub submit_py.sh <script.py> [args...]
#   e.g. qsub submit_py.sh plot_tc_tracks_compare.py --case ian \
#            --makani-parquet ./out/ian/tc_tracks/tc_tracks_makani_ian_end.parquet \
#            --cbottle-parquet /projectnb/.../tc_tracks/tempest/tc_tracks_end.parquet \
#            --out-dir ./out/ian/tc_tracks --survival-npz ./out/ian/survival_ian.npz
#
# Defaults are modest (4 cores / 32 GB / 2 h) -- fine for reading npz/parquet and
# plotting. For a heavier script, override on the qsub line (CLI flags win over the
# embedded directives), e.g.:
#   qsub -l h_rt=04:00:00 -pe omp 16 submit_py.sh <script.py> ...
# The log lands in compare_logs/py_<script-stem>_<jobid>.log.
# ---------------------------------------------------------------------
#$ -P eb-general              # Project
#$ -N Py                     # Job name (override with qsub -N)
#$ -cwd                       # Run from the submission directory
#$ -j y                       # Merge stdout/stderr (redirected below)
#$ -l h_rt=02:00:00           # Walltime (override for heavier scripts)
#$ -pe omp 4                  # 4 CPU cores
#$ -l mem_per_core=8G         # 32 GB total
#$ -m bea                     # Mail on begin / end / abort
#$ -M jlin404@bu.edu

# NOTE: intentionally NO `set -euo pipefail` -- conda's gcc_linux-64 activation
# references SYS_SYSROOT before defining it, so nounset aborts `conda activate`.

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
if [ "$#" -lt 1 ]; then
    echo "usage: qsub submit_py.sh <script.py> [args...]" >&2
    exit 1
fi
STEM="$(basename "$1" .py)"
LOG_DIR="${SCRIPT_DIR}/compare_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/py_${STEM}_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom || { echo "ERROR: cannot activate e2s-custom"; exit 1; }

export OMP_NUM_THREADS="${NSLOTS:-4}"
export MKL_NUM_THREADS="${NSLOTS:-4}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-4}"

echo "[py-job] $(date)  host=$(hostname)  slots=${NSLOTS:-?}"
echo "[py-job] python $*"
python "$@"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "[py-job] FAILED (exit $STATUS) $(date)"
    exit "$STATUS"
fi
echo "[py-job] done $(date)"
