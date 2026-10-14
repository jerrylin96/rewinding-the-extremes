#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE wrapper -- backward-emulator survivor-mask precompute.
#
# Runs survival.py once per case to write the survivor mask
# (survival_<case>.npz).  The mask depends only on t2m, so it is identical
# for every (variable, region) in a case; the dispatcher submits this as a
# root job and -hold_jid's the per-variable compare jobs on it, which then
# LOAD the mask via --survival-npz instead of each rescanning ~50 GB of t2m.
#
# Usage (from this directory):
#   qsub submit_survival.sh \
#       --makani-dir /projectnb/eb-general/jlin404/scratch/backward_emulator_data/ian \
#       --out ./out/ian/survival_ian.npz \
#       --makani-analysis-dir /projectnb/eb-general/jlin404/weather_interpolation/makani-aicr/analysis
# ---------------------------------------------------------------------
#$ -P eb-general              # Project
#$ -N SurvScan               # Job name (overridden by dispatcher -N)
#$ -cwd                       # Run from submission directory
#$ -j y                       # Merge stdout/stderr (redirected below)
#$ -l h_rt=03:00:00           # Walltime (~50 GB t2m stream)
#$ -pe omp 16                 # 16 CPU cores (h5 decode threads)
#$ -l mem_per_core=16G        # region ens + per-member series
#$ -m bea                     # Mail: begin, end, abort
#$ -M jlin404@bu.edu

# NOTE: intentionally NO `set -euo pipefail` -- conda's gcc_linux-64 activation
# references SYS_SYSROOT before defining it, so nounset (`-u`) aborts
# `conda activate` with "SYS_SYSROOT: unbound variable" before any Python runs.
# Exit codes are checked explicitly instead (see below).

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PY="${SCRIPT_DIR}/survival.py"
if [ ! -f "$PY" ]; then
    echo "ERROR: cannot find $PY" >&2
    echo "       Submit from scripts/backward_emulator/: qsub $(basename "$0") ..." >&2
    exit 1
fi
LOG_DIR="${SCRIPT_DIR}/compare_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/survival_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom || { echo "ERROR: cannot activate e2s-custom"; exit 1; }

export OMP_NUM_THREADS="${NSLOTS:-16}"
export MKL_NUM_THREADS="${NSLOTS:-16}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-16}"

echo "[survival-job] $(date)  host=$(hostname)  slots=${NSLOTS:-?}"
echo "[survival-job] python $PY $*"
python "$PY" "$@"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "[survival-job] FAILED (exit $STATUS) $(date)"
    exit "$STATUS"
fi
echo "[survival-job] done $(date)"
