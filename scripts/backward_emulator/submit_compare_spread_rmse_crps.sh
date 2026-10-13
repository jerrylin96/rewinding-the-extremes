#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE wrapper -- backward-emulator spread/RMSE/CRPS comparison.
#
# Runs compare_spread_rmse_crps.py (makani vs cBottle, survivor-censored)
# for ONE (case, variable, region) as a batch job. Heavy pieces: the
# survivor scan streams t2m for all 1000 members (~50 GB I/O) and the
# cBottle overlay regrids the HPX ensemble -> lat/lon batched over members
# -- neither belongs on a login node, hence this job.
#
# Usage (from this directory):
#   qsub submit_compare_spread_rmse_crps.sh \
#       --makani-dir /projectnb/eb-general/jlin404/scratch/backward_emulator_data/ian \
#       --case ian --variable z500 --region-name synoptic \
#       --out-dir ./out/ian/spread_rmse_crps \
#       --makani-analysis-dir /projectnb/eb-general/jlin404/weather_interpolation/makani-aicr/analysis
#
# All args after the script name are forwarded to the Python script, so
# --no-cbottle / --no-censor / --cbottle-size / --batch-size all work here.
# ---------------------------------------------------------------------
#$ -P eb-general              # Project
#$ -N SRCRPScmp              # Job name
#$ -cwd                       # Run from submission directory
#$ -j y                       # Merge stdout/stderr (redirected below)
#$ -l h_rt=06:00:00           # Walltime (survivor scan + cBottle regrid)
#$ -pe omp 16                 # 16 CPU cores (regrid + zarr/h5 decode threads)
#$ -l mem_per_core=16G        # 256 GB total (cBottle regrid batches + region ens)
#$ -m bea                     # Mail: begin, end, abort
#$ -M jlin404@bu.edu

# NOTE: intentionally NO `set -euo pipefail`. conda's gcc_linux-64 activation
# script references SYS_SYSROOT before defining it, so nounset (`-u`) aborts
# `conda activate e2s-custom` with "SYS_SYSROOT: unbound variable" before any
# Python runs. The repo's other submit_*.sh wrappers avoid strict mode for the
# same reason and check exit codes explicitly instead (see conda activate and
# the python call below).

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PY="${SCRIPT_DIR}/compare_spread_rmse_crps.py"
if [ ! -f "$PY" ]; then
    echo "ERROR: cannot find $PY" >&2
    echo "       Submit from scripts/backward_emulator/: qsub $(basename "$0") ..." >&2
    exit 1
fi
LOG_DIR="${SCRIPT_DIR}/compare_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/job_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom || { echo "ERROR: cannot activate e2s-custom"; exit 1; }

export OMP_NUM_THREADS="${NSLOTS:-16}"
export MKL_NUM_THREADS="${NSLOTS:-16}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-16}"

echo "[submit] $(date)  host=$(hostname)  slots=${NSLOTS:-?}"
echo "[submit] python $PY $*"
python "$PY" "$@"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "[submit] FAILED (exit $STATUS) $(date)"
    exit "$STATUS"
fi
echo "[submit] done $(date)"
