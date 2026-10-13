#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE wrapper -- makani TC tracking (TempestExtremes, Text S3).
#
# Runs tc_tracks_compare.py: DetectNodes + StitchNodes per makani member
# (native 0.25 deg fields) + the ERA5 reference (makani truth), writing a
# parquet in the paper's schema for the Ian synoptic-PCA impact target and
# the TC-track aggregator.
#
# PRE-FETCH THE OROGRAPHY FIRST on a login node (network), so the compute
# node reads it from EARTH2STUDIO_CACHE instead of hitting WB2:
#   python smoke_tc_tracks.py --case ian        # fetches + caches, then tracks 2
#
# Usage (from this directory):
#   qsub submit_tc_tracks_compare.sh \
#       --makani-dir /projectnb/eb-general/jlin404/scratch/backward_emulator_data/ian \
#       --case ian --out-dir ./out/ian/tc_tracks
# ---------------------------------------------------------------------
#$ -P eb-general              # Project
#$ -N TCtracksMakani         # Job name (overridden by dispatcher -N)
#$ -cwd                       # Run from submission directory
#$ -j y                       # Merge stdout/stderr (redirected below)
#$ -l h_rt=12:00:00           # Walltime (1000 members x DetectNodes+StitchNodes)
#$ -pe omp 16                 # 16 CPU cores
#$ -l mem_per_core=16G        # 256 GB total (one member slab at a time)
#$ -m bea                     # Mail: begin, end, abort
#$ -M jlin404@bu.edu

# NOTE: intentionally NO `set -euo pipefail`. conda's gcc_linux-64 activation
# references SYS_SYSROOT before defining it, so nounset (`-u`) aborts
# `conda activate e2s-custom`. Exit codes are checked explicitly instead.

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PY="${SCRIPT_DIR}/tc_tracks_compare.py"
if [ ! -f "$PY" ]; then
    echo "ERROR: cannot find $PY" >&2
    echo "       Submit from scripts/backward_emulator/: qsub $(basename "$0") ..." >&2
    exit 1
fi
LOG_DIR="${SCRIPT_DIR}/compare_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/tctracks_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom || { echo "ERROR: cannot activate e2s-custom"; exit 1; }

export OMP_NUM_THREADS="${NSLOTS:-16}"
export MKL_NUM_THREADS="${NSLOTS:-16}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-16}"

echo "[tc-job] $(date)  host=$(hostname)  slots=${NSLOTS:-?}"
echo "[tc-job] python $PY $*"
python "$PY" "$@"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "[tc-job] FAILED (exit $STATUS) $(date)"
    exit "$STATUS"
fi
echo "[tc-job] done $(date)"
