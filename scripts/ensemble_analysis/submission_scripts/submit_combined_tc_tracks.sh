#!/bin/bash -l

# Aggregator for multi-case TC tracks. Reads per-mode
# tc_tracks_<mode>.parquet from each case's <base>/diagnostics/tc_tracks/<tracker>/
# directory and writes a single N x 3 PNG combining all cases.
#$ -P eb-general
#$ -N CombinedTcTracks
#$ -cwd
#$ -j y
#$ -pe omp 2
#$ -l mem_per_core=8G
#$ -m a
#$ -M jlin404@bu.edu

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PYTHON_SCRIPT="${SCRIPT_DIR}/combine_tc_tracks.py"
LOG_DIR="${SCRIPT_DIR}/tc_tracks_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/combined_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom

echo "=========================================================="
echo "Job ID:    $JOB_ID"
echo "Node:      $HOSTNAME"
echo "CPUs:      $NSLOTS"
echo "Arguments: $*"
echo "=========================================================="

python3 "$PYTHON_SCRIPT" "$@"
EXITCODE=$?

if [ $EXITCODE -ne 0 ]; then
    echo "ERROR: combine_tc_tracks.py failed (exit $EXITCODE)"
    exit $EXITCODE
fi

echo "Done."
exit 0
