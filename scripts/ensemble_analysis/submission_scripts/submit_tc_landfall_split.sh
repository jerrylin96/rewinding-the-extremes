#!/bin/bash -l

# End-conditioned landfall-split TC diagnostic (Hurricane Ian).  Reads
# tc_tracks_end.parquet under the --output-dir tracker directory and writes
# tc_landfall_split.png there.  All args are forwarded to the Python worker.
#$ -P eb-general
#$ -N TcLandfallSplit
#$ -cwd
#$ -j y
#$ -pe omp 2
#$ -l mem_per_core=8G
#$ -m a
#$ -M jlin404@bu.edu

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PYTHON_SCRIPT="${SCRIPT_DIR}/tc_landfall_split.py"
LOG_DIR="${SCRIPT_DIR}/tc_tracks_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/landfall_split_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom

python3 "$PYTHON_SCRIPT" "$@"
