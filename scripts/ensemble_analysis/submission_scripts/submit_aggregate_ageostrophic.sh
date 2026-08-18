#!/bin/bash -l

# Aggregator for ageostrophic. Reads per-mode ageostrophic_fraction_<mode>.npz
# under <output-dir> and writes the cross-mode overlay PNG.
#$ -P eb-general
#$ -N AggAgeo
#$ -cwd
#$ -j y
#$ -pe omp 2
#$ -l mem_per_core=8G
#$ -m a
#$ -M jlin404@bu.edu

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PYTHON_SCRIPT="${SCRIPT_DIR}/aggregate_ageostrophic.py"
LOG_DIR="${SCRIPT_DIR}/ageostrophic_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/aggregate_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom

python3 "$PYTHON_SCRIPT" "$@"
EXITCODE=$?
if [ $EXITCODE -ne 0 ]; then
    echo "ERROR: aggregate_ageostrophic.py failed (exit $EXITCODE)"
    exit $EXITCODE
fi
echo "Done."
