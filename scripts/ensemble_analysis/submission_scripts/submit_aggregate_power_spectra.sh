#!/bin/bash -l

# Aggregator for power_spectra. Reads per-mode power_spectra_<mode>.npz
# under <output-root>/<variable>/ and writes the cross-mode overlay PNG.
#$ -P eb-general
#$ -N AggPowerSpec
#$ -cwd
#$ -j y
#$ -pe omp 2
#$ -l mem_per_core=8G
#$ -m a
#$ -M jlin404@bu.edu

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PYTHON_SCRIPT="${SCRIPT_DIR}/aggregate_power_spectra.py"
LOG_DIR="${SCRIPT_DIR}/power_spectra_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/aggregate_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom

ARGS=("$@")
OUTPUT_ROOT=""
VARIABLES=""
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    case "${ARGS[i]}" in
        --output-root) OUTPUT_ROOT="${ARGS[i+1]}"; ((i+=2)) ;;
        --variables)   VARIABLES="${ARGS[i+1]}";   ((i+=2)) ;;
        *)             ((i+=1)) ;;
    esac
done

if [ -z "$OUTPUT_ROOT" ] || [ -z "$VARIABLES" ]; then
    echo "ERROR: Missing --output-root or --variables." >&2
    exit 1
fi

IFS=',' read -ra VAR_ARRAY <<< "$VARIABLES"
FAILED=0
for VAR in "${VAR_ARRAY[@]}"; do
    VAR_DIR="${OUTPUT_ROOT}/${VAR}"
    echo "[aggregate] $VAR -> $VAR_DIR"
    python3 "$PYTHON_SCRIPT" --output-dir "$VAR_DIR" || ((FAILED+=1))
done

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED variable(s) failed"
    exit 1
fi
echo "Done."
