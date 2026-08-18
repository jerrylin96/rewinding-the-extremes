#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE Directives — Free-end-state cross-mode aggregator
#
# Lightweight job: reads per-mode free_end_states_<mode>.npz files and
# produces the aggregated PNG. Dispatched with -hold_jid on the per-mode
# compute jobs.
# ---------------------------------------------------------------------
#$ -P eb-general
#$ -N AggFreeEndStates
#$ -cwd
#$ -j y
#$ -pe omp 2
#$ -l mem_per_core=8G
#$ -m a
#$ -M jlin404@bu.edu

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PYTHON_SCRIPT="${SCRIPT_DIR}/aggregate_free_end_states.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: cannot find $PYTHON_SCRIPT" >&2
    exit 1
fi
LOG_DIR="${SCRIPT_DIR}/free_end_states_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/aggregate_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom

# Expected args (forwarded by dispatcher):
#   --output-root <path>     (root: <case>/diagnostics/free_end_states)
#   --variables <var1,var2,...>

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
    echo "ERROR: Missing --output-root or --variables."
    exit 1
fi

echo "=========================================================="
echo "Aggregator (free_end_states)"
echo "Output root:   $OUTPUT_ROOT"
echo "Variables:     $VARIABLES"
echo "=========================================================="

IFS=',' read -ra VAR_ARRAY <<< "$VARIABLES"
FAILED=0
for VAR in "${VAR_ARRAY[@]}"; do
    VAR_DIR="${OUTPUT_ROOT}/${VAR}"
    echo ""
    echo "[aggregate] $VAR -> $VAR_DIR"
    python3 "$PYTHON_SCRIPT" --output-dir "$VAR_DIR"
    if [ $? -ne 0 ]; then
        echo "WARNING: aggregator failed for $VAR"
        ((FAILED+=1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED variable(s) failed"
    exit 1
fi
echo "Done."
exit 0
