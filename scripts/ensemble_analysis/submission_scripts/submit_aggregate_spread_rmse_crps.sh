#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE Directives — Spread/RMSE/CRPS cross-mode aggregator
#
# Lightweight job: reads per-mode spread_rmse_crps_<mode>.npz files
# and produces the aggregated plots. Dispatched with -hold_jid on the
# three per-mode compute jobs.
# ---------------------------------------------------------------------
#$ -P eb-general
#$ -N AggSpreadRMSE
#$ -cwd
#$ -j y
#$ -pe omp 2
#$ -l mem_per_core=8G
#$ -m a
#$ -M jlin404@bu.edu

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PYTHON_SCRIPT="${SCRIPT_DIR}/aggregate_spread_rmse_crps.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: cannot find $PYTHON_SCRIPT" >&2
    exit 1
fi
LOG_DIR="${SCRIPT_DIR}/spread_rmse_crps_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/aggregate_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom

# Expected args (forwarded by dispatcher):
#   --output-root <path>     (root: <case>/diagnostics/spread_rmse_crps)
#   --variables <var1,var2,...>
#   [--region-name <name>]   (selects the regional npz set to aggregate)
#   [--mask <none|land|sea>] (selects the masked npz set to aggregate)

ARGS=("$@")
OUTPUT_ROOT=""
VARIABLES=""
REGION_ARGS=()

i=0
while [ $i -lt ${#ARGS[@]} ]; do
    case "${ARGS[i]}" in
        --output-root) OUTPUT_ROOT="${ARGS[i+1]}"; ((i+=2)) ;;
        --variables)   VARIABLES="${ARGS[i+1]}";   ((i+=2)) ;;
        --region-name)
            REGION_ARGS+=(--region-name "${ARGS[i+1]}"); ((i+=2)) ;;
        --mask)
            REGION_ARGS+=(--mask "${ARGS[i+1]}"); ((i+=2)) ;;
        *)             ((i+=1)) ;;
    esac
done

if [ -z "$OUTPUT_ROOT" ] || [ -z "$VARIABLES" ]; then
    echo "ERROR: Missing --output-root or --variables."
    exit 1
fi

echo "=========================================================="
echo "Aggregator (spread_rmse_crps)"
echo "Output root:   $OUTPUT_ROOT"
echo "Variables:     $VARIABLES"
echo "Region args:   ${REGION_ARGS[*]:-<global>}"
echo "=========================================================="

IFS=',' read -ra VAR_ARRAY <<< "$VARIABLES"
FAILED=0
for VAR in "${VAR_ARRAY[@]}"; do
    VAR_DIR="${OUTPUT_ROOT}/${VAR}"
    echo ""
    echo "[aggregate] $VAR -> $VAR_DIR"
    python3 "$PYTHON_SCRIPT" --output-dir "$VAR_DIR" "${REGION_ARGS[@]}"
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
