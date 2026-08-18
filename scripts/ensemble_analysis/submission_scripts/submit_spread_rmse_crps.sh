#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE Directives — Spread / RMSE / CRPS, full ensemble
#
# Case-agnostic wrapper: parses --variables out of the dispatcher's
# argv and runs compute_spread_rmse_crps.py once per variable.
# ---------------------------------------------------------------------
#$ -P eb-general              # Project Name
#$ -N SpreadRMSE              # Job Name
#$ -cwd                       # Run from the submission directory
#$ -j y                       # Merge Output/Error (file written to spread_rmse_crps_logs/, see below)
# h_rt is set via qsub -l flags from dispatch_spread_rmse_crps.py
#$ -pe omp 16                 # 16 CPU cores (torch + zarr decode threads)
#$ -l mem_per_core=16G        # 256 GB total — diagnostics_from_zarr loads all-at-once
#$ -m bea                     # Mail at: (b)eginning, (e)nd, (a)bort
#$ -M jlin404@bu.edu          # Email address

# ---------------------------------------------------------------------
# 1a. Locate this script's source directory + self-managed log directory
#
# Under SGE the running script is a copy in /var/spool/sge/.../job_scripts/,
# so $0 (and readlink -f "$0") points at the spool — not at the original
# submission_scripts/ directory that holds the colocated Python source.
# SGE sets SGE_O_WORKDIR to the directory from which `qsub` was invoked;
# combined with `-cwd` above the job runs there too, so we use it as the
# canonical script directory and fall back to resolving $0 for local runs.
#
# SGE's `-o` directive is evaluated at submit time and requires the target
# directory to already exist, so we redirect output from inside the script
# instead — that way the user never has to remember to mkdir before qsub.
# ---------------------------------------------------------------------

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PYTHON_SCRIPT="${SCRIPT_DIR}/compute_spread_rmse_crps.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: cannot find $PYTHON_SCRIPT" >&2
    echo "       Submit this job from the submission_scripts/ directory:" >&2
    echo "         cd .../scripts/ensemble_analysis/submission_scripts && qsub $(basename "$0")" >&2
    exit 1
fi
LOG_DIR="${SCRIPT_DIR}/spread_rmse_crps_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/job_${JOB_ID:-local-$$}.log" 2>&1

# ---------------------------------------------------------------------
# 1b. Environment setup
# ---------------------------------------------------------------------

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment e2s-custom"
    exit 1
fi

export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

# ---------------------------------------------------------------------
# 2. Parse arguments
#
# Expected arguments from dispatch_spread_rmse_crps.py:
#   --ensemble-zarr <path>
#   --era5-zarr <path>
#   --output-root <path>            (root: <case>/diagnostics/spread_rmse_crps)
#   --mode <start|end|both>         (encoded in per-mode output filenames)
#   --variables <var1,var2,...>
#   --case-name <display name>
#   --landfall-frame <int>          (-1 to disable)
#   [--n-members <N>]
# ---------------------------------------------------------------------

ARGS=("$@")

ENSEMBLE_ZARR=""
ERA5_ZARR=""
OUTPUT_ROOT=""
VARIABLES=""
EXTRA_ARGS=()

i=0
while [ $i -lt ${#ARGS[@]} ]; do
    case "${ARGS[i]}" in
        --ensemble-zarr)
            ENSEMBLE_ZARR="${ARGS[i+1]}"; ((i+=2)) ;;
        --era5-zarr)
            ERA5_ZARR="${ARGS[i+1]}"; ((i+=2)) ;;
        --output-root)
            OUTPUT_ROOT="${ARGS[i+1]}"; ((i+=2)) ;;
        --variables)
            VARIABLES="${ARGS[i+1]}"; ((i+=2)) ;;
        *)
            EXTRA_ARGS+=("${ARGS[i]}"); ((i+=1)) ;;
    esac
done

if [ -z "$ENSEMBLE_ZARR" ] || [ -z "$ERA5_ZARR" ] || [ -z "$OUTPUT_ROOT" ] || [ -z "$VARIABLES" ]; then
    echo "ERROR: Missing required arguments."
    echo "Usage: submit_spread_rmse_crps.sh --ensemble-zarr <path> --era5-zarr <path> --output-root <path> --variables <var1,var2,...>"
    exit 1
fi

if [ ! -d "$ENSEMBLE_ZARR" ]; then
    echo "ERROR: Ensemble zarr does not exist: $ENSEMBLE_ZARR"
    exit 1
fi
if [ ! -d "$ERA5_ZARR" ]; then
    echo "ERROR: ERA5 zarr does not exist: $ERA5_ZARR"
    exit 1
fi

echo "=========================================================="
echo "Job ID:        $JOB_ID"
echo "Node:          $HOSTNAME"
echo "CPUs:          $NSLOTS"
echo "Ensemble zarr: $ENSEMBLE_ZARR"
echo "ERA5 zarr:     $ERA5_ZARR"
echo "Output root:   $OUTPUT_ROOT"
echo "Variables:     $VARIABLES"
echo "Extra args:    ${EXTRA_ARGS[*]}"
echo "=========================================================="

# ---------------------------------------------------------------------
# 3. Loop over variables, calling compute_spread_rmse_crps.py once each
# ---------------------------------------------------------------------

IFS=',' read -ra VAR_ARRAY <<< "$VARIABLES"
TOTAL=${#VAR_ARRAY[@]}
FAILED=0

for idx in "${!VAR_ARRAY[@]}"; do
    VAR="${VAR_ARRAY[idx]}"
    VAR_OUTPUT_DIR="${OUTPUT_ROOT}/${VAR}"
    N=$((idx + 1))

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "[$N/$TOTAL] Variable: $VAR"
    echo "  Output: $VAR_OUTPUT_DIR"
    echo "────────────────────────────────────────────────────────"

    python3 "$PYTHON_SCRIPT" \
        --ensemble-zarr "$ENSEMBLE_ZARR" \
        --era5-zarr "$ERA5_ZARR" \
        --output-dir "$VAR_OUTPUT_DIR" \
        --var-name "$VAR" \
        "${EXTRA_ARGS[@]}"

    EXITCODE=$?
    if [ $EXITCODE -ne 0 ]; then
        echo "WARNING: compute_spread_rmse_crps.py failed for $VAR (exit code $EXITCODE)"
        ((FAILED+=1))
    fi
done

echo ""
echo "=========================================================="
echo "Completed: $((TOTAL - FAILED))/$TOTAL variables succeeded"
if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED variable(s) failed"
    exit 1
fi
echo "=========================================================="

echo "Done."
exit 0
