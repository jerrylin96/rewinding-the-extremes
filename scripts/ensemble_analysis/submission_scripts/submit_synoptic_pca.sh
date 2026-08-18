#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE Directives — Synoptic PCA, per-mode worker
#
# Case-agnostic wrapper: all paths/args are forwarded from
# dispatch_synoptic_pca.py.  Run directly only if you want to
# override the defaults that the dispatcher would otherwise compute.
# ---------------------------------------------------------------------
#$ -P eb-general              # Project Name
#$ -N SynPCA                # Job Name
#$ -cwd                       # Run from the submission directory
#$ -j y                       # Merge Output/Error (file written to synoptic_pca_logs/)
# h_rt is set via qsub -l flags from dispatch_synoptic_pca.py
#$ -pe omp 8                  # 8 CPU cores (numpy/torch threads + zarr decode)
#$ -l mem_per_core=16G        # 128 GB total — comfortable for the percentile-member trajectory loads
#$ -m bea                     # Mail at: (b)eginning, (e)nd, (a)bort
#$ -M jlin404@bu.edu          # Email address

# ---------------------------------------------------------------------
# 1a. Locate this script's source directory + self-managed log directory
#
# Under SGE the running script is a copy in /var/spool/sge/.../job_scripts/,
# so $0 points at the spool — not at the original submission_scripts/
# directory that holds the colocated Python source.  SGE sets
# SGE_O_WORKDIR to the directory from which `qsub` was invoked; combined
# with `-cwd` above the job runs there too, so we use it as the
# canonical script directory and fall back to resolving $0 for local
# runs.
#
# SGE's `-o` directive is evaluated at submit time and requires the
# target directory to already exist, so we redirect output from inside
# the script — the user never has to mkdir before qsub.
# ---------------------------------------------------------------------

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PYTHON_SCRIPT="${SCRIPT_DIR}/compute_synoptic_pca.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: cannot find $PYTHON_SCRIPT" >&2
    echo "       Submit this job from the submission_scripts/ directory:" >&2
    echo "         cd .../scripts/ensemble_analysis/submission_scripts && qsub $(basename "$0")" >&2
    exit 1
fi
LOG_DIR="${SCRIPT_DIR}/synoptic_pca_logs"
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
# 2. Forward all arguments to compute_synoptic_pca.py
#
# Expected arguments from dispatch_synoptic_pca.py:
#   --ensemble-zarr <path>
#   --era5-zarr <path>
#   --output-dir <path>
#   --mode <start|end>
#   --case-name <display name>
#   --start-time <ISO 8601>
#   --timezone <IANA name>
#   --variable <name>            (repeatable)
#   --lon-min/max --lat-min/max  (synoptic domain)
#   --max-d <n>                  (number of leading PCs to compute/store)
#   --n-eof <n>                  (number of leading EOFs to analyze)
#   [--conditioning-frame <int> ...]
#   --impact-kind <scalar|track|none>
#   [--impact-variable <name> --impact-bbox L R B T --impact-mask <kind>]
#   [--tc-parquet <path> --tc-tracker <name> --tc-domain L R B T --landfall-frame <n>]
# ---------------------------------------------------------------------

echo "=========================================================="
echo "Job ID:    $JOB_ID"
echo "Node:      $HOSTNAME"
echo "CPUs:      $NSLOTS"
echo "Arguments: $*"
echo "=========================================================="

python3 "$PYTHON_SCRIPT" "$@"
EXITCODE=$?

if [ $EXITCODE -ne 0 ]; then
    echo "ERROR: compute_synoptic_pca.py failed (exit $EXITCODE)"
    exit $EXITCODE
fi

echo "Done."
exit 0
