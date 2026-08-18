#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE Directives for TC Track Anchor Diagnostics (CPU only)
#
# Case-agnostic wrapper: all paths/args are forwarded from
# dispatch_tc_track_anchor_diagnostics.py.  The job reads the tracked
# parquet plus (for the MSL animations) the ensemble zarr; the optional
# DetectNodes candidate overlay shells out to the TempestExtremes
# binaries from the e2s-custom conda env.
# ---------------------------------------------------------------------
#$ -P eb-general              # Project Name
#$ -N TC_AnchorDiag           # Job Name (use qsub -N "name" to customize)
#$ -cwd                       # Run from the submission directory
#$ -j y                       # Merge Output/Error (file written to tc_tracks_logs/, see below)
# h_rt is set via qsub -l flags from dispatch_tc_track_anchor_diagnostics.py
#$ -pe omp 4                  # 4 CPU Cores
#$ -l mem_per_core=16G        # 64 GB Total RAM (per-member HPX regrid + cartopy pages)
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
PYTHON_SCRIPT="${SCRIPT_DIR}/tc_track_anchor_diagnostics.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: cannot find $PYTHON_SCRIPT" >&2
    echo "       Submit this job from the submission_scripts/ directory:" >&2
    echo "         cd .../scripts/ensemble_analysis/submission_scripts && qsub $(basename "$0")" >&2
    exit 1
fi
LOG_DIR="${SCRIPT_DIR}/tc_tracks_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/job_${JOB_ID:-local-$$}.log" 2>&1

# ---------------------------------------------------------------------
# 1b. Environment setup
# ---------------------------------------------------------------------

module load miniconda/25.3.1
module load gcc/12.2.0
module load ffmpeg/8.1
conda activate e2s-custom
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment e2s-custom"
    exit 1
fi

export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

# ---------------------------------------------------------------------
# 2. Forward all arguments to tc_track_anchor_diagnostics.py
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
    echo "ERROR: tc_track_anchor_diagnostics.py failed (exit $EXITCODE)"
    exit $EXITCODE
fi

echo "Done."
exit 0
