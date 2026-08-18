#!/bin/bash -l

# Generic qsub wrapper for animate_ensemble.py (CPU-only MP4 rendering).
# All args are forwarded to the Python worker; used by
# dispatch_tc_landfall_split.py --animate-residual to animate the MSL field
# of the bypass-residual members.
#$ -P eb-general
#$ -N AnimEnsemble
#$ -cwd
#$ -j y
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -m a
#$ -M jlin404@bu.edu

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
# animate_ensemble.py lives one level up, in scripts/ensemble_analysis/.
PYTHON_SCRIPT="${SCRIPT_DIR}/../animate_ensemble.py"
LOG_DIR="${SCRIPT_DIR}/tc_tracks_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/animate_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
# animate_latlon() saves via matplotlib's FFMpegWriter, which shells out to
# the ffmpeg binary; e2s-custom does not bundle it, so load the SCC module.
# Without this the job creates the output dir then fails at anim.save(),
# leaving an empty folder and no MP4s.
module load ffmpeg/8.1
conda activate e2s-custom

python3 "$PYTHON_SCRIPT" "$@"
