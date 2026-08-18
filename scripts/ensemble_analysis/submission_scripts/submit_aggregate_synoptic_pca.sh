#!/bin/bash -l

# Aggregator for synoptic PCA.  Reads per-mode
# synoptic_pca_<mode>.npz under <output-dir> and writes the combined
# precursor->impact figures plus EOF-pattern and PC-scatter supplements.
#$ -P eb-general
#$ -N AggSynPCA
#$ -cwd
#$ -j y
#$ -pe omp 2
#$ -l mem_per_core=8G
#$ -m a
#$ -M jlin404@bu.edu

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PYTHON_SCRIPT="${SCRIPT_DIR}/aggregate_synoptic_pca.py"
LOG_DIR="${SCRIPT_DIR}/synoptic_pca_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/aggregate_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
# ffmpeg provides the writer for the per-EOF MSL videos (render_member_videos).
# The batch job needs the module so libavdevice.so.* is on LD_LIBRARY_PATH --
# without it the binary is on PATH but the encode dies with
# "libavdevice.so.62: cannot open shared object file".
module load ffmpeg/8.1
conda activate e2s-custom

python3 "$PYTHON_SCRIPT" "$@"
EXITCODE=$?
if [ $EXITCODE -ne 0 ]; then
    echo "ERROR: aggregate_synoptic_pca.py failed (exit $EXITCODE)"
    exit $EXITCODE
fi
echo "Done."
