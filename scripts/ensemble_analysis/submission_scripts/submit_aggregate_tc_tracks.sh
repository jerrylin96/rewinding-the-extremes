#!/bin/bash -l

# Aggregator for tc_tracks. Reads per-mode tc_tracks_<mode>.parquet under
# <output-root>/<tracker>/ and writes a 3-panel static comparison PNG.
#$ -P eb-general
#$ -N AggTcTracks
#$ -cwd
#$ -j y
#$ -pe omp 2
#$ -l mem_per_core=8G
#$ -m a
#$ -M jlin404@bu.edu

SCRIPT_DIR="${SGE_O_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}"
PYTHON_SCRIPT="${SCRIPT_DIR}/aggregate_tc_tracks.py"
LOG_DIR="${SCRIPT_DIR}/tc_tracks_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/aggregate_${JOB_ID:-local-$$}.log" 2>&1

module load miniconda/25.3.1
module load gcc/12.2.0
conda activate e2s-custom

# Args (forwarded by dispatcher):
#   --output-root <path>        (root: <case>/diagnostics/tc_tracks)
#   --trackers <tracker1,...>
#   --lon-min --lon-max --lat-min --lat-max
#   --title <str>
ARGS=("$@")
OUTPUT_ROOT=""
TRACKERS=""
PASS_ARGS=()
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    case "${ARGS[i]}" in
        --output-root) OUTPUT_ROOT="${ARGS[i+1]}"; ((i+=2)) ;;
        --trackers)    TRACKERS="${ARGS[i+1]}";    ((i+=2)) ;;
        *)             PASS_ARGS+=("${ARGS[i]}"); ((i+=1)) ;;
    esac
done

if [ -z "$OUTPUT_ROOT" ] || [ -z "$TRACKERS" ]; then
    echo "ERROR: Missing --output-root or --trackers." >&2
    exit 1
fi

IFS=',' read -ra TRACKER_ARRAY <<< "$TRACKERS"
FAILED=0
for TRACKER in "${TRACKER_ARRAY[@]}"; do
    TR_DIR="${OUTPUT_ROOT}/${TRACKER}"
    echo "[aggregate] $TRACKER -> $TR_DIR"
    python3 "$PYTHON_SCRIPT" --output-dir "$TR_DIR" "${PASS_ARGS[@]}" || ((FAILED+=1))
done

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED tracker(s) failed"
    exit 1
fi
echo "Done."
