#!/bin/bash -l

# qsub wrapper for plot_ensemble_member.py --event (CPU-only MP4 rendering).
# All arguments are forwarded to the Python worker, so anything the script
# accepts in event mode works here:
#
#   qsub -N Vid_tele_0_24 submission_scripts/submit_member_videos.sh \
#       --event teleport_atlantic --members 0-24
#
# Shard a large render by submitting one job per disjoint member range; the
# jobs write into the same output/<event>_plots/<mode>/ tree and never collide,
# because the filename carries the member index.
#$ -P eb-general
#$ -N MemberVideos
#$ -cwd
#$ -j y
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -m a
#$ -M jlin404@bu.edu

# Event mode hardcodes its output directory as `output/<event>_plots/<mode>`,
# relative to the working directory, so the run must be anchored to
# scripts/ensemble_analysis/ or the videos land wherever qsub happened to be
# invoked from.  Unlike the dispatcher-submitted wrappers in this directory,
# this one is submitted by hand, so accept either the submission_scripts dir or
# its parent as $SGE_O_WORKDIR and fall back to the script's own location.
ANALYSIS_DIR=""
for candidate in \
    "${SGE_O_WORKDIR:-}/.." \
    "${SGE_O_WORKDIR:-}" \
    "$(cd "$(dirname "$(readlink -f "$0")")/.." 2>/dev/null && pwd)"; do
    if [ -n "$candidate" ] && [ -f "${candidate}/plot_ensemble_member.py" ]; then
        ANALYSIS_DIR="$(cd "$candidate" && pwd)"
        break
    fi
done

if [ -z "$ANALYSIS_DIR" ]; then
    echo "ERROR: could not locate plot_ensemble_member.py."
    echo "  SGE_O_WORKDIR=${SGE_O_WORKDIR:-<unset>}"
    echo "  Submit from scripts/ensemble_analysis/ or its submission_scripts/."
    exit 1
fi

LOG_DIR="${ANALYSIS_DIR}/submission_scripts/member_video_logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/member_videos_${JOB_ID:-local-$$}.log" 2>&1

cd "$ANALYSIS_DIR" || exit 1

module load miniconda/25.3.1
module load gcc/12.2.0
# anim.save(writer="ffmpeg") shells out to the ffmpeg binary; e2s-custom does
# not bundle it, so load the SCC module.  Without this the job creates the
# output directory and then dies at the first save, leaving empty folders.
module load ffmpeg/8.1
conda activate e2s-custom

echo "=========================================================="
echo "Job ID:       ${JOB_ID:-local}"
echo "Working dir:  $(pwd)"
echo "Arguments:    $*"
echo "Videos ->     $(pwd)/output/<event>_plots/<mode>/"
echo "=========================================================="

python3 plot_ensemble_member.py "$@"
