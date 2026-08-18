#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE Directives for Ensemble Merging
# ---------------------------------------------------------------------
#$ -P eb-general              # Project Name
#$ -cwd                       # Run from the submission directory
#$ -j y                       # Merge Output/Error
#$ -o ensemble_logs/merge_$JOB_ID.log
#$ -pe omp 1                  # 1 CPU Core is enough for moving files
#$ -l mem_per_core=4G         # 4 GB Total RAM
#$ -m bea                     # Mail at: (b)eginning, (e)nd, (a)bort
#$ -M jlin404@bu.edu          # Email address

# ---------------------------------------------------------------------

# 1. Environment Setup
module load miniconda/25.3.1
conda activate e2s-custom
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment e2s-custom"
    exit 1
fi

# 2. Run Python Merge Script
python3 merge_ensemble_zarrs.py "$@"

EXITCODE=$?

if [ $EXITCODE -ne 0 ]; then
    echo "ERROR: merge_ensemble_zarrs.py failed with exit code $EXITCODE"
    exit $EXITCODE
fi

echo "Merge done."
exit 0
