#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE Directives
# ---------------------------------------------------------------------
#$ -P eb-general              # Project Name
#$ -N ERA5_Fetch_Raw_Zarr     # Job Name
#$ -j y                       # Merge Output/Error
#$ -o logs/job_$JOB_ID_$TASK_ID.log
#$ -l h_rt=2:30:00            # 2.5 Hours Hard Limit
#$ -l gpus=1                  # Request 1 GPU
#$ -l gpu_c=8.0               # Capability >= 8.0 (Ensures A100/H100/H200 class)
#$ -l gpu_type=H200|A100-80G  # Request high-memory GPUs (H200 is first pref)
#$ -pe omp 4                  # 4 CPU Cores
#$ -l mem_per_core=8G         # 32GB Total RAM
#$ -l h=!scc-a15              # Exclude faulty node scc-a15

# JOB ARRAY: 2001 to 2022 (originally starting in 1959, resubmitting job array at 2001)
# The Task ID ($SGE_TASK_ID) will be the Year
#$ -t 2001-2022               # 2001 - 2022
#$ -tc 10                     # Max 10 GPUs running at once
#$ -m bea                     # Mail at: (b)eginning, (e)nd, (a)bort
#$ -M jlin404@bu.edu          # Email address
# ---------------------------------------------------------------------

# 1. Hardware Isolation & Tuning (The Fix)
# ---------------------------------------------------------------------
# CRITICAL: Force CUDA to order devices by PCI ID.
# This prevents the driver from re-ordering GPUs and bypassing the SGE mask.
# export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Verify SGE has assigned a GPU
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is not set. SGE allocation failed."
    exit 1
fi
export NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# Prevent Numpy/Pytorch from spawning too many threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 2. Environment Setup
module load miniconda/25.3.1
module load gcc/12.2.0
module load cuda/12.8

# Activate conda environment with error checking
conda activate e2s-custom
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment e2s-custom"
    exit 1
fi

# 3. Cache Isolation (Critical for SCC)
# Use the job-specific local scratch space
export EARTH2STUDIO_CACHE="/scratch/$USER/e2_cache_${JOB_ID}_${SGE_TASK_ID}"
# Explicitly set Modulus cache too, as earth2studio uses it under the hood
export MODULUS_CACHE="$EARTH2STUDIO_CACHE" 

mkdir -p "$EARTH2STUDIO_CACHE"

echo "=========================================================="
echo "Job ID: ${JOB_ID} | Task ID (Year): ${SGE_TASK_ID}"
echo "Node: $HOSTNAME"
echo "Assigned GPU (Logical): $CUDA_VISIBLE_DEVICES"
echo "Cache: $EARTH2STUDIO_CACHE"
echo "=========================================================="

# 4. Run Python Script

OUTPUT_DIR="/projectnb/eb-general/shared_data/data/raw/era5_raw_zarr"

# Prevent NVIDIA Modulus/Distributed libraries from probing all GPUs
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

python3 fetch_era5_zarr.py \
    --year $SGE_TASK_ID \
    --output_dir "$OUTPUT_DIR" \
    --chunk_time 1 \
    --compression_level 2

# Capture exit code
EXITCODE=$?

# 5. Cleanup
echo "Cleaning up local cache..."
rm -rf "$EARTH2STUDIO_CACHE"

# Check if job succeeded
if [ $EXITCODE -ne 0 ]; then
    echo "ERROR: fetch_era5_zarr.py failed with exit code $EXITCODE"
    exit $EXITCODE
fi

echo "Done."
exit 0
