#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE Directives for Ensemble Interpolation
# ---------------------------------------------------------------------
#$ -P eb-general              # Project Name
#$ -N Ensemble_Interp         # Job Name (use qsub -N "name" to customize)
#$ -cwd                       # Run from the submission directory
#$ -j y                       # Merge Output/Error
#$ -o ensemble_logs/job_$JOB_ID.log
# h_rt and gpu_type are set via qsub -l flags from dispatch_ensemble.py
#$ -l gpus=1                  # Request 1 GPU
#$ -l gpu_type=A40|A6000|RTX6000|RTX6000ada|L40S|A100
#$ -pe omp 4                  # 4 CPU Cores
#$ -l mem_per_core=16G        # 64 GB Total RAM
#$ -r y                       # Rerunnable: exit 99 below triggers requeue
#$ -m bea                     # Mail at: (b)eginning, (e)nd, (a)bort
#$ -M jlin404@bu.edu          # Email address

# ---------------------------------------------------------------------

# 1. Environment Setup
module load miniconda/25.3.1
module load gcc/12.2.0
module load cuda/12.8
# Activate conda environment with error checking
conda activate e2s-custom
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment e2s-custom"
    exit 1
fi

# Prevent Numpy/Pytorch from spawning too many threads
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

# 2. Cache Isolation (Critical for SCC)
export EARTH2STUDIO_CACHE="/scratch/$USER/e2_cache_$JOB_ID"
export MODULUS_CACHE="$EARTH2STUDIO_CACHE"

mkdir -p "$EARTH2STUDIO_CACHE"

# Save original arguments as array to preserve quoting
ARGS=("$@")

# Extract --output-dir value from arguments so we can write the GPU log there
OUTPUT_DIR=""
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--output-dir" ]]; then
        OUTPUT_DIR="${ARGS[i+1]}"
        break
    fi
done

echo "=========================================================="
echo "Job ID: $JOB_ID"
echo "Node: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Visible GPUs (nvidia-smi -L, honours CUDA_VISIBLE_DEVICES):"
nvidia-smi -L 2>&1 | sed 's/^/  /'
echo "Arguments: ${ARGS[*]}"
echo "Cache: $EARTH2STUDIO_CACHE"
echo "=========================================================="

# 3. Start GPU monitoring in background
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    GPU_LOG="$OUTPUT_DIR/gpu_monitor_${JOB_ID}.csv"
else
    GPU_LOG="gpu_monitor_${JOB_ID}.csv"
fi
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 2 > "$GPU_LOG" &
MONITOR_PID=$!
echo "GPU monitoring started (PID: $MONITOR_PID, log: $GPU_LOG)"

# Cleanup function: kill monitor and remove cache on any exit
cleanup() {
    echo "Cleaning up..."
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null
    rm -rf "$EARTH2STUDIO_CACHE"
}
trap cleanup EXIT

# 4. Preflight GPU health probe.
#    A wedged or mis-assigned GPU manifests as cudaErrorDevicesUnavailable on
#    the first real kernel launch (typically inside model.to(device), well
#    after multi-GB of checkpoints have been downloaded). Catch it early with
#    a cheap CUDA op. Retry twice for genuine handoff races, then exit 99 so
#    SGE requeues this chunk on a different node rather than burning the slot.
PROBE_MAX_ATTEMPTS=3
PROBE_DELAY=30

for ((PROBE_ATTEMPT=1; PROBE_ATTEMPT<=PROBE_MAX_ATTEMPTS; PROBE_ATTEMPT++)); do
    echo "GPU preflight attempt $PROBE_ATTEMPT of $PROBE_MAX_ATTEMPTS..."
    if python3 - <<'PY'
import sys
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
assert torch.cuda.device_count() >= 1, "no CUDA devices visible"
x = torch.zeros(1024, 1024, device="cuda")
(x @ x).sum().item()
torch.cuda.synchronize()
print(f"  preflight ok: {torch.cuda.get_device_name(0)} "
      f"(device_count={torch.cuda.device_count()})")
PY
    then
        break
    fi
    if [ $PROBE_ATTEMPT -ge $PROBE_MAX_ATTEMPTS ]; then
        echo "ERROR: GPU preflight failed after $PROBE_ATTEMPT attempts on $HOSTNAME."
        echo "Exiting 99 to request SGE requeue on a different node."
        exit 99
    fi
    echo "Preflight failed; sleeping ${PROBE_DELAY}s before retry..."
    sleep $PROBE_DELAY
    PROBE_DELAY=$((PROBE_DELAY * 2))
done

# 5. Run Python Script
python3 ensemble_interpolation.py "${ARGS[@]}"
EXITCODE=$?

if [ $EXITCODE -ne 0 ]; then
    echo "ERROR: ensemble_interpolation.py failed with exit code $EXITCODE"
    exit $EXITCODE
fi

echo "Done."
exit 0
