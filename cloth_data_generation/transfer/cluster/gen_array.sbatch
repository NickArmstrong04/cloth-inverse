#!/bin/bash
# Job-array template. submit.py fills the double-underscore placeholders and writes the rendered
# file next to the dataset before calling sbatch.
#
# Task t generates CHUNK samples for slots [t*CHUNK, (t+1)*CHUNK) with
# seed = SEED0 + slot, so every slot is reproducible and independent of how
# the array is cut. Re-submitting the same array for the same OUTDIR resumes:
# passing files are kept, rejected attempts are not re-simulated.
#SBATCH --job-name=cloth_gen
#SBATCH --partition=__PARTITION__
#SBATCH --gpus=__GPUS__
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --array=__ARRAY__
#SBATCH --output=__LOGDIR__/gen_%A_%a.out
#SBATCH --error=__LOGDIR__/gen_%A_%a.err
#SBATCH --requeue
#SBATCH --open-mode=append
__ACCOUNT_LINE__
__EXCLUDE_LINE__

set -euo pipefail
CODE_DIR="__CODE_DIR__"
OUTDIR="__OUTDIR__"
CHUNK=__CHUNK__
SEED0=__SEED0__
NX=__NX__
MATERIAL=__MATERIAL__
SAMPLE_ARGS="__SAMPLE_ARGS__"
JOBS=__JOBS__

SLOT0=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): slots $SLOT0..$((SLOT0 + CHUNK - 1)) nx=$NX material=$MATERIAL seed0=$SEED0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

cd "$CODE_DIR"
# Shared compiled-kernel cache so only the first task pays the compile cost.
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$CODE_DIR/.warp_cache}"
mkdir -p "$OUTDIR" "$WARP_CACHE_PATH"
if ! .venv/bin/python -c "import numpy, h5py, warp, newton" 2>/dev/null; then
  echo "ENVIRONMENT BROKEN: $CODE_DIR/.venv is missing packages. Run: bash cluster/setup_env.sh"; exit 2
fi

# JOBS samples at a time on this GPU (set JOBS_PER_TASK in cluster.env; 1 if the GPU is already saturated).
.venv/bin/python generate_batch.py \
  --n "$CHUNK" --jobs "$JOBS" --slot0 "$SLOT0" --seed0 "$SEED0" --nx "$NX" --material "$MATERIAL" \
  --outdir "$OUTDIR" --resume \
  ${SAMPLE_ARGS:+--sample-args="$SAMPLE_ARGS"}

echo "task $SLURM_ARRAY_TASK_ID done"
