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
__ACCOUNT_LINE__

set -euo pipefail
CODE_DIR="__CODE_DIR__"
OUTDIR="__OUTDIR__"
CHUNK=__CHUNK__
SEED0=__SEED0__
NX=__NX__
SAMPLE_ARGS="__SAMPLE_ARGS__"

SLOT0=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): slots $SLOT0..$((SLOT0 + CHUNK - 1)) nx=$NX seed0=$SEED0"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

cd "$CODE_DIR"
# Shared compiled-kernel cache so only the first task pays the compile cost.
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$CODE_DIR/.warp_cache}"
mkdir -p "$OUTDIR" "$WARP_CACHE_PATH"

# One sample at a time per GPU: parallel samples on one GPU gave no speedup.
.venv/bin/python generate_batch.py \
  --n "$CHUNK" --jobs 1 --slot0 "$SLOT0" --seed0 "$SEED0" --nx "$NX" \
  --outdir "$OUTDIR" --resume \
  ${SAMPLE_ARGS:+--sample-args="$SAMPLE_ARGS"}

echo "task $SLURM_ARRAY_TASK_ID done"
