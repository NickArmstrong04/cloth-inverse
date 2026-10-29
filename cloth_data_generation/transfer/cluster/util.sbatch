#!/bin/bash
# One-shot CPU utility job, so packaging / checks / downloads never run on a login node
# or inside an interactive allocation that can expire under them. Pass the command
# as you would type it from CODE_DIR (transfer/); the venv python is on PATH as `python`:
#   sbatch cluster/util.sbatch python package_splits.py --indir $DATA/ds256_free --outdir $DATA/ds256_free_splits
#   sbatch cluster/util.sbatch python cluster/check_splits.py $DATA/ds256_free_splits
#   sbatch cluster/util.sbatch python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vits14')"
# Output: util_<jobid>.out in transfer/. Chain a training job after it with
#   sbatch --dependency=afterok:<jobid> cluster/run.sbatch ...
#SBATCH --job-name=cloth_util
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=util_%j.out
#SBATCH --error=util_%j.err
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: sbatch cluster/util.sbatch <command...>"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PATH="$SLURM_SUBMIT_DIR/.venv/bin:$PATH"
echo "util: $*  host=$(hostname)  job=$SLURM_JOB_ID"
"$@"
echo "util done"
