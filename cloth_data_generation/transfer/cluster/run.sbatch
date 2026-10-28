#!/bin/bash
# One training configuration as its own job, so experiments run in parallel instead of
# sequentially inside train.sbatch / ladder.sbatch. Pass train.py args verbatim:
#   sbatch cluster/run.sbatch <splits_dir> --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --seed 1
# Optional: --job-name / --partition / --gpus overrides go before the script, e.g.
#   sbatch --job-name=L0s1 --partition=scavenge_gpu cluster/run.sbatch <splits_dir> ...
# Output: run_<jobid>.out in transfer/; metrics in ../inverse/runs/<auto name>/metrics.json
# (pass --out runs/<name> to choose the run directory; record both in cloth_results/JOBS.md).
#SBATCH --job-name=cloth_run
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --exclude=a1125u05n01
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err
set -euo pipefail
SPLITS="${1:?usage: run.sbatch <splits_dir> [train.py args...]}"
shift
CODE_DIR="$SLURM_SUBMIT_DIR"
INV="$(dirname "$CODE_DIR")/inverse"
PY="$CODE_DIR/.venv/bin/python"
cd "$INV"
echo "run: $SPLITS  args='$*'  host=$(hostname)  job=$SLURM_JOB_ID  name=$SLURM_JOB_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY train.py --root "$SPLITS" "$@"
echo "run done"
