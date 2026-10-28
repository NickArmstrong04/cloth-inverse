#!/bin/bash
# Train the inverse models on a packaged dataset at FULL resolution on one GPU.
# Submit from transfer/ (inside or outside an allocation):
#   sbatch cluster/train.sbatch <splits_dir> [epochs] [extra train.py args...]
# e.g.
#   sbatch cluster/train.sbatch /home/naa79/scratch_pi_sz622/naa79/cloth_data/ds768_night_splits 300
#   sbatch --partition=gpu_h100 cluster/train.sbatch <splits_dir> 300 --bs 32
# First run also builds the tensor cache (<split>_pool1.pt, ~15 GB for 1000 768-samples)
# next to the splits; later runs start in seconds.
#SBATCH --job-name=cloth_train
#SBATCH --partition=gpu
#SBATCH --gpus=1                 # any GPU type (cluster/gpu_adapt.sh adds --data-on-cpu on small cards)
#SBATCH --exclude=a1125u05n01
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
set -euo pipefail   # stop at the first failing step (a GPU fault must not look like "train done")
SPLITS="${1:?usage: train.sbatch <splits_dir> [epochs] [extra args]}"
EPOCHS="${2:-300}"
shift; shift 2>/dev/null || true
EXTRA="$*"   # GPU_EXTRA (from gpu_adapt.sh) is appended below
CODE_DIR="$SLURM_SUBMIT_DIR"                                  # transfer/ (where sbatch was run)
INV="$(dirname "$CODE_DIR")/inverse"                           # cloth_data_generation/inverse
PY="$CODE_DIR/.venv/bin/python"
cd "$INV"
echo "train: $SPLITS  epochs=$EPOCHS  extra='$EXTRA'  host=$(hostname)  job=$SLURM_JOB_ID"
source "$CODE_DIR/cluster/gpu_adapt.sh"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { echo "torch missing in venv: run  bash cluster/setup_env.sh --with-torch"; exit 2; }
TAG="$(basename "$SPLITS")"
echo "== baselines"
$PY baselines.py --root "$SPLITS" --out "runs/${TAG}_baselines"
echo "== cnn_heat, dihedral augmentation"
$PY train.py --root "$SPLITS" --model cnn_heat --epochs "$EPOCHS" --no-rot --amp $EXTRA $GPU_EXTRA
echo "== cnn_reg, dihedral augmentation"
$PY train.py --root "$SPLITS" --model cnn_reg --epochs "$EPOCHS" --no-rot --amp $EXTRA $GPU_EXTRA
echo "== cnn_heat, dihedral + 3D rotation augmentation"
$PY train.py --root "$SPLITS" --model cnn_heat --epochs "$EPOCHS" --amp $EXTRA $GPU_EXTRA
echo "train done; metrics in $INV/runs/${TAG}_*/metrics.json"
