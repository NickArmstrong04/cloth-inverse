#!/bin/bash
# Observation ladder (ROADMAP.md): how much accuracy is lost as input privileges are
# removed. One GPU; runs rungs 0-3 on a packaged dataset.
#   sbatch cluster/ladder.sbatch <splits_dir> [epochs]
#SBATCH --job-name=cloth_ladder
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --exclude=a1125u05n01
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=ladder_%j.out
#SBATCH --error=ladder_%j.err
set -uo pipefail  # (no -e: skipped rungs return non-zero from want())
SPLITS="${1:?usage: ladder.sbatch <splits_dir> [epochs] [rungs e.g. 2,3]}"
EPOCHS="${2:-300}"
RUNGS="${3:-0,1,2,3}"
want() { [[ ",$RUNGS," == *",$1,"* ]]; }
CODE_DIR="$SLURM_SUBMIT_DIR"
INV="$(dirname "$CODE_DIR")/inverse"
PY="$CODE_DIR/.venv/bin/python"
cd "$INV"
echo "ladder on $SPLITS epochs=$EPOCHS host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name --format=csv,noheader
want 0 && echo "== rung 0: grid + velocities + known frame (cnn_reg, dihedral aug)" && \
$PY train.py --root "$SPLITS" --model cnn_reg --epochs "$EPOCHS" --no-rot --amp --tta 0
want 1 && echo "== rung 1: grid, positions only, known frame" && \
$PY train.py --root "$SPLITS" --model cnn_reg --epochs "$EPOCHS" --no-rot --amp --tta 0 --no-vel
want 2 && echo "== rung 2: grid, positions only, unknown pose (SO(3) aug, rotated eval, D4-symmetric u,v)" && \
$PY train.py --root "$SPLITS" --model cnn_reg --epochs "$EPOCHS" --amp --tta 8 --no-vel --eval-rotated
want 3 && echo "== rung 3: unordered point cloud, positions only, unknown pose (D4-symmetric u,v)" && \
$PY train.py --root "$SPLITS" --model pointtf --epochs "$EPOCHS" --amp --tta 8 --no-vel --eval-rotated --bs 16 --lr 3e-4
echo "ladder done"
