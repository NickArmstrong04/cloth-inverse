#!/bin/bash
# Re-simulation selection among the flow head's candidates (inverse/select_resim.py), one GPU.
#   sbatch cluster/resim.sbatch <splits_dir> <run_dir> [select_resim.py args...]
#   e.g. sbatch --job-name=M11f2 cluster/resim.sbatch $S2 runs/M11f --limit 50 --k 8
# Shards for the full test split: sbatch --array=0-5 ... (adds --shard $SLURM_ARRAY_TASK_ID/6), then
#   .venv/bin/python ../inverse/select_resim.py --cands <run>/test_candidates.json --root $S2 --merge
# Each candidate is one generate_sample.py run (~45 s at 256): budget K x N x 45 s.
#SBATCH --job-name=resim_select
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --exclude=a1125u05n01,a1125u14n01   # u14n01: uncorrectable ECC errors 2026-10-30
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=resim_%j.out
#SBATCH --error=resim_%j.err
set -euo pipefail
SPLITS="${1:?usage: resim.sbatch <splits_dir> <run_dir> [args...]}"
RUN="${2:?run dir (relative to inverse/, e.g. runs/M11f)}"
shift 2
CODE_DIR="$SLURM_SUBMIT_DIR"
INV="$(dirname "$CODE_DIR")/inverse"
PY="$CODE_DIR/.venv/bin/python"
cd "$INV"
SHARD=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then SHARD="--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"; fi
echo "resim select: $RUN  args='$*' $SHARD  host=$(hostname)  job=$SLURM_JOB_ID"
$PY select_resim.py --cands "$RUN/test_candidates.json" --root "$SPLITS" --out "$RUN/resim_select" "$@" $SHARD
echo "resim done"
