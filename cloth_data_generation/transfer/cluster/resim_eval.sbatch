#!/bin/bash
# Re-simulation error of a finished run (inverse/eval_resim.py), one GPU: the predicted forces,
# the true forces (chaos floor) and the constant predictor (chance) are each simulated with the
# dataset's own physics and compared with the true final state.
#   sbatch cluster/resim_eval.sbatch <splits_dir> <run_dir> [eval_resim.py args...]
#   e.g. sbatch --job-name=M15r cluster/resim_eval.sbatch $S4 runs/M15a --limit 100
# Full test split in parallel: sbatch --array=0-3 ... (adds --shard $SLURM_ARRAY_TASK_ID/4), then
#   .venv/bin/python ../inverse/eval_resim.py --run <run> --root $S --merge
# Budget: 3 simulations per sample, ~10 s each at 256 once the session is built.
#SBATCH --job-name=resim_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --exclude=a1125u05n01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=resim_%j.out
#SBATCH --error=resim_%j.err
set -euo pipefail
SPLITS="${1:?usage: resim_eval.sbatch <splits_dir> <run_dir> [args...]}"
RUN="${2:?run dir (relative to inverse/, e.g. runs/M15a)}"
shift 2
CODE_DIR="$SLURM_SUBMIT_DIR"
INV="$(dirname "$CODE_DIR")/inverse"
PY="$CODE_DIR/.venv/bin/python"
cd "$INV"
SHARD=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then SHARD="--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"; fi
echo "resim eval: $RUN  args='$*' $SHARD  host=$(hostname)  job=$SLURM_JOB_ID"
$PY eval_resim.py --run "$RUN" --root "$SPLITS" "$@" $SHARD
echo "resim done"
