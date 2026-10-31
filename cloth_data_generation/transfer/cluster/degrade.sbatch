#!/bin/bash
# Photo-like copies of a packaged split (Phase C of the rung-4b -> rung-1 plan, JOBS.md M19):
# inverse/degrade_split.py on a GPU node (the renderer is torch). Writes <out> with K files per
# sample and a rewritten dataset_manifest.json; train.py reads it like any split.
#   sbatch --job-name=deg cluster/degrade.sbatch $S3 ${S3}_photo --views 2 --noise measured
#   sbatch --job-name=degv cluster/degrade.sbatch $S3 ${S3}_vis --views 2 --noise none
#SBATCH --job-name=degrade
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --exclude=a1125u05n01,a1125u14n01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=degrade_%j.out
#SBATCH --error=degrade_%j.err
set -euo pipefail
SRC="${1:?usage: degrade.sbatch <splits_dir> <out_dir> [degrade_split.py args...]}"
OUT="${2:?out dir}"
shift 2
CODE_DIR="$SLURM_SUBMIT_DIR"
INV="$(dirname "$CODE_DIR")/inverse"
PY="$CODE_DIR/.venv/bin/python"
cd "$INV"
echo "degrade: $SRC -> $OUT args='$*' host=$(hostname) job=$SLURM_JOB_ID"
$PY degrade_split.py --root "$SRC" --out "$OUT" "$@"
ls "$OUT"/train | wc -l; ls "$OUT"/test | wc -l
