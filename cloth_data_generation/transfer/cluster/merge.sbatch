#!/bin/bash
# CPU post-processing job, chained after the array (--dependency=afterany).
# Merges per-task summaries into batch_summary.json + dataset_summary.json and
# packages 8:1:1 train/val/test splits next to the sample directory.
#SBATCH --job-name=cloth_merge
#SBATCH --partition=__CPU_PARTITION__
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=__LOGDIR__/merge_%j.out
#SBATCH --error=__LOGDIR__/merge_%j.err
__ACCOUNT_LINE__
set -euo pipefail
CODE_DIR="__CODE_DIR__"
OUTDIR="__OUTDIR__"
EXPECT_N=__EXPECT_N__
cd "$CODE_DIR"
.venv/bin/python cluster/merge_summaries.py --indir "$OUTDIR" --expect-n "$EXPECT_N"
.venv/bin/python package_splits.py --indir "$OUTDIR" --outdir "${OUTDIR}_splits"
echo "merge done: ${OUTDIR}_splits"
