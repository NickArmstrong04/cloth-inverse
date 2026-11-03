#!/bin/bash
# Training split whose grids come through the real photo chain (step 3 of the 2026-11-02 plan, JOBS.md
# M21): per shard of source files, photo.py coded-print 20 deg stereo pairs -> VGGT stereo lift
# (sam3env, oracle masks: SAM is only run at test time) -> coded-print decode -> grid_from_views
# --corr decoded --h5-out (degrade_split.py layout). Shard lists <work>/shards/NNN.txt, one
# "<split> <h5 path>" per line, one split per shard. Then cluster/chain_manifest.py.
#   sbatch --array=0-46%20 cluster/chain_split.sbatch <work> <out> [grid_from_views args...]
#SBATCH --job-name=chainsplit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --exclude=a1125u05n01,a1125u14n01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/chainsplit_%A_%a.out
#SBATCH --error=logs/chainsplit_%A_%a.out
set -euo pipefail
WORK="${1:?usage: chain_split.sbatch <work> <out> [grid_from_views args...]}"
OUT="${2:?out}"
shift 2
B=/nfs/roberts/scratch/pi_sz622/naa79/cloth
export HF_HOME=$B/.cache/hf PIP_CACHE_DIR=$B/.cache/pip
SAMPY=$B/sam3env/bin/python
PY="$SLURM_SUBMIT_DIR/.venv/bin/python"
ID=$(printf %03d "$SLURM_ARRAY_TASK_ID")
LIST="$WORK/shards/$ID.txt"
[ -f "$WORK/done/$ID" ] && { echo "shard $ID already done"; exit 0; }
SPLIT=$(head -1 "$LIST" | cut -d' ' -f1)
P="$WORK/shard_$ID/photos"
rm -rf "$WORK/shard_$ID"; mkdir -p "$P" "$OUT/$SPLIT" "$WORK/done"
cd "$SLURM_SUBMIT_DIR/../inverse"
echo "chainsplit shard $ID ($SPLIT, $(wc -l < "$LIST") files) host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID"
$SAMPY -c "import psutil" 2>/dev/null || $SAMPY -m pip install -q psutil
t=$SECONDS; $PY photo.py --h5 $(cut -d' ' -f2 "$LIST") --views 2 --stereo 20 --texture coded --gt --seed "$SLURM_ARRAY_TASK_ID" --out "$P" > /dev/null; echo "render $((SECONDS - t)) s"
t=$SECONDS; $SAMPY lift_stereo.py "$P" --stereo-only > "$WORK/shard_$ID/lift.log"; echo "lift_stereo $((SECONDS - t)) s"
t=$SECONDS; $PY decode_print.py --photos "$P" --oracle-mask --base-only > "$WORK/shard_$ID/decode.log"; echo "decode $((SECONDS - t)) s"
t=$SECONDS; $PY grid_from_views.py --photos "$P" --corr decoded --h5-out "$OUT/$SPLIT" "$@" | tail -3; echo "grids $((SECONDS - t)) s"
mv "$OUT/$SPLIT/chain_photos.json" "$OUT/chain_$ID.json"
rm -rf "$P"  # renders, gt and packages: ~0.4 GB per shard, rebuilt from the seed if needed
touch "$WORK/done/$ID"
echo "chainsplit shard $ID done"
