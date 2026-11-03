#!/bin/bash
# T1 (hackathon tooling plan, item 1): the full photo chain on a photo.py --stereo --texture coded --gt
# set whose SAM masks are already in <photos>/sam3/ (hosted SAM 3.1: inverse/sam_api.py --photos).
# VGGT stereo lift (sam3env) -> coded-print decode -> world-frame grids (oracle and decoded
# correspondences) -> eval_grid.py for each run given.
#   sbatch cluster/t1_chain.sbatch ../../cloth_results/photos/rest20_coded M20e M20g M20f20 M20a2
#SBATCH --job-name=t1chain
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --exclude=a1125u05n01,a1125u14n01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/t1chain_%j.out
#SBATCH --error=logs/t1chain_%j.out
set -euo pipefail
P="${1:?usage: t1_chain.sbatch <photos_dir relative to inverse/> <run>...}"
shift
source "$SLURM_SUBMIT_DIR/cluster/cluster.env"
B="$BASE_DIR"
export HF_HOME=$B/.cache/hf PIP_CACHE_DIR=$B/.cache/pip
SAMPY=$B/sam3env/bin/python
PY="$SLURM_SUBMIT_DIR/.venv/bin/python"
cd "$SLURM_SUBMIT_DIR/../inverse"
echo "t1chain: $P runs=$* host=$(hostname) job=$SLURM_JOB_ID"
$SAMPY -c "import psutil" 2>/dev/null || $SAMPY -m pip install -q psutil
MASKS="${MASKS:-sam}"  # MASKS=oracle: renderer masks (no SAM run), for when SAM is unavailable
SAMFLAG=$([ "$MASKS" = sam ] && echo --sam || true)
t=$SECONDS; $SAMPY lift_stereo.py "$P" $SAMFLAG; echo "lift_stereo $((SECONDS - t)) s"
t=$SECONDS; $PY decode_print.py --photos "$P"; echo "decode_print $((SECONDS - t)) s"
t=$SECONDS; $PY grid_from_views.py --photos "$P" --corr oracle --out "$P/grids"; echo "grids oracle $((SECONDS - t)) s"
t=$SECONDS; $PY grid_from_views.py --photos "$P" --corr decoded --out "$P/grids_dec"; echo "grids decoded $((SECONDS - t)) s"
for r in "$@"; do
  $PY eval_grid.py --run runs/$r --npz "$P/grids" --out "$P/eval_${r}_oraclecorr.json"
  $PY eval_grid.py --run runs/$r --npz "$P/grids_dec" --out "$P/eval_${r}_decoded.json"
done
echo "t1chain done"
