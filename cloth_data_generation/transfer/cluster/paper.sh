#!/bin/bash
# Paper dataset: the two-force free-flight scenario of ds256_const on a creasing paper
# sheet (--material paper: plastic bending hinges, yield 50 /m, yielded hinges softened
# to 10% of edge_ke 200 so folds localize into sharp creases; near-inextensible).
# Same seeds and slots as ds256_const, so the two datasets are paired sample by sample:
# slot k draws the same patches, directions, magnitudes and durations for both materials.
#
# Usage (from cloth_data_generation/transfer):
#   bash cluster/paper.sh smoke  [N] [NX]   workstation/WSL: N samples (default 2) at NX (128)
#   bash cluster/paper.sh devel              cluster: 4 samples at 256 on gpu_devel, then stop
#   bash cluster/paper.sh full               cluster: 3000 slots -> ds256_const_paper, as J-005
#                                             (tasks 0-29 on gpu, 30-59 on scavenge_gpu, no merge)
#   bash cluster/paper.sh status             progress + queue
#   bash cluster/paper.sh merge              merge_summaries + package_splits (after the arrays)
#   bash cluster/paper.sh compare            pass rate / QC stretch / attempts vs ds256_const
#
# Environment overrides: NAME (ds256_const_paper), N (3000), NX (256), SETTLE (0),
# YIELD (50), EDGE_KE (200), SOFTEN (0.1), MAXC (30). SETTLE > 0 adds free flight after the forces end so the
# snapshots also cover the relaxed, creased state; it breaks the time pairing with
# ds256_const (captured at release), so it goes to a differently named dataset.
#
# PROFILE=rest: the still-image dataset (plan 2026-10-31): opposing "scrunch" pushes
# (LOADING scrunch, 0.2-2 N: compression is what creases paper; free directions mostly
# bend it smoothly and it relaxes flat), then the sheet is settled after the forces end
# (air drag DRAG 1/s, stop when the RMS speed < REST_VMAX for REST_FRAMES frames, cap
# SETTLE s), the release state is kept as release_grid and bending stiffness is
# resolution-consistent (--bend-scale-res). Defaults from the 128 sweeps (REPORT.md
# "Capture at rest", DATASETS.md "Rest capture sweep"): SETTLE 3, SETTLE_MODE dynamic,
# DRAG 10, REST_VMAX 0.01 (vrms; at 256 the sheet's RMS speed plateaus at ~0.008 m/s from
# self-contact creep while the crease total is constant from 1.5 s on, so 0.003 never
# triggers and 0.01 fires at ~2.6 s; ~220 s per 256 sample), GAMMA 0 (step softening kept: graded damage at gamma 2
# let the opposing-push sheets relax flat), YIELD 50 (20 keeps more of the shape at the
# price of diffuse creases), SELF_MU unset (Newton 0.5), NAME ds256_paper_rest.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$HERE")"
cd "$CODE_DIR"

STAGE="${1:-}"
PROFILE="${PROFILE:-release}"
LOADING="${LOADING:-free}"
if [ "$PROFILE" = "rest" ]; then
  NAME="${NAME:-ds256_paper_rest}"
  SETTLE="${SETTLE:-3}"
  SETTLE_MODE="${SETTLE_MODE:-dynamic}"
  DRAG="${DRAG:-10}"
  REST_VMAX="${REST_VMAX:-0.01}"
  BEND_SCALE="${BEND_SCALE:-1}"
  LOADING="${LOADING:-scrunch}"
  [ "$LOADING" = "free" ] && LOADING=scrunch
fi
BATCH_ARGS=""
if [ "$LOADING" != "free" ]; then BATCH_ARGS="--loading $LOADING"; fi
NAME="${NAME:-ds256_const_paper}"
N="${N:-3000}"
NX="${NX:-256}"
SETTLE="${SETTLE:-0}"
SETTLE_MODE="${SETTLE_MODE:-dynamic}"
DRAG="${DRAG:-0}"
REST_VMAX="${REST_VMAX:-0}"
REST_FRAMES="${REST_FRAMES:-10}"
REST_METRIC="${REST_METRIC:-vrms}"
# Per-task time limit. cluster.env's 45 min fits 50 release-captured samples at ~50 s; a
# rest-captured 256 sample takes ~220 s, so 50 of them need ~3.5 h plus retries. A task that
# times out is resumed by re-running `full` (only tasks with missing slots are submitted).
if [ "$PROFILE" = "rest" ]; then
  TIME="${TIME:-05:00:00}"
else
  TIME="${TIME:-$(grep '^TIME=' cluster/cluster.env | cut -d= -f2 | cut -d' ' -f1)}"
fi
YIELD="${YIELD:-50}"
EDGE_KE="${EDGE_KE:-200}"
SOFTEN="${SOFTEN:-0.1}"
GAMMA="${GAMMA:-0}"
BEND_SCALE="${BEND_SCALE:-0}"
SELF_MU="${SELF_MU:-}"
MAXC="${MAXC:-30}"
# Pinned in SAMPLE_ARGS (not only in the preset) so submission.json records the physics.
SAMPLE_ARGS="--yield-curvature $YIELD --edge-ke $EDGE_KE --crease-soften $SOFTEN --settle $SETTLE"
if [ "$SETTLE_MODE" != "dynamic" ]; then SAMPLE_ARGS="$SAMPLE_ARGS --settle-mode $SETTLE_MODE"; fi
if [ "$DRAG" != "0" ]; then SAMPLE_ARGS="$SAMPLE_ARGS --drag $DRAG"; fi
if [ "$REST_VMAX" != "0" ]; then SAMPLE_ARGS="$SAMPLE_ARGS --rest-vmax $REST_VMAX --rest-frames $REST_FRAMES --rest-metric $REST_METRIC"; fi
if [ "$GAMMA" != "0" ]; then SAMPLE_ARGS="$SAMPLE_ARGS --damage-gamma $GAMMA"; fi
if [ "$BEND_SCALE" != "0" ]; then SAMPLE_ARGS="$SAMPLE_ARGS --bend-scale-res"; fi
if [ -n "$SELF_MU" ]; then SAMPLE_ARGS="$SAMPLE_ARGS --self-mu $SELF_MU"; fi
if [ "$SETTLE" != "0" ] && [ "$NAME" = "ds256_const_paper" ]; then
  NAME="ds256_const_paper_settle${SETTLE}"
fi
PY=.venv/bin/python
DATA_DIR=$(grep '^DATA_DIR=' cluster/cluster.env | cut -d= -f2)

case "$STAGE" in
  smoke)
    n="${2:-2}"; nx="${3:-128}"
    out="smoke_paper_${nx}"
    if [ "$PROFILE" = "rest" ]; then out="smoke_paper_rest_${nx}"; fi
    echo "smoke: $n paper samples at $nx -> $out  (loading $LOADING; sample args: $SAMPLE_ARGS)"
    $PY generate_batch.py --n "$n" --jobs 2 --nx "$nx" --material paper --seed0 100000 \
      --outdir "$out" --resume $BATCH_ARGS --sample-args="$SAMPLE_ARGS"
    $PY - "$out" <<'PYEOF'
import json, sys, glob, os
d = json.load(open(os.path.join(sys.argv[1], "batch_summary.json")))
for r in d["results"]:
    for i, a in enumerate(r["attempts"]):
        q = a.get("qc", {})
        print(f"slot {r['slot']} attempt {i}: ok={a['ok']} wall={a.get('wall') and round(a['wall'],1)}s "
              f"stretch={q.get('stretch_max')} (release {q.get('stretch_release')}) vmax={q.get('vmax_max')} "
              f"t_rest={q.get('t_rest')} height={q.get('height')} "
              f"pierce={q.get('piercing_pairs')} reasons={a.get('reasons')} "
              + (a['error'][-300:] if 'error' in a else ''))
PYEOF
    ;;
  devel)
    python3 cluster/submit.py --n 4 --nx "$NX" --chunk 2 --material paper \
      --name "${NAME}_devel" --partition gpu_devel --time 00:40:00 \
      --sample-args "$SAMPLE_ARGS" --batch-args "$BATCH_ARGS" --no-merge
    echo "watch: python3 cluster/submit.py --status ${NAME}_devel ; then: tail ${DATA_DIR}/${NAME}_devel/slurm/gen_*.out"
    ;;
  full)
    half=$(( (N + 50 - 1) / 50 / 2 ))     # CHUNK=50 in cluster.env
    last=$(( (N + 50 - 1) / 50 - 1 ))
    python3 cluster/submit.py --n "$N" --nx "$NX" --material paper --name "$NAME" \
      --sample-args "$SAMPLE_ARGS" --batch-args "$BATCH_ARGS" --tasks "0-$((half - 1))" --max-concurrent "$MAXC" --time "$TIME" --no-merge
    python3 cluster/submit.py --n "$N" --nx "$NX" --material paper --name "$NAME" \
      --sample-args "$SAMPLE_ARGS" --batch-args "$BATCH_ARGS" --tasks "${half}-${last}" --partition scavenge_gpu \
      --max-concurrent "$MAXC" --time "$TIME" --no-merge
    echo "monitor: bash cluster/paper.sh status ; when the arrays are done: bash cluster/paper.sh merge"
    ;;
  status)
    python3 cluster/submit.py --status "$NAME"
    ;;
  merge)
    $PY cluster/merge_summaries.py --indir "$DATA_DIR/$NAME" --expect-n "$N"
    $PY package_splits.py --indir "$DATA_DIR/$NAME" --outdir "$DATA_DIR/${NAME}_splits"
    ls "$DATA_DIR/${NAME}_splits"/train/*.h5 | wc -l; ls "$DATA_DIR/${NAME}_splits"/test/*.h5 | wc -l
    ;;
  compare)
    $PY - "$DATA_DIR" "$NAME" <<'PYEOF'
import glob, json, os, sys
import numpy as np
data_dir, name = sys.argv[1], sys.argv[2]
def stats(d):
    res = {}
    for p in glob.glob(os.path.join(d, "batch_summary_*.json")):
        for r in json.load(open(p))["results"]:
            if r["slot"] not in res or (r["ok"] and not res[r["slot"]]["ok"]):
                res[r["slot"]] = r
    att = [a for r in res.values() for a in r["attempts"]]
    st = [a["qc"]["stretch_max"] for a in att if "qc" in a and a["qc"].get("stretch_max") is not None]
    sp = [a["qc"]["vmax_max"] for a in att if "qc" in a and a["qc"].get("vmax_max") is not None]
    nan = sum(1 for a in att if "error" in a or ("qc" in a and not np.isfinite(a["qc"].get("stretch_max", 0))))
    return dict(slots=len(res), passing=sum(r["ok"] for r in res.values()), attempts=len(att),
                rejected=sum(1 for a in att if not a["ok"]), errors_or_nan=nan,
                stretch_p50_p90_max=np.percentile(st, [50, 90, 100]).round(2).tolist() if st else None,
                vmax_p50_max=np.percentile(sp, [50, 100]).round(2).tolist() if sp else None)
for n in ("ds256_const", name):
    print(n, json.dumps(stats(os.path.join(data_dir, n))))
PYEOF
    ;;
  *)
    sed -n 2,22p "$0"; exit 1 ;;
esac
