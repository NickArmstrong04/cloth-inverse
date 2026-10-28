#!/bin/bash
# Stability sweep template: one job per solver setting; re-simulates a fixed set
# of force configurations (ones that failed QC in production) and records
# QC pass/fail, max speed and simulation time for each.
#SBATCH --job-name=cloth_stab
#SBATCH --partition=__PARTITION__
#SBATCH --gpus=__GPUS__
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=__TIME__
#SBATCH --output=__OUTDIR__/slurm_%j.out
#SBATCH --error=__OUTDIR__/slurm_%j.err
__ACCOUNT_LINE__
__EXCLUDE_LINE__
set -uo pipefail
CODE_DIR="__CODE_DIR__"
OUTDIR="__OUTDIR__"
NX=__NX__
SETTING="__SETTING__"            # label, e.g. sub90
EXTRA="__EXTRA__"                # generate_sample.py flags for this setting
FORCES_DIR="__FORCES_DIR__"      # directory of forces_*.json to re-simulate
cd "$CODE_DIR"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$CODE_DIR/.warp_cache}"
mkdir -p "$OUTDIR"
if ! .venv/bin/python -c "import numpy, h5py, warp, newton" 2>/dev/null; then echo "ENVIRONMENT BROKEN"; exit 2; fi
echo "stability $SETTING ($EXTRA) nx=$NX on $(hostname) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name --format=csv,noheader
SCALE=$(python3 -c "print($NX/128)")
MARGIN=$(python3 -c "print(1.0e-3/$SCALE)"); GAP=$(python3 -c "print(0.5e-3/$SCALE)"); RADIUS=$(python3 -c "print(2.0e-3/$SCALE)")
for FJ in "$FORCES_DIR"/forces_*.json; do
  B=$(basename "$FJ" .json)
  OUT="$OUTDIR/${B}.h5"
  S0=$(date +%s.%N)
  .venv/bin/python generate_sample.py --seed 1 --nx $NX --ny $NX --radius $RADIUS --margin $MARGIN --gap $GAP --force-every-substep \
      --force-json "$FJ" --out "$OUT" $EXTRA > "$OUTDIR/${B}.log" 2>&1
  RC=$?
  S1=$(date +%s.%N)
  .venv/bin/python - "$OUTDIR" "$B" "$RC" "$S0" "$S1" "$SETTING" "$EXTRA" "$NX" <<'PY'
import json, os, sys
sys.path.insert(0, ".")
out, b, rc, s0, s1, setting, extra, nx = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7], int(sys.argv[8])
row = {"setting": setting, "extra": extra, "nx": nx, "forces": b, "rc": rc, "wall_total_s": s1 - s0, "ok": False}
p = f"{out}/{b}.h5"
if rc == 0 and os.path.exists(p):
    import h5py, numpy as np
    from cloth_sim.validate import qc_sample_h5
    qc = qc_sample_h5(p, check_pierce=False, stretch_max=(1.3 if "paper" in extra else 2.0))
    row.update({"ok": qc["ok"], "reasons": qc["reasons"], **{k: v for k, v in qc["metrics"].items() if v is not None}})
    with h5py.File(p) as h:
        cfg = json.loads(h["sim_config"][()]); row["sim_wall_s"] = cfg["wall_time_s"]
        if "momentum" in h["sim_log"]:
            mom = h["sim_log/momentum"][()]; fp = json.loads(h["force_params"][()])
            imp = sum(np.array(f["direction"]) / np.linalg.norm(f["direction"]) * f["magnitude"] * f["duration"] * 0.85 for f in fp)
            tend = max(f["duration"] for f in fp); i0 = min(len(mom) - 1, int(round(tend / h["sim_log"].attrs["frame_dt"])) - 1)
            row["impulse_ratio_at_force_end"] = float(np.linalg.norm(mom[i0]) / np.linalg.norm(imp))
    os.remove(p)
else:
    row["error"] = open(f"{out}/{b}.log").read().strip().splitlines()[-1][:200] if os.path.exists(f"{out}/{b}.log") else "no log"
with open(f"{out}/results.jsonl", "a") as f:
    f.write(json.dumps(row) + chr(10))
print(f"{setting} {b}: ok={row['ok']} stretch={row.get('stretch_max')} vmax={row.get('vmax_max')} sim={row.get('sim_wall_s')} impulse_ratio={row.get('impulse_ratio_at_force_end')} {row.get('reasons', row.get('error', ''))}")
PY
done
echo "stability $SETTING done"
