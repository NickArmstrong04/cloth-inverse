#!/bin/bash
# Benchmark job template (one job per GPU type): sweeps resolutions with
# identical forces, records wall time per sample, GPU utilization and memory.
#SBATCH --job-name=cloth_bench
#SBATCH --partition=__PARTITION__
#SBATCH --gpus=__GTYPE__:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --output=__OUTDIR__/slurm_%j.out
#SBATCH --error=__OUTDIR__/slurm_%j.err
__ACCOUNT_LINE__
set -uo pipefail
CODE_DIR="__CODE_DIR__"
OUTDIR="__OUTDIR__"
NX_LIST="__NX_LIST__"
REPEATS=__REPEATS__
BUDGET_S=__BUDGET_S__
SAMPLE_ARGS="__SAMPLE_ARGS__"
GTYPE="__GTYPE__"
CONC=__CONC__

cd "$CODE_DIR"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$CODE_DIR/.warp_cache}"
mkdir -p "$OUTDIR"
if ! .venv/bin/python -c "import numpy, h5py, warp, newton" 2> "$OUTDIR/env_check.err"; then
  echo "ENVIRONMENT BROKEN: $CODE_DIR/.venv is missing packages. Run: bash cluster/setup_env.sh"; cat "$OUTDIR/env_check.err"; exit 2
fi
echo "bench $GTYPE on $(hostname) job $SLURM_JOB_ID start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "{\"job\": \"$SLURM_JOB_ID\", \"gpu\": \"$GTYPE\", \"node\": \"$(hostname)\", \"gpu_name\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\", \"start\": \"$(date -Is)\"}" > "$OUTDIR/job.json"

T0=$(date +%s)
for NX in $NX_LIST; do
  SCALE=$(python3 -c "print($NX/128)")
  MARGIN=$(python3 -c "print(1.0e-3/$SCALE)"); GAP=$(python3 -c "print(0.5e-3/$SCALE)")
  # same physical forces at every resolution: patch size in cells scales with NX
  .venv/bin/python - "$NX" "$OUTDIR/forces_nx$NX.json" <<'PY'
import json, sys
nx, out = int(sys.argv[1]), sys.argv[2]
s = nx / 128
f = [{"u": 0.12, "v": 0.35, "direction": [1, 0, 0.3], "magnitude": 1.0, "duration": 0.5, "sigma_cells": 3*s, "radius_cells": 9*s},
     {"u": 0.88, "v": 0.65, "direction": [-1, 0, -0.3], "magnitude": 1.0, "duration": 0.5, "sigma_cells": 3*s, "radius_cells": 9*s}]
json.dump(f, open(out, "w"))
PY
  if [ "$REPEATS" -le 0 ]; then RMAX=10000; else RMAX=$REPEATS; fi
  for R in $(seq 1 $RMAX); do
    ELAPSED=$(( $(date +%s) - T0 ))
    if [ $ELAPSED -gt $BUDGET_S ]; then echo "budget exhausted at nx=$NX rep=$R after ${ELAPSED}s"; break 2; fi
    # unlimited mode: stop once the next sample would not fit the budget
    if [ "$REPEATS" -le 0 ] && [ -n "${LAST_S:-}" ] && [ $(( ELAPSED + LAST_S )) -gt $BUDGET_S ]; then echo "next sample (~${LAST_S}s) would exceed budget; stopping nx=$NX after $((R-1)) samples"; break 2; fi
    UTIL="$OUTDIR/util_nx${NX}_r${R}.csv"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$UTIL" &
    SMI=$!
    S0=$(date +%s.%N)
    # CONC samples run concurrently on this GPU (one process each, distinct seeds)
    PIDS=()
    for C in $(seq 1 $CONC); do
      SEED=$(( R * 100 + C ))
      .venv/bin/python generate_sample.py --seed $SEED --nx $NX --ny $NX --margin $MARGIN --gap $GAP \
          --force-json "$OUTDIR/forces_nx$NX.json" --out "$OUTDIR/sample_nx${NX}_r${R}_c${C}.h5" $SAMPLE_ARGS \
          > "$OUTDIR/gen_nx${NX}_r${R}_c${C}.log" 2>&1 &
      PIDS+=($!)
    done
    RC=0
    for PID in "${PIDS[@]}"; do wait $PID || RC=1; done
    S1=$(date +%s.%N)
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    LAST_S=$(python3 -c "print(int($S1-$S0)+1)")
    cat "$OUTDIR"/gen_nx${NX}_r${R}_c*.log > "$OUTDIR/gen_nx${NX}_r${R}.log"
    .venv/bin/python - "$OUTDIR" "$NX" "$R" "$RC" "$S0" "$S1" "$GTYPE" "$CONC" <<'PY'
import json, sys, os, h5py, numpy as np
out, nx, r, rc, s0, s1, g, conc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7], int(sys.argv[8])
group_wall = s1 - s0
try:
    u = np.atleast_2d(np.genfromtxt(f"{out}/util_nx{nx}_r{r}.csv", delimiter=",", usecols=(1, 2)))
    util_mean = float(u[:, 0].mean()); mem_peak = float(u[:, 1].max())
    busy = u[u[:, 1] > 500, 0]; util_busy = float(busy.mean()) if len(busy) else None
except Exception:
    util_mean = mem_peak = util_busy = None
sims = []
for c in range(1, conc + 1):
    p = f"{out}/sample_nx{nx}_r{r}_c{c}.h5"
    row = {"gpu": g, "nx": nx, "particles": nx*nx, "rep": r, "conc": conc, "proc": c, "rc": rc,
           "wall_total_s": group_wall, "group_wall_s": group_wall, "sim_wall_s": None, "frames": None,
           "util_mean_pct": util_mean, "util_busy_mean_pct": util_busy, "mem_peak_mb": mem_peak, "nan": None}
    if os.path.exists(p):
        try:
            with h5py.File(p) as h:
                cfg = json.loads(h["sim_config"][()]); row["sim_wall_s"] = cfg["wall_time_s"]; row["nan"] = cfg["nan_time"]
                row["frames"] = int(len(h["sim_log/kinetic_energy"]))
        except Exception as e:
            row["err"] = str(e)
        os.remove(p)
    sims.append(row["sim_wall_s"])
    with open(f"{out}/results.jsonl", "a") as f:
        f.write(json.dumps(row) + chr(10))
print(f"nx={nx} rep={r} conc={conc} rc={rc} group wall {group_wall:.1f}s -> {group_wall/conc:.1f}s per sample; sim walls {sims}; util {util_busy} mem {mem_peak} MB")
PY
    if [ $RC -ne 0 ]; then echo "nx=$NX failed (rc=$RC); see gen_nx${NX}_r${R}.log (CUDA out of memory = GPU too small for this size). Skipping larger sizes."; grep -i -m3 -E "error|memory" "$OUTDIR/gen_nx${NX}_r${R}.log" || tail -3 "$OUTDIR/gen_nx${NX}_r${R}.log"; break 2; fi
  done
done
echo "bench done $(date -Is)"
