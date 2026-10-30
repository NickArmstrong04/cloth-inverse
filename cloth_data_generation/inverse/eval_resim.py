"""Re-simulation error: the task-level metric for an ill-posed inverse problem.

For each test sample the predicted force pair is simulated with the dataset's own physics
(sheet, solver and capture time are read from the test file's ``sim_config``) and the
resulting state is compared with the true final state. Two references put the number on
a scale: the *floor* re-simulates the TRUE forces (solver non-determinism, the best any
prediction can do) and *chance* simulates the constant predictor (the mean test target,
the same "chance" row as the parameter tables). The headline is

    score = (rmse_pred - rmse_floor) / (rmse_chance - rmse_floor)      (0 = perfect, 1 = chance)

on the centred RMSE (rigid centroid shift removed), plus the raw RMSEs in metres and the
fraction of samples where the prediction beats chance.

usage (from inverse/, with the transfer venv):
    python eval_resim.py --run runs/M13b --root <splits_dir> [--limit N] [--shard i/n] [--merge]
Writes <run>/resim/resim_metrics.json and adds a ``test/resim`` entry to <run>/metrics.json.
One SimSession is reused across all simulations (same config), so each one costs the
simulation itself (~10 s at 256) rather than a process start plus graph capture.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLOTH = os.path.dirname(HERE)
if not os.path.exists(os.path.join(CLOTH, "generate_sample.py")):  # repo layout: the simulator lives in transfer/
    CLOTH = os.path.join(CLOTH, "transfer")
if CLOTH not in sys.path:
    sys.path.insert(0, CLOTH)

FORCE_KEYS = ("u", "v", "direction", "magnitude", "duration", "sigma_cells", "radius_cells")


def load_sample(path: str):
    """final_grid, final_vel, sim_config dict, force_params list of one packaged sample."""
    with h5py.File(path, "r") as h:
        grid = h["final_grid"][()]
        vel = h["final_vel"][()] if "final_vel" in h else None
        cfg = h["sim_config"][()]
        fp = h["force_params"][()]
    dec = lambda b: json.loads(b.decode() if isinstance(b, bytes) else b)
    return grid, vel, dec(cfg), dec(fp)


class Simulator:
    """The dataset's simulator, rebuilt from a sample's ``sim_config`` and kept alive
    (sheet, solver and CUDA graph) across calls."""

    def __init__(self, sim_config: dict, workdir: str, device: str = "cuda:0"):
        import warp as wp
        from cloth_sim.sheet import SheetConfig
        from cloth_sim.simulate import SimConfig
        from generate_sample import SessionCache

        wp.init()
        fields = SheetConfig.__dataclass_fields__
        self.sheet = SheetConfig(**{k: v for k, v in sim_config["sheet"].items() if k in fields})
        simd = {k: v for k, v in sim_config["sim"].items() if k in SimConfig.__dataclass_fields__}
        simd["device"] = device
        self.sim = SimConfig(**simd)
        self.cache = SessionCache()
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)

    def run(self, forces: list[dict], t_capture: float, seed: int, keep: str | None = None):
        """Simulate ``forces`` (generator JSON dicts) to ``t_capture``; returns (grid, vel)."""
        from cloth_sim.forces import forces_from_dicts
        from generate_sample import SampleSpec, generate

        sim = copy.copy(self.sim)
        sim.t_capture = t_capture
        sheet = copy.copy(self.sheet)
        sheet.jitter_seed = seed
        out = keep or os.path.join(self.workdir, f"_tmp_{os.getpid()}.h5")
        spec = SampleSpec(forces=forces_from_dicts([{k: v for k, v in f.items() if k in FORCE_KEYS} for f in forces]),
                          sheet=sheet, sim=sim, t_capture=t_capture, seed=seed, out=out)
        generate(spec, cache=self.cache, verbose=False)
        with h5py.File(out, "r") as h:
            grid, vel = h["final_grid"][()], h["final_vel"][()]
        if keep is None:
            os.remove(out)
        return grid, vel


def grid_error(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.linalg.norm(a - b, axis=-1)
    ca, cb = a.reshape(-1, 3).mean(0), b.reshape(-1, 3).mean(0)
    dc = np.linalg.norm((a - ca) - (b - cb), axis=-1)
    return {"rmse_m": float(np.sqrt((d ** 2).mean())), "max_m": float(d.max()),
            "centroid_err_m": float(np.linalg.norm(ca - cb)),
            "rmse_centered_m": float(np.sqrt((dc ** 2).mean()))}


def chance_forces(preds: list[dict]) -> list[dict]:
    """The constant predictor: mean of the true encoded targets over the evaluated samples
    (per force slot), decoded back to force dicts. Same definition as the tables' chance row."""
    from data import decode_forces, encode_forces
    y = np.stack([encode_forces(p["true"]) for p in preds]).mean(0)
    return decode_forces(y)


def dedupe(preds: list[dict]) -> list[dict]:
    """Runs trained before 2026-10-30 with --no-rot wrote every test sample twice."""
    seen, out = set(), []
    for p in preds:
        if p["file"] not in seen:
            seen.add(p["file"])
            out.append(p)
    return out


def _run_dir(args) -> tuple[str, str]:
    if args.run:
        return os.path.abspath(args.run), os.path.join(os.path.abspath(args.run), "test_predictions.json")
    preds = os.path.abspath(args.preds)
    return os.path.dirname(preds), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir containing test_predictions.json")
    ap.add_argument("--preds", default=None, help="(alternative) path to test_predictions.json")
    ap.add_argument("--root", required=True, help="splits dir (reads <root>/test/<file>)")
    ap.add_argument("--out", default=None, help="default <run>/resim")
    ap.add_argument("--limit", type=int, default=None, help="first N test samples")
    ap.add_argument("--shard", default=None, help="i/n: every n-th sample starting at i (job arrays)")
    ap.add_argument("--merge", action="store_true", help="merge <out>/shard_*.json and exit")
    ap.add_argument("--no-floor", action="store_true", help="skip re-simulating the true forces")
    ap.add_argument("--no-chance", action="store_true", help="skip simulating the constant predictor")
    ap.add_argument("--keep", action="store_true", help="keep the simulated .h5 files")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert args.run or args.preds, "give --run <dir> or --preds <json>"
    run_dir, preds_path = _run_dir(args)
    out = os.path.abspath(args.out or os.path.join(run_dir, "resim"))
    os.makedirs(out, exist_ok=True)
    if args.merge:
        rows = sum((json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "shard_*.json")))), [])
        _report(rows, out, run_dir)
        return
    preds = dedupe(json.load(open(preds_path)))[: args.limit]
    chance = None if args.no_chance else chance_forces(preds)
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        preds = preds[i::n]
    simulator, rows = None, []
    for k, item in enumerate(preds):
        name = item["file"]
        true_grid, true_vel, cfg, _ = load_sample(os.path.join(args.root, "test", name))
        if simulator is None:
            simulator = Simulator(cfg, workdir=out, device=args.device)
        base = os.path.join(out, name.replace(".h5", ""))
        keep = (lambda tag: f"{base}_{tag}.h5") if args.keep else (lambda tag: None)
        row = {"file": name, "t_capture": cfg["t_capture"], "param_metrics": item["metrics"]}
        try:
            pg, _ = simulator.run(item["pred"], cfg["t_capture"], seed=900 + k, keep=keep("pred"))
            row["pred"] = grid_error(pg, true_grid)
        except Exception as e:  # a prediction that blows the simulator counts as a miss at the chance level
            print(f"  {name}: predicted forces failed to simulate: {str(e)[-200:]}", flush=True)
            row["pred"] = None
        if not args.no_floor:
            fg, _ = simulator.run(item["true"], cfg["t_capture"], seed=900 + k, keep=keep("floor"))
            row["floor"] = grid_error(fg, true_grid)
        if chance is not None:
            cg, _ = simulator.run(chance, cfg["t_capture"], seed=900 + k, keep=keep("chance"))
            row["chance"] = grid_error(cg, true_grid)
        rows.append(row)
        fmt = lambda r: "%.3f" % r["rmse_centered_m"] if r else "fail"
        print(f"{name}: pred {fmt(row.get('pred'))}  floor {fmt(row.get('floor'))}  chance {fmt(row.get('chance'))} m "
              f"(centred RMSE)  pos {item['metrics']['pos_cm']:.1f} cm", flush=True)
    tag = args.shard.replace("/", "of") if args.shard else "all"
    json.dump(rows, open(os.path.join(out, f"shard_{tag}.json"), "w"))
    _report(rows, out, run_dir)


def _report(rows: list[dict], out: str, run_dir: str) -> dict:
    summ = {"n": len(rows), "n_pred_failed": sum(1 for r in rows if r.get("pred") is None)}
    for key in ("pred", "floor", "chance"):
        vals = [r[key] for r in rows if r.get(key)]
        if vals:
            summ[key] = {m: float(np.mean([v[m] for v in vals])) for m in vals[0]} | \
                        {m + "_median": float(np.median([v[m] for v in vals])) for m in vals[0]}
    if "pred" in summ and "floor" in summ and "chance" in summ:
        c = "rmse_centered_m"
        # failed predictions are scored at the chance level, so a model is not rewarded for blowing up
        pred = np.array([(r["pred"] or r["chance"])[c] for r in rows if r.get("chance")])
        floor = np.array([r["floor"][c] for r in rows if r.get("chance")])
        chance = np.array([r["chance"][c] for r in rows if r.get("chance")])
        summ["score"] = float((pred.mean() - floor.mean()) / max(chance.mean() - floor.mean(), 1e-9))
        summ["score_median"] = float((np.median(pred) - np.median(floor)) / max(np.median(chance) - np.median(floor), 1e-9))
        summ["frac_beats_chance"] = float((pred < chance).mean())
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(out, "resim_metrics.json"), "w"), indent=2)
    mp = os.path.join(run_dir, "metrics.json")
    if os.path.exists(mp):  # make the number visible to summarize_runs.py next to the parameter metrics
        m = json.load(open(mp))
        m["test/resim"] = summ
        json.dump(m, open(mp, "w"), indent=2)
    print(json.dumps(summ, indent=2))
    return summ


if __name__ == "__main__":
    main()
