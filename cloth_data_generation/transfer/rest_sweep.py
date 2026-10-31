"""Rest-capture sweep: does the settled (still-image) state of a creased sheet keep the
force information that the release state carries? (plan 2026-10-31, step 2)

For a few real force draws (taken from existing sample files or force JSONs) and a grid
of settle conditions, simulate loading + settle in one run, then report per condition:

* t_rest (or "not at rest by the cap"), max speed and kinetic energy at the end,
* retained deformation: RMS distance to the best-fit plane and height, rest vs release,
* membrane stretch at release and at rest (what relaxes is stretch, what stays is crease),
* crease statistics (yielded hinges, total crease angle),
* model-free identifiability (``--sens``): finite differences through the simulator.
  Each label of force 1 is perturbed in turn (magnitude x1.1, direction +5 deg, u +1 cm,
  duration x1.05) and the centred RMS change of the REST grid is divided by that of the
  RELEASE grid: the fraction of the shape's sensitivity to that label that survives the
  settle. A 1e-4 relative magnitude perturbation gives the numerical floor (the solver is
  bit-exact run to run, so the floor is the solver's own sensitivity, not noise).

Conditions are given as ``name:flag=value,flag=value`` where flags are generate_sample
CLI names without dashes (settle, settle_mode, drag, rest_vmax, damage_gamma,
bend_scale_res, crease_soften, self_mu, material). Example::

    .venv/bin/python rest_sweep.py --from-h5 'ds256_const_test16/*.h5' --n 6 --nx 128 \
        --cond qs:settle=3,settle_mode=quasistatic,rest_vmax=0.01 \
        --cond drag3:settle=3,drag=3,rest_vmax=0.01 --sens qs --render --out smoke_runs/rest_sweep

Outputs: <out>/<cond>/sample_k.h5, <out>/sweep.json, <out>/sweep.md and, with
``--render``, <out>/render_<cond>_k.png (release | rest, three views each).
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cloth_sim.io import load_sample_h5  # noqa: E402
from cloth_sim.validate import edge_stretch_map, grid_triangles  # noqa: E402


# ----------------------------------------------------------------------------- metrics
def plane_rms(g: np.ndarray) -> float:
    """RMS distance (m) of the centred grid to its best-fit plane (0 = flat)."""
    p = g.reshape(-1, 3).astype(np.float64)
    p = p - p.mean(0)
    n = np.linalg.svd(p, full_matrices=False)[2][2]
    return float(np.sqrt(np.mean((p @ n) ** 2)))


def centred_rms(a: np.ndarray, b: np.ndarray) -> float:
    """RMS vertex distance (m) between two grids after removing the centroid shift."""
    a = a.reshape(-1, 3).astype(np.float64)
    b = b.reshape(-1, 3).astype(np.float64)
    d = (a - a.mean(0)) - (b - b.mean(0))
    return float(np.sqrt(np.mean(np.sum(d**2, axis=1))))


def aligned_rms(a: np.ndarray, b: np.ndarray) -> float:
    """RMS vertex distance (m) after the best rigid alignment (Kabsch) of b onto a: the
    shape change without the rigid drift / spin that a free-floating settle accumulates."""
    a = a.reshape(-1, 3).astype(np.float64)
    b = b.reshape(-1, 3).astype(np.float64)
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(b0.T @ a0)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((b0 @ r - a0) ** 2, axis=1))))


def compaction(g: np.ndarray, rng: np.random.Generator, n_pairs: int = 20000) -> float:
    """Mean over random material pairs of current / rest distance (1 = unfolded)."""
    ny, nx, _ = g.shape
    cell = 0.5 / (nx - 1)
    i = rng.integers(0, nx * ny, size=(n_pairs, 2))
    p = g.reshape(-1, 3).astype(np.float64)
    jj, ii = np.divmod(np.arange(nx * ny), nx)
    rest = cell * np.hypot(ii[i[:, 0]] - ii[i[:, 1]], jj[i[:, 0]] - jj[i[:, 1]])
    ok = rest > 10 * cell
    cur = np.linalg.norm(p[i[ok, 0]] - p[i[ok, 1]], axis=1)
    return float(np.mean(cur / rest[ok]))


def grid_metrics(d: dict) -> dict:
    fin = d["final_grid"]
    rel = d.get("release_grid")
    nx = fin.shape[1]
    cell = 0.5 / (nx - 1)
    log, st = d["sim_log"], d.get("settle", {})
    rng = np.random.default_rng(0)
    m = {
        "t_force": float(st.get("t_force", np.nan)),
        "t_end": float(st.get("t_end", np.nan)),
        "t_rest": (None if st.get("t_rest") is None else float(st["t_rest"])),
        "vmax_end": float(log["max_speed"][-1]),
        "ke_end": float(log["kinetic_energy"][-1]),
        "ke_peak": float(log["kinetic_energy"].max()),
        "plane_rms_rest_cm": 100 * plane_rms(fin),
        "height_rest_cm": 100 * float(np.ptp(fin[..., 2])),
        "stretch_rest": float(edge_stretch_map(fin, cell).max()),
        "compaction_rest": compaction(fin, rng),
    }
    if rel is not None:
        m.update({
            "plane_rms_release_cm": 100 * plane_rms(rel),
            "height_release_cm": 100 * float(np.ptp(rel[..., 2])),
            "stretch_release": float(edge_stretch_map(rel, cell).max()),
            "compaction_release": compaction(rel, rng),
            "release_to_rest_rms_cm": 100 * centred_rms(rel, fin),
        })
    if "plastic" in d:
        pl = d["plastic"]
        m["edges_yielded"] = int(pl["edges_yielded"][-1])
        m["crease_total_rad"] = float(pl["crease_total"][-1])
        rest_ang = np.abs(pl["edge_rest_angle"])
        m["hinges_gt_0p5rad"] = int((rest_ang > 0.5).sum())
    return m


# ----------------------------------------------------------------------------- runs
def parse_cond(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    kv = {}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        kv[k.strip()] = v.strip() if v != "" else "1"
    return name, kv


def cond_argv(kv: dict) -> list[str]:
    out: list[str] = []
    for k, v in kv.items():
        flag = "--" + k.replace("_", "-")
        if v.lower() in ("1", "true") and k in ("bend_scale_res", "deterministic", "table"):
            out.append(flag)
        else:
            out += [flag, v]
    return out


def load_forces(args) -> list[tuple[str, list[dict]]]:
    """Force draws; the patch size (sigma / radius in cells) is rescaled from the source
    resolution to --nx so the patch covers the same area of the sheet."""
    items: list[tuple[str, list[dict]]] = []
    for pat in args.from_h5 or []:
        for p in sorted(glob.glob(os.path.expanduser(pat))):
            d = load_sample_h5(p)
            src_nx = int(d["final_grid"].shape[1])
            fs = copy.deepcopy(d["force_params"])
            for f in fs:
                f["sigma_cells"] = f.get("sigma_cells", 2.5) * args.nx / src_nx
                f["radius_cells"] = f.get("radius_cells", 7.5) * args.nx / src_nx
            items.append((os.path.splitext(os.path.basename(p))[0], fs))
    for pat in args.forces or []:
        for p in sorted(glob.glob(os.path.expanduser(pat))):
            with open(p) as f:
                items.append((os.path.splitext(os.path.basename(p))[0], json.load(f)))
    if not items:
        raise SystemExit("no forces: pass --from-h5 and/or --forces")
    return items[: args.n]


def perturbations(forces: list[dict]) -> dict[str, list[dict]]:
    """Label perturbations of force 1 for the sensitivity test."""
    out: dict[str, list[dict]] = {}

    def mod(fn):
        fs = copy.deepcopy(forces)
        fn(fs[0])
        return fs

    def rot5(f):
        d = np.asarray(f["direction"], dtype=np.float64)
        d /= np.linalg.norm(d)
        # a perpendicular axis, rotate by 5 degrees
        a = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        th = np.deg2rad(5.0)
        d2 = d * np.cos(th) + np.cross(a, d) * np.sin(th) + a * np.dot(a, d) * (1 - np.cos(th))
        f["direction"] = [float(x) for x in d2]

    out["floor(mag x1.0001)"] = mod(lambda f: f.__setitem__("magnitude", f["magnitude"] * 1.0001))
    out["magnitude x1.1"] = mod(lambda f: f.__setitem__("magnitude", f["magnitude"] * 1.1))
    out["direction +5deg"] = mod(rot5)
    out["position u +1cm"] = mod(lambda f: f.__setitem__("u", min(0.98, f["u"] + 0.02)))
    out["duration x1.05"] = mod(lambda f: f.__setitem__("duration", f["duration"] * 1.05))
    return out


def run_sample(gs, cache, base_argv: list[str], forces: list[dict], out_h5: str, seed: int) -> dict:
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    fjson = out_h5[:-3] + ".forces.json"
    with open(fjson, "w") as f:
        json.dump(forces, f)
    argv = ["--seed", str(seed), "--force-json", fjson, "--out", out_h5, *base_argv]
    spec = gs.configure(gs.build_parser().parse_args(argv))
    gs.generate(spec, cache=cache, verbose=False)
    return load_sample_h5(out_h5)


def render_pair(rel: np.ndarray | None, fin: np.ndarray, path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = [("release", rel), ("rest", fin)] if rel is not None else [("capture", fin)]
    fig = plt.figure(figsize=(5 * 3, 4.6 * len(grids)))
    k = 0
    for row, (name, g) in enumerate(grids):
        pts = g.reshape(-1, 3)
        tris = grid_triangles(*g.shape[:2])
        lims = np.stack([pts.min(0), pts.max(0)])
        ctr = lims.mean(0)
        r = float((lims[1] - lims[0]).max()) / 2 or 0.1
        for elev, azim, view in [(20, -60, "iso"), (0, -90, "side"), (90, -90, "top")]:
            k += 1
            ax = fig.add_subplot(len(grids), 3, k, projection="3d")
            ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], triangles=tris, color="burlywood",
                            edgecolor="none", alpha=0.95, shade=True)
            ax.view_init(elev=elev, azim=azim)
            for axis, c in zip("xyz", ctr):
                getattr(ax, f"set_{axis}lim")(c - r, c + r)
            ax.set_title(f"{name} {view}")
            ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-h5", nargs="*", help="sample files whose force_params are reused")
    ap.add_argument("--forces", nargs="*", help="force JSON files")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--material", default="paper")
    ap.add_argument("--base", default="--yield-curvature 50 --edge-ke 200 --crease-soften 0.1",
                    help="generate_sample flags shared by every condition")
    ap.add_argument("--cond", action="append", required=True, help="name:flag=value,...")
    ap.add_argument("--sens", default=None, help="condition name for the sensitivity test (first sample only "
                                                 "unless --sens-all)")
    ap.add_argument("--sens-all", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default="smoke_runs/rest_sweep")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import warp as wp

    import generate_sample as gs

    wp.init()
    cache = gs.SessionCache()
    items = load_forces(args)
    conds = [parse_cond(c) for c in args.cond]
    base = ["--nx", str(args.nx), "--ny", str(args.nx), "--material", args.material, *args.base.split(),
            "--device", args.device]
    os.makedirs(args.out, exist_ok=True)

    results: dict = {"conditions": {}, "sensitivity": {}}
    for cname, kv in conds:
        rows = []
        for k, (sname, forces) in enumerate(items):
            h5 = os.path.join(args.out, cname, f"{sname}.h5")
            d = run_sample(gs, cache, base + cond_argv(kv), forces, h5, seed=1000 + k)
            m = grid_metrics(d)
            m["sample"] = sname
            rows.append(m)
            print(f"[{cname}] {sname}: t_rest={m['t_rest']} vmax_end={m['vmax_end']:.4f} "
                  f"plane_rms {m.get('plane_rms_release_cm', float('nan')):.2f} -> {m['plane_rms_rest_cm']:.2f} cm, "
                  f"stretch {m.get('stretch_release', float('nan')):.2f} -> {m['stretch_rest']:.2f}, "
                  f"yielded {m.get('edges_yielded')}", flush=True)
            if args.render:
                render_pair(d.get("release_grid"), d["final_grid"],
                            os.path.join(args.out, f"render_{cname}_{sname}.png"), f"{cname} {sname}")
        results["conditions"][cname] = {"flags": kv, "rows": rows}

    if args.sens:
        kv = dict(conds)[args.sens]
        sens_items = items if args.sens_all else items[:1]
        for k, (sname, forces) in enumerate(sens_items):
            ref = load_sample_h5(os.path.join(args.out, args.sens, f"{sname}.h5"))
            table = {}
            for pname, pf in perturbations(forces).items():
                h5 = os.path.join(args.out, args.sens, "sens", f"{sname}_{pname.split('(')[0].replace(' ', '_')}.h5")
                d = run_sample(gs, cache, base + cond_argv(kv), pf, h5, seed=1000 + k)
                has_rel = "release_grid" in ref
                drel = centred_rms(ref["release_grid"], d["release_grid"]) if has_rel else float("nan")
                drest = centred_rms(ref["final_grid"], d["final_grid"])
                arel = aligned_rms(ref["release_grid"], d["release_grid"]) if has_rel else float("nan")
                arest = aligned_rms(ref["final_grid"], d["final_grid"])
                table[pname] = {"release_cm": 100 * drel, "rest_cm": 100 * drest,
                                "ratio": (drest / drel if drel > 0 else float("nan")),
                                "release_aligned_cm": 100 * arel, "rest_aligned_cm": 100 * arest,
                                "ratio_aligned": (arest / arel if arel > 0 else float("nan"))}
                print(f"[sens {args.sens}] {sname} {pname}: release {100*drel:.3f} cm, rest {100*drest:.3f} cm, "
                      f"ratio {table[pname]['ratio']:.2f}; aligned {100*arel:.3f} -> {100*arest:.3f} cm, "
                      f"ratio {table[pname]['ratio_aligned']:.2f}", flush=True)
            results["sensitivity"][sname] = table

    with open(os.path.join(args.out, "sweep.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    # markdown summary (medians over samples)
    lines = ["| condition | flags | at rest (n) | t_rest med [s] | vmax_end | plane_rms release -> rest [cm] | "
             "height release -> rest [cm] | stretch release -> rest | compaction release -> rest | yielded | hinges > 0.5 rad |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cname, c in results["conditions"].items():
        rows = c["rows"]
        med = lambda key: float(np.nanmedian([r.get(key, np.nan) if r.get(key) is not None else np.nan for r in rows]))  # noqa: E731
        n_rest = sum(r["t_rest"] is not None for r in rows)
        lines.append(f"| {cname} | {' '.join(f'{k}={v}' for k, v in c['flags'].items())} | {n_rest}/{len(rows)} | "
                     f"{med('t_rest'):.2f} | {med('vmax_end'):.4f} | {med('plane_rms_release_cm'):.2f} -> {med('plane_rms_rest_cm'):.2f} | "
                     f"{med('height_release_cm'):.1f} -> {med('height_rest_cm'):.1f} | {med('stretch_release'):.2f} -> {med('stretch_rest'):.2f} | "
                     f"{med('compaction_release'):.2f} -> {med('compaction_rest'):.2f} | {med('edges_yielded'):.0f} | {med('hinges_gt_0p5rad'):.0f} |")
    if results["sensitivity"]:
        lines += ["", "Sensitivity (centred RMS change of the grid per label perturbation of force 1; ratio = rest / release):", "",
                  "| sample | perturbation | release [cm] | rest [cm] | ratio | release aligned [cm] | rest aligned [cm] | ratio aligned |",
                  "|---|---|---|---|---|---|---|---|"]
        for sname, table in results["sensitivity"].items():
            for pname, t in table.items():
                lines.append(f"| {sname} | {pname} | {t['release_cm']:.3f} | {t['rest_cm']:.3f} | {t['ratio']:.2f} | "
                             f"{t['release_aligned_cm']:.3f} | {t['rest_aligned_cm']:.3f} | {t['ratio_aligned']:.2f} |")
    md = "\n".join(lines)
    with open(os.path.join(args.out, "sweep.md"), "w") as f:
        f.write(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
