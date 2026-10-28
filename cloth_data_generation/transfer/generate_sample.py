"""Generate one deformed-cloth HDF5 sample from a force-parameter JSON or defaults."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import warp as wp

from cloth_sim.forces import ForceParams, default_forces, forces_from_dicts, forces_to_dicts
from cloth_sim.io import save_sample_h5
from cloth_sim.sheet import MATERIALS, build_sheet, sheet_config
from cloth_sim.simulate import SimConfig, run_simulation


def gpu_mem_used_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return float(out.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate one cloth sample.")
    ap.add_argument("--force-json", default=None, help="JSON list of force dicts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sample_seed0.h5")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--t-capture", type=float, default=None,
                    help="default: max force duration + --settle")
    ap.add_argument("--settle", type=float, default=0.7,
                    help="settle delay after forces end when --t-capture unset")
    ap.add_argument("--magnitude", type=float, default=None,
                    help="override both force magnitudes (tuning aid)")
    ap.add_argument("--material", choices=sorted(MATERIALS), default="cloth",
                    help="sheet preset (cloth_sim/sheet.py MATERIALS); flags below override it")
    ap.add_argument("--yield-curvature", type=float, default=None,
                    help="plastic crease yield curvature [1/m] (paper; 0 = elastic)")
    ap.add_argument("--plastic-rate", type=float, default=None)
    ap.add_argument("--edge-ke", type=float, default=None)
    ap.add_argument("--tri-ke", type=float, default=None, help="membrane stiffness (also sets tri_ka)")
    ap.add_argument("--mass-per-area", type=float, default=None, help="kg/m^2 (cloth 0.2, paper 0.08)")
    ap.add_argument("--tri-kd", type=float, default=None)
    ap.add_argument("--table", action="store_true",
                    help="bench scenario: gravity and a ground plane under the sheet")
    ap.add_argument("--gravity", type=float, default=None, help="m/s^2 with --table (default 9.81)")
    ap.add_argument("--friction", type=float, default=None, help="table friction coefficient (default 0.5)")
    ap.add_argument("--substeps", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument("--ramp", type=float, default=0.15)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--force-every-substep", action="store_true",
                    help="apply the external force on every substep (nominal "
                         "impulse); default keeps the original once-per-frame "
                         "write, which delivers ~53%% of the impulse")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    wp.init()
    if args.force_json:
        with open(args.force_json) as f:
            forces = forces_from_dicts(json.load(f))
    else:
        forces = default_forces()
    if args.magnitude is not None:
        for f in forces:
            f.magnitude = args.magnitude

    t_capture = args.t_capture
    if t_capture is None:
        t_capture = max(f.duration for f in forces) + args.settle
    sheet = sheet_config(args.material, nx=args.nx, ny=args.ny)
    sim = SimConfig(t_capture=t_capture, device=args.device,
                    ramp_frac=args.ramp, deterministic=args.deterministic,
                    force_every_substep=args.force_every_substep)
    if args.table:
        sheet.table = True
    if args.gravity is not None:
        sheet.gravity = args.gravity
    if args.friction is not None:
        sheet.table_friction = args.friction
    if args.edge_ke is not None:
        sheet.edge_ke = args.edge_ke
    if args.tri_kd is not None:
        sheet.tri_kd = args.tri_kd
    if args.tri_ke is not None:
        sheet.tri_ke = args.tri_ke
        sheet.tri_ka = args.tri_ke
    if args.mass_per_area is not None:
        sheet.mass_per_area = args.mass_per_area
    if args.yield_curvature is not None:
        sheet.plastic_yield_curvature = args.yield_curvature or None
    if args.plastic_rate is not None:
        sheet.plastic_rate = args.plastic_rate
    if args.margin is not None:
        sheet.self_contact_margin = args.margin
        sim.margin = args.margin
        sheet.particle_radius = max(sheet.particle_radius, args.margin)
    if args.gap is not None:
        sheet.self_contact_gap = args.gap
        sim.gap = args.gap
    if args.substeps is not None:
        sim.substeps = args.substeps
    if args.iterations is not None:
        sim.iterations = args.iterations

    mem_before = gpu_mem_used_mb()
    model, meta = build_sheet(sheet, device=args.device)
    res = run_simulation(model, forces, sim, meta)
    mem_after = gpu_mem_used_mb()

    sim_cfg = {
        "sheet": sheet.__dict__,
        "sim": sim.__dict__,
        "nx": args.nx, "ny": args.ny,
        "t_capture": t_capture,
        "wall_time_s": res["wall_time"],
        "gpu_mem_before_mb": mem_before,
        "gpu_mem_after_mb": mem_after,
        "nan_time": res["nan_time"],
    }
    save_sample_h5(
        args.out,
        final_grid=res["final_grid"], final_vel=res["final_vel"],
        intermediate_grids=res["intermediate_grids"],
        intermediate_times=res["intermediate_times"],
        force_params=forces_to_dicts(forces), sim_config=sim_cfg,
        seed=args.seed, ke=res["ke"], max_speed=res["max_speed"],
        contacts=res["contacts"], frame_dt=res["frame_dt"],
        momentum=res["momentum"], plastic=res["plastic"],
    )
    print(f"saved {args.out}")
    print(f"wall {res['wall_time']:.1f}s frames={len(res['ke'])} "
          f"ke_final={res['ke'][-1]:.3e} ke_max={res['ke'].max():.3e} "
          f"vmax_max={res['max_speed'].max():.2f} "
          f"contacts_max={res['contacts'].max()} contacts_final={res['contacts'][-1]} "
          f"nan={res['nan_time']} gpu_mem={mem_before}->{mem_after} MB")
    if res["plastic"] is not None:
        pl = res["plastic"]
        print(f"plastic: crease_total={pl['crease_total'][-1]:.3f} rad "
              f"edges_yielded={pl['edges_yielded'][-1]}/{len(pl['edge_rest_angle'])} "
              f"yield_angle={pl['yield_angle']:.4f} rad")


if __name__ == "__main__":
    main()
