"""Render one run with live force-vector arrows (dense re-run; default forces or a
dataset force JSON, e.g. a batch's forces_slot0000_seed100000.json)."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # run from transfer/: python tools/<name>.py
import argparse
import os

import warp as wp

import json

from cloth_sim.forces import default_forces, forces_from_dicts, forces_to_dicts
from cloth_sim.sheet import MATERIALS, build_sheet, sheet_config
from cloth_sim.simulate import SimConfig, run_simulation
from cloth_sim.validate import save_force_animation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_seed0_forces.gif")
    ap.add_argument("--capture-every", type=int, default=2)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--material", choices=sorted(MATERIALS), default="cloth")
    ap.add_argument("--force-json", default=None,
                    help="dataset force JSON (generate_batch forces_slot*.json) instead of the defaults")
    ap.add_argument("--res-scaling", action="store_true",
                    help="scale particle radius / margin / gap with the cell as generate_batch does")
    ap.add_argument("--yield-curvature", type=float, default=None, help="override the preset (paper)")
    ap.add_argument("--settle", type=float, default=0.0,
                    help="seconds after the forces end (0 = stop at release, the dataset "
                         "default; paper: try 2-3 to see the creases hold)")
    ap.add_argument("--ramp", type=float, default=0.0,
                    help="force envelope (0 = hard-edged constant; 0.15 = legacy smoothstep)")
    ap.add_argument("--force-once-per-frame", dest="force_every_substep",
                    action="store_false", default=True, help="legacy half-impulse write")
    args = ap.parse_args()

    wp.init()
    if args.force_json:
        forces = forces_from_dicts(json.load(open(args.force_json)))
    else:
        forces = default_forces()
    over = {}
    if args.res_scaling and args.nx != 128:
        rs = args.nx / 128.0
        over = {"particle_radius": 2.0e-3 / rs, "self_contact_margin": 1.0e-3 / rs,
                "self_contact_gap": 0.5e-3 / rs}
    sheet = sheet_config(args.material, nx=args.nx, ny=args.ny, **over)
    if args.yield_curvature is not None:
        sheet.plastic_yield_curvature = args.yield_curvature or None
    t_capture = max(f.duration for f in forces) + args.settle
    sim = SimConfig(t_capture=t_capture, ramp_frac=args.ramp,
                    capture_every=args.capture_every,
                    force_every_substep=args.force_every_substep)
    model, meta = build_sheet(sheet)
    res = run_simulation(model, forces, sim, meta)
    print(f"dense frames: {len(res['dense_times'])} wall={res['wall_time']:.1f}s")
    save_force_animation(res["dense_grids"], res["dense_times"],
                         forces_to_dicts(forces), args.out)
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
