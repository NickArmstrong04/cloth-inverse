"""Render one run with live force-vector arrows (dense re-run, default forces)."""

import argparse
import os

import warp as wp

from cloth_sim.forces import default_forces, forces_to_dicts
from cloth_sim.sheet import SheetConfig, build_sheet
from cloth_sim.simulate import SimConfig, run_simulation
from cloth_sim.validate import save_force_animation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_seed0_forces.gif")
    ap.add_argument("--capture-every", type=int, default=2)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    args = ap.parse_args()

    wp.init()
    forces = default_forces()
    sheet = SheetConfig(nx=args.nx, ny=args.ny)
    t_capture = max(f.duration for f in forces) + 0.7
    sim = SimConfig(t_capture=t_capture, ramp_frac=0.15,
                    capture_every=args.capture_every)
    model, meta = build_sheet(sheet)
    res = run_simulation(model, forces, sim, meta)
    print(f"dense frames: {len(res['dense_times'])} wall={res['wall_time']:.1f}s")
    save_force_animation(res["dense_grids"], res["dense_times"],
                         forces_to_dicts(forces), args.out)
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
