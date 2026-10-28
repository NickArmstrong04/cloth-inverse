"""Validate a sample HDF5: renders, maps, animation, self-intersection count."""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # run from transfer/: python tools/<name>.py
import argparse
import os

import h5py
import numpy as np

from cloth_sim.validate import (count_self_intersections, render_maps,
                                render_views, save_animation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", help="sample HDF5 file")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with h5py.File(args.h5, "r") as h:
        final = h["final_grid"][()]
        inter = h["intermediate_grids"][()]
        times = h["intermediate_times"][()]
        import json

        cfg = json.loads(h["sim_config"][()])
    rest_h = cfg["sheet"]["size_x"] / (cfg["nx"] - 1)
    base = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.h5))[0])
    render_views(final, base + "_views.png")
    render_maps(final, rest_h, base + "_maps.png")
    allg = np.concatenate([inter, final[None]], axis=0)
    allt = np.concatenate([times, [cfg["t_capture"]]])
    save_animation(allg, allt, base + "_anim.gif")
    print("wrote", base + "_views.png", base + "_maps.png", base + "_anim.gif")
    res = count_self_intersections(final)
    print("self-intersections:", res)


if __name__ == "__main__":
    main()
