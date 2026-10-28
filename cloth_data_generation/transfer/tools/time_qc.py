"""Time the piercing check on the most crumpled available states."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # run from transfer/: python tools/<name>.py
import glob
import time

import h5py

from cloth_sim.validate import count_self_intersections

paths = sorted(glob.glob("batch32/sample_seed*.h5"))[-3:]
for p in paths:
    with h5py.File(p, "r") as h:
        g = h["final_grid"][()]
    t0 = time.time()
    r = count_self_intersections(g)
    print(p, r, f"{time.time()-t0:.1f}s")
print("TIME_DONE")
