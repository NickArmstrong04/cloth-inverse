"""Diagnose intersection metric: flat vs folded states, file identity check."""

import h5py
import numpy as np

from cloth_sim.validate import count_self_intersections

for name in ("expA64.h5", "expB64.h5", "expC64.h5"):
    with h5py.File(name, "r") as h:
        final = h["final_grid"][()]
        inter = h["intermediate_grids"][()]
    print(name, "final!=flat check:", float(np.abs(final[..., 2]).max()))
    for tag, g in (("inter0", inter[0]), ("final", final)):
        r = count_self_intersections(g)
        print(" ", tag, r)

with h5py.File("expA64.h5", "r") as h:
    fa = h["final_grid"][()]
with h5py.File("expC64.h5", "r") as h:
    fc = h["final_grid"][()]
print("max|finalA - finalC| =", float(np.abs(fa - fc).max()))

# synthetic perfectly flat sheet (must be zero) + gentle arch (must be ~zero)
flat = np.zeros((64, 64, 3), dtype=np.float32)
ii, jj = np.meshgrid(np.arange(64), np.arange(64))
flat[..., 0] = (ii - 31.5) * 0.00794
flat[..., 1] = (jj - 31.5) * 0.00794
print("synthetic flat:", count_self_intersections(flat))
with h5py.File("smoke32.h5", "r") as h:
    gentle = h["final_grid"][()]
print("gentle arch smoke32:", count_self_intersections(gentle))
print("DIAG_DONE")
