#!/usr/bin/env python3
"""List non-finite samples in a packaged splits dir (fields and counts per file).
Usage (venv, login node is fine): python cluster/check_splits.py <splits_dir> [pool]
Reads the cached <split>_pool<k>.pt if present, else the HDF5 files."""
import glob, os, sys
import numpy as np
root = sys.argv[1]; pool = int(sys.argv[2]) if len(sys.argv) > 2 else 1
for split in ("train", "val", "test"):
    cpath = os.path.join(root, f"{split}_pool{pool}.pt")
    bad = []
    if os.path.exists(cpath):
        import torch
        d = torch.load(cpath, weights_only=False, mmap=True)
        for i, n in enumerate(d["names"]):
            fields = [k for k in ("x", "g", "y") if not torch.isfinite(d[k][i]).all()]
            if fields:
                nx = int((~torch.isfinite(d["x"][i])).sum())
                bad.append(f"{n}: {fields} ({nx} non-finite grid values)")
        src = "cache"
    else:
        import h5py
        for p in sorted(glob.glob(os.path.join(root, split, "*.h5"))):
            with h5py.File(p, "r") as h:
                fields = [k for k in ("final_grid", "final_vel") if k in h and not np.isfinite(h[k][()]).all()]
            if fields:
                bad.append(f"{os.path.basename(p)}: {fields}")
        src = "h5"
    print(f"{split} ({src}): {len(bad)} non-finite sample(s)")
    for b in bad:
        print("   ", b)
