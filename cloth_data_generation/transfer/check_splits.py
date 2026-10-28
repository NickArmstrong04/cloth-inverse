"""Verify split packaging: train keeps intermediates, test strips them."""

import glob
import os

import h5py

for split in ("train", "val", "test"):
    for path in sorted(glob.glob(os.path.join("smoke_dataset", split, "*.h5"))):
        with h5py.File(path, "r") as h:
            keys = sorted(h.keys())
        has_inter = "intermediate_grids" in keys
        print(split, os.path.basename(path), "keys=", keys)
        if split == "test":
            assert not has_inter, "test must not carry intermediates"
            assert "final_grid" in keys and "force_params" in keys
        else:
            assert has_inter, f"{split} must keep intermediates"
print("SPLIT_CHECK_DONE")
