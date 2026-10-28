"""Package batch samples into 8:1:1 train/val/test splits.

Train and validation files are hard-linked (no copy); test files carry ONLY the
final state (final_grid, final_vel) plus labels/config, so the test split
evaluates inference from the end state alone.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import h5py
import numpy as np

KEEP_TEST = ("final_grid", "final_vel", "force_params", "sim_config", "seed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="batch dir with batch_summary.json")
    ap.add_argument("--outdir", required=True, help="dataset output dir")
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--ratios", default="8,1,1")
    args = ap.parse_args()
    if os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir)  # start clean (a previous interrupted run leaves partial files)

    ratios = np.array([float(x) for x in args.ratios.split(",")])
    ratios = ratios / ratios.sum()
    with open(os.path.join(args.indir, "batch_summary.json")) as f:
        summary = json.load(f)
    files = sorted(r["file"] for r in summary["results"] if r["ok"] and r["file"])
    assert files, "no passing samples to package"
    rng = np.random.default_rng(args.split_seed)
    order = rng.permutation(len(files))
    n_test = max(1, int(round(len(files) * ratios[2])))
    n_val = max(1, int(round(len(files) * ratios[1])))
    n_train = len(files) - n_val - n_test
    splits = {
        "train": [files[i] for i in order[:n_train]],
        "val": [files[i] for i in order[n_train:n_train + n_val]],
        "test": [files[i] for i in order[n_train + n_val:]],
    }
    manifest = {"ratios": args.ratios, "split_seed": args.split_seed, "splits": {}}
    for split, names in splits.items():
        sdir = os.path.join(args.outdir, split)
        os.makedirs(sdir, exist_ok=True)
        entries = []
        for name in names:
            src = os.path.join(args.indir, name)
            if split == "test":
                dst = os.path.join(sdir, name)
                with h5py.File(src, "r") as h_in, h5py.File(dst, "w") as h_out:
                    for k in KEEP_TEST:
                        h_out.create_dataset(k, data=h_in[k][()])
                    for k, v in h_in.attrs.items():
                        h_out.attrs[k] = v
                    h_out.attrs["split"] = "test"
                    h_out.attrs["note"] = "intermediates withheld"
            else:
                dst = os.path.join(sdir, name)
                if os.path.exists(dst):
                    os.remove(dst)
                try:
                    os.link(src, dst)  # hard link: instant, no extra disk (same filesystem)
                except OSError:
                    shutil.copy2(src, dst)
            with h5py.File(src, "r") as h:
                fp = h["force_params"][()]
                fp = fp.decode() if isinstance(fp, bytes) else fp
            entries.append({"file": name, "force_params": json.loads(fp)})
        manifest["splits"][split] = entries
        print(f"{split}: {len(entries)} files")
    with open(os.path.join(args.outdir, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", os.path.join(args.outdir, "dataset_manifest.json"))


if __name__ == "__main__":
    main()
