"""Reproducibility + sensitivity analysis for one sample."""

import argparse

import h5py
import numpy as np


def load_grid(path):
    with h5py.File(path, "r") as h:
        return h["final_grid"][()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--sens1", required=True)
    ap.add_argument("--sens2", required=True)
    args = ap.parse_args()

    ref = load_grid(args.ref)
    rep = load_grid(args.rerun)
    d = np.abs(ref - rep)
    print(f"reproducibility: max per-point diff = {d.max():.3e} m, "
          f"mean = {d.mean():.3e} m")

    for tag, p in (("F1+1%", args.sens1), ("F2+1%", args.sens2)):
        s = load_grid(p)
        dd = np.linalg.norm((s - ref).reshape(-1, 3), axis=1)
        print(f"sensitivity {tag}: mean per-point diff = {dd.mean():.3e} m, "
              f"max = {dd.max():.3e} m")


if __name__ == "__main__":
    main()
