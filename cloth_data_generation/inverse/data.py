"""Dataset loading, target encoding and physics-exact augmentation.

Input per sample: the material-indexed final state, a (6, ny, nx) tensor of
centroid-centred positions (m) and velocities (m/s), plus a 6-vector of
global features (centroid displacement, mean velocity). Grid axis0 = v (j),
axis1 = u (i), matching the HDF5 convention.

Target per force: (u, v, dx, dy, dz, log magnitude, duration) -> 7 numbers,
two forces -> (2, 7). The pair is unordered; the loss handles permutation.

Augmentations are exact symmetries of the simulation (no gravity, no ground,
isotropic sheet centred at the origin):
  * any rotation R of the world: positions, velocities, centroid, mean
    velocity and force directions all rotate by R; (u, v) unchanged.
  * the square sheet's dihedral group D4 acting on material coordinates with
    the matching world reflection/swap: flip u <-> 1-u with x -> -x, flip
    v <-> 1-v with y -> -y, transpose (u, v) <-> (v, u) with x <-> y.
"""

from __future__ import annotations

import glob
import json
import os

import h5py
import numpy as np
import torch

SHEET_SIZE = 0.5  # m, square sheet


def load_split(root: str, split: str) -> dict:
    files = sorted(glob.glob(os.path.join(root, split, "*.h5")))
    assert files, f"no files in {root}/{split}"
    X, G, Y, names = [], [], [], []
    for p in files:
        with h5py.File(p, "r") as h:
            pos = h["final_grid"][()].astype(np.float32)  # (ny, nx, 3)
            vel = h["final_vel"][()].astype(np.float32)
            fp = h["force_params"][()]
            fp = json.loads(fp.decode() if isinstance(fp, bytes) else fp)
        c = pos.reshape(-1, 3).mean(0)
        mv = vel.reshape(-1, 3).mean(0)
        x = np.concatenate([pos - c, vel], axis=-1).transpose(2, 0, 1)  # (6, ny, nx)
        X.append(x)
        G.append(np.concatenate([c, mv]))
        Y.append(encode_forces(fp))
        names.append(os.path.basename(p))
    return {"x": torch.tensor(np.stack(X)), "g": torch.tensor(np.stack(G), dtype=torch.float32),
            "y": torch.tensor(np.stack(Y), dtype=torch.float32), "names": names}


def encode_forces(fp: list[dict]) -> np.ndarray:
    out = np.zeros((2, 7), dtype=np.float32)
    for k, f in enumerate(fp[:2]):
        d = np.asarray(f["direction"], dtype=np.float64)
        d = d / np.linalg.norm(d)
        out[k] = [f["u"], f["v"], d[0], d[1], d[2], np.log(f["magnitude"]), f["duration"]]
    return out


def decode_forces(y: np.ndarray) -> list[dict]:
    """(2, 7) target row -> force dicts in the generator's JSON format."""
    out = []
    for row in y:
        d = np.asarray(row[2:5], dtype=np.float64)
        d = d / max(np.linalg.norm(d), 1e-8)
        out.append({"u": float(np.clip(row[0], 0.02, 0.98)),
                    "v": float(np.clip(row[1], 0.02, 0.98)),
                    "direction": [float(v) for v in d],
                    "magnitude": float(np.exp(row[5])),
                    "duration": float(np.clip(row[6], 0.1, 1.0)),
                    "sigma_cells": 3.0, "radius_cells": 9.0})
    return out


class TargetScaler:
    """Z-score the regression targets (u, v, logmag, duration); directions
    are left as unit vectors."""

    def __init__(self, y: torch.Tensor):
        flat = y.reshape(-1, 7)
        self.mean = flat.mean(0)
        self.std = flat.std(0).clamp_min(1e-3)
        self.mean[2:5] = 0.0
        self.std[2:5] = 1.0

    def to(self, dev):
        self.mean, self.std = self.mean.to(dev), self.std.to(dev)
        return self

    def norm(self, y):
        return (y - self.mean) / self.std

    def denorm(self, z):
        return z * self.std + self.mean


def random_rotation(n: int, dev) -> torch.Tensor:
    """Uniform random rotation matrices (n, 3, 3) via QR of Gaussians."""
    a = torch.randn(n, 3, 3, device=dev)
    q, r = torch.linalg.qr(a)
    s = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * s.unsqueeze(-2)
    det = torch.det(q)
    q[det < 0, :, 0] *= -1
    return q


def augment(x, g, y, rotate=True, dihedral=True):
    """Apply a random exact symmetry to a batch. x: (B,6,ny,nx), g: (B,6), y: (B,2,7)."""
    B, dev = x.shape[0], x.device
    x, g, y = x.clone(), g.clone(), y.clone()
    if dihedral:
        for b in range(B):
            if torch.rand(1).item() < 0.5:  # flip u <-> 1-u, x -> -x
                x[b] = x[b].flip(-1)
                x[b, 0] *= -1
                x[b, 3] *= -1
                g[b, 0] *= -1
                g[b, 3] *= -1
                y[b, :, 0] = 1.0 - y[b, :, 0]
                y[b, :, 2] *= -1
            if torch.rand(1).item() < 0.5:  # flip v <-> 1-v, y -> -y
                x[b] = x[b].flip(-2)
                x[b, 1] *= -1
                x[b, 4] *= -1
                g[b, 1] *= -1
                g[b, 4] *= -1
                y[b, :, 1] = 1.0 - y[b, :, 1]
                y[b, :, 3] *= -1
            if torch.rand(1).item() < 0.5:  # transpose (u,v), swap x<->y
                x[b] = x[b].transpose(-1, -2).clone()[[1, 0, 2, 4, 3, 5]]
                g[b] = g[b][[1, 0, 2, 4, 3, 5]]
                y[b, :, [0, 1]] = y[b, :, [1, 0]]
                y[b, :, [2, 3]] = y[b, :, [3, 2]]
    if rotate:
        R = random_rotation(B, dev)  # (B,3,3)
        pos = torch.einsum("bij,bjhw->bihw", R, x[:, 0:3])
        vel = torch.einsum("bij,bjhw->bihw", R, x[:, 3:6])
        x = torch.cat([pos, vel], 1)
        g = torch.cat([torch.einsum("bij,bj->bi", R, g[:, 0:3]),
                       torch.einsum("bij,bj->bi", R, g[:, 3:6])], 1)
        y[:, :, 2:5] = torch.einsum("bij,bkj->bki", R, y[:, :, 2:5])
    return x, g, y
