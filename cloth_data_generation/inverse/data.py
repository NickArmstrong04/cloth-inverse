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
import math
import torch

SHEET_SIZE = 0.5  # m, square sheet


def _pool(a: np.ndarray, k: int) -> np.ndarray:
    """Average k x k blocks of a (ny, nx, C) grid in material space (k=1: no-op)."""
    if k <= 1:
        return a
    ny, nx, c = a.shape
    ny2, nx2 = ny // k, nx // k
    return a[:ny2 * k, :nx2 * k].reshape(ny2, k, nx2, k, c).mean((1, 3))


def force_field_map(fp: list[dict], ny: int, nx: int, pool: int = 1) -> np.ndarray:
    """Dense auxiliary target (3, ny//pool, nx//pool): the applied force field's
    direction, weighted by each force's Gaussian patch (peak 1, same sigma/radius
    the simulator used, scaled to the pooled grid). Carries location and direction
    of both forces at every cell: ~65k labels per sample instead of 14."""
    ny2, nx2 = ny // pool, nx // pool
    out = np.zeros((3, ny2, nx2), dtype=np.float32)
    jj, ii = np.mgrid[0:ny2, 0:nx2]
    for f in fp[:2]:
        cx, cy = f["u"] * (nx2 - 1), f["v"] * (ny2 - 1)
        sig = float(f.get("sigma_cells", 3.0)) / pool
        rad = float(f.get("radius_cells", 9.0)) / pool
        d2 = (ii - cx) ** 2 + (jj - cy) ** 2
        w = np.where(d2 <= rad ** 2, np.exp(-d2 / (2.0 * sig ** 2)), 0.0)
        d = np.asarray(f["direction"], dtype=np.float64)
        d = d / np.linalg.norm(d)
        out += (w[None] * d[:, None, None]).astype(np.float32)
    return out


def _force_params(path: str) -> list[dict]:
    with h5py.File(path, "r") as h:
        fp = h["force_params"][()]
    return json.loads(fp.decode() if isinstance(fp, bytes) else fp)


def load_split(root: str, split: str, pool: int = 1, cache: bool = True) -> dict:
    """Load a packaged split at full resolution (default) or block-averaged by
    ``pool``. The assembled tensors are cached next to the split as
    <split>_pool<k>.pt and reused when the file list is unchanged, so a 768
    dataset (~14 MB per sample) is read from HDF5 only once."""
    files = sorted(glob.glob(os.path.join(root, split, "*.h5")))
    assert files, f"no files in {root}/{split}"
    manifest = os.path.join(root, "dataset_manifest.json")
    if os.path.exists(manifest):  # written last by package_splits.py: the split must match it exactly
        want = sorted(e["file"] for e in json.load(open(manifest))["splits"][split])
        have = sorted(os.path.basename(f) for f in files)
        assert want == have, (f"{root}/{split} has {len(have)} files but the manifest lists {len(want)}: "
                              "packaging still in progress or incomplete; wait for package_splits.py to finish")
    names = [os.path.basename(f) for f in files]
    cpath = os.path.join(root, f"{split}_pool{pool}.pt")
    if cache and os.path.exists(cpath):
        d = torch.load(cpath, weights_only=False, mmap=True)
        if d.get("names") == names:
            if "a" not in d:  # cache from before the dense target existed: add it (reads force_params only)
                ny2, nx2 = d["x"].shape[-2:]
                d = dict(d)
                d["a"] = torch.stack([torch.from_numpy(force_field_map(_force_params(p), ny2 * pool, nx2 * pool, pool))
                                      for p in files])
                try:  # the tensors are mmapped from cpath: write a new file and rename, never save over it
                    d = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}
                    torch.save(d, cpath + ".tmp")
                    os.replace(cpath + ".tmp", cpath)
                except Exception as e:
                    print(f"  (cache not rewritten: {e})")
            return d
    with h5py.File(files[0], "r") as h:
        ny, nx = h["final_grid"].shape[:2]
    ny2, nx2 = ny // pool, nx // pool
    X = torch.empty((len(files), 6, ny2, nx2), dtype=torch.float32)  # preallocate (~14 MB/sample at 768)
    G = torch.empty((len(files), 6), dtype=torch.float32)
    Y = torch.empty((len(files), 2, 7), dtype=torch.float32)
    A = torch.empty((len(files), 3, ny2, nx2), dtype=torch.float32)  # dense force-field target
    for i, p in enumerate(files):
        with h5py.File(p, "r") as h:
            pos = h["final_grid"][()].astype(np.float32)  # (ny, nx, 3)
            vel = h["final_vel"][()].astype(np.float32)
            fp = h["force_params"][()]
            fp = json.loads(fp.decode() if isinstance(fp, bytes) else fp)
        c = pos.reshape(-1, 3).mean(0)
        mv = vel.reshape(-1, 3).mean(0)
        pos, vel = _pool(pos, pool), _pool(vel, pool)
        X[i] = torch.from_numpy(np.concatenate([pos - c, vel], axis=-1).transpose(2, 0, 1).copy())
        G[i] = torch.from_numpy(np.concatenate([c, mv]).astype(np.float32))
        Y[i] = torch.from_numpy(encode_forces(fp))
        A[i] = torch.from_numpy(force_field_map(fp, ny, nx, pool))
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{len(files)} {split}", flush=True)
    d = {"x": X, "g": G, "y": Y, "a": A, "names": names}
    if cache:
        try:
            torch.save(d, cpath)
            print(f"  cached {split} -> {cpath} ({X.numel() * 4 / 1e9:.1f} GB)", flush=True)
        except Exception as e:
            print(f"  (cache not written: {e})")
    return d


def load_frames(root: str, split: str, pool: int = 1, cache: bool = True, mode: str = "post") -> dict:
    """Extra training samples from the intermediate states the generator stores
    (5 evenly spaced grids before t_capture; train/val files keep them, test files
    do not). Each frame is labelled with the sample's forces. ``mode="post"`` keeps
    only frames after both forces have ended (a settling shape produced by the same
    impulse); ``"all"`` also keeps frames taken while a force was still acting.
    Velocities are not stored for frames, so their velocity channels are zero
    (use with --no-vel). Returns x/g/y/a plus ``base`` (index of the parent
    sample in the split's file order) and names like ``<file>@t0.83``."""
    files = sorted(glob.glob(os.path.join(root, split, "*.h5")))
    names = [os.path.basename(f) for f in files]
    cpath = os.path.join(root, f"{split}_pool{pool}_frames_{mode}.pt")
    if cache and os.path.exists(cpath):
        d = torch.load(cpath, weights_only=False, mmap=True)
        if d.get("parent_names") == names:
            return d
    xs, gs, ys, as_, base, fnames = [], [], [], [], [], []
    for i, p in enumerate(files):
        with h5py.File(p, "r") as h:
            if "intermediate_grids" not in h:
                continue
            times = h["intermediate_times"][()]
            fp = _force_params(p)
            t_end = max(f["duration"] for f in fp[:2])
            keep = [k for k, t in enumerate(times) if mode == "all" or t > t_end + 1e-6]
            if not keep:
                continue
            grids = h["intermediate_grids"][()].astype(np.float32)  # (K, ny, nx, 3)
        y = torch.from_numpy(encode_forces(fp))
        a = torch.from_numpy(force_field_map(fp, grids.shape[1], grids.shape[2], pool))
        for k in keep:
            pos = grids[k]
            c = pos.reshape(-1, 3).mean(0)
            pos = _pool(pos, pool)
            x = np.concatenate([pos - c, np.zeros_like(pos)], axis=-1).transpose(2, 0, 1)
            xs.append(torch.from_numpy(x.copy()))
            gs.append(torch.from_numpy(np.concatenate([c, np.zeros(3, np.float32)]).astype(np.float32)))
            ys.append(y); as_.append(a); base.append(i)
            fnames.append(f"{names[i]}@t{float(times[k]):.2f}")
        if (i + 1) % 100 == 0:
            print(f"  frames {i + 1}/{len(files)} {split} ({len(xs)} so far)", flush=True)
    if not xs:
        d = {"x": torch.empty(0), "g": torch.empty(0), "y": torch.empty(0), "a": torch.empty(0),
             "base": torch.empty(0, dtype=torch.long), "names": [], "parent_names": names}
    else:
        d = {"x": torch.stack(xs), "g": torch.stack(gs), "y": torch.stack(ys), "a": torch.stack(as_),
             "base": torch.tensor(base), "names": fnames, "parent_names": names}
    if cache and xs:
        try:
            torch.save(d, cpath)
            print(f"  cached {split} frames ({mode}) -> {cpath} ({d['x'].numel() * 4 / 1e9:.1f} GB)", flush=True)
        except Exception as e:
            print(f"  (cache not written: {e})")
    return d


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


def random_rotation(n: int, dev, yaw_only: bool = False) -> torch.Tensor:
    """Uniform random rotation matrices (n, 3, 3) via QR of Gaussians.
    yaw_only: rotations about the sheet normal (z) only, i.e. the support plane is known
    and only the in-plane heading is unknown (rung 1.5, a bench with a table)."""
    if yaw_only:
        th = torch.rand(n, device=dev) * (2 * math.pi)
        c, s = torch.cos(th), torch.sin(th)
        R = torch.zeros(n, 3, 3, device=dev)
        R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
        return R
    a = torch.randn(n, 3, 3, device=dev)
    q, r = torch.linalg.qr(a)
    s = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * s.unsqueeze(-2)
    det = torch.det(q)
    q[det < 0, :, 0] *= -1
    return q


def augment(x, g, y, rotate=True, dihedral=True, yaw_only=False):
    """Apply a random exact symmetry to a batch. x: (B,6,ny,nx), g: (B,6), y: (B,2,7)."""
    B, dev = x.shape[0], x.device
    x, g, y = x.clone(), g.clone(), y.clone()
    nvec = x.shape[1] // 3  # vector fields on the grid: positions, velocities, extra (e.g. force field)
    ng = g.shape[1] // 3  # global vectors (centroid, mean velocity); may be fewer than nvec
    xs = [3 * k for k in range(nvec)]  # channel index of the x-component of each vector
    swap = sum([[3 * k + 1, 3 * k, 3 * k + 2] for k in range(nvec)], [])  # x<->y per vector
    if dihedral:
        for b in range(B):
            if torch.rand(1).item() < 0.5:  # flip u <-> 1-u, x -> -x
                x[b] = x[b].flip(-1)
                for c in xs:
                    x[b, c] *= -1
                for k in range(ng):
                    g[b, 3 * k] *= -1
                y[b, :, 0] = 1.0 - y[b, :, 0]
                y[b, :, 2] *= -1
            if torch.rand(1).item() < 0.5:  # flip v <-> 1-v, y -> -y
                x[b] = x[b].flip(-2)
                for c in xs:
                    x[b, c + 1] *= -1
                for k in range(ng):
                    g[b, 3 * k + 1] *= -1
                y[b, :, 1] = 1.0 - y[b, :, 1]
                y[b, :, 3] *= -1
            if torch.rand(1).item() < 0.5:  # transpose (u,v), swap x<->y
                x[b] = x[b].transpose(-1, -2).clone()[swap]
                g[b] = g[b][swap[:3 * ng]]
                y[b, :, [0, 1]] = y[b, :, [1, 0]]
                y[b, :, [2, 3]] = y[b, :, [3, 2]]
    if rotate:
        R = random_rotation(B, dev, yaw_only)  # (B,3,3)
        for k in range(nvec):
            x[:, 3 * k:3 * k + 3] = torch.einsum("bij,bjhw->bihw", R, x[:, 3 * k:3 * k + 3])
        for k in range(ng):
            g[:, 3 * k:3 * k + 3] = torch.einsum("bij,bj->bi", R, g[:, 3 * k:3 * k + 3])
        y[:, :, 2:5] = torch.einsum("bij,bkj->bki", R, y[:, :, 2:5])
    return x, g, y
