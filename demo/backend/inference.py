"""M16-a inference wrapper for the demo backend.

Reproduces the rung-1 input pipeline from ``cloth_data_generation/inverse``
exactly (see ``eval_grid.py GridRun`` + ``grid_input`` + ``data.py``)::

* X = (pos - centroid, vel) on the indexed material grid, G = (centroid, mean vel)
* hybrid model takes raw metres and standardises internally with its stored
  ``x_scale`` / ``g_scale`` (== ``xstd`` / ``gstd`` from ``model.pt``)
* ``--no-rot`` (no Kabsch), no TTA, denormalise with stored ``scaler``
* metrics via ``metrics.pair_metrics`` with symmetry off (known frame)

``model.pt`` is a dict with ``state``, ``args``, ``scaler`` (mean, std),
``xstd``, ``gstd``. The M16-a checkpoint lives on the cluster only; Nick
copies ``model.pt`` + ``metrics.json`` to ``demo/backend/checkpoints/M16a/``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_INVERSE = os.path.join(_REPO, "cloth_data_generation", "inverse")

# M16-a reference numbers (RESULTS.md M16 table: mean pos/dir, median rel force).
REF_POS_CM = 1.0
REF_DIR_DEG = 4.3
REF_IMPULSE_REL_MEDIAN = 0.156

EXPECTED = {"model": "cnn_reg", "hybrid": True, "kabsch": False, "no_vel": False}


def _load_torch_modules():
    sys.path.insert(0, _INVERSE)
    import torch  # noqa: E402

    from data import _pool, decode_forces, encode_forces  # noqa: E402
    from metrics import NAMES, pair_metrics, summarize  # noqa: E402
    from models import build  # noqa: E402
    return torch, _pool, decode_forces, encode_forces, NAMES, pair_metrics, summarize, build


def load_checkpoint(ckpt_dir: str, device=None):
    """Load ``model.pt`` from ``ckpt_dir``. Returns (ckpt, torch)."""
    torch, _pool, decode_forces, encode_forces, NAMES, pair_metrics, summarize, build = _load_torch_modules()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(pt):
        raise FileNotFoundError(
            f"{pt} not found. Ask Nick to copy model.pt + metrics.json from "
            "/home/naa79/scratch_pi_sz622/naa79/cloth-inverse/"
            "cloth_data_generation/inverse/runs/M16a/ to demo/backend/checkpoints/M16a/")
    ck = torch.load(pt, map_location=device, weights_only=False)
    a = ck["args"]
    for k, v in EXPECTED.items():
        got = bool(a.get(k)) if isinstance(v, bool) else a.get(k)
        if got != v:
            print(f"WARNING: checkpoint args[{k!r}] = {a.get(k)!r}, expected {v!r} (M16-a recipe)")
    state = {k[2:] if k.startswith("m.") else k: v for k, v in ck["state"].items()}
    xstd, gstd = ck["xstd"].to(device), ck["gstd"].to(device)
    mean, std = ck["scaler"]
    mean, std = mean.to(device), std.to(device)
    n_blocks = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("enc.blocks."))
    in_size = 8 * 2 ** n_blocks if n_blocks <= 5 else 16 * 2 ** n_blocks
    model = build(
        a["model"], uv_scale=(mean[:2], std[:2]), w=a.get("width", 32),
        in_size=in_size, cin=xstd.shape[1], gdim=gstd.shape[1],
        aux=(a.get("aux_w") or 0) > 0, intrinsic=bool(a.get("intrinsic")),
        pool="max" if a.get("pool_max") else "mean",
        strain_scale=a.get("strain_scale") or 10.0, hybrid=bool(a.get("hybrid")),
        intrinsic_rel=bool(a.get("intrinsic_rel")),
        patch_head=a.get("patch_head") or 0).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if not bool(a.get("hybrid")):
        missing = [k for k in missing if k not in ("x_scale", "g_scale")]
    assert not missing, f"missing tensors: {missing[:5]}"
    assert not unexpected, f"unexpected tensors: {unexpected[:5]}"
    model.eval()
    ckpt = {"model": model, "mean": mean, "std": std, "xstd": xstd, "gstd": gstd,
            "in_size": in_size, "hybrid": bool(a.get("hybrid")), "args": dict(a),
            "torch": torch, "decode_forces": decode_forces,
            "encode_forces": encode_forces, "pair_metrics": pair_metrics,
            "summarize": summarize, "NAMES": NAMES, "device": device}
    return ckpt, torch


def grid_input(pos: np.ndarray, vel: np.ndarray | None, torch=None):
    """train.py's X / G from a (ny, nx, 3) world-frame grid (cf. eval_grid.grid_input)."""
    if torch is None:
        _load_torch_modules()
        import torch as _t
        torch = _t
    sys.path.insert(0, _INVERSE)
    from data import _pool
    c = pos.reshape(-1, 3).mean(0)
    v = vel if vel is not None else np.zeros_like(pos)
    mv = v.reshape(-1, 3).mean(0) if vel is not None else np.zeros(3, np.float32)
    x = _pool(np.concatenate([pos - c, v], -1).astype(np.float32), 1).transpose(2, 0, 1)
    g = np.concatenate([c, mv]).astype(np.float32)
    return x, g


def upsample_indexed(pos: np.ndarray, vel: np.ndarray | None, size: int = 256):
    """Bilinear upsample of indexed material grids to ``size`` (demo step 2: 128 -> 256)."""
    torch, *_ = _load_torch_modules()
    import torch.nn.functional as F
    h = pos.shape[0]
    assert pos.shape[0] == pos.shape[1], pos.shape
    if h == size:
        return pos, vel
    x = torch.from_numpy(pos.transpose(2, 0, 1)).unsqueeze(0).float()
    up = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=True)[0].numpy().transpose(1, 2, 0)
    vup = None
    if vel is not None:
        xv = torch.from_numpy(vel.transpose(2, 0, 1)).unsqueeze(0).float()
        vup = F.interpolate(xv, size=(size, size), mode="bilinear", align_corners=True)[0].numpy().transpose(1, 2, 0)
    return up.astype(np.float32), (vup.astype(np.float32) if vup is not None else None)


@np.errstate(all="ignore")
def _noop():
    pass


def predict_forces(ckpt, pos: np.ndarray, vel: np.ndarray):
    """Full-res final state -> (2, 7) raw-scale prediction + decoded force dicts."""
    torch = ckpt["torch"]
    n = pos.shape[0]
    pool = n // ckpt["in_size"]
    assert pool * ckpt["in_size"] == n, f"grid {n} vs model input {ckpt['in_size']}"
    sys.path.insert(0, _INVERSE)
    from data import _pool
    c = pos.reshape(-1, 3).mean(0)
    mv = vel.reshape(-1, 3).mean(0)
    x = _pool(np.concatenate([pos - c, vel], -1).astype(np.float32), pool).transpose(2, 0, 1)
    g = np.concatenate([c, mv]).astype(np.float32)
    model = ckpt["model"]
    dev = ckpt["device"]
    with torch.no_grad():
        xt = torch.from_numpy(x[:ckpt["xstd"].shape[1]]).unsqueeze(0).to(dev)
        gt = torch.from_numpy(g[:ckpt["gstd"].shape[1]]).unsqueeze(0).to(dev)
        if ckpt["hybrid"]:
            pred, _ = model(xt, gt, None)
        else:
            pred, _ = model(xt / ckpt["xstd"], gt / ckpt["gstd"], None)
        pred = pred.float() * ckpt["std"] + ckpt["mean"]
        pred[..., 2:5] = torch.nn.functional.normalize(pred[..., 2:5], dim=-1)
        out = pred.cpu().numpy()[0]
    forces = ckpt["decode_forces"](out)
    for f in forces:
        f["impulse"] = f["magnitude"] * f["duration"]
    return out, forces


def score_forces(ckpt, pred: np.ndarray, true_fp: list[dict]):
    """Per-force rows + mean, exactly as ``metrics.pair_metrics`` (sym off).

    ``pair_metrics`` returns the mean over the better of the two force
    pairings; here the best pairing is selected with the identical cost
    (position + direction/10) and the two rows are kept as well.
    """
    sys.path.insert(0, _INVERSE)
    from metrics import _err_one  # same error terms as pair_metrics
    y = ckpt["encode_forces"](true_fp)
    best, best_cost = None, None
    for pairing in ((0, 1), (1, 0)):
        e = np.stack([_err_one(pred[0], y[pairing[0]]), _err_one(pred[1], y[pairing[1]])])
        cost = e[:, 0].sum() + e[:, 1].sum() / 10
        if best_cost is None or cost < best_cost:
            best, best_cost = e, cost
    names = ckpt["NAMES"]
    per_force = [{n: float(best[k, i]) for i, n in enumerate(names)} for k in range(2)]
    mean = {n: float(best[:, i].mean()) for i, n in enumerate(names)}
    return per_force, mean


def load_h5(path: str):
    import h5py
    with h5py.File(path, "r") as h:
        pos = h["final_grid"][()].astype(np.float32)
        vel = h["final_vel"][()].astype(np.float32) if "final_vel" in h else None
        raw = h["force_params"][()]
        fp = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    return pos, vel, fp


def main():
    ap = argparse.ArgumentParser(description="Verify the M16-a wrapper on local test samples.")
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "checkpoints", "M16a"))
    ap.add_argument("--h5", default=[], nargs="*")
    args = ap.parse_args()
    files = []
    for pat in args.h5:
        files.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    files = [f for f in files if f.endswith(".h5")]
    assert files, "no .h5 files given (pass --h5 <dir>/*.h5)"
    ckpt, torch = load_checkpoint(args.ckpt)
    print(f"checkpoint {args.ckpt}: model={ckpt['args']['model']} "
          f"hybrid={ckpt['hybrid']} in_size={ckpt['in_size']}")
    rows = []
    for f in files:
        pos, vel, fp = load_h5(f)
        assert vel is not None, f"{f}: no final_vel (velocities required for rung 1)"
        pred, _ = predict_forces(ckpt, pos, vel)
        _, mean = score_forces(ckpt, pred, fp)
        rows.append(mean)
        print(f"{os.path.basename(f)}: pos {mean['pos_cm']:.2f} cm  dir {mean['dir_deg']:.1f} deg  "
              f"impulse_rel {mean['impulse_rel']:.3f}")
    pos = float(np.mean([r["pos_cm"] for r in rows]))
    dgr = float(np.mean([r["dir_deg"] for r in rows]))
    imp = float(np.median([r["impulse_rel"] for r in rows]))
    print(f"n={len(rows)}  pos {pos:.2f} cm (ref {REF_POS_CM})  dir {dgr:.1f} deg (ref {REF_DIR_DEG})  "
          f"impulse_rel median {imp:.3f} (ref {REF_IMPULSE_REL_MEDIAN})")


if __name__ == "__main__":
    main()
