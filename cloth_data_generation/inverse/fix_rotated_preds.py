"""Return the saved test predictions of a --eval-rotated run to the frame of the packaged files.

Until 2026-10-30 train.py saved ``test_predictions.json`` (and ``test_candidates.json`` for --flow)
in the rotated evaluation frame: val/test inputs and targets are rotated by a seeded random
rotation per sample (rung 2, unknown pose), and the predicted directions were written as the
model saw them. eval_resim.py / select_resim.py apply the saved forces to the un-rotated sample,
so every rung-2 run re-simulated at or below chance (M16-d/e/f3, M13-e). (u, v), magnitude and
duration are frame-free; only the direction needs R^T.

    python fix_rotated_preds.py --run runs/M16d --root $S3 [--dry]

The rotations are regenerated from the same seed and split sizes as train.py; the script checks
them against the files' own force_params before writing (the saved "true" direction, un-rotated,
must match the nominal direction to < 0.5 deg; 0.04 deg observed, 171 deg before) and refuses otherwise. Idempotent: a run whose
"true" already matches the files is left alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch

from train import _rot_from_gen


def _n_files(root: str, split: str) -> int:
    files = glob.glob(os.path.join(root, split, "*.h5"))
    assert files, f"no files in {root}/{split}"
    return len(files)


def _true_dirs(root: str, name: str) -> np.ndarray:
    import h5py
    with h5py.File(os.path.join(root, "test", name), "r") as h:
        fp = h["force_params"][()]
    fp = json.loads(fp.decode() if isinstance(fp, bytes) else fp)
    d = np.array([f["direction"] for f in fp[:2]], dtype=np.float64)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1))).max())


def _unrot(force: dict, Rt: np.ndarray) -> dict:
    g = dict(force)
    g["direction"] = [float(v) for v in Rt @ np.asarray(force["direction"], dtype=np.float64)]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", required=True, help="splits dir the run was trained on")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    res = json.load(open(os.path.join(args.run, "metrics.json")))
    if not res.get("eval_rotated"):
        print(f"{args.run}: not an --eval-rotated run; nothing to do")
        return
    n_val, n_test = _n_files(args.root, "val"), _n_files(args.root, "test")
    gen = torch.Generator().manual_seed(1234)  # train.py: val first, then test, from one generator
    _rot_from_gen(n_val, gen, res.get("yaw_only", False))
    R = _rot_from_gen(n_test, gen, res.get("yaw_only", False)).numpy().astype(np.float64)

    pp = os.path.join(args.run, "test_predictions.json")
    preds = json.load(open(pp))
    if len(preds) == 2 * n_test:  # runs before 2026-10-30 wrote every test sample twice (tta 0 twice)
        seen, uniq = set(), []
        for e in preds:
            if e["file"] not in seen:
                seen.add(e["file"]); uniq.append(e)
        preds = uniq
        print(f"{args.run}: duplicated test rows removed ({2 * n_test} -> {len(preds)})")
    assert len(preds) == n_test, f"{len(preds)} predictions for {n_test} test files"
    # check: the saved "true" direction is the rotated nominal one (R d_file); un-rotated it must match the file
    worst_rot, worst_id = 0.0, 0.0
    for i, e in enumerate(preds):
        d_file = _true_dirs(args.root, e["file"])
        d_saved = np.array([f["direction"] for f in e["true"]], dtype=np.float64)
        worst_id = max(worst_id, _angle(d_saved, d_file))
        worst_rot = max(worst_rot, _angle(d_saved @ R[i], d_file))  # rows: (R^T d)_k = d_j R_jk
    TOL = 0.5  # deg; the saved directions went through float32 / bf16 and a unit-normalisation (~0.04 deg)
    if worst_id < TOL:
        print(f"{args.run}: saved forces already in the file frame (max {worst_id:.4f} deg); nothing to do")
        return
    assert worst_rot < TOL, (f"{args.run}: regenerated rotations do not reproduce the saved frame "
                              f"(max {worst_rot:.3f} deg after un-rotation, {worst_id:.1f} deg before); not writing")
    print(f"{args.run}: {n_test} test items, saved frame = seeded rotation (check {worst_rot:.5f} deg); "
          f"un-rotating directions")
    if args.dry:
        return
    shutil.copy(pp, pp + ".rotframe")
    for i, e in enumerate(preds):
        Rt = R[i].T
        e["pred"] = [_unrot(f, Rt) for f in e["pred"]]
        e["true"] = [_unrot(f, Rt) for f in e["true"]]
        if "pred_raw" in e:
            raw = np.asarray(e["pred_raw"], dtype=np.float64)
            raw[:, 2:5] = raw[:, 2:5] @ R[i]
            e["pred_raw"] = raw.tolist()
    json.dump(preds, open(pp, "w"), indent=2)
    cp = os.path.join(args.run, "test_candidates.json")
    if os.path.exists(cp):
        cands = json.load(open(cp))
        assert len(cands) == n_test
        shutil.copy(cp, cp + ".rotframe")
        for i, e in enumerate(cands):
            Rt = R[i].T
            e["candidates"] = [[_unrot(f, Rt) for f in c] for c in e["candidates"]]
            e["true"] = [_unrot(f, Rt) for f in e["true"]]
        json.dump(cands, open(cp, "w"))
        print(f"  candidates un-rotated too ({len(cands[0]['candidates'])} per item)")
    # the old resim result was computed in the wrong frame
    rd = os.path.join(args.run, "resim")
    if os.path.isdir(rd):
        shutil.move(rd, rd + "_rotframe")
        print("  old resim/ moved to resim_rotframe/ (wrong frame); re-run eval_resim.py")
    print("done")


if __name__ == "__main__":
    main()
