"""Physical error metrics with optimal pairing of predicted and true forces."""

from __future__ import annotations

import numpy as np

SHEET_SIZE = 0.5


def _err_one(p, t):
    pos = np.hypot(p[0] - t[0], p[1] - t[1]) * SHEET_SIZE * 100.0  # cm on sheet
    d = p[2:5] / max(np.linalg.norm(p[2:5]), 1e-8)
    ang = np.degrees(np.arccos(np.clip(np.dot(d, t[2:5]), -1, 1)))
    mag_rel = abs(np.exp(p[5]) - np.exp(t[5])) / np.exp(t[5])
    dur = abs(p[6] - t[6])
    imp_rel = abs(np.exp(p[5]) * p[6] - np.exp(t[5]) * t[6]) / (np.exp(t[5]) * t[6])
    return np.array([pos, ang, mag_rel, dur, imp_rel])


def pair_metrics(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """pred/true: (N,2,7) raw-scale. Returns (N,5) mean-over-forces metrics
    [pos_cm, dir_deg, mag_rel, dur_s, impulse_rel] using the better pairing
    (chosen by position + direction cost)."""
    out = []
    for p, t in zip(pred, true):
        a = np.stack([_err_one(p[0], t[0]), _err_one(p[1], t[1])])
        b = np.stack([_err_one(p[0], t[1]), _err_one(p[1], t[0])])
        ca, cb = a[:, 0].sum() + a[:, 1].sum() / 10, b[:, 0].sum() + b[:, 1].sum() / 10
        out.append((a if ca <= cb else b).mean(0))
    return np.stack(out)


NAMES = ["pos_cm", "dir_deg", "mag_rel", "dur_s", "impulse_rel"]


def summarize(m: np.ndarray) -> dict:
    return {f"{n}_mean": float(m[:, i].mean()) for i, n in enumerate(NAMES)} | \
           {f"{n}_median": float(np.median(m[:, i])) for i, n in enumerate(NAMES)}
