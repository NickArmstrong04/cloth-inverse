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
    mag_n = abs(np.exp(p[5]) - np.exp(t[5]))  # absolute force error in newtons (comparable across papers)
    return np.array([pos, ang, mag_rel, dur, imp_rel, mag_n])


# The square sheet's symmetry group (D4) acting on material coordinates. Under an
# unknown rigid pose these 8 relabelings are indistinguishable (90-degree in-plane
# turns and flipping the sheet over are proper 3D rotations), so (u, v) can only be
# predicted up to this group; world-frame directions are unaffected.
D4 = [lambda u, v: (u, v), lambda u, v: (1 - u, v), lambda u, v: (u, 1 - v), lambda u, v: (1 - u, 1 - v),
      lambda u, v: (v, u), lambda u, v: (1 - v, u), lambda u, v: (v, 1 - u), lambda u, v: (1 - v, 1 - u)]


def _relabel(p: np.ndarray, f) -> np.ndarray:
    q = p.copy()
    q[:, 0], q[:, 1] = f(p[:, 0], p[:, 1])
    return q


def pair_metrics(pred: np.ndarray, true: np.ndarray, sym: bool = False) -> np.ndarray:
    """pred/true: (N,2,7) raw-scale. Returns (N,6) mean-over-forces metrics
    [pos_cm, dir_deg, mag_rel, dur_s, impulse_rel, mag_N] using the better pairing
    (chosen by position + direction cost). With ``sym=True`` the prediction's
    (u, v) may additionally be relabeled by any element of D4 (unknown-pose
    evaluation)."""
    out = []
    relabels = D4 if sym else D4[:1]
    for p, t in zip(pred, true):
        best, best_cost = None, None
        for f in relabels:
            q = _relabel(p, f)
            for pairing in ((0, 1), (1, 0)):
                e = np.stack([_err_one(q[0], t[pairing[0]]), _err_one(q[1], t[pairing[1]])])
                cost = e[:, 0].sum() + e[:, 1].sum() / 10
                if best_cost is None or cost < best_cost:
                    best, best_cost = e, cost
        out.append(best.mean(0))
    return np.stack(out)


NAMES = ["pos_cm", "dir_deg", "mag_rel", "dur_s", "impulse_rel", "mag_N"]


def summarize(m: np.ndarray) -> dict:
    return {f"{n}_mean": float(m[:, i].mean()) for i, n in enumerate(NAMES)} | \
           {f"{n}_median": float(np.median(m[:, i])) for i, n in enumerate(NAMES)}
