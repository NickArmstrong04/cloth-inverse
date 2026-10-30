"""How identifiable is the force magnitude from the data itself? Read-only check on a split's
HDF5 files (no model): two oracle estimators that use the TRUE position / direction / duration
and only have to recover the magnitude, so their error is a floor for any learned model.

  momentum oracle : the simulator's momentum log at capture, P, against the nominal impulse
                    sum_i F_i t_i d_i; least squares in the two magnitudes given t_i, d_i.
                    Also reports |P| / |J_nominal| (delivered fraction) and its spread: a
                    spread of the fraction is irreducible magnitude error for a velocity reader.
  strain oracle   : peak in-plane strain (trace of g - I) within the patch radius of the true
                    (u, v), fit log F = a + b log s (2-fold cross-validated).

usage: python identifiability.py <splits_dir>/test [--limit N] [--radius 9]
"""
import argparse, glob, json, os
import numpy as np
import h5py

SHEET = 0.5


def strain_trace(grid):
    """grid (ny, nx, 3) -> (ny, nx) trace of the metric minus identity (g_uu + g_vv - 2)."""
    ny, nx = grid.shape[:2]
    hu, hv = SHEET / (nx - 1), SHEET / (ny - 1)
    tv, tu = np.gradient(grid, hv, hu, axis=(0, 1), edge_order=1)
    return (tu * tu).sum(-1) + (tv * tv).sum(-1) - 2.0


def rel(e):
    return float(np.median(np.abs(e)))


def cv_fit(x, y):
    """2-fold CV of y = a + b x; returns the out-of-fold prediction."""
    n = len(x); idx = np.random.default_rng(0).permutation(n); out = np.empty(n)
    for half in (idx[: n // 2], idx[n // 2:]):
        other = np.setdiff1d(idx, half)
        b, a = np.polyfit(x[other], y[other], 1)
        out[half] = a + b * x[half]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--radius", type=float, default=9.0, help="patch radius (cells) for the strain peak")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.split_dir, "*.h5")))
    if args.limit:
        files = files[: args.limit]
    F, T, S, ratio, ang, Fmom = [], [], [], [], [], []
    for p in files:
        with h5py.File(p, "r") as h:
            grid = h["final_grid"][()].astype(np.float64)
            fp = h["force_params"][()]
            fp = json.loads(fp.decode() if isinstance(fp, bytes) else fp)[:2]
            if "sim_log" in h and "momentum" in h["sim_log"]:
                P = h["sim_log"]["momentum"][()][-1].astype(np.float64)
            else:  # test files keep no log; free flight: total momentum at capture = delivered impulse
                cfg = h["sim_config"][()]
                sh = json.loads(cfg.decode() if isinstance(cfg, bytes) else cfg)["sheet"]
                m_p = sh["mass_per_area"] * sh["size_x"] * sh["size_y"] / (sh["nx"] * sh["ny"])
                P = m_p * h["final_vel"][()].reshape(-1, 3).astype(np.float64).sum(0)
        ny, nx = grid.shape[:2]
        st = strain_trace(grid)
        jj, ii = np.mgrid[0:ny, 0:nx]
        d = np.array([np.asarray(f["direction"], float) / np.linalg.norm(f["direction"]) for f in fp])
        t = np.array([f["duration"] for f in fp]); f_true = np.array([f["magnitude"] for f in fp])
        for k, f in enumerate(fp):
            cx, cy = f["u"] * (nx - 1), f["v"] * (ny - 1)
            m = (ii - cx) ** 2 + (jj - cy) ** 2 <= args.radius ** 2
            S.append(max(float(st[m].max()), 1e-6)); F.append(f_true[k]); T.append(t[k])
        if P is not None:
            Jn = (f_true * t) @ d
            ratio.append(np.linalg.norm(P) / np.linalg.norm(Jn))
            ang.append(np.degrees(np.arccos(np.clip(P @ Jn / (np.linalg.norm(P) * np.linalg.norm(Jn)), -1, 1))))
            A = (t[:, None] * d).T  # (3, 2): P = A @ F
            x, *_ = np.linalg.lstsq(A, P, rcond=None)
            Fmom.extend(x / f_true - 1.0)
    F, T, S = map(np.array, (F, T, S))
    print(f"{len(files)} files, {len(F)} forces; F range {F.min():.2f}-{F.max():.2f} N")
    print(f"constant predictor (geometric mean): median |rel err| {rel(np.exp(np.log(F).mean()) / F - 1):.3f}")
    s_pred = cv_fit(np.log(S), np.log(F))
    r = np.corrcoef(np.log(S), np.log(F))[0, 1]
    print(f"strain oracle   : corr(log peak strain, log F) = {r:+.2f}; "
          f"median |rel err| {rel(np.exp(s_pred) / F - 1):.3f}  (peak strain median {np.median(S):.4f})")
    if ratio:
        ratio, ang, Fmom = map(np.array, (ratio, ang, Fmom))
        print(f"momentum at capture / nominal impulse: median {np.median(ratio):.3f}, "
              f"IQR {np.percentile(ratio, 25):.3f}-{np.percentile(ratio, 75):.3f}, angle median {np.median(ang):.1f} deg")
        print(f"momentum oracle : median |rel err| {rel(Fmom):.3f} (raw least squares, no calibration); "
              f"after dividing by the median fraction {rel((Fmom + 1) / np.median(ratio) - 1):.3f}")
    else:
        print("momentum oracle : no momentum available")


if __name__ == "__main__":
    main()
