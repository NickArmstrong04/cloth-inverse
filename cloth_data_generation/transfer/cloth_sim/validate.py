"""Validation: renders, material-space maps, animation, self-intersection count."""

from __future__ import annotations

import numpy as np


def grid_triangles(ny: int, nx: int) -> np.ndarray:
    """Triangle indices matching ModelBuilder.add_cloth_grid winding."""
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v0 = j * nx + i
            v1 = j * nx + i + 1
            v2 = (j + 1) * nx + i + 1
            v3 = (j + 1) * nx + i
            tris.append((v0, v1, v3))
            tris.append((v1, v2, v3))
    return np.array(tris, dtype=np.int64)


def edge_stretch_map(grid: np.ndarray, rest_h: float) -> np.ndarray:
    """Mean current/rest edge length per node over right/up neighbors."""
    du = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=-1) / rest_h
    dv = np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=-1) / rest_h
    s = np.zeros(grid.shape[:2])
    c = np.zeros(grid.shape[:2])
    s[:, :-1] += du
    c[:, :-1] += 1
    s[:, 1:] += du
    c[:, 1:] += 1
    s[:-1, :] += dv
    c[:-1, :] += 1
    s[1:, :] += dv
    c[1:, :] += 1
    return s / np.maximum(c, 1)


def curvature_map(grid: np.ndarray, rest_h: float) -> np.ndarray:
    """Dimensionless fold indicator: |4-point Laplacian| / rest spacing.

    One-sided boundary stencils are biased, so the outer rim is NaN.
    """
    lap = np.zeros_like(grid)
    lap[1:-1, 1:-1] = (
        grid[1:-1, 2:] + grid[1:-1, :-2] + grid[2:, 1:-1] + grid[:-2, 1:-1]
        - 4.0 * grid[1:-1, 1:-1]
    )
    out = np.full(grid.shape[:2], np.nan)
    out[1:-1, 1:-1] = np.linalg.norm(lap[1:-1, 1:-1], axis=-1) / 4.0 / rest_h
    return out


def _seg_tri_pierce(a: np.ndarray, b: np.ndarray, t: np.ndarray,
                      bary_tol: float = 1e-6) -> bool:
    """True if segment a->b properly pierces triangle t (Moller-Trumbore).

    Computed in float64 with touch-excluding margins so resting contact
    (gap ~ 0) never counts: only true crossings are reported.
    """
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    t = t.astype(np.float64, copy=False)
    d = b - a
    e1 = t[1] - t[0]
    e2 = t[2] - t[0]
    p = np.cross(d, e2)
    det = float(e1 @ p)
    if abs(det) < 1e-24:
        return False  # parallel
    inv = 1.0 / det
    s = a - t[0]
    u = float(s @ p) * inv
    if u < bary_tol or u > 1.0 - bary_tol:
        return False
    q = np.cross(s, e1)
    v = float(d @ q) * inv
    if v < bary_tol or u + v > 1.0 - bary_tol:
        return False
    lam = float(e2 @ q) * inv
    return 1e-7 < lam < 1.0 - 1e-7


def _triangles_pierce(t1: np.ndarray, t2: np.ndarray) -> bool:
    """True if an edge of one triangle properly pierces the other."""
    edges1 = ((t1[0], t1[1]), (t1[1], t1[2]), (t1[2], t1[0]))
    edges2 = ((t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0]))
    for a, b in edges1:
        if _seg_tri_pierce(a, b, t2):
            return True
    for a, b in edges2:
        if _seg_tri_pierce(a, b, t1):
            return True
    return False


def _shrunk_abs(t: np.ndarray, eps: float) -> np.ndarray:
    """Erode triangle by absolute distance eps toward its centroid."""
    c = t.mean(axis=0, keepdims=True)
    d = t - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return c + d * np.maximum(n - eps, 0.0) / np.maximum(n, 1e-18)


DEPTH_LEVELS = (0.0001, 0.0005, 0.001)  # penetration-depth thresholds [m]


def count_self_intersections(grid: np.ndarray, max_pairs: int = 2_000_000) -> dict:
    """Count non-neighbor triangle pairs with a proper edge piercing.

    A pair counts only if an edge of one triangle passes strictly through the
    interior of the other (Moller-Trumbore with touch-excluding margins), so
    resting contact and coplanar touching never count. Broadphase: sort by
    min-x, sweep with active list on max-x overlap, then full AABB overlap.
    Triangle pairs whose material cells touch (|di| <= 1 and |dj| <= 1) are
    skipped as topological neighbors.
    """
    ny, nx, _ = grid.shape
    tris = grid_triangles(ny, nx)
    pts = grid.reshape(-1, 3)
    T = pts[tris]
    mn = T.min(axis=1)
    mx = T.max(axis=1)
    order = np.argsort(mn[:, 0], kind="stable")
    # material cell of each triangle's first vertex (for neighbor exclusion)
    ci = (tris[:, 0] % nx).astype(np.int64)
    cj = (tris[:, 0] // nx).astype(np.int64)

    checked = 0
    piercing = 0
    active: list[int] = []
    for t in order:
        x0 = mn[t, 0]
        active = [a for a in active if mx[a, 0] >= x0]
        for a in active:
            if checked >= max_pairs:
                return {"piercing_pairs": piercing,
                        "pairs_checked": checked, "truncated": True}
            if abs(int(ci[t]) - int(ci[a])) <= 1 and abs(int(cj[t]) - int(cj[a])) <= 1:
                continue
            if np.any(mx[t] < mn[a]) or np.any(mx[a] < mn[t]):
                continue
            checked += 1
            if _triangles_pierce(T[t], T[a]):
                piercing += 1
        active.append(t)
    return {"piercing_pairs": piercing,
            "pairs_checked": checked, "truncated": False}


def qc_sample_h5(path: str, stretch_max: float = 2.0,
                 ke_final_max: float = 1.0, vmax_max: float = 15.0,
                 pierce_max: int = 100, check_pierce: bool = True) -> dict:
    """Automatic quality gate for one sample HDF5.

    Fails on NaN/explosion (sim nan flag, extreme final KE or speed),
    tearing (max local stretch), and gross self-penetration (piercing
    triangle pairs). Thresholds calibrated on healthy runs (stretch ~1.15,
    KE_final ~1e-2, piercings 0); limits are deliberately loose.
    """
    import h5py

    reasons = []
    with h5py.File(path, "r") as h:
        final = h["final_grid"][()]
        ke = h["sim_log"]["kinetic_energy"][()]
        vm = h["sim_log"]["max_speed"][()]
        import json

        cfg = json.loads(h["sim_config"][()])
    rest_h = cfg["sheet"]["size_x"] / (cfg["nx"] - 1)
    metrics = {
        "nan_time": cfg.get("nan_time"),
        "ke_final": float(ke[-1]),
        "vmax_max": float(vm.max()),
        "stretch_max": float(edge_stretch_map(final, rest_h).max()),
    }
    if metrics["nan_time"] is not None:
        reasons.append(f"NaN at t={metrics['nan_time']}")
    if metrics["ke_final"] > ke_final_max:
        reasons.append(f"ke_final {metrics['ke_final']:.2e} > {ke_final_max}")
    if metrics["vmax_max"] > vmax_max:
        reasons.append(f"vmax_max {metrics['vmax_max']:.2f} > {vmax_max}")
    if metrics["stretch_max"] > stretch_max:
        reasons.append(f"stretch_max {metrics['stretch_max']:.2f} > {stretch_max}")
    if check_pierce:
        pierce = count_self_intersections(final)
        metrics["piercing_pairs"] = pierce["piercing_pairs"]
        metrics["pairs_checked"] = pierce["pairs_checked"]
        if pierce["piercing_pairs"] > pierce_max:
            reasons.append(f"piercing {pierce['piercing_pairs']} > {pierce_max}")
    else:
        metrics["piercing_pairs"] = None  # skipped (CPU check too slow at high resolution)
        metrics["pairs_checked"] = None
    return {"ok": not reasons, "metrics": metrics, "reasons": reasons}


def render_views(grid: np.ndarray, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = grid.reshape(-1, 3)
    tris = grid_triangles(*grid.shape[:2])
    lims = np.stack([pts.min(0), pts.max(0)])
    ctr = lims.mean(0)
    r = float((lims[1] - lims[0]).max()) / 2 or 0.1
    fig = plt.figure(figsize=(15, 5))
    for k, (elev, azim, title) in enumerate(
        [(20, -60, "view iso"), (0, -90, "view side (y)"), (90, -90, "view top")]
    ):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], triangles=tris,
                        color="steelblue", edgecolor="none", alpha=0.9,
                        shade=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.set_box_aspect((1, 1, 1))
    fig.suptitle(f"deformed sheet max|z|={np.abs(pts[:,2]).max():.3f} m")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_maps(grid: np.ndarray, rest_h: float, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = grid[..., 2]
    s = edge_stretch_map(grid, rest_h)
    c = curvature_map(grid, rest_h)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axs[0].imshow(z, origin="lower", cmap="coolwarm")
    axs[0].set_title("z over (u,v) [m]")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(s, origin="lower", cmap="viridis",
                        vmin=0.8, vmax=1.2)
    axs[1].set_title("local stretch (current/rest)")
    fig.colorbar(im1, ax=axs[1])
    im2 = axs[2].imshow(c, origin="lower", cmap="magma")
    axs[2].set_title("fold indicator (|lap|/h)")
    fig.colorbar(im2, ax=axs[2])
    for ax in axs:
        ax.set_xlabel("u (i)")
        ax.set_ylabel("v (j)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_force_animation(grids: np.ndarray, times: np.ndarray,
                         forces: list[dict], path: str,
                         arrow_m_per_n: float = 0.12,
                         scatter_stride: int = 4) -> None:
    """Animate the deforming sheet with live force-vector arrows.

    Each force is drawn as a 3D arrow anchored at its patch centroid (which
    rides the deforming mesh), labeled with its magnitude, and shown only
    while the force is active (t < duration). Patch particles are red.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    from .forces import patch_weights

    ny, nx = grids.shape[1:3]
    masks = []
    for f in forces:
        w = patch_weights(nx, ny, f["u"], f["v"], f["sigma_cells"],
                          f["radius_cells"])
        masks.append(w > 0.05 * w.max())
    dirs = [np.asarray(f["direction"], dtype=float) /
            np.linalg.norm(f["direction"]) for f in forces]

    lo = grids.min(axis=(0, 1, 2))
    hi = grids.max(axis=(0, 1, 2))
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    def draw(k):
        ax.clear()
        g = grids[k]
        t = float(times[k])
        p = g[::scatter_stride, ::scatter_stride, :].reshape(-1, 3)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.0, c="steelblue")
        status = []
        for m, (f, d) in enumerate(zip(forces, dirs)):
            active = t < f["duration"]
            status.append(f"F{m + 1}:{'ON' if active else 'off'}")
            if not active:
                continue
            c = g[masks[m]].mean(axis=0)
            v = d * f["magnitude"] * arrow_m_per_n
            ax.quiver(c[0], c[1], c[2], v[0], v[1], v[2],
                      color="red", linewidth=2.5, arrow_length_ratio=0.25)
            ax.text(c[0] + v[0], c[1] + v[1], c[2] + v[2],
                    f"F{m + 1} {f['magnitude']:.1f}N", color="red")
            pm = g[masks[m]][::2]
            ax.scatter(pm[:, 0], pm[:, 1], pm[:, 2], s=2.0, c="red")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.view_init(elev=20, azim=-60)
        ax.set_title(f"t={t:.2f}s  " + "  ".join(status))

    ani = animation.FuncAnimation(fig, draw, frames=len(grids), interval=110)
    ani.save(path, writer="pillow", dpi=80)
    plt.close(fig)


def save_animation(grids: np.ndarray, times: np.ndarray, path: str,
                   stride: int = 1) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    sel = range(0, len(grids), stride)
    frames = [grids[k].reshape(-1, 3) for k in sel]
    lims = np.stack([f.min(0) for f in frames] + [f.max(0) for f in frames])
    lo, hi = lims.min(0), lims.max(0)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    def draw(k):
        ax.clear()
        p = frames[k]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.3, c="steelblue")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_title(f"t={times[list(sel)[k]]:.2f}s")

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=120)
    ani.save(path, writer="pillow", dpi=80)
    plt.close(fig)
