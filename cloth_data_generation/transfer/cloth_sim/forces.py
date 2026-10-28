"""Two linear force vectors applied as Gaussian patches in material space.

Each force is defined by an application point (u, v) in [0, 1]^2, a 3D unit
direction, a magnitude in newtons, and a duration in seconds. Particles are
selected by grid indices (material coordinates), never by world position.
Patch weights are normalized so the vector sum of applied forces equals
``magnitude * direction`` exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ForceParams:
    u: float  # material coordinate in [0, 1] along x
    v: float  # material coordinate in [0, 1] along y
    direction: tuple[float, float, float]  # unit vector (normalized on build)
    magnitude: float  # newtons, total over the patch
    duration: float  # seconds, active on [0, duration)
    sigma_cells: float = 2.5  # Gaussian sigma in particle spacings
    radius_cells: float = 7.5  # cutoff radius (3 sigma) in particle spacings


def _normalize_direction(d) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n == 0.0:
        raise ValueError("force direction must be nonzero")
    return d / n


def patch_weights(nx: int, ny: int, u: float, v: float, sigma_cells: float,
                  radius_cells: float) -> np.ndarray:
    """Normalized Gaussian weights over the (ny, nx) material grid.

    Returns weights with axis 0 = v (j), axis 1 = u (i); sums to 1.
    """
    cx = u * (nx - 1)
    cy = v * (ny - 1)
    jj, ii = np.mgrid[0:ny, 0:nx]
    d2 = (ii - cx) ** 2 + (jj - cy) ** 2
    mask = d2 <= radius_cells**2
    w = np.zeros((ny, nx), dtype=np.float64)
    w[mask] = np.exp(-d2[mask] / (2.0 * sigma_cells**2))
    total = w.sum()
    if total <= 0.0:
        raise ValueError("empty force patch; check (u, v) and radius")
    return w / total


def _ramp_factor(time: float, duration: float, ramp_frac: float) -> float:
    """Smoothstep on/off envelope over the first/last ramp_frac of duration."""
    if not 0.0 < ramp_frac < 0.5:
        return 1.0
    r = ramp_frac * duration
    s = min(max(time / r, 0.0), 1.0)
    e = min(max((duration - time) / r, 0.0), 1.0)
    s = s * s * (3.0 - 2.0 * s)
    e = e * e * (3.0 - 2.0 * e)
    return s * e


def build_force_array(nx: int, ny: int, forces: list[ForceParams],
                      time: float, ramp_frac: float = 0.0) -> np.ndarray:
    """Total external force per particle at time ``time``, shape (nx*ny, 3).

    Only forces with ``time < duration`` contribute. Flat index is j*nx+i.
    """
    total = np.zeros((ny, nx, 3), dtype=np.float32)
    for f in forces:
        if time < f.duration:
            w = patch_weights(nx, ny, f.u, f.v, f.sigma_cells, f.radius_cells)
            d = _normalize_direction(f.direction).astype(np.float32)
            total += ((w * f.magnitude) * _ramp_factor(time, f.duration, ramp_frac))[..., None] * d[None, None, :]
    return total.reshape(nx * ny, 3)


def forces_to_dicts(forces: list[ForceParams]) -> list[dict]:
    out = []
    for f in forces:
        d = asdict(f)
        d["direction"] = list(d["direction"])
        out.append(d)
    return out


def forces_from_dicts(dicts: list[dict]) -> list[ForceParams]:
    return [ForceParams(**d) for d in dicts]


def default_forces() -> list[ForceParams]:
    """First-sample forces: two points near opposite edges pushed toward
    each other with opposing out-of-plane components (twist/S-fold seed).
    Tuned at 64x64: 1 N over a sigma=3-cell patch ramps folding without
    exploding the sheet; see REPORT for the tuning experiments.
    """
    return [
        ForceParams(u=0.12, v=0.35, direction=(1.0, 0.15, 0.45),
                    magnitude=1.0, duration=0.5,
                    sigma_cells=3.0, radius_cells=9.0),
        ForceParams(u=0.88, v=0.65, direction=(-1.0, -0.15, -0.45),
                    magnitude=1.0, duration=0.5,
                    sigma_cells=3.0, radius_cells=9.0),
    ]
