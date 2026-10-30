"""Cloth sheet construction with explicit material-coordinate mapping.

Material coordinates: particle index (i, j) with i along x (u = i/(nx-1))
and j along y (v = j/(ny-1)). Newton builds the grid row-major, so the
flat particle index is ``p = j * nx + i``. The saved (ny, nx, 3) arrays use
axis 0 = v (j), axis 1 = u (i).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import warp as wp

import newton


@dataclass
class SheetConfig:
    size_x: float = 0.5
    size_y: float = 0.5
    nx: int = 128  # particles along x
    ny: int = 128  # particles along y
    mass_per_area: float = 0.2  # kg/m^2, light shirting fabric
    tri_ke: float = 1.0e3  # stretch stiffness (VBD), from Newton cloth example
    tri_ka: float = 1.0e3  # area stiffness (VBD), from Newton cloth example
    tri_kd: float = 1.0e2  # membrane damping, from Newton cloth example
    edge_ke: float = 10.0  # bending stiffness, from Newton cloth example
    edge_kd: float = 0.0  # bending damping
    particle_radius: float = 0.002  # ~50% of particle spacing at 128x128
    self_contact_margin: float = 0.001  # ~25% of particle spacing at 128x128
    self_contact_gap: float = 0.0005  # ~12% of particle spacing at 128x128
    material: str = "cloth"
    # Plastic bending (see plasticity.py). None = purely elastic (cloth).
    plastic_yield_curvature: float | None = None  # 1/m; bend past this leaves a crease
    plastic_rate: float = 1.0  # fraction of the excess bend made permanent per substep
    # Bench scenario: sheet resting on a table under gravity (the real-world setting the
    # inverse model is meant for). False = the original free-floating, zero-gravity sheet.
    table: bool = False
    gravity: float = 9.81  # m/s^2, only with table=True
    table_friction: float = 0.5  # ground-plane friction coefficient
    # A perfectly flat sheet compressed in its plane never buckles (no out-of-plane
    # force to seed the fold). Random initial height noise, as a fraction of the cell,
    # applied with table=True; seeded per sample.
    z_jitter: float = 0.1
    jitter_seed: int = 0
    contact_ke: float | None = None  # particle-shape (table) contact stiffness; None = Newton default 1e3


# Overrides of the SheetConfig defaults per material. "cloth" is the original
# dataset sheet and must stay empty so existing datasets reproduce.
MATERIALS: dict[str, dict] = {
    "cloth": {},
    # Light printer-paper-like sheet: lighter, much stiffer in plane (near
    # inextensible) and in bending, and it creases: bends sharper than the
    # yield curvature become permanent, so the end state cannot be unfolded
    # back to flat by elasticity alone. Values are VBD units tuned at 128x128
    # (see REPORT.md, paper section), not measured paper constants.
    # Yield 50 /m (2 cm radius) from the 2026-10-28 sweep: 20 creases 45% of the
    # edges diffusely, 100 barely creases; 50 gives localized crease lines.
    # (The preset carried 20 until 2026-10-30; no dataset was generated with it.)
    "paper": {
        "mass_per_area": 0.08,
        "tri_ke": 1.0e4,
        "tri_ka": 1.0e4,
        "tri_kd": 1.0e2,
        "edge_ke": 50.0,
        "edge_kd": 0.0,
        "plastic_yield_curvature": 50.0,
        "plastic_rate": 1.0,
    },
}


def sheet_config(material: str = "cloth", **overrides) -> SheetConfig:
    if material not in MATERIALS:
        raise ValueError(f"unknown material {material!r}; choose from {sorted(MATERIALS)}")
    return SheetConfig(material=material, **(MATERIALS[material] | overrides))


def particle_spacing(cfg: SheetConfig) -> tuple[float, float]:
    return cfg.size_x / (cfg.nx - 1), cfg.size_y / (cfg.ny - 1)


def build_sheet(cfg: SheetConfig, device: str = "cuda:0"):
    """Build a flat sheet at z=0 centered on the origin.

    Default: free-floating, zero gravity, no ground plane, no fixed particles.
    ``cfg.table``: gravity along -z and a ground plane just under the sheet, so it
    rests on a table and forces scrunch it against the table (bench scenario).
    Returns ``(model, meta)`` where meta records the grid mapping.
    """
    nx, ny = cfg.nx, cfg.ny
    cell_x, cell_y = particle_spacing(cfg)
    mass_p = cfg.mass_per_area * cfg.size_x * cfg.size_y / (nx * ny)

    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.gravity = [0.0, 0.0, -cfg.gravity] if cfg.table else [0.0, 0.0, 0.0]
        if cfg.table:
            # plane one particle radius below the sheet so the resting sheet is in contact
            builder.default_shape_cfg.mu = cfg.table_friction
            builder.add_ground_plane(height=-cfg.particle_radius)
        builder.add_cloth_grid(
            pos=wp.vec3(-cfg.size_x / 2.0, -cfg.size_y / 2.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=nx - 1,
            dim_y=ny - 1,
            cell_x=cell_x,
            cell_y=cell_y,
            mass=mass_p,
            tri_ke=cfg.tri_ke,
            tri_ka=cfg.tri_ka,
            tri_kd=cfg.tri_kd,
            edge_ke=cfg.edge_ke,
            edge_kd=cfg.edge_kd,
            particle_radius=cfg.particle_radius,
        )
        # Required before finalize() when using SolverVBD.
        builder.color(include_bending=True)
        model = builder.finalize()
        if cfg.contact_ke is not None:
            model.soft_contact_ke = cfg.contact_ke
        q_flat = model.particle_q.numpy().copy()
        q0 = initial_positions(cfg, q_flat, cfg.jitter_seed)
        if q0 is not q_flat:
            model.particle_q.assign(q0)

    meta = {
        "nx": nx,
        "ny": ny,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "mass_per_particle": mass_p,
        "sheet": asdict(cfg),
        "q_flat": q_flat,  # unjittered rest positions (per-sample jitter: initial_positions)
    }
    return model, meta


def initial_positions(cfg: SheetConfig, q_flat: np.ndarray, seed: int) -> np.ndarray:
    """Per-sample initial positions: the flat grid, plus seeded height noise with
    ``cfg.table`` (``z_jitter`` x cell). Returns ``q_flat`` itself when no jitter
    applies, so a reused SimSession can reset to it per sample."""
    if not (cfg.table and cfg.z_jitter > 0.0):
        return q_flat
    cell_x, cell_y = particle_spacing(cfg)
    rng = np.random.default_rng(seed)
    q = q_flat.copy()
    q[:, 2] += rng.uniform(0.0, cfg.z_jitter * min(cell_x, cell_y), size=len(q)).astype(q.dtype)
    return q


def grid_index(i: int, j: int, nx: int) -> int:
    """Flat particle index for material coordinate (i, j)."""
    return j * nx + i
