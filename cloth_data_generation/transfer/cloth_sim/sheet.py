"""Cloth sheet construction with explicit material-coordinate mapping.

Material coordinates: particle index (i, j) with i along x (u = i/(nx-1))
and j along y (v = j/(ny-1)). Newton builds the grid row-major, so the
flat particle index is ``p = j * nx + i``. The saved (ny, nx, 3) arrays use
axis 0 = v (j), axis 1 = u (i).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

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


def particle_spacing(cfg: SheetConfig) -> tuple[float, float]:
    return cfg.size_x / (cfg.nx - 1), cfg.size_y / (cfg.ny - 1)


def build_sheet(cfg: SheetConfig, device: str = "cuda:0"):
    """Build a free-floating flat sheet at z=0 centered on the origin.

    Zero gravity, no ground plane, no fixed particles. Returns
    ``(model, meta)`` where meta records the grid mapping.
    """
    nx, ny = cfg.nx, cfg.ny
    cell_x, cell_y = particle_spacing(cfg)
    mass_p = cfg.mass_per_area * cfg.size_x * cfg.size_y / (nx * ny)

    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.gravity = [0.0, 0.0, 0.0]
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

    meta = {
        "nx": nx,
        "ny": ny,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "mass_per_particle": mass_p,
        "sheet": asdict(cfg),
    }
    return model, meta


def grid_index(i: int, j: int, nx: int) -> int:
    """Flat particle index for material coordinate (i, j)."""
    return j * nx + i
