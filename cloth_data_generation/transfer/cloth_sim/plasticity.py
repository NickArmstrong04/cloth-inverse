"""Plastic bending hinges: permanent creases for paper-like sheets.

VBD's bending energy is ``0.5 * k * L * (theta - rest)^2`` per interior edge
and it reads ``model.edge_rest_angle`` on every solve, so plasticity is a
return-mapping update of that array between substeps: wherever an edge is bent
further than a yield angle from its rest angle, the rest angle follows it
(``rate`` = 1 is perfect plasticity). Once the force is gone the sheet relaxes
to the new rest angles instead of flat, i.e. the crease stays.

The yield is set as a curvature (1/m) and converted to a per-edge angle with
the cell spacing, so a crease of a given sharpness yields at the same load at
every resolution.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def plastic_hinge_update(
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    rest_angle: wp.array[float],
    yield_angle: float,
    rate: float,
    max_angle: float,
    plastic_work: wp.array[float],
    bend_props: wp.array2d[float],
    ke_soft: float,
):
    e = wp.tid()
    # boundary edges have no opposite vertex and no bending term
    if edge_indices[e, 0] == -1 or edge_indices[e, 1] == -1:
        return
    x0 = pos[edge_indices[e, 0]]  # opposite 0
    x1 = pos[edge_indices[e, 1]]  # opposite 1
    x2 = pos[edge_indices[e, 2]]  # edge start
    x3 = pos[edge_indices[e, 3]]  # edge end
    # same angle convention as evaluate_dihedral_angle_based_bending_force_hessian
    n1 = wp.cross(x2 - x0, x3 - x0)
    n2 = wp.cross(x3 - x1, x2 - x1)
    edge = x3 - x2
    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(edge)
    if n1_norm < 1.0e-12 or n2_norm < 1.0e-12 or e_norm < 1.0e-12:
        return
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    theta = wp.atan2(wp.dot(wp.cross(n1, n2), edge / e_norm), wp.dot(n1, n2))

    rest = rest_angle[e]
    d = theta - rest
    excess = wp.abs(d) - yield_angle
    if excess > 0.0:
        step = rate * excess
        # keep the rest angle off the +-pi branch cut of atan2 (VBD does not wrap)
        new_rest = wp.clamp(rest + wp.sign(d) * step, -max_angle, max_angle)
        rest_angle[e] = new_rest
        plastic_work[e] = plastic_work[e] + wp.abs(new_rest - rest)
        # crease softening: a yielded hinge is weaker, so further bending
        # concentrates there (VBD reads edge_bending_properties every solve)
        if ke_soft >= 0.0 and bend_props[e, 0] > ke_soft:
            bend_props[e, 0] = ke_soft


class PlasticHinges:
    """Holds the per-edge plastic state for one model; call ``update`` after each substep."""

    def __init__(self, model, yield_curvature: float, cell: float, rate: float = 1.0,
                 max_angle: float = 2.8, soften: float = 1.0):
        self.model = model
        self.yield_angle = float(yield_curvature * cell)
        self.rate = float(rate)
        self.max_angle = float(max_angle)
        self.rest0 = model.edge_rest_angle.numpy().copy()
        self.plastic_work = wp.zeros(model.edge_count, dtype=float, device=model.device)
        # softening: yielded hinges drop to soften * edge_ke (ke_soft < 0 = disabled)
        self.props0 = model.edge_bending_properties.numpy().copy()
        self.soften = float(soften)
        ke0 = float(self.props0[:, 0].max()) if len(self.props0) else 0.0
        self.ke_soft = ke0 * self.soften if 0.0 <= self.soften < 1.0 else -1.0

    def reset(self) -> None:
        """Forget all creases (rest angles and hinge stiffness back to the flat sheet)."""
        self.model.edge_rest_angle.assign(self.rest0)
        self.plastic_work.zero_()
        if self.ke_soft >= 0.0:
            self.model.edge_bending_properties.assign(self.props0)

    def update(self, state) -> None:
        wp.launch(plastic_hinge_update, dim=self.model.edge_count,
                  inputs=[state.particle_q, self.model.edge_indices, self.model.edge_rest_angle,
                          self.yield_angle, self.rate, self.max_angle, self.plastic_work,
                          self.model.edge_bending_properties, self.ke_soft],
                  device=self.model.device)

    def summary(self) -> tuple[float, int]:
        """(sum |rest - rest0| over edges [rad], number of edges that have yielded)."""
        rest = self.model.edge_rest_angle.numpy()
        work = self.plastic_work.numpy()
        return float(np.abs(rest - self.rest0).sum()), int((work > 0.0).sum())

    def crease_map(self, nx: int, ny: int) -> np.ndarray:
        """Per-particle crease strength (ny, nx): mean |rest angle| of incident edges.

        Each interior edge contributes to its two edge-end vertices.
        """
        idx = self.model.edge_indices.numpy()
        rest = np.abs(self.model.edge_rest_angle.numpy() - self.rest0)
        ok = (idx[:, 0] >= 0) & (idx[:, 1] >= 0)
        acc = np.zeros(nx * ny)
        cnt = np.zeros(nx * ny)
        for col in (2, 3):
            np.add.at(acc, idx[ok, col], rest[ok])
            np.add.at(cnt, idx[ok, col], 1.0)
        return (acc / np.maximum(cnt, 1.0)).reshape(ny, nx).astype(np.float32)
