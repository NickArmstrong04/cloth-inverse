"""Time integration of the cloth sheet with VBD + self-contact on CUDA."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from .forces import ForceParams, build_force_array
from .plasticity import PlasticHinges


@dataclass
class SimConfig:
    fps: int = 60  # logging / capture frame rate
    substeps: int = 30  # VBD substeps per frame
    iterations: int = 20  # VBD iterations per substep
    t_capture: float = 1.0  # end-state capture time (s)
    n_intermediate: int = 5  # evenly spaced states before t_capture
    margin: float | None = None  # self-contact interaction distance (m)
    gap: float | None = None  # self-contact extra detection reach (m)
    buffer_vt: int = 64  # per-vertex vertex-triangle contact capacity
    buffer_ee: int = 128  # per-edge edge-edge contact capacity
    ramp_frac: float = 0.15  # smooth force on/off over this fraction of duration
    deterministic: bool = False  # RUN_TO_RUN bit-exact repeats (slower)
    capture_every: int = 0  # store every Nth frame densely (0 = intermediates only)
    # Re-apply the external force to the input state on EVERY substep. The
    # original (False) wrote it to state_0 once per frame; since the two state
    # buffers swap each substep, only half the substeps saw the force (~53% of
    # the nominal impulse, measured). True gives the nominal impulse.
    force_every_substep: bool = False
    device: str = "cuda:0"


def _host(arr) -> np.ndarray:
    return np.asarray(arr.numpy())


def run_simulation(model, forces: list[ForceParams], sim: SimConfig,
                   meta: dict) -> dict:
    """Step the model to t_capture; return grids, velocities, and per-frame log.

    Force activity is evaluated per frame (frame time < duration => active).
    """
    nx, ny = meta["nx"], meta["ny"]
    frame_dt = 1.0 / sim.fps
    sim_dt = frame_dt / sim.substeps
    n_frames = int(round(sim.t_capture / frame_dt))

    with wp.ScopedDevice(sim.device):
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        margin = sim.margin if sim.margin is not None \
            else meta["sheet"]["self_contact_margin"]
        gap = sim.gap if sim.gap is not None \
            else meta["sheet"]["self_contact_gap"]
        sc_kwargs: dict = {
            "particle_enable_self_contact": True,
            "particle_vertex_contact_buffer_size": sim.buffer_vt,
            "particle_edge_contact_buffer_size": sim.buffer_ee,
            "particle_self_contact_margin": margin,
            "particle_self_contact_gap": gap,
        }
        sim.margin, sim.gap = margin, gap  # record resolved values
        if sim.deterministic:
            sc_kwargs["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
        solver = newton.solvers.SolverVBD(
            model=model,
            iterations=sim.iterations,
            **sc_kwargs,
        )
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        masses = _host(model.particle_mass).ravel().astype(np.float64)
        sheet = meta["sheet"]
        hinges = None
        if sheet.get("plastic_yield_curvature"):
            hinges = PlasticHinges(model, sheet["plastic_yield_curvature"],
                                   min(meta["cell_x"], meta["cell_y"]),
                                   rate=sheet.get("plastic_rate", 1.0))

        def self_contact_count() -> int:
            # VBD detects self-contact internally; read its detector counts.
            # Falls back to the pipeline soft-contact counter (particle-shape).
            try:
                det = solver.trimesh_collision_detector
                info = det.collision_info
                vt = _host(info.vertex_colliding_triangles_count).ravel()
                ee = _host(info.edge_colliding_edges_count).ravel()
                cap_vt = _host(info.vertex_collision_buffer_sizes).ravel() \
                    if hasattr(info, "vertex_collision_buffer_sizes") else None
                n_vt = (np.minimum(vt, cap_vt).sum() if cap_vt is not None
                        else vt.sum())
                return int(n_vt + ee.sum())
            except Exception:
                return int(_host(contacts.soft_contact_count).ravel()[0])

        ke_log: list[float] = []
        mom_log: list[np.ndarray] = []
        vmax_log: list[float] = []
        contact_log: list[int] = []
        plastic_log: list[tuple[float, int]] = []
        intermediates: list[np.ndarray] = []
        inter_times: list[float] = []
        dense: list[np.ndarray] = []
        dense_times: list[float] = []
        # ~5 evenly spaced earlier times, excluding t=0, plus t_capture.
        keep = {int(round(k * n_frames / (sim.n_intermediate + 1)))
                for k in range(1, sim.n_intermediate + 1)}
        keep.add(n_frames)

        t0 = time.time()
        nan_hit = None
        for frame in range(1, n_frames + 1):
            t = frame * frame_dt
            f_np = build_force_array(nx, ny, forces, (frame - 1) * frame_dt,
                                       ramp_frac=sim.ramp_frac)
            f_dev = (wp.array(f_np, dtype=wp.vec3, device=sim.device)
                     if np.any(f_np) else None)
            if f_dev is not None:
                wp.copy(state_0.particle_f, f_dev)
            else:
                state_0.clear_forces()
            for k_sub in range(sim.substeps):
                if sim.force_every_substep and k_sub > 0:
                    if f_dev is not None:
                        wp.copy(state_0.particle_f, f_dev)
                    else:
                        state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0
                if hinges is not None:
                    hinges.update(state_0)

            q = _host(state_0.particle_q).astype(np.float64)
            qd = _host(state_0.particle_qd).astype(np.float64)
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
                nan_hit = t
                break
            speed = np.linalg.norm(qd, axis=1)
            ke_log.append(float(0.5 * np.sum(masses * speed**2)))
            mom_log.append((masses[:, None] * qd).sum(0))
            vmax_log.append(float(speed.max()))
            contact_log.append(self_contact_count())
            if hinges is not None:
                plastic_log.append(hinges.summary())
            if frame in keep:
                intermediates.append(q.reshape(ny, nx, 3).astype(np.float32))
                inter_times.append(t)
            if sim.capture_every > 0 and frame % sim.capture_every == 0:
                dense.append(q.reshape(ny, nx, 3).astype(np.float32))
                dense_times.append(t)
        wall = time.time() - t0

        final_q = _host(state_0.particle_q).reshape(ny, nx, 3).astype(np.float32)
        final_qd = _host(state_0.particle_qd).reshape(ny, nx, 3).astype(np.float32)

    return {
        "final_grid": final_q,
        "final_vel": final_qd,
        "intermediate_grids": np.stack(intermediates).astype(np.float32),
        "intermediate_times": np.array(inter_times, dtype=np.float64),
        "dense_grids": (np.stack(dense).astype(np.float32) if dense
                        else np.zeros((0, ny, nx, 3), dtype=np.float32)),
        "dense_times": np.array(dense_times, dtype=np.float64),
        "ke": np.array(ke_log, dtype=np.float64),
        "momentum": (np.stack(mom_log) if mom_log
                     else np.zeros((0, 3), dtype=np.float64)),
        "max_speed": np.array(vmax_log, dtype=np.float64),
        "contacts": np.array(contact_log, dtype=np.int64),
        "frame_dt": frame_dt,
        "wall_time": wall,
        "nan_time": nan_hit,
        # plastic state (None for elastic sheets): per-frame (total crease
        # angle [rad], yielded edges), final per-edge rest angles, crease map
        "plastic": None if hinges is None else {
            "crease_total": np.array([p[0] for p in plastic_log], dtype=np.float64),
            "edges_yielded": np.array([p[1] for p in plastic_log], dtype=np.int64),
            "edge_rest_angle": _host(model.edge_rest_angle).astype(np.float32),
            "edge_indices": _host(model.edge_indices).astype(np.int32),
            "crease_map": hinges.crease_map(nx, ny),
            "yield_angle": hinges.yield_angle,
        },
    }
