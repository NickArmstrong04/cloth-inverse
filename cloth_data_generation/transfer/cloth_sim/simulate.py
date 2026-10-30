"""Time integration of the cloth sheet with VBD + self-contact on CUDA.

Model (2026-10-29): a constant external force is held on each patch for its
``duration`` and the sample is captured at release, so the deformation is
recorded before the stored energy relaxes. State snapshots (positions and
velocities) are taken at k/n_snapshots of t_capture.

Performance: one frame of substeps is recorded as a CUDA graph and replayed
(the Python launch overhead was ~80% of the wall time), and a ``SimSession``
(model + solver + state buffers + graph) can be reused across samples in one
process, which removes the per-sample solver setup (~4 s at 128x128).
"""

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
    # Snapshots of the state (positions + velocities) at k/n_snapshots of
    # t_capture, k = 1..n_snapshots; the last one is the capture itself. With
    # the constant-force model (settle 0) t_capture is the deformation window,
    # so 5 = every fifth of the way through the deformation.
    n_snapshots: int = 5
    margin: float | None = None  # self-contact interaction distance (m)
    gap: float | None = None  # self-contact extra detection reach (m)
    buffer_vt: int = 64  # per-vertex vertex-triangle contact capacity
    buffer_ee: int = 128  # per-edge edge-edge contact capacity
    # Force envelope. 0 = hard-edged constant force over [0, duration) (current
    # model: hold a constant force, capture at release so the deformation is
    # caught before it relaxes). >0 = legacy smoothstep on/off over this
    # fraction of the duration (datasets up to ds768_night used 0.15).
    ramp_frac: float = 0.0
    deterministic: bool = False  # RUN_TO_RUN bit-exact repeats (slower)
    capture_every: int = 0  # store every Nth frame densely (0 = snapshots only)
    # Re-apply the external force to the input state on EVERY substep (True =
    # the force really is constant in time, nominal impulse). The legacy
    # setting (False) wrote it to state_0 once per frame; since the two state
    # buffers swap each substep, only half the substeps saw the force (~53% of
    # the nominal impulse, measured; datasets up to ds768_night).
    force_every_substep: bool = True
    # Record one frame (all substeps) as a CUDA graph and replay it per frame.
    # Falls back to eager launches if capture fails; off in deterministic mode
    # (RUN_TO_RUN allocates sort scratch per launch, ~1.6 GB per captured
    # substep). Results are kernel-for-kernel identical to eager launches.
    use_graph: bool = True
    device: str = "cuda:0"


def _host(arr) -> np.ndarray:
    return np.asarray(arr.numpy())


class SimSession:
    """Solver, state buffers, force buffer and CUDA graph for one model.

    Build once per (sheet, sim) configuration and call :meth:`reset` before
    each sample; everything the solver reads is restored, so a reused session
    reproduces a fresh one bit for bit (checked in deterministic mode).
    """

    def __init__(self, model, meta: dict, sim: SimConfig):
        self.model, self.meta, self.sim = model, meta, sim
        with wp.ScopedDevice(sim.device):
            self.state_0 = model.state()
            self.state_1 = model.state()
            self.control = model.control()
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
            # Newton: construct CollisionPipeline -> Contacts -> SolverVBD before capture
            self.pipeline = newton.CollisionPipeline(model)
            self.contacts = self.pipeline.contacts()
            self.solver = newton.solvers.SolverVBD(
                model=model, iterations=sim.iterations, **sc_kwargs)
            self.masses = _host(model.particle_mass).ravel().astype(np.float64)
            sheet = meta["sheet"]
            self.hinges = None
            if sheet.get("plastic_yield_curvature"):
                self.hinges = PlasticHinges(model, sheet["plastic_yield_curvature"],
                                            min(meta["cell_x"], meta["cell_y"]),
                                            rate=sheet.get("plastic_rate", 1.0),
                                            soften=sheet.get("plastic_soften", 1.0))
            # positions the model was built with (reset default)
            self.q0 = _host(model.particle_q).copy()
            # External force lives in a device buffer the (captured) frame copies
            # from; the host only refreshes it when the envelope changes.
            self.f_buf = wp.zeros(model.particle_count, dtype=wp.vec3, device=sim.device)
            # Free-floating sheet: no shapes, so the particle-shape collision pass
            # has nothing to do (self-contact is handled inside VBD).
            self.do_collide = model.shape_count > 0
            self.sim_dt = (1.0 / sim.fps) / sim.substeps
            self.bufs = (self.state_0, self.state_1)
            self.cur = 0
            self.graphs: dict[int, object] = {}
            if sim.use_graph and sim.deterministic:
                print("note: CUDA graph disabled in deterministic mode")
            elif sim.use_graph and wp.get_device(sim.device).is_cuda:
                try:
                    # one graph per starting buffer; with an even substep count
                    # the result lands back in the starting buffer: one graph
                    for key in (0, 1):
                        if key == 1 and sim.substeps % 2 == 0:
                            break
                        with wp.ScopedCapture(device=sim.device) as cap:
                            self._substeps_from(self.bufs[key], self.bufs[key ^ 1])
                        self.graphs[key] = cap.graph
                except Exception as exc:  # depends on driver / newton version
                    print(f"warning: CUDA graph capture failed ({exc!r}); running eagerly")
                    self.graphs = {}
        self.reset()

    # -- per-frame stepping -------------------------------------------------
    def _substeps_from(self, s_in, s_out):
        """One frame of substeps starting in s_in; returns (result, scratch)."""
        sim = self.sim
        for k_sub in range(sim.substeps):
            if k_sub == 0 or sim.force_every_substep:
                wp.copy(s_in.particle_f, self.f_buf)
            if self.do_collide:
                self.pipeline.collide(s_in, self.contacts)
            self.solver.step(s_in, s_out, self.control, self.contacts, self.sim_dt)
            s_in, s_out = s_out, s_in
            if self.hinges is not None:
                self.hinges.update(s_in)
        return s_in, s_out

    def step_frame(self):
        """Advance one frame; returns the state holding the result."""
        if self.graphs:
            wp.capture_launch(self.graphs[self.cur])
            if self.sim.substeps % 2 == 1:
                self.cur ^= 1
        else:
            s_res, _ = self._substeps_from(self.bufs[self.cur], self.bufs[self.cur ^ 1])
            self.cur = 0 if s_res is self.state_0 else 1
        return self.bufs[self.cur]

    def reset(self, q0: np.ndarray | None = None) -> None:
        """Restore the rest state: positions (``q0`` or the build positions),
        zero velocities/forces, and the unyielded plastic hinges."""
        q = self.q0 if q0 is None else np.asarray(q0, dtype=np.float32).reshape(-1, 3)
        for s in self.bufs:
            s.particle_q.assign(q)
            s.particle_qd.zero_()
            s.particle_f.zero_()
        self.f_buf.zero_()
        if self.hinges is not None:
            self.hinges.reset()
        self.cur = 0

    def self_contact_count(self) -> int:
        # VBD detects self-contact internally; read its detector counts.
        # Falls back to the pipeline soft-contact counter (particle-shape).
        try:
            info = self.solver.trimesh_collision_detector.collision_info
            vt = _host(info.vertex_colliding_triangles_count).ravel()
            ee = _host(info.edge_colliding_edges_count).ravel()
            cap_vt = _host(info.vertex_collision_buffer_sizes).ravel() \
                if hasattr(info, "vertex_collision_buffer_sizes") else None
            n_vt = (np.minimum(vt, cap_vt).sum() if cap_vt is not None
                    else vt.sum())
            return int(n_vt + ee.sum())
        except Exception:
            return int(_host(self.contacts.soft_contact_count).ravel()[0])


def run_simulation(model, forces: list[ForceParams], sim: SimConfig,
                   meta: dict, session: SimSession | None = None,
                   q0: np.ndarray | None = None) -> dict:
    """Step the model to t_capture; return grids, velocities, and per-frame log.

    Force activity is evaluated per frame (frame time < duration => active).
    Snapshots (positions and velocities) are taken at k/n_snapshots of
    t_capture; the final state is the last snapshot. Pass a ``session`` built
    on the same model to skip the solver setup; it is reset (to ``q0`` if
    given) before the run.
    """
    nx, ny = meta["nx"], meta["ny"]
    frame_dt = 1.0 / sim.fps
    n_frames = int(round(sim.t_capture / frame_dt))
    if session is None:
        session = SimSession(model, meta, sim)
    elif session.model is not model:
        raise ValueError("session was built for a different model")
    session.reset(q0)
    masses, hinges = session.masses, session.hinges

    ke_log: list[float] = []
    mom_log: list[np.ndarray] = []
    vmax_log: list[float] = []
    contact_log: list[int] = []
    plastic_log: list[tuple[float, int]] = []
    intermediates: list[np.ndarray] = []
    inter_vels: list[np.ndarray] = []
    inter_times: list[float] = []
    dense: list[np.ndarray] = []
    dense_times: list[float] = []
    # snapshot frames at k/n_snapshots of the run (the last is t_capture)
    n_snap = max(int(sim.n_snapshots), 1)
    keep = {int(round(k * n_frames / n_snap)) for k in range(1, n_snap + 1)}
    keep.add(n_frames)

    with wp.ScopedDevice(sim.device):
        f_cache: np.ndarray | None = None
        t0 = time.time()
        nan_hit = None
        for frame in range(1, n_frames + 1):
            t = frame * frame_dt
            f_np = build_force_array(nx, ny, forces, (frame - 1) * frame_dt,
                                     ramp_frac=sim.ramp_frac)
            if f_cache is None or not np.array_equal(f_np, f_cache):
                session.f_buf.assign(f_np)  # once for a constant force
                f_cache = f_np
            state = session.step_frame()

            q = _host(state.particle_q).astype(np.float64)
            qd = _host(state.particle_qd).astype(np.float64)
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
                nan_hit = t
                break
            speed = np.linalg.norm(qd, axis=1)
            ke_log.append(float(0.5 * np.sum(masses * speed**2)))
            mom_log.append((masses[:, None] * qd).sum(0))
            vmax_log.append(float(speed.max()))
            contact_log.append(session.self_contact_count())
            if hinges is not None:
                plastic_log.append(hinges.summary())
            if frame in keep:
                intermediates.append(q.reshape(ny, nx, 3).astype(np.float32))
                inter_vels.append(qd.reshape(ny, nx, 3).astype(np.float32))
                inter_times.append(t)
            if sim.capture_every > 0 and frame % sim.capture_every == 0:
                dense.append(q.reshape(ny, nx, 3).astype(np.float32))
                dense_times.append(t)
        wall = time.time() - t0

        state = session.bufs[session.cur]
        final_q = _host(state.particle_q).reshape(ny, nx, 3).astype(np.float32)
        final_qd = _host(state.particle_qd).reshape(ny, nx, 3).astype(np.float32)

    return {
        "final_grid": final_q,
        "final_vel": final_qd,
        "intermediate_grids": np.stack(intermediates).astype(np.float32),
        "intermediate_vels": np.stack(inter_vels).astype(np.float32),
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
        "graph": bool(session.graphs),
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
