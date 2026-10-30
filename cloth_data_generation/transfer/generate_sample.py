"""Generate one deformed-cloth HDF5 sample from a force-parameter JSON or defaults.

Also importable: ``build_parser`` / ``configure`` / ``generate`` let
``generate_batch.py`` run many samples in one process, reusing a
``SimSession`` (solver + CUDA graph) across samples of the same configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import warp as wp

from cloth_sim.forces import ForceParams, default_forces, forces_from_dicts, forces_to_dicts
from cloth_sim.io import save_sample_h5
from cloth_sim.sheet import MATERIALS, SheetConfig, build_sheet, initial_positions, sheet_config
from cloth_sim.simulate import SimConfig, SimSession, run_simulation


def gpu_mem_used_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return float(out.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate one cloth sample.")
    ap.add_argument("--force-json", default=None, help="JSON list of force dicts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sample_seed0.h5")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--t-capture", type=float, default=None,
                    help="default: max force duration + --settle, i.e. capture at the "
                         "instant the constant force is released")
    ap.add_argument("--settle", type=float, default=0.0,
                    help="extra free-flight time after the forces end when --t-capture "
                         "unset (0 = capture at release; legacy datasets used 0.7)")
    ap.add_argument("--magnitude", type=float, default=None,
                    help="override both force magnitudes (tuning aid)")
    ap.add_argument("--material", choices=sorted(MATERIALS), default="cloth",
                    help="sheet preset (cloth_sim/sheet.py MATERIALS); flags below override it")
    ap.add_argument("--yield-curvature", type=float, default=None,
                    help="plastic crease yield curvature [1/m] (paper; 0 = elastic)")
    ap.add_argument("--plastic-rate", type=float, default=None)
    ap.add_argument("--crease-soften", type=float, default=None,
                    help="yielded hinges keep this fraction of edge_ke (paper; 1 = none)")
    ap.add_argument("--edge-ke", type=float, default=None)
    ap.add_argument("--tri-ke", type=float, default=None, help="membrane stiffness (also sets tri_ka)")
    ap.add_argument("--mass-per-area", type=float, default=None, help="kg/m^2 (cloth 0.2, paper 0.08)")
    ap.add_argument("--tri-kd", type=float, default=None)
    ap.add_argument("--table", action="store_true",
                    help="bench scenario: gravity and a ground plane under the sheet")
    ap.add_argument("--gravity", type=float, default=None, help="m/s^2 with --table (default 9.81)")
    ap.add_argument("--friction", type=float, default=None, help="table friction coefficient (default 0.5)")
    ap.add_argument("--contact-ke", type=float, default=None,
                    help="table contact stiffness (Newton soft_contact_ke, default 1e3); sweep aid")
    ap.add_argument("--z-jitter", type=float, default=None,
                    help="initial height noise as a fraction of the cell with --table (default 0.1)")
    ap.add_argument("--substeps", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--radius", type=float, default=None,
                    help="particle (self-contact) radius [m]; default 2 mm = 50%% of the 128-grid "
                         "spacing. Scale with the cell at other resolutions (generate_batch does)")
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument("--ramp", type=float, default=0.0,
                    help="force envelope: 0 = hard-edged constant force (default); "
                         ">0 = legacy smoothstep on/off over this fraction of duration")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--force-every-substep", dest="force_every_substep",
                    action="store_true", default=True,
                    help="apply the external force on every substep (default; "
                         "nominal impulse)")
    ap.add_argument("--force-once-per-frame", dest="force_every_substep",
                    action="store_false",
                    help="legacy: write the force once per frame, which delivers "
                         "~53%% of the nominal impulse (datasets up to ds768_night)")
    ap.add_argument("--snapshots", type=int, default=5,
                    help="state snapshots at k/N of t_capture (default 5 = every fifth of the deformation)")
    ap.add_argument("--no-graph", action="store_true",
                    help="disable CUDA graph replay of the substep loop (debug / timing reference)")
    ap.add_argument("--device", default="cuda:0")
    return ap


@dataclass
class SampleSpec:
    forces: list[ForceParams]
    sheet: SheetConfig
    sim: SimConfig
    t_capture: float
    seed: int
    out: str

    def session_key(self) -> tuple:
        """Everything a SimSession depends on (the seed only moves the table jitter)."""
        return (json.dumps({k: v for k, v in self.sheet.__dict__.items() if k != "jitter_seed"},
                           sort_keys=True),
                json.dumps({k: v for k, v in self.sim.__dict__.items()
                            if k not in ("t_capture", "capture_every", "n_snapshots")},
                           sort_keys=True))


def configure(args: argparse.Namespace) -> SampleSpec:
    """Resolve CLI args into forces, sheet and sim configs (no GPU work)."""
    if args.force_json:
        with open(args.force_json) as f:
            forces = forces_from_dicts(json.load(f))
    else:
        forces = default_forces()
    if args.magnitude is not None:
        for f in forces:
            f.magnitude = args.magnitude

    t_capture = args.t_capture
    if t_capture is None:
        t_capture = max(f.duration for f in forces) + args.settle
    sheet = sheet_config(args.material, nx=args.nx, ny=args.ny)
    sim = SimConfig(t_capture=t_capture, device=args.device,
                    ramp_frac=args.ramp, deterministic=args.deterministic,
                    force_every_substep=args.force_every_substep,
                    n_snapshots=args.snapshots, use_graph=not args.no_graph)
    if args.table:
        sheet.table = True
    sheet.jitter_seed = args.seed
    if args.gravity is not None:
        sheet.gravity = args.gravity
    if args.friction is not None:
        sheet.table_friction = args.friction
    if args.contact_ke is not None:
        sheet.contact_ke = args.contact_ke
    if args.z_jitter is not None:
        sheet.z_jitter = args.z_jitter
    if args.edge_ke is not None:
        sheet.edge_ke = args.edge_ke
    if args.tri_kd is not None:
        sheet.tri_kd = args.tri_kd
    if args.tri_ke is not None:
        sheet.tri_ke = args.tri_ke
        sheet.tri_ka = args.tri_ke
    if args.mass_per_area is not None:
        sheet.mass_per_area = args.mass_per_area
    if args.yield_curvature is not None:
        sheet.plastic_yield_curvature = args.yield_curvature or None
    if args.plastic_rate is not None:
        sheet.plastic_rate = args.plastic_rate
    if args.crease_soften is not None:
        sheet.plastic_soften = args.crease_soften
    if args.radius is not None:
        sheet.particle_radius = args.radius
    if args.margin is not None:
        sheet.self_contact_margin = args.margin
        sim.margin = args.margin
        sheet.particle_radius = max(sheet.particle_radius, args.margin)
    if args.gap is not None:
        sheet.self_contact_gap = args.gap
        sim.gap = args.gap
    if args.substeps is not None:
        sim.substeps = args.substeps
    if args.iterations is not None:
        sim.iterations = args.iterations
    return SampleSpec(forces=forces, sheet=sheet, sim=sim, t_capture=t_capture,
                      seed=args.seed, out=args.out)


class SessionCache:
    """One SimSession per configuration, kept for the life of the process."""

    def __init__(self) -> None:
        self.key = None
        self.session: SimSession | None = None
        self.meta: dict | None = None

    def get(self, spec: SampleSpec) -> tuple[SimSession, dict]:
        key = spec.session_key()
        if self.session is None or key != self.key:
            self.session, self.meta = None, None  # free the old one first
            model, meta = build_sheet(spec.sheet, device=spec.sim.device)
            self.session = SimSession(model, meta, spec.sim)
            self.meta, self.key = meta, key
        # per-sample pieces of the config
        self.session.sim.t_capture = spec.sim.t_capture
        self.session.sim.n_snapshots = spec.sim.n_snapshots
        self.session.sim.capture_every = spec.sim.capture_every
        return self.session, self.meta


def generate(spec: SampleSpec, cache: SessionCache | None = None,
             verbose: bool = True) -> dict:
    """Simulate one sample and write ``spec.out``; returns the run summary dict."""
    mem_before = gpu_mem_used_mb()
    if cache is None:
        model, meta = build_sheet(spec.sheet, device=spec.sim.device)
        session = SimSession(model, meta, spec.sim)
    else:
        session, meta = cache.get(spec)
    # table jitter is seeded per sample; everything else resets to the build state
    q0 = initial_positions(spec.sheet, meta["q_flat"], spec.seed)
    res = run_simulation(session.model, spec.forces, session.sim, meta,
                         session=session, q0=q0)
    mem_after = gpu_mem_used_mb()

    sim_cfg = {
        "sheet": spec.sheet.__dict__,
        "sim": session.sim.__dict__,
        "nx": spec.sheet.nx, "ny": spec.sheet.ny,
        "t_capture": spec.t_capture,
        "wall_time_s": res["wall_time"],
        "gpu_mem_before_mb": mem_before,
        "gpu_mem_after_mb": mem_after,
        "nan_time": res["nan_time"],
        "cuda_graph": res["graph"],
        "session_reused": cache is not None,
    }
    save_sample_h5(
        spec.out,
        final_grid=res["final_grid"], final_vel=res["final_vel"],
        intermediate_grids=res["intermediate_grids"],
        intermediate_times=res["intermediate_times"],
        intermediate_vels=res["intermediate_vels"],
        force_params=forces_to_dicts(spec.forces), sim_config=sim_cfg,
        seed=spec.seed, ke=res["ke"], max_speed=res["max_speed"],
        contacts=res["contacts"], frame_dt=res["frame_dt"],
        momentum=res["momentum"], plastic=res["plastic"],
    )
    if verbose:
        print(f"saved {spec.out}")
        print(f"wall {res['wall_time']:.1f}s graph={res['graph']} frames={len(res['ke'])} "
              f"snapshots={[round(float(t), 4) for t in res['intermediate_times']]} "
              f"ke_final={res['ke'][-1]:.3e} ke_max={res['ke'].max():.3e} "
              f"vmax_max={res['max_speed'].max():.2f} "
              f"contacts_max={res['contacts'].max()} contacts_final={res['contacts'][-1]} "
              f"nan={res['nan_time']} gpu_mem={mem_before}->{mem_after} MB")
        if res["plastic"] is not None:
            pl = res["plastic"]
            print(f"plastic: crease_total={pl['crease_total'][-1]:.3f} rad "
                  f"edges_yielded={pl['edges_yielded'][-1]}/{len(pl['edge_rest_angle'])} "
                  f"yield_angle={pl['yield_angle']:.4f} rad")
    return {"wall_time": res["wall_time"], "nan_time": res["nan_time"],
            "graph": res["graph"], "frames": len(res["ke"])}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    wp.init()
    generate(configure(args))


if __name__ == "__main__":
    main()
