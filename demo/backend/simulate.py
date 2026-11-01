"""Demo simulation wrapper: user forces -> simulated deformation, in-process.

Wraps ``cloth_data_generation/transfer/generate_sample.py`` exactly
(``build_parser`` / ``configure`` / ``SessionCache`` + ``run_simulation``)
so the demo reproduces the ``ds256_const`` dataset physics at any ``nx``:

* ``res_scale = nx / 128``; ``sigma_cells = 3.0 * res_scale``,
  ``radius_cells = 9.0 * res_scale``; radius / margin / gap divided by
  ``res_scale`` (cf. ``generate_batch.py`` lines 289-293)
* ``--snapshots 30``, ``--settle 0``, constant force every substep
  (default), zero gravity (no ``--table``), ``--device cuda:0``
* ``--t-capture`` = longer of the two durations (the default)

A long-lived :class:`SimService` keeps the Warp session warm across calls;
only ``t_capture`` / ``n_snapshots`` (PER_SAMPLE fields) change per run.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TRANSFER = os.path.join(_REPO, "cloth_data_generation", "transfer")

sys.path.insert(0, _TRANSFER)

# Training ranges (ds256_const) enforced on user input.
U_RANGE = (0.05, 0.95)
V_RANGE = (0.05, 0.95)
MAG_RANGE = (0.15, 1.25)
DUR_RANGE = (0.3, 0.7)
MIN_SEP = 0.2


def validate_forces(forces: list[dict]) -> list[dict]:
    """Check the demo force schema and training ranges; normalise directions."""
    if len(forces) != 2:
        raise ValueError(f"demo needs exactly 2 forces, got {len(forces)}")
    out = []
    for i, f in enumerate(forces):
        for k in ("u", "v", "direction", "magnitude", "duration"):
            if k not in f:
                raise ValueError(f"force {i}: missing {k!r}")
        u, v = float(f["u"]), float(f["v"])
        if not (U_RANGE[0] <= u <= U_RANGE[1]):
            raise ValueError(f"force {i}: u={u} outside {U_RANGE}")
        if not (V_RANGE[0] <= v <= V_RANGE[1]):
            raise ValueError(f"force {i}: v={v} outside {V_RANGE}")
        d = np.asarray(f["direction"], dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if not np.isfinite(n) or n < 1e-8:
            raise ValueError(f"force {i}: direction must be nonzero")
        mag = float(f["magnitude"])
        if not (MAG_RANGE[0] <= mag <= MAG_RANGE[1]):
            raise ValueError(f"force {i}: magnitude={mag} outside {MAG_RANGE}")
        dur = float(f["duration"])
        if not (DUR_RANGE[0] <= dur <= DUR_RANGE[1]):
            raise ValueError(f"force {i}: duration={dur} outside {DUR_RANGE}")
        out.append({"u": u, "v": v, "direction": [float(x) for x in d / n],
                    "magnitude": mag, "duration": dur})
    du = ((out[0]["u"] - out[1]["u"]) ** 2 + (out[0]["v"] - out[1]["v"]) ** 2) ** 0.5
    if du < MIN_SEP:
        raise ValueError(f"patch separation {du:.3f} < {MIN_SEP}")
    return out


class SimService:
    """Warm simulator: one CUDA session per grid, reused across forces."""

    def __init__(self, device: str = "cuda:0"):
        from generate_sample import SessionCache  # noqa: E402
        import warp as wp  # noqa: E402
        wp.init()
        self._caches: dict[int, SessionCache] = {}
        self._SessionCache = SessionCache
        self.device = device

    def _args(self, forces: list[dict], nx: int, snapshots: int, seed: int):
        import generate_sample  # noqa: E402
        res_scale = nx / 128.0
        full = [dict(f, sigma_cells=3.0 * res_scale, radius_cells=9.0 * res_scale)
                for f in forces]
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(full, fh)
        argv = ["--force-json", path, "--nx", str(nx), "--ny", str(nx),
                "--snapshots", str(snapshots), "--settle", "0",
                "--device", self.device, "--seed", str(seed)]
        if nx != 128:
            argv += ["--radius", f"{2.0e-3 / res_scale:.6g}",
                     "--margin", f"{1.0e-3 / res_scale:.6g}",
                     "--gap", f"{0.5e-3 / res_scale:.6g}"]
        args = generate_sample.build_parser().parse_args(argv)
        return args, path, full

    def run(self, forces: list[dict], nx: int = 128, snapshots: int = 30,
            seed: int = 0) -> dict:
        """Simulate; returns frames (K,ny,nx,3), final pos/vel, timings."""
        import generate_sample  # noqa: E402
        from cloth_sim.sheet import initial_positions  # noqa: E402
        from cloth_sim.simulate import run_simulation  # noqa: E402
        forces = validate_forces(forces)
        args, path, full = self._args(forces, nx, snapshots, seed)
        try:
            spec = generate_sample.configure(args)
        finally:
            os.remove(path)
        cache = self._caches.get(nx)
        if cache is None:
            cache = self._caches[nx] = self._SessionCache()
        session, meta = cache.get(spec)
        q0 = initial_positions(spec.sheet, meta["q_flat"], spec.seed)
        res = run_simulation(session.model, spec.forces, session.sim, meta,
                             session=session, q0=q0)
        if res["nan_time"] is not None:
            raise RuntimeError(f"simulation went NaN at t={res['nan_time']}")
        frames = np.asarray(res["intermediate_grids"], dtype=np.float32)
        return {"frames": frames,
                "frame_times": np.asarray(res["intermediate_times"], dtype=np.float64),
                "final_pos": np.asarray(res["final_grid"], dtype=np.float32),
                "final_vel": np.asarray(res["final_vel"], dtype=np.float32),
                "wall_s": float(res["wall_time"]),
                "t_capture": spec.t_capture,
                "forces": full,
                "nx": nx}


def downsample_frames(frames: np.ndarray, size: int = 64) -> np.ndarray:
    """Block-average (K,ny,nx,3) indexed grids to (K,size,size,3)."""
    k, ny, nx, _ = frames.shape
    assert ny == nx and ny % size == 0, (frames.shape, size)
    f = ny // size
    return frames.reshape(k, size, f, size, f, 3).mean(axis=(2, 4)).astype(np.float32)


def encode_frames(frames64: np.ndarray) -> str:
    """(30,64,64,3) float32 -> float16 base64 string."""
    buf = io.BytesIO()
    np.save(buf, frames64.astype(np.float16), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


def decode_frames(s: str) -> np.ndarray:
    buf = io.BytesIO(base64.b64decode(s.encode()))
    return np.load(buf).astype(np.float32)
