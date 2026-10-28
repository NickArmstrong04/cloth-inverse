"""HDF5 I/O for cloth samples."""

from __future__ import annotations

import json
import platform
import sys

import h5py
import numpy as np


def collect_versions() -> dict:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for name in ("warp", "newton", "h5py", "matplotlib"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "missing"
    try:
        import warp as wp

        versions["warp_devices"] = str(wp.get_devices()) if hasattr(wp, "get_devices") else ""
    except Exception:
        pass
    return versions


def save_sample_h5(path: str, *, final_grid: np.ndarray, final_vel: np.ndarray,
                   intermediate_grids: np.ndarray, intermediate_times: np.ndarray,
                   force_params: list[dict], sim_config: dict, seed: int,
                   ke: np.ndarray, max_speed: np.ndarray, contacts: np.ndarray,
                   frame_dt: float, momentum: np.ndarray | None = None) -> None:
    with h5py.File(path, "w") as h:
        h.create_dataset("final_grid", data=final_grid)
        h.create_dataset("final_vel", data=final_vel)
        h.create_dataset("intermediate_grids", data=intermediate_grids)
        h.create_dataset("intermediate_times", data=intermediate_times)
        h.create_dataset("force_params", data=json.dumps(force_params))
        h.create_dataset("sim_config", data=json.dumps(sim_config))
        h.create_dataset("seed", data=np.int64(seed))
        log = h.create_group("sim_log")
        log.create_dataset("kinetic_energy", data=ke)
        log.create_dataset("max_speed", data=max_speed)
        log.create_dataset("soft_contacts", data=contacts)
        if momentum is not None:
            log.create_dataset("momentum", data=momentum)
        log.attrs["frame_dt"] = frame_dt
        for k, v in collect_versions().items():
            h.attrs[f"pkg_{k}"] = str(v)[:512]
        h.attrs["grid_convention"] = (
            "axis0=v(j) rows, axis1=u(i) cols; flat particle p=j*nx+i"
        )


def load_sample_h5(path: str) -> dict:
    with h5py.File(path, "r") as h:
        out = {k: h[k][()] for k in
               ("final_grid", "final_vel", "intermediate_grids",
                "intermediate_times", "seed")}
        out["force_params"] = json.loads(h["force_params"][()].decode()
                                         if isinstance(h["force_params"][()], bytes)
                                         else h["force_params"][()])
        out["sim_config"] = json.loads(h["sim_config"][()].decode()
                                       if isinstance(h["sim_config"][()], bytes)
                                       else h["sim_config"][()])
        out["sim_log"] = {k: h["sim_log"][k][()] for k in h["sim_log"]}
        out["attrs"] = dict(h.attrs)
    return out
