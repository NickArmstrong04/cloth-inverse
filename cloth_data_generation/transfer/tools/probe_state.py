"""Probe Newton 1.7.0 state/contact APIs on CUDA (tiny 4x3-cell cloth)."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # run from transfer/: python tools/<name>.py
import numpy as np
import warp as wp

import newton

wp.init()

builder = newton.ModelBuilder()
builder.gravity = [0.0, 0.0, 0.0]
builder.add_cloth_grid(
    pos=wp.vec3(0.0, 0.0, 0.0),
    rot=wp.quat_identity(),
    vel=wp.vec3(0.0, 0.0, 0.0),
    dim_x=3,
    dim_y=2,
    cell_x=0.1,
    cell_y=0.1,
    mass=0.01,
    tri_ke=1.0e3,
    tri_ka=1.0e3,
    tri_kd=1.0e2,
    edge_ke=10.0,
    edge_kd=0.0,
    particle_radius=0.02,
)
print("builder particles:", len(builder.particle_q))
builder.color(include_bending=True)
model = builder.finalize()
print("model gravity:", model.gravity)
print("model particle count:", model.particle_count)

s0 = model.state()
s1 = model.state()
ctrl = model.control()
print("state particle attrs:", [a for a in dir(s0) if "particle" in a])

# readback pattern test
try:
    q = s0.particle_q.numpy()
    print("direct .numpy() OK, shape:", q.shape, "dtype:", q.dtype)
except Exception as e:
    print("direct .numpy() failed:", type(e).__name__, str(e)[:200])

host = wp.empty(s0.particle_q.shape, dtype=s0.particle_q.dtype, device="cpu")
wp.copy(host, s0.particle_q)
qn = host.numpy()
print("copy-to-host OK, shape:", qn.shape)
print("q[0]=", qn[0], "q[1]=", qn[1], "q[4]=", qn[4])
print("expected: q[1]-q[0] ~ [0.1,0,0], q[4]-q[0] ~ [0,0.1,0]")

mp = wp.empty(model.particle_mass.shape, dtype=model.particle_mass.dtype, device="cpu")
wp.copy(mp, model.particle_mass)
print("masses:", mp.numpy().ravel()[:6])

solver = newton.solvers.SolverVBD(
    model=model,
    iterations=2,
    particle_enable_self_contact=True,
    particle_self_contact_margin=0.02,
    particle_self_contact_gap=0.01,
)
cp = newton.CollisionPipeline(model)
contacts = cp.contacts()
print("has soft_contact_count:", hasattr(contacts, "soft_contact_count"))
print("has rigid_contact_count:", hasattr(contacts, "rigid_contact_count"))

# one collide + step, then read counts
s0.clear_forces()
cp.collide(s0, contacts)
solver.step(s0, s1, ctrl, contacts, 1.0 / 600.0)
for name in ("rigid_contact_count", "soft_contact_count"):
    arr = getattr(contacts, name)
    h = wp.empty(arr.shape, dtype=arr.dtype, device="cpu")
    wp.copy(h, arr)
    print(name, "=", h.numpy().ravel())

# force-write pattern test
f = np.zeros((model.particle_count, 3), dtype=np.float32)
f[5] = [0.0, 0.0, 1.0]
fdev = wp.array(f.reshape(-1), dtype=wp.float32, device="cuda:0")
fdev3 = wp.array(f, dtype=wp.vec3, device="cuda:0")
wp.copy(s0.particle_f, fdev3)
fh = wp.empty(s0.particle_f.shape, dtype=s0.particle_f.dtype, device="cpu")
wp.copy(fh, s0.particle_f)
print("force write OK, f[5]=", fh.numpy()[5])
print("PROBE_ALL_DONE")
