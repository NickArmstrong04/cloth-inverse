"""Unit tests for the piercing metric (synthetic ground truth)."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # run from transfer/: python tools/<name>.py
import numpy as np

from cloth_sim.validate import _triangles_pierce, count_self_intersections

# crossing: vertical triangle pierced by a horizontal triangle edge
t1 = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
t2 = np.array([[0.2, 0.2, -2.0], [0.2, 0.2, 0.0], [0.9, 0.1, -0.5]])
assert _triangles_pierce(t1, t2), "must detect real crossing"

# touching at a point only
t3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
t4 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]])
assert not _triangles_pierce(t3, t4), "shared-vertex touch must not count"

# parallel separated
t5 = t3 + np.array([0.0, 0.0, 0.5])
assert not _triangles_pierce(t3, t5), "parallel separated must not count"

# coplanar overlapping (folded-flat contact analogue)
t6 = np.array([[0.1, 0.1, 1.0], [0.9, 0.1, 1.0], [0.1, 0.9, 1.0]])
assert not _triangles_pierce(t3, t6), "coplanar overlap must not count"

# edge lying in plane (crease pressing into layer)
t7 = np.array([[-1.0, 0.5, 1.0], [2.0, 0.5, 1.0], [0.5, 0.5, 2.0]])
assert not _triangles_pierce(t3, t7), "in-plane edge must not count"

print("pierce unit tests passed")

# grid-level: two stacked sheets 1mm apart -> 0; pushed through -> >0
n = 8
base = np.zeros((n, n, 3), dtype=np.float32)
ii, jj = np.meshgrid(np.arange(n), np.arange(n))
base[..., 0] = ii * 0.01
base[..., 1] = jj * 0.01
stacked = base.copy()
stacked[4:, :, 2] = 0.001  # upper half lifted 1mm (resting-contact analogue)
r = count_self_intersections(np.concatenate([base, stacked], axis=0).reshape(2 * n, n, 3))
print("stepped sheet (expect 0):", r)
