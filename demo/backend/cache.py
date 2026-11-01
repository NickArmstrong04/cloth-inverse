"""Precomputed-result cache for the live demo (step 6).

If a GPU job fails or exceeds 90 s, the service returns the nearest cached
result with ``cached: true`` so the demo never ends on an empty screen.
Nearest = smallest summed (u, v) distance over the better force pairing.

``make_cache.py`` builds the entries; each ``*.json`` holds the exact job
result payload plus the ``true_forces`` it was simulated with. On fallback
the service re-scores the cached prediction against the USER's forces, so
every metric stays truthful, and marks ``cached: true``.
"""

from __future__ import annotations

import glob
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "cache")


def _pair_cost(a: list[dict], b: list[dict]) -> float:
    def d(f, g):
        return ((f["u"] - g["u"]) ** 2 + (f["v"] - g["v"]) ** 2) ** 0.5
    return min(d(a[0], b[0]) + d(a[1], b[1]),
               d(a[0], b[1]) + d(a[1], b[0]))


def load_cache(cache_dir: str = CACHE_DIR) -> list[dict]:
    entries = []
    for p in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        with open(p) as fh:
            entries.append(json.load(fh))
    return entries


def nearest(entries: list[dict], forces: list[dict]) -> dict | None:
    """Entry whose true forces are closest in (u, v); None when empty."""
    if not entries:
        return None
    return min(entries, key=lambda e: _pair_cost(e["true_forces"], forces))
