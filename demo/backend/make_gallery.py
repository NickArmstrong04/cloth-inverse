"""Camera-path gallery for the frontend, from a finished chain run (no new compute but M21-c).

Takes the hosted SAM 3.1 run of the 100 rest-paper test sheets (RESULTS.md M21: photo.py coded stereo
pairs -> sam_api.py -> t1_chain.sbatch -> grid_from_views --corr decoded --scale-fix 0.15) and writes,
per sheet, one JSON with the photo, the SAM mask over it, the camera's grid, the true mesh and M21-c's
predictions from both (camera.package). Sheets are the first N in name order whose base view 0 has a
camera grid: no selection on the result. index.json holds the list and the gallery means.

  python make_gallery.py --photos $DATA/photos/rest_test100_sam --run runs/M21c --n 20 \\
      --out ../frontend/public/gallery
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import camera


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--grids", default="grids_dec_fix", help="grid_from_views output dir inside --photos")
    ap.add_argument("--run", default=camera.CAMERA_CKPT)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = camera.Model(a.run)
    G = os.path.join(a.photos, a.grids)
    recs = {r["stem"]: r for r in json.load(open(os.path.join(G, "grids.json")))["records"]}
    man = {os.path.splitext(v["png"])[0]: v for v in json.load(open(os.path.join(a.photos, "manifest.json")))["views"]}
    files = sorted(glob.glob(os.path.join(G, "*_v0_grids.npz")))[: a.n]
    index = []
    for i, f in enumerate(files):
        stem = os.path.basename(f)[: -len("_grids.npz")]
        e = camera.package(model, f, os.path.join(a.photos, f"{stem}.png"),
                           os.path.join(a.photos, "sam3", f"{stem}__fabric.png"), man[stem]["force_params"][:2], recs.get(stem))
        e["id"] = f"g{i:02d}"
        e["live"] = False
        json.dump(e, open(os.path.join(a.out, f"{e['id']}.json"), "w"))
        cm, mm = e["camera"]["metrics"]["mean"], e["mesh"]["metrics"]["mean"]
        index.append({"id": e["id"], "stem": stem, "camera": cm, "mesh": mm, "coverage": e["coverage"]})
        print(f"{e['id']} {stem}: camera {cm['pos_cm']:.1f} cm / {cm['dir_deg']:.0f} deg | mesh {mm['pos_cm']:.1f} / {mm['dir_deg']:.0f}"
              f" | decoded {e['coverage']['decoded'] or 0:.2f}", flush=True)
    summ = {k: {m: float(np.mean([r[k][m] for r in index])) for m in camera.NAMES_SHOWN} for k in ("camera", "mesh")}
    for k in ("camera", "mesh"):
        summ[k]["pos_cm_median"] = float(np.median([r[k]["pos_cm"] for r in index]))
    json.dump({"entries": index, "summary": summ, "reference": camera.REFERENCE, "n": len(index),
               "source": "first sheets in name order of the 100-sheet hosted SAM 3.1 test run (RESULTS.md M21)"},
              open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print("gallery means:", json.dumps(summ))


if __name__ == "__main__":
    main()
