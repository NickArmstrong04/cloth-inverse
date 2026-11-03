"""T0 (hackathon tooling plan): hosted SAM 3.1 vs the oracle mask, and one Muse Spark image call.

Scores every `<stem>_v<k>.png` that has a `_gt.npz` (written by `photo.py --gt`) with the
same metric as `segment.py`: the highest-coverage instance and the union of all instances,
IoU / precision / recall against the oracle mask. Synthetic renders only. Key: META_API_KEY.

  python t0_api_check.py --photos ../../cloth_results/photos/free256 --prompt fabric \\
      --out ../../cloth_results/photos/sam31_api
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import time
import urllib.request

import numpy as np

import sam_api


def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    inter = float((pred & gt).sum())
    return {"iou": inter / max(float((pred | gt).sum()), 1.0),
            "precision": inter / max(float(pred.sum()), 1.0),
            "recall": inter / max(float(gt.sum()), 1.0)}


def spark(png: str, question: str, model: str) -> tuple[str, float]:
    with open(png, "rb") as f:
        url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    body = {"model": model, "stream": False, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": question}, {"type": "input_image", "image_url": url}]}]}
    req = urllib.request.Request(sam_api.URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {os.environ['META_API_KEY']}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    text = "".join(c.get("text", "") for o in resp.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []))
    return text, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--prompt", default="fabric")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spark-model", default="muse-spark-1.3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pngs = sorted(p for d in a.photos for p in glob.glob(os.path.join(d, "*_v[0-9].png"))
                  if os.path.exists(p[:-4] + "_gt.npz"))
    if a.limit:
        pngs = pngs[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for p in pngs:
        gt = np.load(p[:-4] + "_gt.npz")["mask"]
        try:
            insts, dt = sam_api.segment(p, a.prompt)
        except Exception as e:                       # keep going; a failed call is a result
            rows.append({"png": os.path.basename(p), "error": repr(e)})
            print(f"{os.path.basename(p)}: ERROR {e!r}")
            continue
        row = {"png": os.path.basename(p), "k": len(insts), "api_s": round(dt, 3)}
        if insts:
            best = max(insts, key=lambda i: int((i["mask"] & gt).sum()))
            row["best"] = score(best["mask"], gt)
            row["union"] = score(np.logical_or.reduce([i["mask"] for i in insts]), gt)
        rows.append(row)
        print(f"{row['png']}: K={row['k']} {dt:.2f}s IoU {row.get('best', {}).get('iou', 0):.4f}")

    ok = [r for r in rows if r.get("k")]
    ious = np.array([r["best"]["iou"] for r in ok]) if ok else np.zeros(0)
    lat = np.array([r["api_s"] for r in rows if "api_s" in r])
    summary = {"prompt": a.prompt, "n": len(rows), "found": len(ok),
               "errors": sum("error" in r for r in rows),
               "iou_median": float(np.median(ious)) if ious.size else None,
               "iou_min": float(ious.min()) if ious.size else None,
               "iou_ge_0.95": int((ious >= 0.95).sum()),
               "api_s_median": float(np.median(lat)) if lat.size else None,
               "api_s_max": float(lat.max()) if lat.size else None}

    q = ("This is a rendered photo. In one sentence: what object is in the centre, and is it "
         "flat, folded, or crumpled?")
    try:
        text, dt = spark(pngs[0], q, a.spark_model)
        summary["spark"] = {"model": a.spark_model, "png": os.path.basename(pngs[0]), "question": q,
                            "answer": text, "s": round(dt, 2)}
    except Exception as e:
        summary["spark"] = {"model": a.spark_model, "error": repr(e)}
    print(json.dumps(summary, indent=1))
    with open(os.path.join(a.out, "scores.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
