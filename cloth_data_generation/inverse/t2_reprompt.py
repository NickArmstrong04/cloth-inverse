"""T2 (meta_comp_improvements.md item 2): Muse Spark as the SAM 3.1 re-prompting agent.

Per view: run hosted SAM 3.1 with a start phrase; show Muse Spark the photo, a red overlay of
the masks (or the bare photo when K = 0), K and the mask fraction; it answers JSON
{"action": "accept" | "reprompt" | "box", "prompt": str, "box": [x1, y1, x2, y2], "why": str}.
`reprompt` reruns SAM with its phrase, `box` sends its box as the prompt; at most --rounds SAM calls
after the first. Scored against the oracle mask like t0_api_check.py. The agent never sees the
oracle. Baselines: --mode fixed (start phrase, then a hard-coded fallback list) and --mode single.

  python t2_reprompt.py --photos ../../cloth_results/photos/free256 --start paper --mode agent \\
      --out ../../cloth_results/photos/sam31_api/t2_agent_paper.json
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import re
import time
import urllib.request

import numpy as np
from PIL import Image

import sam_api
from t0_api_check import score

FALLBACK = ["fabric", "a piece of cloth", "cloth"]

SYSTEM = (
    "You supervise a segmentation model (SAM 3.1) that must find the one thin sheet (cloth or paper) "
    "in a photo, so its 3D shape can be measured. You see the photo and, if SAM found anything, the "
    "same photo with SAM's masks in red. Decide: 'accept' if the red mask covers the whole sheet and "
    "nothing else; 'reprompt' with a short noun phrase SAM is more likely to match (SAM matches simple "
    "nouns better than descriptions; adjectives hurt); or 'box' with a pixel box [x1, y1, x2, y2] "
    "around the sheet if phrases keep failing. Answer with JSON only: "
    '{"action": ..., "prompt": ..., "box": ..., "why": "<10 words>"}')


def overlay(png: str, masks: list[np.ndarray]) -> bytes:
    im = np.asarray(Image.open(png).convert("RGB")).astype(np.float32)
    if masks:
        m = np.logical_or.reduce(masks)
        im[m] = 0.5 * im[m] + 0.5 * np.array([255, 0, 0])
    buf = io.BytesIO()
    Image.fromarray(im.astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _data_url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def ask_spark(png: str, insts: list[dict], prompt: str, history: list[str], model: str) -> tuple[dict, float]:
    with open(png, "rb") as f:
        raw = f.read()
    h, w = insts[0]["mask"].shape if insts else Image.open(png).size[::-1]
    frac = float(np.logical_or.reduce([i["mask"] for i in insts]).mean()) if insts else 0.0
    status = (f"Image {w}x{h} px. Phrases tried so far: {history}. Last phrase '{prompt}': SAM found "
              f"{len(insts)} instance(s), mask covers {frac:.1%} of the image"
              + (f", boxes {[i['box'] for i in insts]}." if insts else "."))
    content = [{"type": "input_text", "text": SYSTEM + "\n\n" + status},
               {"type": "input_image", "image_url": _data_url(raw)}]
    if insts:
        content.append({"type": "input_image", "image_url": _data_url(overlay(png, [i["mask"] for i in insts]))})
    body = {"model": model, "stream": False, "input": [{"type": "message", "role": "user", "content": content}]}
    req = urllib.request.Request(sam_api.URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {os.environ['META_API_KEY']}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    text = "".join(c.get("text", "") for o in resp.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []))
    m = re.search(r"\{.*\}", text, re.S)
    try:
        dec = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        dec = {}
    dec.setdefault("action", "accept")
    dec["raw"] = text[:300]
    return dec, time.time() - t0


def segment_box(png: str, box: list[int]) -> tuple[list[dict], float]:
    """SAM 3.1 with a box prompt: the box token rides in the text lane (README 'SAM API input')."""
    w, h = Image.open(png).size
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(x1, min(x2, w - 1)), max(y1, min(y2, h - 1))
    return sam_api.segment(png, f"<0f>0<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}|>")


def run_view(png: str, gt: np.ndarray, a) -> dict:
    prompt, history, steps, t_spark, t_sam = a.start, [], [], 0.0, 0.0
    insts, dt = sam_api.segment(png, prompt)
    t_sam += dt
    history.append(prompt)
    fallback = [p for p in FALLBACK if p != a.start]
    for _ in range(a.rounds):
        if a.mode == "single":
            break
        if a.mode == "fixed":
            if insts or not fallback:
                break
            prompt = fallback.pop(0)
            insts, dt = sam_api.segment(png, prompt)
            t_sam += dt
            history.append(prompt)
            steps.append({"action": "reprompt", "prompt": prompt, "k": len(insts)})
            continue
        dec, dt = ask_spark(png, insts, prompt, history, a.spark_model)
        t_spark += dt
        step = {k: dec.get(k) for k in ("action", "prompt", "box", "why")}
        if dec["action"] == "reprompt" and dec.get("prompt"):
            prompt = str(dec["prompt"])
            insts, dt = sam_api.segment(png, prompt)
            history.append(prompt)
        elif dec["action"] == "box" and isinstance(dec.get("box"), list) and len(dec["box"]) == 4:
            insts, dt = segment_box(png, dec["box"])
            history.append(f"box {dec['box']}")
        else:
            steps.append(step)
            break
        t_sam += dt
        step["k"] = len(insts)
        steps.append(step)
    row = {"png": os.path.basename(png), "final_prompt": history[-1], "sam_calls": len(history),
           "k": len(insts), "steps": steps, "sam_s": round(t_sam, 2), "spark_s": round(t_spark, 2)}
    if insts:
        best = max(insts, key=lambda i: int((i["mask"] & gt).sum()))
        row["best"] = score(best["mask"], gt)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--start", default="paper")
    ap.add_argument("--mode", choices=["agent", "fixed", "single"], default="agent")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="every n-th view (cheap controls)")
    ap.add_argument("--spark-model", default="muse-spark-1.3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pngs = sorted(p for d in a.photos for p in glob.glob(os.path.join(d, "*_v[0-9].png"))
                  if os.path.exists(p[:-4] + "_gt.npz"))[::a.stride]
    if a.limit:
        pngs = pngs[:a.limit]
    rows = []
    for p in pngs:
        gt = np.load(p[:-4] + "_gt.npz")["mask"]
        try:
            row = run_view(p, gt, a)
        except Exception as e:
            row = {"png": os.path.basename(p), "error": repr(e)}
        rows.append(row)
        print(f"{row['png']}: {row.get('final_prompt')!r} calls={row.get('sam_calls')} "
              f"K={row.get('k')} IoU {row.get('best', {}).get('iou', 0):.3f} "
              f"{[s.get('action') + ':' + str(s.get('prompt') or s.get('box')) for s in row.get('steps', [])]}",
              flush=True)
    good = [r for r in rows if r.get("best", {}).get("iou", 0) >= 0.95]
    wrong = [r for r in rows if r.get("k") and r.get("best", {}).get("iou", 0) < 0.5]
    summary = {"start": a.start, "mode": a.mode, "rounds": a.rounds, "n": len(rows),
               "iou_ge_0.95": len(good), "found_but_iou_lt_0.5": len(wrong),
               "errors": sum("error" in r for r in rows),
               "sam_calls_mean": float(np.mean([r["sam_calls"] for r in rows if "sam_calls" in r])),
               "wall_s_mean": float(np.mean([r["sam_s"] + r["spark_s"] for r in rows if "sam_s" in r])),
               "final_prompts": sorted({r.get("final_prompt", "") for r in rows})}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
