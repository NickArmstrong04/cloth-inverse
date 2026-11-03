"""T3 (hackathon tooling plan, item 3): Muse Spark writes the plain-language verdict of a demo result.

Input: demo/backend/cache/entry_*.json (the /jobs/{id} payload). We hand Muse Spark only
pre-rounded numbers (positions in cm on the 50 cm sheet, angles, impulses, per-push errors, the
model's test averages), then check the text: every number it writes must be one of the numbers we
gave (or a trivial restatement: 50 cm, 2 pushes, percentages of given fractions). No image: the mesh
path has no photo, and in T0 Muse Spark called the same render "folded" once and "crumpled" once.

  python t3_narrate.py --cache /tmp/t0/cache --out ../../cloth_results/photos/sam31_api/t3_narration.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time
import urllib.request

import sam_api

REF = {"pos_cm": 1.0, "dir_deg": 4.3, "impulse_rel": 0.156}   # demo/backend/inference.py (M16-a)
SHEET_CM = 50.0

PROMPT = (
    "You explain a physics demo to a non-expert in exactly two sentences. A user pushed a 50 cm square "
    "sheet twice; a neural network looked only at the deformed sheet and guessed where, which way and how "
    "hard each push was. Using ONLY the numbers below (copy them as written, do not compute new ones), "
    "say how close the guess was and whether that is typical for this model, in plain words. "
    "No lists, no markdown.\n\n")


def angle(a, b) -> float:
    c = sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def facts(e: dict) -> dict:
    t, p = e["true_forces"], e["predicted_forces"]

    def err(pi, ti):
        a, b = p[pi], t[ti]
        return {"pos_cm": SHEET_CM * math.hypot(a["u"] - b["u"], a["v"] - b["v"]),
                "dir_deg": angle(a["direction"], b["direction"]),
                "impulse_rel": abs(a["impulse"] - b["impulse"]) / b["impulse"]}
    pairs = min(((0, 0), (1, 1)), ((0, 1), (1, 0)),
                key=lambda pr: sum(err(*x)["pos_cm"] + err(*x)["dir_deg"] / 10 for x in pr))
    pushes = []
    for k, (pi, ti) in enumerate(sorted(pairs, key=lambda x: x[1])):
        er = err(pi, ti)
        pushes.append({
            "push": k + 1,
            "true_position_cm": [round(SHEET_CM * t[ti]["u"]), round(SHEET_CM * t[ti]["v"])],
            "guessed_position_cm": [round(SHEET_CM * p[pi]["u"]), round(SHEET_CM * p[pi]["v"])],
            "position_error_cm": round(er["pos_cm"], 1),
            "direction_error_deg": round(er["dir_deg"]),
            "true_strength_Ns": round(t[ti]["impulse"], 2),
            "guessed_strength_Ns": round(p[pi]["impulse"], 2),
            "strength_error_percent": round(100 * er["impulse_rel"])})
    m = e["metrics_mean"]
    return {"pushes": pushes,
            "average_position_error_cm": round(m["pos_cm"], 1),
            "average_direction_error_deg": round(m["dir_deg"]),
            "average_strength_error_percent": round(100 * m["impulse_rel"]),
            "model_typical_position_error_cm": REF["pos_cm"],
            "model_typical_direction_error_deg": REF["dir_deg"],
            "model_typical_strength_error_percent": round(100 * REF["impulse_rel"])}


def numbers_in(o) -> set[float]:
    if isinstance(o, dict):
        return set().union(*(numbers_in(v) for v in o.values()))
    if isinstance(o, list):
        return set().union(*(numbers_in(v) for v in o)) if o else set()
    return {float(o)} if isinstance(o, (int, float)) else set()


def check(text: str, given: set[float]) -> dict:
    allowed = given | {50.0, 2.0, 1.0}           # sheet size, two pushes, "push 1"
    found = [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text.replace(",", ""))]
    bad = [x for x in found if not any(abs(x - g) < 1e-6 for g in allowed)]
    sentences = len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])
    return {"numbers": found, "unsupported": bad, "sentences": sentences}


def spark(text: str, model: str) -> tuple[str, float]:
    body = {"model": model, "stream": False,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]}
    req = urllib.request.Request(sam_api.URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {os.environ['META_API_KEY']}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    out = "".join(c.get("text", "") for o in resp.get("output", []) if o.get("type") == "message"
                  for c in o.get("content", []))
    return out.strip(), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--spark-model", default="muse-spark-1.3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.cache, "entry_*.json"))):
        with open(f) as fh:
            fa = facts(json.load(fh))
        try:
            text, dt = spark(PROMPT + json.dumps(fa, indent=1), a.spark_model)
        except Exception as e:
            rows.append({"entry": os.path.basename(f), "error": repr(e)})
            continue
        row = {"entry": os.path.basename(f), "facts": fa, "text": text, "s": round(dt, 2),
               **check(text, numbers_in(fa))}
        rows.append(row)
        print(f"{row['entry']} ({dt:.1f}s, {row['sentences']} sent., unsupported {row['unsupported']}): {text}\n",
              flush=True)
    ok = [r for r in rows if "text" in r and not r["unsupported"] and r["sentences"] <= 2]
    summary = {"n": len(rows), "pass": len(ok), "errors": sum("error" in r for r in rows),
               "with_unsupported_numbers": sum(bool(r.get("unsupported")) for r in rows),
               "over_two_sentences": sum(r.get("sentences", 0) > 2 for r in rows),
               "s_median": sorted(r["s"] for r in rows if "s" in r)[len(rows) // 2] if rows else None}
    print(json.dumps(summary, indent=1))
    with open(a.out, "w") as fh:
        json.dump({"summary": summary, "prompt": PROMPT, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
