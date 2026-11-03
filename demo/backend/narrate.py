"""Muse Spark (Meta Model API) writes the verdict of a camera-path result for the demo's step strip.

The model gets only pre-rounded numbers (what the camera saw, how far off both guesses were, what
is typical) and the text is checked the way t3_narrate.py checks it: every number it writes must be
one we gave. One retry on a failed check; after that ``ok`` is False and the frontend shows its
fixed sentence instead. Needs META_API_KEY; without it ``verdict`` returns ``ok: False`` at once.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
INVERSE = os.path.join(_HERE, "..", "..", "cloth_data_generation", "inverse")
if INVERSE not in sys.path:
    sys.path.insert(0, INVERSE)

import sam_api  # noqa: E402  (URL, Meta Model API)
from t3_narrate import check, numbers_in  # noqa: E402

MODEL = os.environ.get("SPARK_MODEL", "muse-spark-1.3")

PROMPT = (
    "You explain a live physics demo to a non-expert in exactly three short sentences, present tense, "
    "plain words, no lists, no markdown. A user pushed a 50 cm square sheet of paper twice. A camera "
    "pipeline then worked only from two photographs: SAM 3.1 found the sheet in the photos, VGGT turned "
    "the pair into depth, the printed pattern on the sheet told which point of the sheet each pixel "
    "shows, and a neural network guessed where and which way each push was. Sentence 1: what the "
    "camera pipeline saw (how much of the sheet, how the sheet was found). Sentence 2: how close the "
    "camera's guess was and whether that is typical. Sentence 3: how the same network does when it "
    "sees the exact simulated sheet instead of photos. Use ONLY the numbers below, copied as written; "
    "do not compute new ones.\n\n")


def facts(out: dict) -> dict:
    cam, mesh = out["camera"]["metrics"]["mean"], out["mesh"]["metrics"]["mean"]
    ref, cov = out.get("reference", {}), out.get("coverage", {})
    f = {"camera_guess_position_error_cm": round(cam["pos_cm"], 1),
         "camera_guess_direction_error_deg": round(cam["dir_deg"]),
         "typical_camera_position_error_cm": ref.get("camera", {}).get("pos_cm", 7.4),
         "typical_camera_direction_error_deg": round(ref.get("camera", {}).get("dir_deg", 17.0)),
         "exact_sheet_guess_position_error_cm": round(mesh["pos_cm"], 1),
         "exact_sheet_guess_direction_error_deg": round(mesh["dir_deg"]),
         "typical_exact_sheet_position_error_cm": ref.get("mesh", {}).get("pos_cm", 3.4),
         "typical_exact_sheet_direction_error_deg": round(ref.get("mesh", {}).get("dir_deg", 7.3)),
         "sheet_found_by": "SAM 3.1" if not out.get("mask_fallback") else "the renderer's own mask (SAM 3.1 unavailable)"}
    if cov.get("camera_sees") is not None:
        f["percent_of_sheet_the_camera_sees"] = round(100 * cov["camera_sees"])
    if cov.get("decoded") is not None:
        f["percent_of_sheet_points_read_from_the_print"] = round(100 * cov["decoded"])
    return f


def spark(text: str, timeout: float) -> str:
    body = {"model": MODEL, "stream": False,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]}
    req = urllib.request.Request(sam_api.URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {os.environ['META_API_KEY']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    return "".join(c.get("text", "") for o in resp.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", [])).strip()


def verdict(out: dict, timeout: float = 45.0) -> dict:
    """{text, ok, model, latency_s, checks, error?}; ok only when every number in the text was given."""
    if not os.environ.get("META_API_KEY"):
        return {"text": None, "ok": False, "model": MODEL, "error": "META_API_KEY not set"}
    f = facts(out)
    given = numbers_in(f)
    t0 = time.time()
    last = None
    for attempt in range(2):
        text = spark(PROMPT + json.dumps(f, indent=1), timeout)
        c = check(text, given)
        last = {"text": text, "ok": not c["unsupported"] and 1 <= c["sentences"] <= 4, "model": MODEL,
                "latency_s": round(time.time() - t0, 1), "checks": c, "attempt": attempt + 1}
        if last["ok"]:
            return last
    assert last is not None
    last["error"] = f"unsupported numbers {last['checks']['unsupported']} after 2 tries"
    last["text"] = None
    return last
