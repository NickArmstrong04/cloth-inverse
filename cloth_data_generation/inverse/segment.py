"""Rung 4b, step 2: segment the sheet in the synthetic photos with SAM 3 and score it.

Runs in the SAM 3 venv (cluster/setup_sam3.sh; `hf auth login` once for the gated
checkpoints), not in the cloth venv. Needs the `_gt.npz` files written by `photo.py --gt`.

For every photo and every text prompt: SAM 3 concept segmentation -> K instance masks.
"The sheet" is the highest-scoring mask; the union of all K is scored too (a sheet split
into several instances is a failure mode worth seeing). Scores against the oracle mask:
IoU, precision, recall, and the coverage of oracle-visible vertices (what the point model
will actually lose). `--box-oracle` adds a box prompt from the oracle mask as an upper
bound on prompting (not available on real photos).

Outputs under <photos>/sam3/: <stem>__<prompt-slug>.png mask PNGs (gitignored),
scores.json, contact_<slug>.png sheets (photo | oracle | SAM), summary.md (prompt table by
material). The prompts of each round live in cloth_results/photos/prompts.txt.

  python segment.py --photos ../../cloth_results/photos/cloth256 ../../cloth_results/photos/paper256 \\
      --prompt-file ../../cloth_results/photos/prompts.txt --box-oracle
  python segment.py --smoke        # the SAM 3 repo test image, prints K and scores
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

DEFAULT_PROMPTS = ["cloth", "fabric", "a piece of cloth", "a crumpled sheet of fabric", "a checkered cloth",
                   "a folded blue and white checkered cloth", "paper", "a crumpled sheet of paper"]


# ----------------------------------------------------------------------------- SAM 3

def sam3_root() -> str:
    import sam3
    return os.path.dirname(os.path.dirname(os.path.abspath(sam3.__file__)))


def load_sam3(threshold: float):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model()  # BPE vocab from the package; checkpoint from HF (gated: hf auth login)
    return Sam3Processor(model, confidence_threshold=threshold)


def run_prompt(proc, state, prompt: str | None, box=None):
    """Text prompt, or an oracle box (cxcywh normalised). Returns masks (K,H,W) bool numpy, scores (K,)."""
    proc.reset_all_prompts(state)
    if box is not None:
        out = proc.add_geometric_prompt(box, True, state)
    else:
        out = proc.set_text_prompt(prompt, state)
    out = out if isinstance(out, dict) else state
    masks = out.get("masks")
    if masks is None or len(masks) == 0:
        return np.zeros((0, state["original_height"], state["original_width"]), bool), np.zeros(0)
    masks = masks.squeeze(1).bool().cpu().numpy()
    scores = out["scores"].float().cpu().numpy()
    return masks, scores


# ----------------------------------------------------------------------------- scoring

def iou_pr(pred: np.ndarray, gt: np.ndarray):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter / union if union else 0.0, inter / pred.sum() if pred.sum() else 0.0,
            inter / gt.sum() if gt.sum() else 0.0)


def coverage(pred: np.ndarray, vpix: np.ndarray, vis: np.ndarray) -> float:
    """Fraction of oracle-visible vertices whose pixel is inside the predicted mask."""
    col, row = vpix[vis, 0], vpix[vis, 1]
    return float(pred[row, col].mean()) if vis.any() else 0.0


def oracle_box(mask: np.ndarray):
    rows, cols = np.nonzero(mask)
    H, W = mask.shape
    x0, x1, y0, y1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
    return [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


# ----------------------------------------------------------------------------- report

def overlay(photo: Image.Image, mask: np.ndarray, colour=(255, 60, 60)) -> Image.Image:
    im = photo.convert("RGB").copy()
    tint = Image.new("RGB", im.size, colour)
    alpha = Image.fromarray((mask * 110).astype(np.uint8))
    im.paste(tint, (0, 0), alpha)
    return im


def contact_sheet(rows: list, path: str, thumb: int = 192) -> None:
    """rows: list of (title, photo PIL, oracle mask, sam mask, text)."""
    n = len(rows)
    W, H = 3 * thumb, thumb + 18
    sheet = Image.new("RGB", (W, n * H), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (title, photo, gt, pred, text) in enumerate(rows):
        y = i * H
        for j, im in enumerate([photo, overlay(photo, gt, (60, 220, 60)), overlay(photo, pred)]):
            sheet.paste(im.resize((thumb, thumb)), (j * thumb, y + 18))
        draw.text((4, y + 2), f"{title}  {text}", fill=(230, 230, 230))
    sheet.save(path)


def summarize(records: list, prompts: list, path: str) -> str:
    lines = ["| prompt | material | n | mean IoU | median IoU | found (IoU>0.5) | mean coverage | mean K | union IoU |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in prompts:
        for mat in sorted({r["material"] for r in records}):
            rs = [r for r in records if r["prompt"] == p and r["material"] == mat]
            if not rs:
                continue
            iou = np.array([r["iou"] for r in rs])
            lines.append(f"| {p} | {mat} | {len(rs)} | {iou.mean():.3f} | {np.median(iou):.3f} | "
                         f"{(iou > 0.5).mean():.2f} | {np.mean([r['coverage'] for r in rs]):.3f} | "
                         f"{np.mean([r['K'] for r in rs]):.1f} | {np.mean([r['union_iou'] for r in rs]):.3f} |")
    text = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write("# SAM 3 on synthetic photos\n\n" + text + "\n")
    return text


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="*", default=[], help="photo set directories (PNG + _gt.npz)")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--prompt-file", default=None, help="one prompt per line; # comments")
    ap.add_argument("--box-oracle", action="store_true", help="also prompt with the oracle mask's box")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="default <photos>/sam3 per set; one dir for all sets if given")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="segment the SAM 3 repo test image")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    proc = load_sam3(args.threshold)
    print(f"SAM 3 loaded in {time.time() - t0:.0f}s")
    if args.smoke:
        im = Image.open(os.path.join(sam3_root(), "assets", "images", "test_image.jpg"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(im)
            for p in ["person", "shoe", "cloth"]:
                m, s = run_prompt(proc, state, p)
                print(f"{p!r}: K={len(m)} scores={np.round(s, 3).tolist()} image {im.size}")
        return
    prompts = list(args.prompts or [])
    if args.prompt_file:
        with open(args.prompt_file) as fh:
            prompts += [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    prompts = prompts or DEFAULT_PROMPTS
    if args.box_oracle:
        prompts.append("[oracle box]")
    all_records = []
    for pdir in args.photos:
        out = args.out or os.path.join(pdir, "sam3")
        os.makedirs(out, exist_ok=True)
        gts = sorted(glob.glob(os.path.join(pdir, "*_gt.npz")))
        if args.limit:
            gts = gts[: args.limit]
        records, sheets = [], {p: [] for p in prompts}
        for gp in gts:
            stem = os.path.basename(gp)[: -len("_gt.npz")]
            g = np.load(gp)
            gt_mask, vpix, vis = g["mask"], g["vpix"], g["vis"]
            material = str(g["material"])
            photo = Image.open(os.path.join(pdir, stem + ".png")).convert("RGB")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = proc.set_image(photo)
                for p in prompts:
                    box = oracle_box(gt_mask) if p == "[oracle box]" else None
                    masks, scores = run_prompt(proc, state, None if box else p, box)
                    if len(masks):
                        k = int(np.argmax(scores))
                        best, union = masks[k], masks.any(0)
                    else:
                        best = union = np.zeros_like(gt_mask)
                    iou, prec, rec = iou_pr(best, gt_mask)
                    rec_ = {"set": os.path.basename(pdir), "stem": stem, "material": material, "prompt": p,
                            "K": int(len(masks)), "scores": np.round(scores, 4).tolist(), "iou": iou,
                            "precision": prec, "recall": rec, "union_iou": iou_pr(union, gt_mask)[0],
                            "coverage": coverage(best, vpix, vis), "found": bool(iou > 0.5)}
                    records.append(rec_)
                    Image.fromarray((best * 255).astype(np.uint8)).save(os.path.join(out, f"{stem}__{slug(p)}.png"))
                    if len(sheets[p]) < 16:
                        sheets[p].append((stem, photo, gt_mask, best, f"K={len(masks)} IoU={iou:.2f} cov={rec_['coverage']:.2f}"))
            print(f"{stem}: " + "  ".join(f"{slug(p)[:12]}={r['iou']:.2f}" for p, r in
                                          zip(prompts, records[-len(prompts):])))
        for p, rows in sheets.items():
            if rows:
                contact_sheet(rows, os.path.join(out, f"contact_{slug(p)}.png"))
        with open(os.path.join(out, "scores.json"), "w") as fh:
            json.dump({"prompts": prompts, "threshold": args.threshold, "records": records}, fh, indent=1)
        print(summarize(records, prompts, os.path.join(out, "summary.md")))
        all_records += records
    if args.out and len(args.photos) > 1:
        summarize(all_records, prompts, os.path.join(args.out, "summary.md"))
    print(f"{len(all_records)} image x prompt runs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
