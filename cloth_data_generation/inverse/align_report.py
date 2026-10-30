"""Per-view alignment report: what the renderer knows about a photo vs what the perception stack found.

For every photo in a set we hold the truth (photo.py --gt): the sheet mask, the metric depth per
pixel, the vertex behind each pixel and the camera. The perception stack produced a SAM 3 mask
(segment.py, one prompt) and a VGGT depth / point cloud restricted to that mask (lift.py). This
script lines the two up per view and writes

  <out>/<set>__<stem>.png   one panel per view: photo | oracle mask | SAM mask | mask difference
                            oracle depth | VGGT depth (one scale fitted) | signed depth error | clouds
  <out>/report.md           summary per set + one row per view, worst first (by the similarity RMSE)
  <out>/report.json         the per-view numbers
  <out>/worst.png           the panels of the worst views, stacked

Numbers per view: mask IoU / precision / recall, visible-vertex coverage (fraction of the oracle's
visible vertices whose pixel the SAM mask keeps), VGGT depth error after a single median scale
(signed, cm: median absolute, p90 absolute, bias), relative depth error, point RMSE after scale +
centring and after the best similarity (Umeyama), the fitted scale, the visible fraction.

The VGGT depth map is rebuilt from the saved cloud: lift.py unprojects the pixels of
`sam_mask & oracle_depth > 0` in np.nonzero order and stores the cloud minus its centroid plus the
centroid in g[:3], so the per-pixel depth is -(pts + g[:3])[:, 2].

  python align_report.py --photos ../../cloth_results/photos/cloth256 ../../cloth_results/photos/paper256 \
      --out ../../cloth_results/photos/align
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from package_views import unproject  # noqa: E402


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def umeyama(a: np.ndarray, b: np.ndarray):
    """Similarity (s, R, t) minimising |s R a + t - b|; returns the aligned a and the RMSE in cm."""
    ma, mb = a.mean(0), b.mean(0)
    a0, b0 = a - ma, b - mb
    cov = b0.T @ a0 / len(a)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a0 ** 2).sum(1).mean()
    al = s * (a0 @ R.T) + mb
    return al, float(np.sqrt(((al - b) ** 2).sum(1).mean()) * 100)


def load_view(pdir: str, stem: str, prompt: str):
    g = np.load(os.path.join(pdir, f"{stem}_gt.npz"))
    photo = np.asarray(Image.open(os.path.join(pdir, f"{stem}.png")).convert("RGB"))
    mp = os.path.join(pdir, "sam3", f"{stem}__{slug(prompt)}.png")
    sam = np.asarray(Image.open(mp).convert("L")) > 127 if os.path.exists(mp) else None
    vp = os.path.join(pdir, "views", f"{stem}_pts_vggt.npz")
    vg = np.load(vp) if os.path.exists(vp) else None
    return g, photo, sam, vg


def analyse(g, sam, vg):
    """All per-view numbers plus the maps the panel draws."""
    omask, depth = g["mask"] & (g["depth"] > 0), g["depth"]
    vis, vpix = g["vis"], g["vpix"]
    r = {"visible_frac": float(vis.mean()), "n_oracle_px": int(omask.sum()), "material": str(g["material"])}
    maps = {"omask": omask, "odepth": depth}
    if sam is None:
        r["sam_found"] = False
        return r, maps
    inter, union = (sam & omask).sum(), (sam | omask).sum()
    r.update(sam_found=bool(sam.sum() > 0), n_sam_px=int(sam.sum()),
             iou=float(inter / max(union, 1)), precision=float(inter / max(sam.sum(), 1)),
             recall=float(inter / max(omask.sum(), 1)))
    vp = vpix[vis]  # (n, 2) col,row of the visible vertices, image rows down
    rows = np.clip(vp[:, 1], 0, sam.shape[0] - 1)
    cols = np.clip(vp[:, 0], 0, sam.shape[1] - 1)
    r["coverage"] = float(sam[rows, cols].mean()) if len(vp) else 0.0
    maps["sam"] = sam
    if vg is None:
        return r, maps
    m = sam & (depth > 0)
    rr, cc = np.nonzero(m)
    pts = vg["pts"] + vg["g"][:3]
    if len(pts) != len(rr):
        r["vggt_note"] = f"cloud has {len(pts)} points, mask {len(rr)} pixels: pixel order unknown"
        return r, maps
    s = float(vg["scale_to_oracle"])
    dv = -pts[:, 2] * s  # VGGT depth per masked pixel, scaled to the oracle median
    do = depth[rr, cc]
    err = dv - do
    vdepth = np.zeros_like(depth)
    vdepth[rr, cc] = dv
    emap = np.full_like(depth, np.nan)
    emap[rr, cc] = err * 100
    ref = unproject(m, depth, float(g["fov_deg"]))
    ps = pts * s
    rmse = float(np.sqrt((((ps - ps.mean(0)) - (ref - ref.mean(0))) ** 2).sum(1).mean()) * 100)
    al, sim = umeyama(pts, ref)
    r.update(scale=s, depth_abs_err_med_cm=float(np.median(np.abs(err)) * 100),
             depth_abs_err_p90_cm=float(np.percentile(np.abs(err), 90) * 100),
             depth_bias_cm=float(np.median(err) * 100),
             depth_rel_err_med=float(np.median(np.abs(err) / do)),
             rmse_scaled_cm=rmse, rmse_similarity_cm=sim,
             depth_range_oracle_cm=float((do.max() - do.min()) * 100))
    maps.update(vdepth=vdepth, emap=emap, ref=ref, lifted=ps - ps.mean(0) + ref.mean(0), aligned=al)
    return r, maps


def panel(photo, r, maps, title, path, max_pts=4000):
    fig, ax = plt.subplots(2, 4, figsize=(18, 9.5))
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.title.set_fontsize(9)
    ax[0, 0].imshow(photo)
    ax[0, 0].set_title("photo")
    ax[0, 1].imshow(maps["omask"], cmap="gray")
    ax[0, 1].set_title(f"oracle mask ({r['n_oracle_px']} px, visible {r['visible_frac']:.2f})")
    if "sam" in maps:
        sam, om = maps["sam"], maps["omask"]
        ax[0, 2].imshow(sam, cmap="gray")
        ax[0, 2].set_title(f"SAM mask ({r['n_sam_px']} px)")
        diff = np.zeros(sam.shape + (3,), np.uint8) + 40
        diff[sam & om] = (200, 200, 200)
        diff[om & ~sam] = (230, 40, 40)  # missed sheet
        diff[sam & ~om] = (40, 90, 230)  # extra background
        ax[0, 3].imshow(diff)
        ax[0, 3].set_title(f"IoU {r['iou']:.3f}  recall {r['recall']:.3f}  precision {r['precision']:.3f}\n"
                           f"red = missed sheet, blue = extra; vertex coverage {r['coverage']:.3f}")
    else:
        ax[0, 2].set_title("no SAM mask")
    od = maps["odepth"]
    inside = od > 0
    lo, hi = (od[inside].min(), od[inside].max()) if inside.any() else (0, 1)
    ax[1, 0].imshow(np.where(inside, od, np.nan), cmap="viridis", vmin=lo, vmax=hi)
    ax[1, 0].set_title(f"oracle depth {lo:.2f}-{hi:.2f} m")
    if "vdepth" in maps:
        vd = maps["vdepth"]
        ax[1, 1].imshow(np.where(vd > 0, vd, np.nan), cmap="viridis", vmin=lo, vmax=hi)
        ax[1, 1].set_title(f"VGGT depth x {r['scale']:.2f}\n(same colour scale)")
        lim = max(3.0, r["depth_abs_err_p90_cm"])
        im = ax[1, 2].imshow(maps["emap"], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[1, 2].set_title(f"VGGT - oracle depth (cm)\n|err| median {r['depth_abs_err_med_cm']:.1f}, "
                           f"p90 {r['depth_abs_err_p90_cm']:.1f}, bias {r['depth_bias_cm']:+.1f}")
        plt.colorbar(im, ax=ax[1, 2], fraction=0.046)
        ref, lifted, al = maps["ref"], maps["lifted"], maps["aligned"]
        sel = np.random.RandomState(0).permutation(len(ref))[:max_pts]
        a = ax[1, 3]
        a.scatter(ref[sel, 0], ref[sel, 2], s=1, c="k", label="oracle")
        a.scatter(lifted[sel, 0], lifted[sel, 2], s=1, c="tab:red", alpha=0.6, label=f"VGGT scaled+centred ({r['rmse_scaled_cm']:.1f} cm)")
        a.scatter(al[sel, 0], al[sel, 2], s=1, c="tab:blue", alpha=0.6, label=f"VGGT best similarity ({r['rmse_similarity_cm']:.1f} cm)")
        a.set_aspect("equal")
        a.legend(loc="upper right", fontsize=8, markerscale=6)
        a.set_title("clouds from above: x (right) vs z (toward camera)")
    else:
        ax[1, 1].set_title(r.get("vggt_note", "no VGGT cloud"))
    fig.suptitle(title)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=70)
    plt.close(fig)


def summary_figure(recs, path):
    """When does lifting fail? Depth error and point RMSE against how oblique the view is."""
    ok = [r for r in recs if r.get("sam_found") and "rmse_similarity_cm" in r]
    if not ok:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    col = ["tab:orange" if r["material"] == "paper" else "tab:blue" for r in ok]
    vf = [r["visible_frac"] for r in ok]
    dr = [r["depth_range_oracle_cm"] for r in ok]
    ax[0].scatter(vf, [r["depth_abs_err_med_cm"] for r in ok], c=col)
    ax[0].set_xlabel("visible fraction of the sheet (small = oblique / folded)")
    ax[0].set_ylabel("VGGT |depth error| median (cm)")
    ax[1].scatter(dr, [r["rmse_scaled_cm"] for r in ok], c=col, marker="o", label="scale + centring")
    ax[1].scatter(dr, [r["rmse_similarity_cm"] for r in ok], c=col, marker="x", label="best similarity")
    ax[1].set_xlabel("oracle depth range inside the mask (cm)")
    ax[1].set_ylabel("point RMSE vs oracle (cm)")
    ax[1].legend()
    ax[2].scatter([r["iou"] for r in ok], [r["coverage"] for r in ok], c=col)
    ax[2].set_xlabel("SAM mask IoU")
    ax[2].set_ylabel("visible-vertex coverage")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("blue = cloth, orange = paper")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fmt(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--prompt", default="fabric")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worst", type=int, default=6, help="panels stacked into worst.png")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = []
    for pdir in args.photos:
        sname = os.path.basename(os.path.normpath(pdir))
        gts = sorted(glob.glob(os.path.join(pdir, "*_gt.npz")))[: args.limit or None]
        for gp in gts:
            stem = os.path.basename(gp)[: -len("_gt.npz")]
            g, photo, sam, vg = load_view(pdir, stem, args.prompt)
            r, maps = analyse(g, sam, vg)
            r.update(set=sname, stem=stem, panel=f"{sname}__{stem}.png")
            title = f"{sname} / {stem} ({r['material']})"
            panel(photo, r, maps, title, os.path.join(args.out, r["panel"]))
            recs.append(r)
            print(f"{sname:9s} {stem:36s} IoU {fmt(r.get('iou'))}  cov {fmt(r.get('coverage'))}  "
                  f"|dz| med {fmt(r.get('depth_abs_err_med_cm'), 1)} cm  rmse {fmt(r.get('rmse_scaled_cm'), 1)} / "
                  f"sim {fmt(r.get('rmse_similarity_cm'), 1)} cm")
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(recs, fh, indent=1)

    def key(r):  # worst first: lifting error, then segmentation
        return -(r.get("rmse_similarity_cm", 1e9) if r.get("sam_found") else 1e9 + (1 - r.get("iou", 0)))

    order = sorted(recs, key=key)
    lines = [f"# Alignment report: truth vs perception, prompt \"{args.prompt}\"", "",
             "Per view: the renderer's truth (mask, metric depth, visible vertices) against the SAM 3 mask and",
             "the VGGT cloud lifted inside it. Panels: `<set>__<stem>.png`. Worst first by the similarity RMSE;",
             "views SAM did not find come first.", ""]
    for sname in sorted({r["set"] for r in recs}):
        rs = [r for r in recs if r["set"] == sname]
        ok = [r for r in rs if r.get("sam_found") and "rmse_similarity_cm" in r]
        med = lambda k: float(np.median([r[k] for r in ok])) if ok else float("nan")
        lines += [f"## {sname}: {len(rs)} views, SAM found {sum(bool(r.get('sam_found')) for r in rs)}, "
                  f"lifted {len(ok)}", "",
                  "| metric (median over lifted views) | value |", "|---|---|",
                  f"| mask IoU | {med('iou'):.3f} |", f"| mask recall / precision | {med('recall'):.3f} / {med('precision'):.3f} |",
                  f"| visible-vertex coverage | {med('coverage'):.3f} |",
                  f"| VGGT depth abs. error, median / p90 (cm) | {med('depth_abs_err_med_cm'):.1f} / {med('depth_abs_err_p90_cm'):.1f} |",
                  f"| VGGT depth bias (cm) | {med('depth_bias_cm'):+.1f} |",
                  f"| oracle depth range inside the mask (cm) | {med('depth_range_oracle_cm'):.1f} |",
                  f"| point RMSE, scale + centring / best similarity (cm) | {med('rmse_scaled_cm'):.1f} / {med('rmse_similarity_cm'):.1f} |",
                  f"| fitted scale (VGGT -> metres) | {med('scale'):.2f} |", ""]
    lines += ["## Per view (worst first)", "",
              "| set | view | mat | visible | IoU | recall | precision | coverage | |dz| med cm | |dz| p90 cm | bias cm | depth range cm | RMSE scaled cm | RMSE sim cm | scale | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in order:
        note = "SAM: no detection" if not r.get("sam_found") else r.get("vggt_note", "")
        lines.append("| " + " | ".join([r["set"], f"[{r['stem']}]({r['panel']})", r["material"][:5], fmt(r["visible_frac"], 2),
                                        fmt(r.get("iou")), fmt(r.get("recall")), fmt(r.get("precision")), fmt(r.get("coverage")),
                                        fmt(r.get("depth_abs_err_med_cm"), 1), fmt(r.get("depth_abs_err_p90_cm"), 1),
                                        fmt(r.get("depth_bias_cm"), 1), fmt(r.get("depth_range_oracle_cm"), 1),
                                        fmt(r.get("rmse_scaled_cm"), 1), fmt(r.get("rmse_similarity_cm"), 1),
                                        fmt(r.get("scale"), 2), note]) + " |")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    summary_figure(recs, os.path.join(args.out, "summary.png"))
    worst = [Image.open(os.path.join(args.out, r["panel"])) for r in order[: args.worst]]
    if worst:
        W, H = worst[0].size
        sheet = Image.new("RGB", (W, H * len(worst)), "white")
        for i, im in enumerate(worst):
            sheet.paste(im, (0, i * H))
        sheet.save(os.path.join(args.out, "worst.png"))
    print(f"{len(recs)} views -> {args.out}/report.md")


if __name__ == "__main__":
    main()
