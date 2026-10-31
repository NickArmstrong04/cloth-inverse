"""Phase B2: grid correspondences from the coded print (photo.py --texture coded), classic CV.

Per photo: SAM mask -> pixel colour class (chromaticity, nearest of the print colours; dark = grid
line) -> connected components inside the lines = cells -> each cell's 8 neighbours through the
lines, arranged into a 3x3 window on the local lattice axes -> code-book lookup (the print's 3x3
windows are unique under the 8 dihedral transforms, so orientation and the back side are free)
-> consistency filter against the neighbours' decoded indices. Decoded cells give (u, v) <-> pixel
correspondences at the cell centres; the 256-grid vertices get a pixel by a local affine fit over
the decoded cells around them. Scored against the renderer's truth (`_gt.npz`: per-pixel uv, the
visible vertices and their pixels).

  python decode_print.py --photos ../../cloth_results/photos/coded_test1024 [--oracle-mask] [--limit N]
Writes <photos>/decode/<stem>_corr.npz (cells: ij, pixel, ok; vertices: vpix_est, has_est) and
decode/scores.json; --panels writes an overlay PNG per view.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def dihedral_lookup(code: np.ndarray) -> dict:
    """window tuple (9 colour ids, row-major in the local frame) -> (i, j, ii, jj): the centre cell's
    index and the (3, 3) code indices of every local window position (so a decoded cell can label its
    neighbours whatever the in-plane rotation or mirroring of the print in the photo)."""
    cells = code.shape[0]
    table = {}
    for i in range(cells - 2):
        for j in range(cells - 2):
            w = code[i:i + 3, j:j + 3]
            ii, jj = np.meshgrid(np.arange(i, i + 3), np.arange(j, j + 3), indexing="ij")
            for r in range(4):
                for t in (False, True):
                    v, vi, vj = np.rot90(w, r), np.rot90(ii, r), np.rot90(jj, r)
                    if t:
                        v, vi, vj = v.T, vi.T, vj.T
                    table[tuple(v.ravel())] = (i + 1, j + 1, vi.copy(), vj.copy())
    return table


DIHEDRAL = []  # the 8 transforms as (3,3) offset arrays: local window position -> (di, dj) in the code
_oi, _oj = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
for _r in range(4):
    for _t in (False, True):
        _vi, _vj = np.rot90(_oi, _r), np.rot90(_oj, _r)
        DIHEDRAL.append((_vi.T.copy(), _vj.T.copy()) if _t else (_vi.copy(), _vj.copy()))


def classify(img: np.ndarray, mask: np.ndarray, colours: np.ndarray, dark: float = 0.16):
    """Per-pixel class: -1 outside, -2 grid line, else colour index (nearest chromaticity)."""
    cls = np.full(mask.shape, -1, np.int16)
    rgb = img[mask]
    line = rgb.max(1) < dark
    chroma = rgb / np.maximum(rgb.sum(1, keepdims=True), 1e-6)
    cc = colours / colours.sum(1, keepdims=True)
    d = ((chroma[:, None, :] - cc[None]) ** 2).sum(-1)
    k = d.argmin(1).astype(np.int16)
    k[line] = -2
    cls[mask] = k
    return cls


def decode_view(img, mask, code, colours, table, min_area=6, max_cells=4000):
    cells = code.shape[0]
    cls = classify(img, mask, colours)
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    props = []  # (centroid row, col, area, colour)
    n = 0
    for k in range(len(colours)):
        l, m = ndi.label(cls == k)
        if m == 0:
            continue
        areas = ndi.sum(np.ones_like(l), l, np.arange(1, m + 1))
        cents = ndi.center_of_mass(np.ones_like(l), l, np.arange(1, m + 1))
        keep = np.nonzero(areas >= min_area)[0]
        for q in keep:
            n += 1
            lab[l == q + 1] = n
            props.append((cents[q][0], cents[q][1], float(areas[q]), k))
        if n > max_cells:
            break
    if n < 9:
        return cls, lab, props, {}
    cent = np.array([(p[0], p[1]) for p in props])
    area = np.array([p[2] for p in props])
    col = np.array([p[3] for p in props])
    # neighbours on the local lattice: the 12 nearest centroids are candidates; the two lattice axes
    # come from the nearest candidate and the nearest roughly perpendicular one, refined by least
    # squares over the candidates that land on integer lattice sites (perspective and curvature
    # change the spacing smoothly, so a local affine lattice holds over a 3x3 window)
    tree = cKDTree(cent)
    nn_d, nn_i = tree.query(cent, k=min(13, n))
    decoded = {}  # b -> (i, j, ii, jj): label and frame from a direct code-book hit
    nbrs, sites = {}, {}
    stats = {"few_cand": 0, "no_axis": 0, "ambiguous": 0, "incomplete": 0, "miss": 0, "hit": 0, "hit1": 0, "nbr_hist": {}}
    for b in range(1, n + 1):
        cand = nn_i[b - 1, 1:]
        dist = nn_d[b - 1, 1:]
        if len(cand) < 8 or not np.isfinite(dist).all():
            stats["few_cand"] += 1
            continue
        rel = cent[cand] - cent[b - 1]
        a1 = rel[0] / dist[0]
        a2 = None
        for o in range(1, len(cand)):
            if dist[o] > 2.5 * dist[0]:
                break
            u = rel[o] / dist[o]
            if abs(u @ a1) < 0.7:
                a2 = rel[o]
                break
        if a2 is None:
            stats["no_axis"] += 1
            continue
        A = np.stack([rel[0], a2], 1)
        try:
            coords = np.linalg.solve(A, rel.T).T
        except np.linalg.LinAlgError:
            continue
        for _ in range(2):  # refine the lattice from the candidates on integer sites, then re-express
            on = (np.abs(coords - np.round(coords)) < 0.35).all(1) & (np.abs(np.round(coords)) <= 1.5).all(1)
            if on.sum() < 3:
                break
            ints = np.round(coords[on])
            A, *_ = np.linalg.lstsq(ints, rel[on], rcond=None)  # rel ~ ints @ A
            try:
                coords = np.linalg.solve(A.T, rel.T).T
            except np.linalg.LinAlgError:
                break
        win = np.full((3, 3), -1, int)
        win[1, 1] = col[b - 1]
        site = {}
        ok = True
        for q, (x, y) in zip(cand, coords):
            xi, yi = int(round(x)), int(round(y))
            if abs(x - xi) > 0.35 or abs(y - yi) > 0.35 or abs(xi) > 1 or abs(yi) > 1 or (xi == 0 and yi == 0):
                continue
            if win[yi + 1, xi + 1] >= 0:
                ok = False  # two blobs on one lattice site: ambiguous window
                break
            win[yi + 1, xi + 1] = col[q]
            site[(yi + 1, xi + 1)] = q + 1
        nbrs[b] = set(site.values())
        stats["nbr_hist"][len(site)] = stats["nbr_hist"].get(len(site), 0) + 1
        if not ok:
            stats["ambiguous"] += 1
            continue
        sites[b] = site
        missing = np.argwhere(win < 0)
        if len(missing) > 1:
            stats["incomplete"] += 1
            continue
        if len(missing) == 1:  # one unknown cell: accept if exactly one colour gives a code-book hit
            hits = []
            for k in range(len(colours)):
                w = win.copy()
                w[tuple(missing[0])] = k
                h = table.get(tuple(w.ravel()))
                if h is not None:
                    hits.append(h)
            if len(hits) == 1:
                decoded[b] = hits[0]
                stats["hit1"] += 1
            else:
                stats["miss"] += 1
            continue
        hit = table.get(tuple(win.ravel()))
        if hit is not None:
            decoded[b] = hit
            stats["hit"] += 1
        else:
            stats["miss"] += 1
    # propagation: a cell with a frame labels its 8 lattice neighbours; votes are tallied; a labelled
    # cell without a frame recovers one from >= 2 labelled neighbours (the unique dihedral transform
    # consistent with them) and labels onward. Final labels need >= 2 votes with >= 75% agreement.
    label = {b: (d[0], d[1]) for b, d in decoded.items()}
    frame = {b: (d[2], d[3]) for b, d in decoded.items()}
    for _ in range(8):
        votes = {}
        for b, (fi, fj) in frame.items():
            votes.setdefault(b, []).append(label[b])
            for (a, c), q in sites.get(b, {}).items():
                votes.setdefault(q, []).append((int(fi[a, c]), int(fj[a, c])))
        changed = False
        for q, vs in votes.items():
            best, cnt = max(((v, vs.count(v)) for v in set(vs)), key=lambda t: t[1])
            if cnt >= 2 and cnt >= 0.75 * len(vs) and label.get(q) != best:
                label[q] = best
                changed = True
        for q in list(label):  # frames for labelled cells that have a site map but no frame yet
            if q in frame or q not in sites:
                continue
            i0, j0 = label[q]
            lab_n = [((a, c), label[p]) for (a, c), p in sites[q].items() if p in label]
            if len(lab_n) < 2:
                continue
            fits = []
            for di, dj in DIHEDRAL:
                if all(label_n == (i0 + di[a, c], j0 + dj[a, c]) for (a, c), label_n in lab_n):
                    fits.append((i0 + di, j0 + dj))
            if len(fits) == 1:
                frame[q] = fits[0]
                changed = True
        if not changed:
            break
    good = {}
    for q, vs in votes.items():
        if q in label:
            cnt = vs.count(label[q])
            need = 2 if q in decoded else 3  # a propagated-only label needs one more agreeing neighbour
            if cnt >= need and cnt >= 0.75 * len(vs):
                good[q] = label[q]
    stats["propagated"] = len(good) - len([b for b in decoded if b in good])
    print("   stats", {k: v for k, v in stats.items() if k != "nbr_hist"}, "nbrs", dict(sorted(stats["nbr_hist"].items())), "good", len(good))
    return cls, lab, props, good


def vertex_pixels(good: dict, props, cells: int, ny: int, nx: int, radius_cells: float = 1.6):
    """Pixel estimate for every grid vertex by a local affine fit (uv -> pixel) over the decoded cell
    centres within radius_cells of it (needs >= 4). Returns (ny*nx, 2) col,row and a validity mask."""
    if len(good) < 4:
        return np.zeros((ny * nx, 2), np.float32), np.zeros(ny * nx, bool)
    bs = np.array(list(good.keys()))
    ij = np.array([good[b] for b in bs], float)
    uv_c = np.stack([(ij[:, 1] + 0.5) / cells, (ij[:, 0] + 0.5) / cells], 1)  # (u, v) of cell centres
    pix = np.array([(props[b - 1][1], props[b - 1][0]) for b in bs])  # (col, row)
    tree = cKDTree(uv_c)
    vv, uu = np.meshgrid(np.linspace(0, 1, ny), np.linspace(0, 1, nx), indexing="ij")
    q = np.stack([uu.ravel(), vv.ravel()], 1)
    est = np.zeros((len(q), 2), np.float32)
    has = np.zeros(len(q), bool)
    groups = tree.query_ball_point(q, r=radius_cells / cells)
    for t, g in enumerate(groups):
        if len(g) < 4:
            continue
        g = np.array(g)
        for _ in range(3):  # affine fit with outlier rejection (a wrongly decoded cell is far off)
            P = np.concatenate([uv_c[g] - q[t], np.ones((len(g), 1))], 1)
            M, res, rank, _ = np.linalg.lstsq(P, pix[g], rcond=None)
            if rank < 3:
                g = None
                break
            r = np.linalg.norm(P @ M - pix[g], axis=1)
            bad = r > max(2.5, 0.25 * np.median(np.linalg.norm(pix[g] - pix[g].mean(0), axis=1)))
            if not bad.any() or (~bad).sum() < 4:
                break
            g = g[~bad]
        if g is None or len(g) < 4 or np.median(r) > 3.0:
            continue
        est[t] = M[2]
        has[t] = True
    return est, has


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--oracle-mask", action="store_true")
    ap.add_argument("--min-area", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--panels", action="store_true")
    args = ap.parse_args()
    for pdir in args.photos:
        man = json.load(open(os.path.join(pdir, "manifest.json")))
        assert "coded" in man, f"{pdir}: not a coded-print render (photo.py --texture coded)"
        code = np.array(man["coded"]["code"])
        colours = np.array(man["coded"]["colours"], np.float32)
        cells = code.shape[0]
        table = dihedral_lookup(code)
        out = os.path.join(pdir, "decode")
        os.makedirs(out, exist_ok=True)
        recs = []
        for gp in sorted(glob.glob(os.path.join(pdir, "*_gt.npz")))[: args.limit or None]:
            stem = os.path.basename(gp)[: -len("_gt.npz")]
            gt = np.load(gp)
            img = np.asarray(Image.open(os.path.join(pdir, stem + ".png")).convert("RGB"), np.float32) / 255
            mask = gt["mask"]
            if not args.oracle_mask:
                mp = os.path.join(pdir, "sam3", f"{stem}__fabric.png")
                if os.path.exists(mp):
                    mask = np.asarray(Image.open(mp).convert("L")) > 127
            t0 = time.time()
            cls, lab, props, good = decode_view(img, mask, code, colours, table, args.min_area)
            # truth per decoded cell: the cell under the blob centroid
            uv = gt["uv"]  # (2, S, S)
            n_ok = 0
            cell_rows = []
            for b, (i, j) in good.items():
                r, c = int(round(props[b - 1][0])), int(round(props[b - 1][1]))
                u, v = float(uv[0, r, c]), float(uv[1, r, c])
                it, jt = min(int(v * cells), cells - 1), min(int(u * cells), cells - 1)
                ok = (it == i and jt == j)
                n_ok += ok
                cell_rows.append((i, j, props[b - 1][1], props[b - 1][0], ok))
            N = int(np.sqrt(len(gt["vis"])))
            est, has = vertex_pixels(good, props, cells, N, N)
            vis, vpix = gt["vis"], gt["vpix"].astype(np.float32)
            sel = vis & has
            err = np.linalg.norm(est[sel] - vpix[sel], axis=1) if sel.any() else np.array([])
            cell_px = float(np.median(np.sqrt([p[2] for p in props]))) / (1 - 0.2) if props else 0.0  # ~ cell size in px
            rec = dict(stem=stem, n_blobs=len(props), n_decoded=len(good), n_correct=n_ok,
                       decode_precision=(n_ok / len(good)) if good else 0.0,
                       cell_px=cell_px, visible_frac=float(vis.mean()),
                       vertex_coverage=float(sel.sum() / max(vis.sum(), 1)),
                       vertex_err_px_median=float(np.median(err)) if len(err) else float("nan"),
                       vertex_err_px_p90=float(np.percentile(err, 90)) if len(err) else float("nan"),
                       vertex_within_2px=float((err < 2).mean()) if len(err) else 0.0,
                       vertex_within_half_cell=float((err < 0.5 * cell_px).mean()) if len(err) else 0.0,
                       seconds=time.time() - t0)
            recs.append(rec)
            np.savez_compressed(os.path.join(out, f"{stem}_corr.npz"), cells=np.array(cell_rows, np.float32),
                                vpix_est=est, has_est=has, cells_n=cells)
            if args.panels:
                ov = (img * 255).astype(np.uint8).copy()
                for (i, j, c, r, ok) in cell_rows:
                    rr, cc = int(r), int(c)
                    ov[max(rr - 1, 0):rr + 2, max(cc - 1, 0):cc + 2] = (0, 255, 0) if ok else (255, 0, 255)
                Image.fromarray(ov).save(os.path.join(out, f"{stem}_decode.png"))
            print(f"{stem:34s} cell {cell_px:4.1f}px blobs {len(props):5d} decoded {len(good):4d} correct {n_ok:4d} "
                  f"({rec['decode_precision']:.2f}) | vertices: coverage {rec['vertex_coverage']:.2f} err med "
                  f"{rec['vertex_err_px_median']:4.1f} px p90 {rec['vertex_err_px_p90']:5.1f} within 2px {rec['vertex_within_2px']:.2f} "
                  f"| {rec['seconds']:.1f}s")
        med = lambda k: float(np.nanmedian([r[k] for r in recs]))
        summary = {k: med(k) for k in ["cell_px", "n_decoded", "decode_precision", "vertex_coverage", "vertex_err_px_median",
                                       "vertex_err_px_p90", "vertex_within_2px", "vertex_within_half_cell"]}
        summary["n"] = len(recs)
        print(os.path.basename(os.path.normpath(pdir)), json.dumps(summary))
        with open(os.path.join(out, "scores.json"), "w") as fh:
            json.dump({"args": vars(args), "records": recs, "summary": summary}, fh, indent=1)


if __name__ == "__main__":
    main()
