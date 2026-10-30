"""Train an inverse model (final cloth state -> two force vectors)."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

import numpy as np
import math
import torch

from data import TargetScaler, augment, decode_forces, load_frames, load_split, random_rotation
from geometry import shade
from metrics import pair_metrics, summarize
from models import PointTF, SonataReg, build, equivalent_targets, pair_loss
from render import view_batch

VIEW = None  # set in main() when --view: dict(size, extent, splat, norm, shade)
FLOW = None  # set in main() when --flow: dict(scaler, sym_uv); forward_batch hands the model its equivalent targets


def _set_flow_targets(net, y):
    m = net.m
    if FLOW is not None and getattr(m, "flow", None) is not None:
        m.flow_targets = equivalent_targets(FLOW["scaler"].norm(y), FLOW["sym_uv"])


def forward_batch(net, x, g, y, a=None, R=None, vis_only=True):
    """One model call with the observation-ladder preprocessing applied.
    Without --view: net(x, g, a). With --view: rotate into the camera frame R (random if None),
    render a depth+mask image (and the force field into image space); image models see the
    image, the point model sees the camera-frame points restricted to the visible ones.
    Returns pred (B,2,7), aux loss or None, and the (possibly rotated) target y."""
    if VIEW is None:
        _set_flow_targets(net, y)
        return (*net(x, g, a), y)
    dev = g.device
    x = x.to(dev, non_blocking=True)
    a = a.to(dev, non_blocking=True) if a is not None else None
    if R is None:
        R = random_rotation(x.shape[0], dev)
    img, gc, yc, a_img, vis, pts = view_batch(x, g, y, a, R, VIEW["size"], VIEW["extent"], VIEW["splat"], VIEW["norm"])
    _set_flow_targets(net, yc)
    if isinstance(net.m, (PointTF, SonataReg)):
        B, N, _ = pts.shape
        xp = pts.transpose(1, 2).reshape(B, 3, x.shape[-2], x.shape[-1])
        ar = torch.einsum("bij,bjhw->bihw", R, a) if a is not None else None
        pred, aux = net(xp, gc, ar, vis=vis if vis_only else None)
    else:
        if VIEW["shade"]:
            img = shade(img, VIEW["extent"])
        pred, aux = net(img, gc, a_img)
    return pred, aux, yc


def _rot_from_gen(n, gen, yaw_only=False):
    if yaw_only:
        th = torch.rand(n, generator=gen) * (2 * math.pi)
        c, s = torch.cos(th), torch.sin(th)
        q = torch.zeros(n, 3, 3)
        q[:, 0, 0], q[:, 0, 1], q[:, 1, 0], q[:, 1, 1], q[:, 2, 2] = c, -s, s, c, 1.0
        return q
    a = torch.randn(n, 3, 3, generator=gen)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q[torch.det(q) < 0, :, 0] *= -1
    return q


def _rotate(x, g, y, R):
    """Apply per-sample rotations R (N,3,3) to a dataset; works for 3 or 6 channels."""
    x, g, y = x.clone(), g.clone(), y.clone()
    nvec = x.shape[1] // 3
    for k in range(nvec):
        x[:, 3 * k:3 * k + 3] = torch.einsum("bij,bjhw->bihw", R, x[:, 3 * k:3 * k + 3])
        g[:, 3 * k:3 * k + 3] = torch.einsum("bij,bj->bi", R, g[:, 3 * k:3 * k + 3])
    y[:, :, 2:5] = torch.einsum("bij,bkj->bki", R, y[:, :, 2:5])
    return x, g, y


def batched_loss(net, x, g, y, scaler, bs, sym_uv=None, a=None, R=None, aux_w=0.0, w_mag=1.0):
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        ai = a[i:i + bs] if a is not None else None
        Ri = R[i:i + bs] if R is not None else None
        p, aux, yb = forward_batch(net, x[i:i + bs], g[i:i + bs], y[i:i + bs], ai, Ri)
        l = pair_loss(p.float(), scaler.norm(yb), sym_uv=sym_uv, w_mag=w_mag)
        if aux is not None and aux_w > 0:
            l = l + aux_w * aux
        fl = getattr(net.m, "last_flow_loss", None)
        if fl is not None:
            l = l + fl
        tot += l.item() * len(p)
        n += len(p)
    return tot / max(n, 1)


def predict(model, x, g, scaler, tta=0, bs=16, yaw_only=False, R=None):
    outs = []
    for i in range(0, len(x), bs):
        if VIEW is not None:  # fixed camera per sample; predictions come back in the camera frame
            model.eval()
            with torch.no_grad():
                p, _, _ = forward_batch(model, x[i:i + bs], g[i:i + bs], torch.zeros(len(x[i:i + bs]), 2, 7, device=g.device),
                                        None, R[i:i + bs])
                p = scaler.denorm(p.float())
                p[:, :, 2:5] = torch.nn.functional.normalize(p[:, :, 2:5], dim=-1)
            outs.append(p)
            continue
        outs.append(_predict_batch(model, x[i:i + bs], g[i:i + bs], scaler, tta, yaw_only))
    return torch.cat(outs)


def _predict_batch(model, x, g, scaler, tta=0, yaw_only=False):
    """Return raw-scale predictions (N,2,7). With tta>0, average over random
    world rotations (directions rotated back; u,v,mag,dur rotation-invariant)."""
    model.eval()
    x = x.to(next(model.parameters()).device, non_blocking=True)
    with torch.no_grad():
        outs = []
        p, _ = model(x, g)
        p = p.float()
        outs.append(scaler.denorm(p))
        for _ in range(tta):
            R = random_rotation(x.shape[0], x.device, yaw_only)
            xr, gr, _ = _rotate(x, g, torch.zeros(x.shape[0], 2, 7, device=x.device), R)
            p = scaler.denorm(model(xr, gr)[0].float())
            p[:, :, 2:5] = torch.einsum("bji,bkj->bki", R, p[:, :, 2:5])  # R^T d
            # align slot order to the first prediction before averaging
            d0 = outs[0][:, :, :2]
            swap = ((p[:, :, :2] - d0) ** 2).sum((-1, -2)) > ((p.flip(1)[:, :, :2] - d0) ** 2).sum((-1, -2))
            p[swap] = p[swap].flip(1)
            outs.append(p)
        out = torch.stack(outs).mean(0)
        out[:, :, 2:5] = torch.nn.functional.normalize(out[:, :, 2:5], dim=-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--pool", type=int, default=1, help="block-average factor on load (1 = full resolution)")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (recommended at 768)")
    ap.add_argument("--data-on-cpu", action="store_true",
                    help="keep the dataset in host RAM and move batches (if it does not fit the GPU)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--model", default="cnn_reg",
                    choices=["cnn_reg", "cnn_heat", "cnn_field", "pointtf", "dino_reg", "sonata_reg"])
    ap.add_argument("--intrinsic", action="store_true",
                    help="grid models: rotation-invariant inputs (strain + curvature in material coordinates) and "
                         "directions predicted in the local tangent frame (geometry.py). Pose-invariant by construction, "
                         "so --eval-rotated costs nothing for position; world-frame globals g are ignored")
    ap.add_argument("--flow", action="store_true",
                    help="grid models: add a conditional flow-matching head that samples K candidate force pairs "
                         "(multimodal posterior); candidates for the test split go to test_candidates.json for "
                         "select_resim.py. Metrics report best-of-K (oracle) and mean-of-K")
    ap.add_argument("--flow-k", type=int, default=8, help="--flow: candidates per test sample")
    ap.add_argument("--flow-steps", type=int, default=32, help="--flow: Euler steps when sampling")
    ap.add_argument("--view-shade", action="store_true",
                    help="--view with image models: add a Lambertian shading channel computed from the depth image "
                         "(depth, mask, shade), closer to a photograph for pretrained vision backbones")
    ap.add_argument("--dino-repo", default="facebookresearch/dinov2",
                    help="torch.hub repo for dino_reg (facebookresearch/dinov2 or facebookresearch/dinov3)")
    ap.add_argument("--dino-weights", default=None, help="dino_reg: local weights file (DINOv3 weights are gated)")
    ap.add_argument("--sonata-ckpt", default="sonata", help="sonata_reg: checkpoint name (HF facebook/sonata) or .pth path")
    ap.add_argument("--sonata-grid", type=float, default=0.004, help="sonata_reg: voxel size (m) for the point cloud")
    ap.add_argument("--no-finetune", action="store_true", help="sonata_reg: freeze the backbone")
    ap.add_argument("--aux-w", type=float, default=0.0,
                    help="weight of the dense force-field auxiliary loss (per-cell direction map of both force "
                         "patches); 0 = off. Supported by cnn_reg, pointtf (per point) and dino_reg (per patch)")
    ap.add_argument("--frames", choices=["off", "post", "all"], default="off",
                    help="add the stored intermediate states of each TRAIN sample as extra samples with the same "
                         "labels: post = only frames after both forces ended, all = every stored frame. "
                         "Frames carry no velocities (use with --no-vel)")
    ap.add_argument("--view", action="store_true",
                    help="rung 4a: observe a single orthographic view from a random camera (depth+mask image for "
                         "image models, visible points only for pointtf); fixed seeded cameras for val/test. "
                         "Implies unknown pose (D4-symmetric loss/metrics, TTA off)")
    ap.add_argument("--view-size", type=int, default=224, help="--view image size in px (multiple of 14 for dino)")
    ap.add_argument("--view-extent", type=float, default=0.6, help="--view metric window (m) on the image")
    ap.add_argument("--view-splat", type=int, default=3, help="--view dilation (px) closing pin holes")
    ap.add_argument("--view-norm", action="store_true",
                    help="--view: rescale by the visible extent (no absolute scale, as in a monocular photo)")
    ap.add_argument("--view-all-points", action="store_true",
                    help="--view with pointtf: feed every point (camera frame only; measures pose vs occlusion)")
    ap.add_argument("--dino-arch", default="dinov2_vits14", help="dino_reg backbone (torch.hub facebookresearch/dinov2)")
    ap.add_argument("--dino-finetune", action="store_true", help="dino_reg: unfreeze the backbone")
    ap.add_argument("--init", default=None,
                    help="initialise from a saved model.pt (curriculum: e.g. rung 3 -> rung 4a); shapes must match")
    ap.add_argument("--no-vel", action="store_true",
                    help="observation ladder rung 1: drop velocity channels and mean velocity (positions only)")
    ap.add_argument("--eval-rotated", action="store_true",
                    help="rung 2: evaluate val/test under a fixed random world rotation per sample (unknown pose)")
    ap.add_argument("--n-points", type=int, default=2048, help="pointtf: points sampled per step")
    ap.add_argument("--pt-dim", type=int, default=128, help="pointtf: embedding width")
    ap.add_argument("--pt-layers", type=int, default=4, help="pointtf: transformer layers")
    ap.add_argument("--sym", choices=["auto", "on", "off"], default="auto",
                    help="treat (u,v) as defined only up to the sheet's D4 symmetry in loss and metrics "
                         "(auto: on when --eval-rotated, i.e. the pose is unknown)")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--pool-max", action="store_true",
                    help="cnn_reg: concatenate per-channel max pooling to the mean pooling of the final map "
                         "(keeps the local peak strain that sets the force magnitude; M14)")
    ap.add_argument("--w-mag", type=float, default=1.0, help="weight of the log-magnitude term in the loss (M14)")
    ap.add_argument("--strain-scale", type=float, default=10.0,
                    help="--intrinsic: multiplier on the strain channels before the learnable affine (M14)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training: evaluate the --init checkpoint and write metrics.json (recover a run whose eval crashed)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--yaw-only", action="store_true",
                    help="rung 1.5: world rotation (aug, --eval-rotated, TTA) about the sheet normal only; "
                         "known support plane, unknown in-plane heading")
    ap.add_argument("--no-rot", action="store_true",
                    help="dihedral (material-symmetry) augmentation only, no 3D world rotation")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=0,
                    help="use only the first N training samples (seeded shuffle); 0 = all. "
                         "Data-scaling control: train on a 1100-sample subset of a larger dataset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global VIEW, FLOW
    if args.view:
        VIEW = {"size": args.view_size, "extent": args.view_extent, "splat": args.view_splat, "norm": args.view_norm,
                "shade": args.view_shade}
        args.tta = 0
    grid_model = args.model in ("cnn_reg", "cnn_field")
    if args.intrinsic:
        assert grid_model and not args.view, "--intrinsic needs the material grid (cnn_reg / cnn_field, no --view)"
        args.tta = 0  # invariant by construction
    if args.flow:
        assert grid_model and not args.view, "--flow is implemented for cnn_reg / cnn_field without --view"
    out = args.out or (f"runs/{os.path.basename(os.path.normpath(args.root))}_{args.model}_s{args.seed}_e{args.epochs}"
                       + (f"_pool{args.pool}" if args.pool > 1 else "")
                       + ("_intr" if args.intrinsic else "") + ("_flow" if args.flow else "")
                       + (f"_d{args.pt_dim}L{args.pt_layers}n{args.n_points}" if args.model == "pointtf" else "")
                       + ("_novel" if args.no_vel else "") + ("_evalrot" if args.eval_rotated else "") + ("_yaw" if args.yaw_only else "")
                       + ("_noaug" if args.no_aug else "") + ("_norot" if args.no_rot else ""))
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB", flush=True)

    tr, va, te = (load_split(args.root, s, pool=args.pool) for s in ("train", "val", "test"))
    for split, d in (("train", tr), ("val", va), ("test", te)):  # a single non-finite sample makes the split loss NaN
        ok = (torch.isfinite(d["x"]).flatten(1).all(1) & torch.isfinite(d["g"]).all(1) & torch.isfinite(d["y"]).flatten(1).all(1))
        if not bool(ok.all()):
            bad = [d["names"][i] for i in (~ok).nonzero().flatten().tolist()]
            print(f"WARNING: dropping {len(bad)} non-finite {split} sample(s): {bad[:10]}{' ...' if len(bad) > 10 else ''}", flush=True)
            keep = ok.nonzero().flatten()
            for k in ("x", "g", "y", "a"):
                d[k] = d[k][keep].contiguous()
            d["names"] = [d["names"][i] for i in keep.tolist()]
    keep = None
    if args.n_train and args.n_train < len(tr["names"]):
        keep = torch.randperm(len(tr["names"]), generator=torch.Generator().manual_seed(args.seed))[: args.n_train]
        keep = keep.sort().values
        tr = {"x": tr["x"][keep].contiguous(), "g": tr["g"][keep].contiguous(), "y": tr["y"][keep].contiguous(),
              "a": tr["a"][keep].contiguous(), "names": [tr["names"][i] for i in keep.tolist()]}
    if args.frames != "off":  # extra training samples from the stored intermediate states
        fr = load_frames(args.root, "train", pool=args.pool, mode=args.frames)
        if keep is not None and len(fr["names"]):
            sel = torch.isin(fr["base"], keep)
            fr = {k: (v[sel] if torch.is_tensor(v) and v.ndim else v) for k, v in fr.items()}
            fr["names"] = [n for n, ok in zip(fr["names"], sel.tolist()) if ok]
        if len(fr["names"]):
            okf = torch.isfinite(fr["x"]).flatten(1).all(1)
            print(f"frames ({args.frames}): +{int(okf.sum())} training samples from intermediate states "
                  f"({len(tr['names'])} finals)", flush=True)
            tr = {"x": torch.cat([tr["x"], fr["x"][okf]]), "g": torch.cat([tr["g"], fr["g"][okf]]),
                  "y": torch.cat([tr["y"], fr["y"][okf]]), "a": torch.cat([tr["a"], fr["a"][okf]]),
                  "names": tr["names"] + [n for n, ok in zip(fr["names"], okf.tolist()) if ok]}
        else:
            print("frames: none found in the train files (older dataset?)", flush=True)
    if args.no_vel:  # rung 1: positions only (keep tensors contiguous for later indexing)
        for d in (tr, va, te):
            d["x"] = d["x"][:, :3].contiguous()
            d["g"] = d["g"][:, :3].contiguous()
    cin, gdim = tr["x"].shape[1], tr["g"].shape[1]
    if args.eval_rotated:  # rung 2: a fixed, seeded random rotation per val/test sample
        gen = torch.Generator().manual_seed(1234)
        for d in (va, te):
            R = _rot_from_gen(len(d["x"]), gen, args.yaw_only)
            d["x"], d["g"], d["y"] = _rotate(d["x"], d["g"], d["y"], R)
            d["a"] = torch.einsum("bij,bjhw->bihw", R, d["a"])
    # --view: a fixed seeded camera per val/test sample (the unknown pose is the camera frame)
    gen_cam = torch.Generator().manual_seed(4321)
    Rva = _rot_from_gen(len(va["x"]), gen_cam) if args.view else None
    Rte = _rot_from_gen(len(te["x"]), gen_cam) if args.view else None
    in_size = tr["x"].shape[-1]
    print(f"data: train {len(tr['names'])}  val {len(va['names'])}  test {len(te['names'])}  "
          f"input {tuple(tr['x'].shape[1:])}  ({tr['x'].numel() * 4 / 1e9:.1f} GB train)", flush=True)
    scaler = TargetScaler(tr["y"]).to(dev)
    # per-channel input scale from a subsample (full std over 15 GB is slow and unnecessary)
    sub = tr["x"][:: max(1, len(tr["x"]) // 64)]
    xstd = sub.std((0, 2, 3), keepdim=True).to(dev)
    gstd = tr["g"].std(0, keepdim=True).to(dev) + 1e-6
    data_dev = torch.device("cpu") if args.data_on_cpu else dev

    if args.model == "cnn_field" and args.aux_w < 1.0:
        print(f"cnn_field: the dense maps are the primary output; --aux-w raised from {args.aux_w} to 1.0", flush=True)
        args.aux_w = 1.0
    use_aux = args.aux_w > 0 and args.model != "cnn_heat"

    def prep(d):
        x = d["x"].to(data_dev)
        if args.data_on_cpu:
            x = x.pin_memory()
        a = d["a"].to(data_dev) if use_aux else None
        return x, d["g"].to(dev), d["y"].to(dev), a

    xtr, gtr, ytr, atr = prep(tr)
    xva, gva, yva, ava = prep(va)
    xte, gte, yte, ate = prep(te)
    if Rva is not None:
        Rva, Rte = Rva.to(dev), Rte.to(dev)

    uv_scale = (scaler.mean[:2], scaler.std[:2])
    if args.sym == "auto":
        # With intrinsic inputs the material indexing is part of the input, so (u, v) are
        # identifiable; the D4 relabeling is only needed when the indexing is lost (points) or
        # the pose enters through world coordinates. Pass --sym on to grade on the lenient metric.
        sym = (args.eval_rotated or args.view) and not args.intrinsic
    else:
        sym = args.sym == "on"
    sym_uv = uv_scale if sym else None
    if sym and args.tta:
        print("note: TTA disabled under the D4-symmetric metric (passes may settle on different relabelings)")
        args.tta = 0
    print(f"symmetric (D4) (u,v) loss/metrics: {sym}", flush=True)
    if args.flow:
        FLOW = {"scaler": scaler, "sym_uv": sym_uv}
    if args.view and args.model not in ("pointtf", "sonata_reg"):  # image models see the rendered image
        in_size, cin = args.view_size, (3 if args.view_shade else 2)
    model = build(args.model, uv_scale=uv_scale, w=args.width, in_size=in_size, cin=cin, gdim=gdim,
                  n_points=args.n_points, d=args.pt_dim, layers=args.pt_layers, aux=use_aux,
                  arch=args.dino_arch, size=args.view_size, finetune=args.dino_finetune,
                  repo=args.dino_repo, weights=args.dino_weights, intrinsic=args.intrinsic, flow=args.flow,
                  flow_steps=args.flow_steps, ckpt=args.sonata_ckpt, grid_size=args.sonata_grid,
                  pool="max" if args.pool_max else "mean", strain_scale=args.strain_scale).to(dev)
    if args.model == "sonata_reg":
        model.finetune = not args.no_finetune
        for p in model.backbone.parameters():
            p.requires_grad = model.finetune
    if use_aux:
        model.aux_m0 = float((atr[:: max(1, len(atr) // 256)].float() ** 2).mean().clamp_min(1e-8))
        print(f"aux force-field loss: weight {args.aux_w}, normaliser m0 {model.aux_m0:.2e}", flush=True)
    if args.init:
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        state = {k[2:] if k.startswith("m.") else k: v for k, v in ck["state"].items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init from {args.init}: {len(state) - len(unexpected)} tensors loaded, "
              f"{len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if (args.amp and dev.type == "cuda") else contextlib.nullcontext()
    nparam = sum(p.numel() for p in model.parameters())

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def _raw(self):
            return getattr(self.m, "intrinsic", False) or isinstance(self.m, SonataReg)

        def _scale(self, x):
            if self._raw():
                return x  # raw metres: the model derives invariant features / voxelises itself
            return x / xstd if x.shape[1] == xstd.shape[1] else x / img_scale[:, : x.shape[1]]  # rendered image

        def forward(self, x, g, a=None, **kw):
            a = a.to(dev, non_blocking=True) if a is not None else None
            x = self._scale(x.to(dev, non_blocking=True))
            return self.m(x, g if self._raw() else g / gstd, a, **kw)

        @torch.no_grad()
        def sample(self, x, g, K):
            self.m.eval()
            return self.m.sample(self._scale(x.to(dev, non_blocking=True)), g if self._raw() else g / gstd, K)

    img_scale = torch.tensor([args.view_extent, 1.0, 1.0], device=dev).view(1, 3, 1, 1)
    if args.view:
        assert args.no_vel, "--view needs --no-vel (a view gives positions only)"
    net = Wrapped(model)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(xtr) // args.bs)
    if args.eval_only:
        assert args.init, "--eval-only needs --init <model.pt>"
        args.epochs = 0
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, args.epochs * steps_per_epoch),
                                                pct_start=0.1)
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xtr), device=dev)
        tot = 0.0
        for k in range(steps_per_epoch):
            idx = perm[k * args.bs:(k + 1) * args.bs]
            x = xtr[idx.cpu()].to(dev, non_blocking=True) if args.data_on_cpu else xtr[idx]
            g, y = gtr[idx], ytr[idx]
            a = (atr[idx.cpu()].to(dev, non_blocking=True) if args.data_on_cpu else atr[idx]) if use_aux else None
            if not args.no_aug:
                if a is not None:  # the force field is a vector field on the grid: augment it with x
                    xa, g, y = augment(torch.cat([x, a], 1), g, y, rotate=not args.no_rot and not args.view,
                                       yaw_only=args.yaw_only)
                    x, a = xa[:, :-3], xa[:, -3:]
                else:
                    x, g, y = augment(x, g, y, rotate=not args.no_rot and not args.view, yaw_only=args.yaw_only)
            with amp:
                pred, aux, y = forward_batch(net, x, g, y, a, None, not args.view_all_points)
            loss = pair_loss(pred.float(), scaler.norm(y), sym_uv=sym_uv, w_mag=args.w_mag)
            if aux is not None:
                loss = loss + args.aux_w * aux
            fl = getattr(net.m, "last_flow_loss", None)
            if fl is not None:
                loss = loss + fl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad(), amp:
                vl = batched_loss(net, xva, gva, yva, scaler, args.bs, sym_uv, ava, Rva, args.aux_w, args.w_mag)
                ntr = min(len(xtr), 256)  # train loss on a fixed subset (full pass at 768 is slow)
                Rtr = _rot_from_gen(ntr, torch.Generator().manual_seed(999)).to(dev) if args.view else None
                tl = batched_loss(net, xtr[:ntr], gtr[:ntr], ytr[:ntr], scaler, args.bs, sym_uv,
                                  atr[:ntr] if use_aux else None, Rtr, args.aux_w, args.w_mag)
            hist.append({"epoch": ep + 1, "train_aug": tot / steps_per_epoch, "train": tl, "val": vl})
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}
            if (ep + 1) % 25 == 0:
                print(f"ep {ep+1:4d} loss_aug {tot/steps_per_epoch:.4f} train {tl:.4f} val {vl:.4f} "
                      f"best {best:.4f} [{time.time()-t0:.0f}s]", flush=True)
    if args.eval_only:
        best_state, best = net.state_dict(), float("nan")
    if best_state is None:
        raise RuntimeError(f"validation loss was never finite (best={best}); check the val split for NaN inputs/targets")
    net.load_state_dict(best_state)
    if not args.eval_only:
        torch.save({"state": best_state, "args": vars(args), "scaler": (scaler.mean.cpu(), scaler.std.cpu()),
                "xstd": xstd.cpu(), "gstd": gstd.cpu()}, os.path.join(out, "model.pt"))

    res = {"model": args.model, "params": nparam, "best_val_loss": best, "train_time_s": time.time() - t0,
           "sym_d4": sym, "no_vel": args.no_vel, "eval_rotated": args.eval_rotated, "yaw_only": args.yaw_only,
           "view": args.view, "frames": args.frames, "aux_w": args.aux_w, "n_train": len(tr["names"]),
           "intrinsic": args.intrinsic, "flow": args.flow, "history": hist}
    if args.flow:  # K candidates per val/test item: oracle (best-of-K by position) and mean-of-K
        for nm, x, g, y, d in (("val", xva, gva, yva, va), ("test", xte, gte, yte, te)):
            cands = []
            for i in range(0, len(x), args.bs):
                with amp:
                    z = net.sample(x[i:i + args.bs], g[i:i + args.bs], args.flow_k).float()
                cands.append(scaler.denorm(z.view(-1, 2, 7)).view(z.shape[0], args.flow_k, 2, 7))
            cands = torch.cat(cands).cpu().numpy()  # (N, K, 2, 7)
            yy = y.cpu().numpy()
            mk = np.stack([pair_metrics(cands[:, k], yy, sym=sym) for k in range(args.flow_k)], 1)  # (N, K, 5)
            best_k = mk[:, :, 0].argmin(1)
            res[f"{nm}/flow_bestK"] = summarize(mk[np.arange(len(mk)), best_k])
            res[f"{nm}/flow_meanK"] = summarize(mk.mean(1))
            if nm == "test":
                with open(os.path.join(out, "test_candidates.json"), "w") as fh:
                    json.dump([{"file": n, "candidates": [decode_forces(c) for c in cands[i]], "true": decode_forces(yy[i]),
                                "cand_metrics": mk[i].tolist()} for i, n in enumerate(d["names"])], fh)
            print(f"{nm} flow: best-of-{args.flow_k} pos {res[f'{nm}/flow_bestK']['pos_cm_mean']:.1f} cm / "
                  f"dir {res[f'{nm}/flow_bestK']['dir_deg_mean']:.1f} deg;  mean-of-K pos "
                  f"{res[f'{nm}/flow_meanK']['pos_cm_mean']:.1f} cm", flush=True)
    preds_out = []
    ntr_eval = min(len(xtr), 1024) if args.frames != "off" else len(xtr)  # frames inflate train; keep eval bounded
    Rtr_eval = _rot_from_gen(ntr_eval, torch.Generator().manual_seed(999)).to(dev) if args.view else None
    for nm, x, g, y, d, R in (("train", xtr[:ntr_eval], gtr[:ntr_eval], ytr[:ntr_eval], tr, Rtr_eval),
                              ("val", xva, gva, yva, va, Rva), ("test", xte, gte, yte, te, Rte)):
        for tta in (0, args.tta if not args.no_rot else 0):
            with amp:
                p = predict(net, x, g, scaler, tta=tta, bs=args.bs, yaw_only=args.yaw_only, R=R).cpu().numpy()
            yy = y
            if args.view:  # targets in the same camera frame as the predictions
                yy = y.clone()
                yy[:, :, 2:5] = torch.einsum("bij,bkj->bki", R, y[:, :, 2:5])
            m = pair_metrics(p, yy.cpu().numpy(), sym=sym)
            res[f"{nm}/tta{tta}"] = summarize(m)
            if nm == "test" and tta == args.tta:
                for i, name in enumerate(d["names"]):
                    preds_out.append({"file": name, "metrics": dict(zip(["pos_cm", "dir_deg", "mag_rel", "dur_s", "impulse_rel"], m[i].tolist())),
                                      "pred": decode_forces(p[i]), "true": decode_forces(yy[i].cpu().numpy()),
                                      "pred_raw": p[i].tolist()})
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    with open(os.path.join(out, "test_predictions.json"), "w") as fh:
        json.dump(preds_out, fh, indent=2)
    print(f"\n{args.model} ({nparam/1e6:.2f}M params) best val loss {best:.4f}, {time.time()-t0:.0f}s")
    for k in [k for k in res if "/tta" in k]:
        v = res[k]
        print(f"{k:12s} pos {v['pos_cm_mean']:5.1f} cm  dir {v['dir_deg_mean']:5.1f} deg  "
              f"mag {v['mag_rel_mean']:.2f}  dur {v['dur_s_mean']:.3f} s  impulse {v['impulse_rel_mean']:.2f}")


if __name__ == "__main__":
    main()
