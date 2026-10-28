"""Train an inverse model (final cloth state -> two force vectors)."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from data import TargetScaler, augment, decode_forces, load_split, random_rotation
from metrics import pair_metrics, summarize
from models import build, pair_loss


def predict(model, x, g, scaler, tta=0):
    """Return raw-scale predictions (N,2,7). With tta>0, average over random
    world rotations (directions rotated back; u,v,mag,dur rotation-invariant)."""
    model.eval()
    with torch.no_grad():
        outs = []
        p, _ = model(x, g)
        outs.append(scaler.denorm(p))
        for _ in range(tta):
            R = random_rotation(x.shape[0], x.device)
            xr = torch.cat([torch.einsum("bij,bjhw->bihw", R, x[:, 0:3]),
                            torch.einsum("bij,bjhw->bihw", R, x[:, 3:6])], 1)
            gr = torch.cat([torch.einsum("bij,bj->bi", R, g[:, 0:3]),
                            torch.einsum("bij,bj->bi", R, g[:, 3:6])], 1)
            p = scaler.denorm(model(xr, gr)[0])
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
    ap.add_argument("--model", default="cnn_reg", choices=["cnn_reg", "cnn_heat"])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-rot", action="store_true",
                    help="dihedral (material-symmetry) augmentation only, no 3D world rotation")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or (f"runs/{args.model}_s{args.seed}_e{args.epochs}"
                       + ("_noaug" if args.no_aug else "") + ("_norot" if args.no_rot else ""))
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, te = (load_split(args.root, s) for s in ("train", "val", "test"))
    scaler = TargetScaler(tr["y"]).to(dev)
    xstd = tr["x"].std((0, 2, 3), keepdim=True).to(dev)  # per-channel input scale
    gstd = tr["g"].std(0, keepdim=True).to(dev) + 1e-6

    def prep(d):
        return d["x"].to(dev), d["g"].to(dev), d["y"].to(dev)

    xtr, gtr, ytr = prep(tr)
    xva, gva, yva = prep(va)
    xte, gte, yte = prep(te)

    uv_scale = (scaler.mean[:2], scaler.std[:2])
    model = build(args.model, uv_scale=uv_scale, w=args.width).to(dev)
    nparam = sum(p.numel() for p in model.parameters())

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, g):
            return self.m(x / xstd, g / gstd)

    net = Wrapped(model)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(xtr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.1)
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xtr), device=dev)
        tot = 0.0
        for k in range(steps_per_epoch):
            idx = perm[k * args.bs:(k + 1) * args.bs]
            x, g, y = xtr[idx], gtr[idx], ytr[idx]
            if not args.no_aug:
                x, g, y = augment(x, g, y, rotate=not args.no_rot)
            pred, _ = net(x, g)
            loss = pair_loss(pred, scaler.norm(y))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                vl = pair_loss(net(xva, gva)[0], scaler.norm(yva)).item()
                tl = pair_loss(net(xtr, gtr)[0], scaler.norm(ytr)).item()
            hist.append({"epoch": ep + 1, "train_aug": tot / steps_per_epoch, "train": tl, "val": vl})
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}
            if (ep + 1) % 25 == 0:
                print(f"ep {ep+1:4d} loss_aug {tot/steps_per_epoch:.4f} train {tl:.4f} val {vl:.4f} "
                      f"best {best:.4f} [{time.time()-t0:.0f}s]", flush=True)
    net.load_state_dict(best_state)
    torch.save({"state": best_state, "args": vars(args), "scaler": (scaler.mean.cpu(), scaler.std.cpu()),
                "xstd": xstd.cpu(), "gstd": gstd.cpu()}, os.path.join(out, "model.pt"))

    res = {"model": args.model, "params": nparam, "best_val_loss": best, "train_time_s": time.time() - t0,
           "history": hist}
    preds_out = []
    for nm, x, g, y, d in (("train", xtr, gtr, ytr, tr), ("val", xva, gva, yva, va), ("test", xte, gte, yte, te)):
        for tta in (0, args.tta if not args.no_rot else 0):
            p = predict(net, x, g, scaler, tta=tta).cpu().numpy()
            m = pair_metrics(p, y.cpu().numpy())
            res[f"{nm}/tta{tta}"] = summarize(m)
            if nm == "test" and tta == args.tta:
                for i, name in enumerate(d["names"]):
                    preds_out.append({"file": name, "metrics": dict(zip(["pos_cm", "dir_deg", "mag_rel", "dur_s", "impulse_rel"], m[i].tolist())),
                                      "pred": decode_forces(p[i]), "true": decode_forces(y[i].cpu().numpy()),
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
