"""Mixed clean + photo-like split (JOBS.md M20-g): hard links, no copies.

train = photo-like train files + clean train files; val/test = photo-like only.
  python cluster/make_mix.py --clean $S5 --photo ${S5}_photo --out ${S5}_mix
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--clean", required=True)
ap.add_argument("--photo", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

if os.path.exists(args.out):
    raise SystemExit(f"{args.out} exists; refusing to mix into an old set")
clean = json.load(open(os.path.join(args.clean, "dataset_manifest.json")))
photo = json.load(open(os.path.join(args.photo, "dataset_manifest.json")))
mix = dict(photo)
mix["splits"] = {s: list(e) for s, e in photo["splits"].items()}
mix["splits"]["train"] = photo["splits"]["train"] + clean["splits"]["train"]
mix["mix"] = (f"train = photo-like copies ({args.photo}) + the clean train files ({args.clean}); "
              "val/test photo-like only (cluster/make_mix.py)")
for split, entries in mix["splits"].items():
    os.makedirs(os.path.join(args.out, split))
    for e in entries:
        src_root = args.photo if (split != "train" or "degraded_from" in e) else args.clean
        os.link(os.path.join(src_root, split, e["file"]), os.path.join(args.out, split, e["file"]))
    print(f"{split}: {len(entries)} files")
json.dump(mix, open(os.path.join(args.out, "dataset_manifest.json"), "w"), indent=1)
