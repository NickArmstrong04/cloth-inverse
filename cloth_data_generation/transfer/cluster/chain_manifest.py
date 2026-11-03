"""dataset_manifest.json for a chain_split.sbatch output: every <src stem>_v<k>.h5 under <out>/<split>/
gets its source sample's manifest entry with file and degraded_from set (degrade_split.py convention).
  python cluster/chain_manifest.py --src $S5 --out ${S5}_chain
"""
import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
man = json.load(open(os.path.join(args.src, "dataset_manifest.json")))
for split, entries in man["splits"].items():
    by_file = {e["file"]: e for e in entries}
    new = []
    for f in sorted(glob.glob(os.path.join(args.out, split, "*.h5"))):
        name = os.path.basename(f)
        src = re.sub(r"_v\d+\.h5$", ".h5", name)
        e = dict(by_file[src])
        e["file"], e["degraded_from"] = name, src
        new.append(e)
    print(f"{split}: {len(new)} files from {len({e['degraded_from'] for e in new})} of {len(entries)} sources")
    man["splits"][split] = new
man["chain"] = "grids through the photo chain (cluster/chain_split.sbatch): coded-print stereo render, VGGT, decode, grid_from_views --corr decoded"
json.dump(man, open(os.path.join(args.out, "dataset_manifest.json"), "w"), indent=1)
