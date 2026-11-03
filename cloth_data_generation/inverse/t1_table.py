"""Table of the photo-chain evals in one photo set (t1_chain.sbatch + scale-fix outputs), stdlib only:
  python3 t1_table.py <photos dir> <run> [<run> ...]
Columns: oracle grid, hidden (true visible + fill), photo (oracle correspondences), photo_dec (fully
photo-derived), each without / with the scale fix; pos cm / dir deg means (medians for photo_dec)."""
import json
import os
import sys

P, runs = sys.argv[1], sys.argv[2:]


def get(run, tag, var):
    names = {"or": ["oraclecorr", "grids"], "dec": ["decoded", "grids_dec"],
             "or_fix": ["oraclecorr_fix", "grids_fix"], "dec_fix": ["decoded_fix", "grids_dec_fix"]}[tag]
    for n in names:
        f = os.path.join(P, f"eval_{run}_{n}.json")
        if os.path.exists(f):
            for k, r in json.load(open(f))["rows"].items():
                if k.split()[-1] == var or k == var:
                    return r
    return None


def fmt(r, med=False):
    if r is None:
        return "-"
    s = f"{r['pos_cm_mean']:.1f} / {r['dir_deg_mean']:.1f}"
    return s + (f" (med {r['pos_cm_median']:.1f} / {r['dir_deg_median']:.1f})" if med else "")


print("| model | n | oracle grid | hidden | photo (oracle corr.) | photo + fix | photo_dec | **photo_dec + fix** |")
print("|---|---|---|---|---|---|---|---|")
for run in runs:
    o = get(run, "or", "oracle")
    print(f"| {run} | {o['n'] if o else '-'} | {fmt(o)} | {fmt(get(run, 'or', 'hidden'))} | {fmt(get(run, 'or', 'photo'))} | "
          f"{fmt(get(run, 'or_fix', 'photo'))} | {fmt(get(run, 'dec', 'photo_dec'))} | {fmt(get(run, 'dec_fix', 'photo_dec'), True)} |")
