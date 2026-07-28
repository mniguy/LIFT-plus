#!/usr/bin/env python
"""Pool the rarity-variant seeds and test them against baseline / global centering.

Seed 0 for each variant was run earlier under a different output root (center_geom25 for tail,
center_kappa25 for kappa50/logcount); run_rarity_seeds.sh adds seeds 1-4 under rarity_seeds25.
All of them share identical settings (semantic init, scale 25, 5 ep, mda+tte, aux off), so this
just collects them by (dataset, variant, seed) and reports mean/std plus a paired t-test of the
per-seed differences against both anchors.

    python scripts/agg_rarity_seeds.py
    python scripts/agg_rarity_seeds.py --seeds 0 1 2      # partial run, while the rest is training
"""
import argparse
import glob
import os
import re
from statistics import mean, stdev

COLS = ["All", "Head", "Med", "Few"]

# variant -> {seed: run dir}, "{s}" is substituted with the seed
SOURCES = {
    "baseline": ["output/seed_ablation 25/{data}/baseline_seed{s}"],
    "global":   ["output/prompt_center25/{data}/center",                 # seed 0 only
                 "output/center_seeds25/{data}/center_seed{s}"],
    "tail":     ["output/center_geom25/{data}/tail",                     # seed 0 only
                 "output/rarity_seeds25/{data}/tail_seed{s}"],
    "kappa50":  ["output/center_kappa25/{data}/kappa50",                 # seed 0 only
                 "output/rarity_seeds25/{data}/kappa50_seed{s}"],
    "logcount": ["output/center_kappa25/{data}/logcount",                # seed 0 only
                 "output/rarity_seeds25/{data}/logcount_seed{s}"],
}
# the seed-0-only entries above have no {s}; they are valid for seed 0 exclusively
SEED0_ONLY = ("output/prompt_center25", "output/center_geom25", "output/center_kappa25")


def read_result(run_dir):
    for f in reversed(sorted(glob.glob(os.path.join(run_dir, "log-*.txt")))):
        txt = open(f, errors="ignore").read()
        m = re.findall(r"\* Many:\s*([\d.]+)%\s*Med:\s*([\d.]+)%\s*Few:\s*([\d.]+)%", txt)
        if not m:
            continue
        ov = re.findall(r"\* Overall accuracy:\s*([\d.]+)%", txt)
        head, med, few = map(float, m[-1])
        return [float(ov[-1]) if ov else float("nan"), head, med, few]
    return None


def load(variant, data, seed):
    for pat in SOURCES[variant]:
        if "{s}" not in pat and seed != 0:
            continue                      # seed-0-only source
        if "{s}" not in pat and not pat.startswith(SEED0_ONLY):
            continue
        r = read_result(pat.format(data=data, s=seed))
        if r:
            return r
    return None


def ttest(diffs):
    """One-sample t on the per-seed differences (paired by seed)."""
    n = len(diffs)
    if n < 2:
        return float("nan")
    sd = stdev(diffs)
    return mean(diffs) / (sd / n ** 0.5) if sd > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="*", default=["imagenet_lt", "places_lt"])
    args = ap.parse_args()

    for data in args.datasets:
        vals = {}                                     # variant -> {seed: [4 metrics]}
        for v in SOURCES:
            got = {s: r for s in args.seeds if (r := load(v, data, s))}
            if got:
                vals[v] = got
        if "baseline" not in vals:
            print(f"\n{data}: no baseline runs found, skipping")
            continue

        print(f"\n{'='*92}\n{data}\n{'='*92}")
        print(f"  {'variant':10} {'n':>2} " + " ".join(f"{c:>14}" for c in COLS))
        for v, got in vals.items():
            seeds = sorted(got)
            col = lambda i: [got[s][i] for s in seeds]
            cells = " ".join(f"{mean(col(i)):8.2f}±{(stdev(col(i)) if len(seeds) > 1 else 0):4.2f}"
                             for i in range(4))
            print(f"  {v:10} {len(seeds):2d} {cells}   seeds={seeds}")

        for anchor in ("baseline", "global"):
            if anchor not in vals:
                continue
            print(f"\n  --- Δ vs {anchor} (paired by seed; t on per-seed differences) ---")
            print(f"  {'variant':10} {'n':>2} " + " ".join(f"{c:>16}" for c in COLS))
            for v, got in vals.items():
                if v == anchor:
                    continue
                shared = sorted(set(got) & set(vals[anchor]))
                if not shared:
                    continue
                cells = []
                for i in range(4):
                    d = [got[s][i] - vals[anchor][s][i] for s in shared]
                    cells.append(f"{mean(d):+7.2f}(t{ttest(d):+5.1f})")
                print(f"  {v:10} {len(shared):2d} " + " ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
