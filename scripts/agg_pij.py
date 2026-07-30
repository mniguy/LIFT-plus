#!/usr/bin/env python
"""Pool and test the P/I/J cells produced by run_pij_close.sh.

Two tables per dataset:
  FROZEN     -- P2 with its negative control. frozen center should show a large Few gain over
                frozen baseline; frozen randdir should show ~none. That contrast is what makes
                the freeze intervention selective rather than "any perturbation helps".
  TRAINABLE  -- the J controls at multiple seeds, so the selectivity claim carries error bars.
                Seed 0 is pooled from output/center_control25 (identical settings).

    python scripts/agg_pij.py
    python scripts/agg_pij.py --seeds 0 1        # partial, while runs are still training
"""
import argparse
import glob
import os
import re
from statistics import mean, stdev

COLS = ["All", "Head", "Med", "Few"]

FROZEN = {  # variant -> path patterns tried in order ({data}, {s})
    "baseline": ["output/pij_frozen25/{data}/baseline_seed{s}", "output/freeze_center25/{data}/baseline@0"],
    "center":   ["output/pij_frozen25/{data}/center_seed{s}",   "output/freeze_center25/{data}/center@0"],
    "randdir":  ["output/pij_frozen25/{data}/randdir_seed{s}"],
}
TRAINABLE = {
    "baseline":      ["output/seed_ablation 25/{data}/baseline_seed{s}"],
    "center":        ["output/prompt_center25/{data}/center@0", "output/center_seeds25/{data}/center_seed{s}"],
    "randdir":       ["output/pij_control25/{data}/randdir_seed{s}",       "output/center_control25/{data}/randdir@0"],
    "headonly":      ["output/pij_control25/{data}/headonly_seed{s}",      "output/center_control25/{data}/headonly@0"],
    "perclass_rand": ["output/pij_control25/{data}/perclass_rand_seed{s}", "output/center_control25/{data}/perclass_rand@0"],
}


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


def load(table, variant, data, seed):
    """'path@0' marks a legacy dir that is only valid for seed 0."""
    for pat in table[variant]:
        if pat.endswith("@0"):
            if seed != 0:
                continue
            pat = pat[:-2]
        r = read_result(pat.format(data=data, s=seed))
        if r:
            return r
    return None


def ttest(d):
    n = len(d)
    if n < 2:
        return float("nan")
    sd = stdev(d)
    return mean(d) / (sd / n ** 0.5) if sd > 0 else float("inf")


def report(title, table, data, seeds, anchor="baseline"):
    vals = {}
    for v in table:
        got = {s: r for s in seeds if (r := load(table, v, data, s))}
        if got:
            vals[v] = got
    if anchor not in vals:
        print(f"\n  {title}: anchor '{anchor}' not found -- skipped")
        return

    print(f"\n  {title}")
    print(f"    {'variant':16} {'n':>2} " + " ".join(f"{c:>14}" for c in COLS))
    for v, got in vals.items():
        ss = sorted(got)
        col = lambda i: [got[s][i] for s in ss]
        cells = " ".join(f"{mean(col(i)):8.2f}±{(stdev(col(i)) if len(ss) > 1 else 0):4.2f}" for i in range(4))
        print(f"    {v:16} {len(ss):2d} {cells}")

    print(f"\n    Δ vs {anchor} (paired by seed)")
    print(f"    {'variant':16} {'n':>2} " + " ".join(f"{c:>16}" for c in COLS))
    for v, got in vals.items():
        if v == anchor:
            continue
        sh = sorted(set(got) & set(vals[anchor]))
        if not sh:
            continue
        cells = []
        for i in range(4):
            d = [got[s][i] - vals[anchor][s][i] for s in sh]
            cells.append(f"{mean(d):+7.2f}(t{ttest(d):+5.1f})")
        print(f"    {v:16} {len(sh):2d} " + " ".join(f"{c:>16}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--datasets", nargs="*", default=["imagenet_lt", "places_lt"])
    a = ap.parse_args()
    for data in a.datasets:
        print(f"\n{'='*92}\n{data}\n{'='*92}")
        report("FROZEN classifier (P2 + its negative control)", FROZEN, data, a.seeds)
        report("TRAINABLE (J selectivity, multi-seed)", TRAINABLE, data, a.seeds)
    print("\nreading:")
    print("  frozen: center >> baseline on Few, randdir ~ baseline  -> the freeze effect is specific")
    print("          to removing mu, not to perturbing the init.")
    print("  trainable: every J stays at/below baseline Few with error bars -> I's selectivity holds.")


if __name__ == "__main__":
    main()
