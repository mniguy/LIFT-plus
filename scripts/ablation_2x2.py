#!/usr/bin/env python
"""E7 -- 2x2 grouping x shrinkage ablation, to ISOLATE grouping as the causal lever.

The four cells are already-implemented estimators (same decision rule, differ only in pi):
                 no-shrink        + KL-gated shrink
  per-class :    em               em_shrink   (= #4)
  grouped   :    em_group         em_group_shrink

Reads the SAME saved logits as compare_baselines / make_beta_tables and reports All
accuracy (mean +/- std over seed roots) per cell, plus the two MARGINAL effects:
  grouping effect = grouped - per-class   (averaged over the shrink axis)
  shrink   effect = +shrink  - no-shrink  (averaged over the grouping axis)

Claim it closes (C4): grouping's marginal >> shrink's marginal, and shrink ALONE
(em_shrink) still collapses at large C (iNat) -- so grouping is the cause, shrink is a
secondary no-shift-protection knob.  PASS if grouping-marginal is positive on every
dataset and dominant on iNat; KILL if shrink alone matches em_group on iNat.

  python scripts/ablation_2x2.py \
    ImageNet-LT:output/test_agnostic_ms/imagenet_lt/seed0,...seed1,...seed2 \
    Places-LT:...  iNat2018:output/test_agnostic/inat2018/lift+ \
    --out output/paper/tables_ablation_2x2.tex
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured_prior_adapt import run_eval

# cell -> estimator name; grid[grouping][shrink]
GRID = {"per-class": {"no-shrink": "em", "shrink": "em_shrink"},
        "grouped":   {"no-shrink": "em_group", "shrink": "em_group_shrink"}}
METHODS = ["no-adapt", "em", "em_shrink", "em_group", "em_group_shrink"]
GROUPINGS = ["per-class", "grouped"]
SHRINKS = ["no-shrink", "shrink"]


def per_root(root, trials, total, gmode, K):
    logits = np.load(os.path.join(root, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(root, "y_true.npy")).astype(int)
    cn = np.load(os.path.join(root, "cls_num_list.npy")).astype(np.float64)
    C = logits.shape[1]
    tr = min(trials, 3) if C > 3000 else trials
    acc, _, cols, _ = run_eval(logits, y, cn, METHODS, 1.0, 1.0, gmode, K,
                               tr, total, 0, compute_l1=False)
    allm = {m: float(np.mean([np.nanmean(acc[m][c]["All"]) for c in cols])) for m in METHODS}
    base_few = np.mean([np.nanmean(acc["no-adapt"][c]["Few"]) for c in cols])
    few_d = {m: float(np.mean([np.nanmean(acc[m][c]["Few"]) for c in cols]) - base_few)
             for m in METHODS}
    return allm, few_d, C


def fmt(vals):
    v = np.array(vals)
    return f"{v.mean():.2f}" if len(v) < 2 else f"{v.mean():.2f}\\tiny$\\pm${v.std(ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="Name:root1,root2,... per dataset")
    ap.add_argument("--out", default="output/paper/tables_ablation_2x2.tex")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--total", type=int, default=15000)
    ap.add_argument("--group-mode", default="quantile")
    ap.add_argument("--n-groups", type=int, default=5)
    args = ap.parse_args()

    data = []  # (name, allmean{m:[per-seed]}, few_d{m:[..]}, C)
    for spec in args.specs:
        name, roots = spec.split(":", 1)
        roots = roots.split(",")
        am = {m: [] for m in METHODS}; fd = {m: [] for m in METHODS}; C = None
        for r in roots:
            a, f, C = per_root(r, args.trials, args.total, args.group_mode, args.n_groups)
            for m in METHODS:
                am[m].append(a[m]); fd[m].append(f[m])
        data.append((name, am, fd, C))

    # ---- console: 2x2 grid + marginals per dataset ----
    for name, am, fd, C in data:
        print(f"\n=== {name} (C={C}) 2x2 All accuracy ===")
        print(f"  {'':10s} {'no-shrink':>16s} {'+shrink':>16s}")
        for g in GROUPINGS:
            cells = [fmt(am[GRID[g][s]]) for s in SHRINKS]
            print(f"  {g:10s} " + " ".join(f"{c:>16s}" for c in cells))
        grp = np.mean([np.mean(am[GRID['grouped'][s]]) - np.mean(am[GRID['per-class'][s]])
                       for s in SHRINKS])
        shr = np.mean([np.mean(am[GRID[g]['shrink']]) - np.mean(am[GRID[g]['no-shrink']])
                       for g in GROUPINGS])
        print(f"  marginal: grouping {grp:+.2f}   shrink {shr:+.2f}"
              f"   (per-class+shrink[#4] Few-d {np.mean(fd['em_shrink']):+.2f}, "
              f"grouped+shrink Few-d {np.mean(fd['em_group_shrink']):+.2f})")

    # ---- LaTeX: cells x datasets (All), then marginal-effect table ----
    L = [f"% E7 2x2 grouping x shrinkage  group={args.group_mode}/{args.n_groups} "
         f"trials={args.trials} (mean over forward/uniform/backward; +/- over seeds)",
         "\\begin{tabular}{ll" + "c" * len(data) + "}\\toprule",
         "Grouping & Shrink & " + " & ".join(n for n, _, _, _ in data) + "\\\\\\midrule"]
    for g in GROUPINGS:
        for s in SHRINKS:
            tag = " (\\#4)" if (g, s) == ("per-class", "shrink") else ""
            cells = [fmt(am[GRID[g][s]]) for _, am, _, _ in data]
            L.append(f"{g if s == 'no-shrink' else ''} & {s}{tag} & " + " & ".join(cells) + "\\\\")
        L.append("\\midrule")
    # marginal effects row
    grp_row, shr_row = [], []
    for _, am, _, _ in data:
        grp_row.append(f"{np.mean([np.mean(am[GRID['grouped'][s]]) - np.mean(am[GRID['per-class'][s]]) for s in SHRINKS]):+.2f}")
        shr_row.append(f"{np.mean([np.mean(am[GRID[g]['shrink']]) - np.mean(am[GRID[g]['no-shrink']]) for g in GROUPINGS]):+.2f}")
    L.append("\\multicolumn{2}{l}{marginal: grouping} & " + " & ".join(grp_row) + "\\\\")
    L.append("\\multicolumn{2}{l}{marginal: shrink} & " + " & ".join(shr_row) + "\\\\")
    L.append("\\bottomrule\\end{tabular}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
