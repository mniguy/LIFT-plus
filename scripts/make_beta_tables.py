#!/usr/bin/env python
"""Emit the beta result LaTeX tables from saved logits. Multi-seed ready:
pass several comma-separated seed roots per dataset and it reports mean +/- std
across seeds (resampling variance is averaged inside each root by run_eval).

  # single seed (now):
  python scripts/make_beta_tables.py \
    ImageNet-LT:output/test_agnostic/imagenet_lt/lift+ \
    Places-LT:output/test_agnostic/places_lt/lift+ \
    --out output/paper/tables_beta.tex

  # multi-seed (after dumping seed1,seed2 logits -- see scripts/run_dump_logits.sh):
  python scripts/make_beta_tables.py \
    ImageNet-LT:output/test_agnostic_ms/imagenet_lt/seed0,.../seed1,.../seed2 ...
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured_prior_adapt import run_eval

METHODS = ["no-adapt", "bbse", "em_shrink", "em_group", "em_group_shrink", "oracle"]
PRETTY = {"no-adapt": "LIFT+ (no-adapt)", "bbse": "BBSE", "em_shrink": "EM-shrink (\\#4)",
          "em_group": "EM-group (ours)", "em_group_shrink": "EM-group+shrink (ours)",
          "oracle": "Oracle prior"}
SPLITS = ["Many", "Med", "Few", "All"]


def per_root_stats(root, trials, total, gmode, K):
    logits = np.load(os.path.join(root, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(root, "y_true.npy")).astype(int)
    cn = np.load(os.path.join(root, "cls_num_list.npy")).astype(np.float64)
    acc, _, cols, _ = run_eval(logits, y, cn, METHODS, 1.0, 1.0, gmode, K,
                               trials, total, 0, compute_l1=False)
    allmean = {m: float(np.mean([np.nanmean(acc[m][c]["All"]) for c in cols])) for m in METHODS}
    base = {s: np.mean([np.nanmean(acc["no-adapt"][c][s]) for c in cols]) for s in SPLITS}
    delta = {m: {s: float(np.mean([np.nanmean(acc[m][c][s]) for c in cols]) - base[s]) for s in SPLITS}
             for m in METHODS}
    return allmean, delta


def agg(roots, trials, total, gmode, K):
    am = {m: [] for m in METHODS}
    dl = {m: {s: [] for s in SPLITS} for m in METHODS}
    for r in roots:
        a, d = per_root_stats(r, trials, total, gmode, K)
        for m in METHODS:
            am[m].append(a[m])
            for s in SPLITS:
                dl[m][s].append(d[m][s])
    return am, dl


def fmt(vals):
    v = np.array(vals)
    return f"{v.mean():.2f}" if len(v) < 2 else f"{v.mean():.2f}\\tiny$\\pm${v.std(ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="Name:root1,root2,... per dataset")
    ap.add_argument("--out", default="output/paper/tables_beta.tex")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--total", type=int, default=15000)
    ap.add_argument("--group-mode", default="quantile")
    ap.add_argument("--n-groups", type=int, default=5)
    args = ap.parse_args()

    datasets = []
    for spec in args.specs:
        name, roots = spec.split(":", 1)
        datasets.append((name, roots.split(",")))

    results = {name: agg(roots, args.trials, args.total, args.group_mode, args.n_groups)
               for name, roots in datasets}
    nseed = max(len(r) for _, r in datasets)
    L = []
    L.append(f"% beta tables  group={args.group_mode}/{args.n_groups}  trials={args.trials}  "
             f"seeds={nseed}  (mean over forward/uniform/backward)")

    # T1: All accuracy per dataset
    L.append("% ===== T1: test-agnostic All accuracy (mean over shift priors) =====")
    L.append("\\begin{tabular}{l" + "c" * len(datasets) + "}\\toprule")
    L.append("Method & " + " & ".join(n for n, _ in datasets) + "\\\\\\midrule")
    for m in METHODS:
        row = [PRETTY[m]] + [fmt(results[n][0][m]) for n, _ in datasets]
        if m == "oracle":
            L.append("\\midrule")
        L.append(" & ".join(row) + "\\\\")
    L.append("\\bottomrule\\end{tabular}")

    # T2: split breakdown (delta vs no-adapt) for the two headline methods on each dataset
    L.append("")
    L.append("% ===== T2: per-split delta vs LIFT+ no-adapt (mean over shift priors) =====")
    L.append("\\begin{tabular}{ll" + "c" * len(SPLITS) + "}\\toprule")
    L.append("Dataset & Method & " + " & ".join(SPLITS) + "\\\\\\midrule")
    for n, _ in datasets:
        for m in ["em_shrink", "em_group", "em_group_shrink"]:
            row = [n if m == "em_shrink" else "", PRETTY[m]]
            row += [("$" + ("+" if np.mean(results[n][1][m][s]) >= 0 else "") +
                     fmt(results[n][1][m][s]) + "$") for s in SPLITS]
            L.append(" & ".join(row) + "\\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule\\end{tabular}"

    tex = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(tex)
    print(tex)
    print(f"\n[written] {args.out}  ({nseed} seed(s) per dataset)")


if __name__ == "__main__":
    main()
