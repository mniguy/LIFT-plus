#!/usr/bin/env python
"""Plot the gamma frontier for em_group_shrink: All-accuracy vs the split 'crater'
(worst per-split delta vs no-adapt) as gamma varies. Shows the All-vs-balance knob,
with EM-shrink (#4) and EM-group (gamma->0) as reference points.

  python scripts/plot_beta_frontier.py --root output/test_agnostic/imagenet_lt/lift+ \
      --title ImageNet-LT --out output/paper/fig_frontier_imagenet.pdf
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured_prior_adapt import run_eval

SPLITS = ["Many", "Med", "Few", "All"]


def stats(acc, cols):
    am = float(np.mean([np.nanmean(acc["em_group_shrink"][c]["All"]) for c in cols]))
    base = {s: np.mean([np.nanmean(acc["no-adapt"][c][s]) for c in cols]) for s in ["Many", "Med", "Few"]}
    crater = min(float(np.mean([np.nanmean(acc["em_group_shrink"][c][s]) for c in cols]) - base[s])
                 for s in ["Many", "Med", "Few"])
    return am, crater


def ref_point(acc, cols, m):
    am = float(np.mean([np.nanmean(acc[m][c]["All"]) for c in cols]))
    base = {s: np.mean([np.nanmean(acc["no-adapt"][c][s]) for c in cols]) for s in ["Many", "Med", "Few"]}
    crater = min(float(np.mean([np.nanmean(acc[m][c][s]) for c in cols]) - base[s])
                 for s in ["Many", "Med", "Few"])
    return am, crater


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--group-mode", default="quantile")
    ap.add_argument("--n-groups", type=int, default=5)
    args = ap.parse_args()

    logits = np.load(os.path.join(args.root, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(args.root, "y_true.npy")).astype(int)
    cn = np.load(os.path.join(args.root, "cls_num_list.npy")).astype(np.float64)
    M = ["no-adapt", "em_shrink", "em_group", "em_group_shrink"]
    common = dict(group_mode=args.group_mode, n_groups=args.n_groups,
                  trials=args.trials, total=15000, seed=0, compute_l1=False)

    gammas = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    xs, ys = [], []
    for g in gammas:
        acc, _, cols, _ = run_eval(logits, y, cn, M, 1.0, g, **common)
        am, cr = stats(acc, cols)
        xs.append(cr); ys.append(am)
    # reference points (use last acc for #4 and em_group, gamma-independent)
    sh_am, sh_cr = ref_point(acc, cols, "em_shrink")
    gr_am, gr_cr = ref_point(acc, cols, "em_group")

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot(xs, ys, "-o", color="tab:blue", label="EM-group+shrink (vary $\\gamma$)")
    for g, x, yv in zip(gammas, xs, ys):
        ax.annotate(f"$\\gamma$={g}", (x, yv), textcoords="offset points",
                    xytext=(5, -2), fontsize=7, color="tab:blue")
    ax.scatter([sh_cr], [sh_am], marker="s", s=70, color="tab:red", zorder=5, label="EM-shrink (\\#4)")
    ax.scatter([gr_cr], [gr_am], marker="^", s=80, color="tab:green", zorder=5, label="EM-group ($\\gamma{\\to}0$)")
    ax.set_xlabel("worst per-split $\\Delta$ vs no-adapt  (crater; $\\to 0$ better)")
    ax.set_ylabel("All accuracy (mean over shift priors)")
    if args.title:
        ax.set_title(args.title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=150)
    print(f"[written] {args.out} (+ .png)")
    print(f"  gammas {gammas}\n  crater {[round(x,2) for x in xs]}\n  All    {[round(v,2) for v in ys]}")
    print(f"  #4 (All={sh_am:.2f}, crater={sh_cr:.2f}) | em_group (All={gr_am:.2f}, crater={gr_cr:.2f})")


if __name__ == "__main__":
    main()
