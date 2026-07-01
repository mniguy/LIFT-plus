#!/usr/bin/env python
"""Sweeps for direction beta: (1) data-driven grouping robustness, (2) tau, (3) gamma.

Imports run_eval from structured_prior_adapt so the estimator logic stays single-source.

  python scripts/sweep_beta.py --root output/test_agnostic/imagenet_lt/lift+ --trials 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured_prior_adapt import run_eval


def all_mean(acc, m, cols):
    return float(np.mean([np.nanmean(acc[m][c]["All"]) for c in cols]))


def split_delta(acc, m, cols, s):
    base = np.mean([np.nanmean(acc["no-adapt"][c][s]) for c in cols])
    return float(np.mean([np.nanmean(acc[m][c][s]) for c in cols]) - base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--total", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logits = np.load(os.path.join(args.root, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(args.root, "y_true.npy")).astype(int)
    cls_num = np.load(os.path.join(args.root, "cls_num_list.npy")).astype(np.float64)
    common = dict(trials=args.trials, total=args.total, seed=args.seed, compute_l1=False)

    print(f"\n################ SWEEPS  {args.root}  (C={logits.shape[1]}, trials={args.trials}) ################")

    # (1) data-driven grouping robustness: em_group vs the circular 'fixed' baseline + #4.
    print("\n[A] grouping robustness -- em_group All-mean (de-circularized = quantile/logwidth)")
    print(f"  {'grouping':16s} {'All-mean':>9s} {'Few-d':>8s} {'Med-d':>8s}")
    ref_methods = ["no-adapt", "em_shrink", "em_group"]
    for mode, K in [("fixed", 3), ("quantile", 3), ("quantile", 5), ("quantile", 10),
                    ("logwidth", 5), ("logwidth", 10)]:
        acc, _, cols, _ = run_eval(logits, y, cls_num, ref_methods, 1.0, 1.0, mode, K, **common)
        print(f"  {mode+'/'+str(K):16s} {all_mean(acc,'em_group',cols):9.3f} "
              f"{split_delta(acc,'em_group',cols,'Few'):+8.2f} {split_delta(acc,'em_group',cols,'Med'):+8.2f}")
    print(f"  (#4 em_shrink ref All-mean = {all_mean(acc,'em_shrink',cols):.3f}, "
          f"no-adapt = {all_mean(acc,'no-adapt',cols):.3f})")

    # fix a good data-driven grouping for the tau/gamma sweeps
    GMODE, GK = "quantile", 5
    M = ["no-adapt", "em_group", "em_group_shrink"]

    # (2) tau sweep (gamma fixed = 1)
    print(f"\n[B] tau sweep (group={GMODE}/{GK}, gamma=1) -- All-mean")
    print(f"  {'tau':>5s} {'em_group':>10s} {'em_grp_shr':>11s}")
    for tau in [0.5, 1.0, 1.5, 2.0]:
        acc, _, cols, _ = run_eval(logits, y, cls_num, M, tau, 1.0, GMODE, GK, **common)
        print(f"  {tau:5.1f} {all_mean(acc,'em_group',cols):10.3f} {all_mean(acc,'em_group_shrink',cols):11.3f}")

    # (3) gamma sweep (tau fixed = 1) -- All-mean + Few-delta (does more shrink balance splits?)
    print(f"\n[C] gamma sweep (group={GMODE}/{GK}, tau=1) -- em_group_shrink")
    print(f"  {'gamma':>5s} {'All-mean':>9s} {'Many-d':>8s} {'Med-d':>8s} {'Few-d':>8s}")
    for g in [0.3, 0.5, 1.0, 2.0, 3.0]:
        acc, _, cols, _ = run_eval(logits, y, cls_num, M, 1.0, g, GMODE, GK, **common)
        print(f"  {g:5.1f} {all_mean(acc,'em_group_shrink',cols):9.3f} "
              f"{split_delta(acc,'em_group_shrink',cols,'Many'):+8.2f} "
              f"{split_delta(acc,'em_group_shrink',cols,'Med'):+8.2f} "
              f"{split_delta(acc,'em_group_shrink',cols,'Few'):+8.2f}")


if __name__ == "__main__":
    main()
