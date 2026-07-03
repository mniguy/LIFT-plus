#!/usr/bin/env python
"""Head-to-head vs LSC (ICML'24, generalized-BBSE / self-training per-class prior).

Reimplements LSC's core estimator faithfully from github.com/Stomach-ache/label-shift-correction
(methods/pda.py): accumulate the (confidence-filtered) mean softmax prediction as a FULL
per-class prior, applied as a logit adjustment, iterated as self-training. NO grouping,
NO shrinkage -- so the C/N variance argument predicts it, like plain EM, collapses at
large class counts. This script tests exactly that on the saved LIFT+ logits, under the
same forward/uniform/backward protocol and the SAME decision rule as our methods.

  python scripts/lsc_baseline.py --root output/test_agnostic/inat2018/lift+ --trials 2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured_prior_adapt import (softmax, make_priors, resample_idx, em,
                                     shrink_to_uniform, split_masks, build_groups)


def lsc_pi(logits, tau=1.0, thresh=0.0, iters=50, tol=1e-6):
    """LSC-style estimator: self-trained, confidence-filtered per-class prediction marginal."""
    N, C = logits.shape
    logu = np.log(1.0 / C)
    pi = np.full(C, 1.0 / C)
    for _ in range(iters):
        probs = softmax(logits + tau * (np.log(pi) - logu))
        conf = probs.max(1) > thresh
        if conf.sum() == 0:
            conf = np.ones(N, bool)
        new = probs[conf].mean(0)
        new /= new.sum()
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    return pi


def logadj(pi, C):
    return np.log(np.clip(pi, 1e-12, None)) - np.log(1.0 / C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--group-mode", default="quantile")
    ap.add_argument("--n-groups", type=int, default=5)
    ap.add_argument("--lsc-thresh", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--total", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logits = np.load(f"{args.root}/logits.npy").astype(np.float64)
    y = np.load(f"{args.root}/y_true.npy").astype(int)
    cn = np.load(f"{args.root}/cls_num_list.npy").astype(np.float64)
    C = logits.shape[1]
    masks = split_masks(cn)
    groups = build_groups(cn, args.group_mode, args.n_groups)
    priors = make_priors(cn)
    cols = ["forward", "uniform", "backward"]
    splits = ["Many", "Med", "Few", "All"]
    methods = ["no-adapt", "per-class EM", "LSC (ICML'24)", "GPA", "GPA-S", "oracle"]
    rng = np.random.default_rng(args.seed)
    by = [np.where(y == c)[0] for c in range(C)]

    acc = {m: {c: {s: [] for s in splits} for c in cols} for m in methods}
    for _ in range(args.trials):
        for c in cols:
            sel = resample_idx(by, priors[c], args.total, rng)
            zl, yl = logits[sel], y[sel]
            probs = softmax(zl)
            pi_em = em(probs)
            pi_grp = em(probs, groups=groups)
            adj = {
                "no-adapt": np.zeros(C),
                "per-class EM": logadj(pi_em, C),
                "LSC (ICML'24)": logadj(lsc_pi(zl, args.tau, args.lsc_thresh), C),
                "GPA": logadj(pi_grp, C),
                "GPA-S": logadj(shrink_to_uniform(pi_grp, args.gamma), C),
                "oracle": logadj(priors[c], C),
            }
            for m in methods:
                pred = (zl + args.tau * adj[m]).argmax(1)
                ok = (pred == yl)
                for s, msk in masks.items():
                    idx = msk[yl]
                    acc[m][c][s].append(ok[idx].mean() * 100 if idx.sum() else np.nan)
                acc[m][c]["All"].append(ok.mean() * 100)

    cell = np.nanmean
    print(f"\n=== LSC head-to-head  {args.root}  (C={C}, trials={args.trials}) ===")
    print("\n[All accuracy, mean over forward/uniform/backward]")
    base_all = np.mean([cell(acc['no-adapt'][c]['All']) for c in cols])
    for m in methods:
        v = [cell(acc[m][c]["All"]) for c in cols]
        d = np.mean(v) - base_all
        print(f"  {m:16s} F/U/B = {v[0]:6.2f} {v[1]:6.2f} {v[2]:6.2f}   mean {np.mean(v):6.2f}  (Δ {d:+.2f})")
    print("\n[Δ vs no-adapt by split, mean/3]")
    print(f"  {'method':16s} {'Many':>7s} {'Med':>7s} {'Few':>7s} {'All':>7s}")
    base = {s: np.mean([cell(acc['no-adapt'][c][s]) for c in cols]) for s in splits}
    for m in methods:
        ds = [np.mean([cell(acc[m][c][s]) for c in cols]) - base[s] for s in splits]
        print(f"  {m:16s} " + " ".join(f"{x:+7.2f}" for x in ds))


if __name__ == "__main__":
    main()
