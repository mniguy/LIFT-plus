#!/usr/bin/env python
"""Per-split (all/head/med/few) accuracy for each variant subdir under --root,
plus Δ vs a baseline variant. Generic dev tool (gate_controls / gate_collapse / ...).

Usage:
    python scripts/summarize_splits.py --root output/gate_collapse/imagenet_lt --baseline fixed
"""
import argparse
import glob
import os
import sys

import numpy as np


def splits(c):
    c = np.asarray(c)
    return (c > 100), ((c >= 20) & (c <= 100)), (c < 20)


def row(d, h, m, fw):
    a = np.load(os.path.join(d, "cls_accs.npy")).astype(float)
    return np.array([a.mean(), a[h].mean(), a[m].mean(), a[fw].mean()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", default="fixed")
    ap.add_argument("--order", nargs="*", default=None)
    args = ap.parse_args()

    variants = {os.path.basename(d): d
                for d in sorted(glob.glob(os.path.join(args.root, "*")))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "cls_accs.npy"))}
    if not variants:
        sys.exit(f"no variants (with cls_accs.npy) under {args.root}")

    num = np.load(os.path.join(next(iter(variants.values())), "cls_num_list.npy"))
    h, m, fw = splits(num)
    cols = ["all", "head", "med", "few"]

    if args.order:
        order = [v for v in args.order if v in variants]
    elif args.baseline in variants:
        order = [args.baseline] + [v for v in variants if v != args.baseline]
    else:
        order = list(variants)

    R = {v: row(variants[v], h, m, fw) for v in order}
    print(f"\n=== {args.root}  (head={int(h.sum())} med={int(m.sum())} few={int(fw.sum())}) ===")
    print(f"{'variant':12s} " + " ".join(f"{c:>8s}" for c in cols))
    for v in order:
        print(f"{v:12s} " + " ".join(f"{R[v][i]:8.3f}" for i in range(4)))

    if args.baseline in R:
        b = R[args.baseline]
        print(f"\n-- Δ vs {args.baseline} --")
        for v in order:
            if v != args.baseline:
                print(f"{v:12s} " + " ".join(f"{R[v][i] - b[i]:+8.3f}" for i in range(4)))


if __name__ == "__main__":
    main()
