#!/usr/bin/env python
"""Method ablation verdict: per-split accuracy (mean±std over seeds) for each rung,
and the PAIRED Δ(target - baseline) across matched seeds that tells whether the
method beats LIFT+ OUTSIDE the run-noise band.

Run dirs are named "<rung>_seed<N>" (or "<rung>" = single seed).

Usage:
    python scripts/analyze_method_ablation.py --root output/method_ablation/imagenet_lt \
        --baseline lift+ --target full
"""
import argparse
import glob
import os
import sys

import numpy as np


def splits(cls_num):
    c = np.asarray(cls_num)
    return (c > 100), ((c >= 20) & (c <= 100)), (c < 20)


def discover(root):
    rungs = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "cls_accs.npy")):
            continue
        base = os.path.basename(d)
        name, seed = base.rsplit("_seed", 1) if "_seed" in base else (base, "0")
        rungs.setdefault(name, {})[seed] = d
    return rungs


def split_accs(d, head, med, few):
    a = np.load(os.path.join(d, "cls_accs.npy")).astype(float)
    return np.array([a.mean(), a[head].mean(), a[med].mean(), a[few].mean()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", default="lift+")
    ap.add_argument("--target", default="full")
    ap.add_argument("--order", nargs="*", default=None, help="rung display order")
    args = ap.parse_args()

    rungs = discover(args.root)
    if not rungs:
        sys.exit(f"no rungs (with cls_accs.npy) found under {args.root}")

    any_dir = next(iter(next(iter(rungs.values())).values()))
    num = np.load(os.path.join(any_dir, "cls_num_list.npy"))
    head, med, few = splits(num)
    cols = ["all", "head", "med", "few"]

    order = args.order or sorted(rungs.keys(), key=lambda r: (r != args.baseline, r))

    print(f"\n=== {args.root}  (head={int(head.sum())} med={int(med.sum())} few={int(few.sum())}) ===")
    print(f"{'rung':12s} {'seeds':>5s}   " + "   ".join(f"{c:>12s}" for c in cols))
    per_rung = {}
    for r in order:
        seeds = sorted(rungs[r])
        M = np.stack([split_accs(rungs[r][s], head, med, few) for s in seeds], 0)  # [S,4]
        per_rung[r] = (seeds, M)
        mean = M.mean(0)
        std = M.std(0, ddof=1) if len(seeds) > 1 else np.zeros(4)
        cells = "   ".join(f"{mean[i]:6.3f}±{std[i]:4.2f}" for i in range(4))
        print(f"{r:12s} {len(seeds):5d}   {cells}")

    # ---- paired verdict: target vs baseline on matched seeds ----
    if args.baseline not in per_rung or args.target not in per_rung:
        print(f"\n[skip verdict] need both '{args.baseline}' and '{args.target}' rungs.")
        return
    sb, Mb = per_rung[args.baseline]
    st, Mt = per_rung[args.target]
    common = sorted(set(sb) & set(st))
    if not common:
        print("\n[skip verdict] no matched seeds between baseline and target.")
        return

    ib = [sb.index(s) for s in common]
    it = [st.index(s) for s in common]
    D = Mt[it] - Mb[ib]  # [C,4] per-seed paired delta
    md = D.mean(0)
    sd = D.std(0, ddof=1) if len(common) > 1 else np.zeros(4)
    se = sd / np.sqrt(len(common)) if len(common) > 1 else np.zeros(4)

    print(f"\n--- paired Δ ({args.target} − {args.baseline}), {len(common)} matched seed(s) {common} ---")
    print(f"{'mean Δ±std':12s}   " + "   ".join(f"{md[i]:+6.3f}±{sd[i]:4.2f}" for i in range(4)))
    if len(common) > 1:
        flags = []
        for i, c in enumerate(cols):
            if md[i] > 2 * se[i] and md[i] > 0:
                flags.append(f"{c}: WIN (outside noise)")
            elif md[i] < -2 * se[i]:
                flags.append(f"{c}: LOSE")
            else:
                flags.append(f"{c}: within noise")
        print("  verdict (Δ vs 2·SE, heuristic): " + " | ".join(flags))
    else:
        print("  (only 1 matched seed — cannot judge noise band; run >=2 seeds)")


if __name__ == "__main__":
    main()
