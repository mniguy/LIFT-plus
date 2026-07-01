#!/usr/bin/env python
"""Collapse-recovery figure from EXISTING gate_collapse data (no new runs needed).

Grouped bars of head/med/few accuracy at the strong-regularization point, for
fixed (no gate) vs gated variants, per dataset. Shows fixed collapses (esp. few)
while gating recovers.

  python scripts/plot_collapse_bars.py \
      --bases output/gate_collapse/imagenet_lt output/gate_collapse/places_lt \
      --out output/paper/collapse_bars
"""
import argparse
import os

import numpy as np


def splits(c):
    c = np.asarray(c)
    return (c > 100), ((c >= 20) & (c <= 100)), (c < 20)


def grp(d, h, m, fw):
    a = np.load(os.path.join(d, "cls_accs.npy")).astype(float)
    return np.array([a[h].mean(), a[m].mean(), a[fw].mean()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", required=True, help="gate_collapse/<ds> dirs")
    ap.add_argument("--variants", nargs="+", default=["fixed", "agreement", "freq_inv"])
    ap.add_argument("--out", default="output/paper/collapse_bars")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    groups = ["head", "med", "few"]
    colors = {"fixed": "crimson", "agreement": "steelblue", "freq_inv": "seagreen"}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.bases), figsize=(6 * len(args.bases), 4.2))
    if len(args.bases) == 1:
        axes = [axes]

    for ax, base in zip(axes, args.bases):
        num = np.load(os.path.join(base, "fixed", "cls_num_list.npy"))
        h, m, fw = splits(num)
        ds = os.path.basename(base.rstrip("/"))
        x = np.arange(len(groups))
        w = 0.8 / len(args.variants)
        for k, v in enumerate(args.variants):
            d = os.path.join(base, v)
            if not os.path.exists(os.path.join(d, "cls_accs.npy")):
                continue
            vals = grp(d, h, m, fw)
            ax.bar(x + (k - (len(args.variants) - 1) / 2) * w, vals, w,
                   label=v, color=colors.get(v))
            print(f"{ds:14s} {v:10s} head {vals[0]:6.2f} med {vals[1]:6.2f} few {vals[2]:6.2f}")
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel("accuracy"); ax.set_title(ds)
        lo = min(grp(os.path.join(base, "fixed"), h, m, fw))
        ax.set_ylim(lo - 3, None)  # zoom so the collapse/recovery is visible
        ax.legend(title="gate", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Strong text-prior regularization collapses head/med/few (fixed); "
                 "per-class gating recovers it")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\n[save] {args.out}.png")


if __name__ == "__main__":
    main()
