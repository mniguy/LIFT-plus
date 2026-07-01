#!/usr/bin/env python
"""Interpretability figure: what the per-class agreement gate looks like.

Plots the minmax-normalized agreement gate g_c vs class train frequency, colored
by head/med/few. Illustrates that the gate (image-text agreement) is NOT a
restatement of frequency -- it down-weights low-agreement classes across all
frequency groups.

  python scripts/plot_gate.py --agreement output/agreement/_meta/agreement.npy \
      --cls-num output/agreement/_meta/cls_num_list.npy --out output/paper/gate
"""
import argparse
import os

import numpy as np
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agreement", required=True)
    ap.add_argument("--cls-num", required=True)
    ap.add_argument("--out", default="output/paper/gate")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    a = np.load(args.agreement).astype(float)
    cn = np.load(args.cls_num).astype(float)
    valid = np.isfinite(a)
    a, cn = a[valid], cn[valid]
    g = (a - a.min()) / (a.max() - a.min())  # minmax gate in [0,1]
    h, m, fw = (cn > 100), ((cn >= 20) & (cn <= 100)), (cn < 20)
    rho, p = spearmanr(g, cn)
    print(f"Spearman(gate, frequency) = {rho:+.3f} (p={p:.1e})  "
          f"-> gate {'≈' if abs(rho)<0.3 else ''} {'NOT ' if abs(rho)<0.3 else ''}explained by frequency")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for mask, lab, col in [(h, "head", "crimson"), (m, "med", "darkorange"), (fw, "few", "steelblue")]:
        ax.scatter(cn[mask], g[mask], s=14, alpha=0.5, color=col, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("class train frequency (log)")
    ax.set_ylabel("agreement gate  $g_c$")
    ax.set_title(f"Per-class agreement gate vs frequency  (Spearman ρ={rho:+.2f})")
    ax.legend(title="split")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    print(f"[save] {args.out}.png")


if __name__ == "__main__":
    main()
