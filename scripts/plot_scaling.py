#!/usr/bin/env python
"""Slide/paper figure: the gain over no-adapt as a function of #classes.
The SOTA post-hoc estimator LSC (ICML'24) -- a full per-class estimator -- collapses
below zero as C grows, while GPA-S stays positive: the C/N-variance prediction made visible.

Numbers are the All-split deltas vs LIFT+ (no-adapt), mean over forward/uniform/backward,
from scripts/lsc_baseline.py (LSC vs GPA-S head-to-head on the saved logits).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (dataset, #classes, LSC delta, GPA-S delta) -- All vs no-adapt, 3-seed (compare_baselines.py)
DATA = [("Places-LT", 365, 2.30, 3.74),
        ("ImageNet-LT", 1000, -0.49, 1.40),
        ("iNat2018", 8142, -6.27, 2.30)]

labels = [f"{n}\n(C={c})" for n, c, _, _ in DATA]
em = [d[2] for d in DATA]
gpas = [d[3] for d in DATA]
x = np.arange(len(DATA)); w = 0.36

fig, ax = plt.subplots(figsize=(6.0, 3.7))
b1 = ax.bar(x - w/2, em, w, label="LSC (ICML'24, SOTA)", color="tab:red")
b2 = ax.bar(x + w/2, gpas, w, label="GPA-S (ours)", color="tab:blue")
ax.axhline(0, color="black", lw=0.8)
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.annotate(f"{h:+.1f}", (b.get_x() + b.get_width()/2, h),
                ha="center", va="bottom" if h >= 0 else "top",
                fontsize=8, xytext=(0, 2 if h >= 0 else -2), textcoords="offset points")
# highlight the divergence at scale
ax.annotate("gap +8.6", (x[2], 2.30), xytext=(x[2]-0.15, 4.4), fontsize=9,
            color="tab:blue", ha="center",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("$\\Delta$ accuracy vs LIFT+ (no adaptation)")
ax.set_title("SOTA estimator (LSC) collapses as #classes grows", fontsize=11)
ax.set_ylim(-7.6, 5.8)
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
out = "output/paper/fig_scaling_bars.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print(f"[written] {out} (+ .png)")
