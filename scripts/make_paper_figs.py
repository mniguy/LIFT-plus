#!/usr/bin/env python
"""
Paper figures for the prototype-centering study. Numbers are the verified aggregates
from output/ (see comments for provenance). Colorblind-safe (Okabe-Ito), IEEE-column size.

    /opt/anaconda3/bin/python scripts/make_paper_figs.py
Outputs: output/paper/fig_pca_ucurve.{pdf,png}, output/paper/fig_breadth_predictor.{pdf,png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "output/paper"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito (CVD-safe). blue = ImageNet / helps ; orange = Places ; vermillion = hurts
BLUE, ORANGE, VERM, GRAY = "#0072B2", "#E69F00", "#D55E00", "#8a8a8a"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "font.family": "sans-serif",
})


def style_ax(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


# ============ Fig 1: PCA-removal U-curve (Delta Few over no-centering baseline) ============
# provenance: output/pca_sweep25/*  (Few), baseline = seed_ablation 25 (IN 73.58 / PL 51.62)
# whiten (all PCs, ZCA): IN 73.19 / PL 51.94 -> reported in caption, off this k-axis (different op)
k_lab = ["0", "1", "2", "5", "10", "20"]
x = np.arange(len(k_lab))
base_in, base_pl = 73.58, 51.62
in_few = np.array([75.12, 75.15, 75.13, 75.18, 74.49, 73.44]) - base_in
pl_few = np.array([53.58, 53.08, 52.73, 52.73, 52.13, 51.03]) - base_pl

fig, ax = plt.subplots(figsize=(3.5, 2.7))
style_ax(ax)
ax.axhline(0, color=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.text(x[0], -0.12, "no-centering baseline", color=GRAY, fontsize=7, ha="left", va="top")
for y, c, name, ly in [(in_few, BLUE, "ImageNet-LT", 1.38), (pl_few, ORANGE, "Places-LT", 1.98)]:
    ax.plot(x, y, "-", color=c, lw=2, marker="o", ms=5, zorder=3)
    ax.text(x[0] + 0.12, ly, name, color=c, fontsize=8, ha="left", va="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(k_lab)
ax.set_xlabel(r"principal components removed  $k$")
ax.set_ylabel(r"$\Delta$ Few acc. vs baseline (pp)")
ax.margins(x=0.08)
ax.set_ylim(-0.85, 2.25)
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT}/fig_pca_ucurve.pdf"); fig.savefig(f"{OUT}/fig_pca_ucurve.png")
plt.close(fig)

# ============ Fig 2: breadth predictor -- tail init-persistence predicts benefit ============
# provenance: breadth25 (ΔFew), analyze_anisotropy.py (Few drift), + IN/PL from center_seeds25
# (dataset, tail_drift M2, ΔFew pp, baseline Few%) -- all at each dataset's matched protocol
# (iNat = 15 epochs, matching LIFT+; its tail drifts ~0.85 from init, so it is not frozen)
pts = [
    ("CIFAR-IR100", 0.037, +2.24, 79.63),
    ("Places-LT",   0.065, +1.62, 51.62),
    ("ImageNet-LT", 0.074, +1.59, 73.58),
    ("CIFAR-IR50",  0.049, -0.27, 84.44),
    ("iNat2018",    0.853, -0.23, 82.36),
]
fig, ax = plt.subplots(figsize=(3.7, 2.7))
style_ax(ax)
ax.set_xscale("log")
ax.axhline(0, color=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=1)
# per-point label: (text_x, text_y, ha, display text)
lab = {
    "CIFAR-IR100": (0.036, 2.62, "center", "CIFAR-IR100"),
    "Places-LT":   (0.073, 1.98, "left",   "Places-LT"),
    "ImageNet-LT": (0.081, 1.30, "left",   "ImageNet-LT"),
    "CIFAR-IR50":  (0.049, -0.66, "center", "CIFAR-IR50 (saturated)"),
    "iNat2018":    (0.85,  0.44, "center", "iNat2018 (not frozen)"),
}
for name, drift, dfew, bfew in pts:
    c = BLUE if dfew > 0.5 else GRAY          # helps vs. neutral (nothing actually hurts)
    s = (100 - bfew) * 9                      # marker size ~ tail headroom (100 - baseline Few)
    ax.scatter(drift, dfew, s=s, color=c, alpha=0.85, edgecolor="white", lw=1.2, zorder=3)
    tx, ty, ha, disp = lab[name]
    ax.annotate(disp, (drift, dfew), xytext=(tx, ty), fontsize=7.3, ha=ha, va="center", color="#222")
ax.text(0.050, 3.02, "frozen tail + headroom\n$\\rightarrow$ centering helps", fontsize=7.8, color=BLUE, ha="center", va="center")
ax.text(0.30, -0.72, "not frozen or saturated $\\rightarrow$ neutral", fontsize=7.8, color=GRAY, ha="center", va="center")
ax.text(0.30, 2.55, "marker size $\\propto$ tail headroom", fontsize=6.5, color=GRAY, ha="center", va="center")
ax.set_xlabel(r"tail weight-drift from init  $1-\cos(W_{\mathrm{final}},W_{\mathrm{init}})$  (log)")
ax.set_ylabel(r"centering $\Delta$ Few acc. (pp)")
ax.set_xlim(0.03, 1.15)
ax.set_ylim(-1.05, 3.35)
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT}/fig_breadth_predictor.pdf"); fig.savefig(f"{OUT}/fig_breadth_predictor.png")
plt.close(fig)

# ============ Fig 3 (schematic): the mechanism ============
# uniform anisotropy at init -> head fine-tunes out of the cone, tail is frozen ->
# centering de-anisotropizes the init the tail is stuck with.
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D

rng = np.random.default_rng(3)
c0 = 90.0                                                     # init-cone center angle (deg), pointing up
head_init = c0 + rng.uniform(-11, 11, 6)
tail_init = c0 + rng.uniform(-11, 11, 6)
head_ft = np.array([22, 48, 68, 112, 138, 162]) + rng.uniform(-3, 3, 6)   # head escapes (same in b,c)
tail_froz = c0 + rng.uniform(-11, 11, 6)                                   # tail frozen near init
tail_spr = np.array([35, 58, 80, 100, 125, 150]) + rng.uniform(-3, 3, 6)   # tail spreads with centering


def circ(ax, title):
    t = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(t), np.sin(t), color="#d3d8de", lw=1.0, zorder=0)
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.5); ax.axis("off")
    ax.set_title(title, fontsize=8.5, pad=1)


def dots(ax, ang, color):
    a = np.deg2rad(ang)
    ax.scatter(np.cos(a), np.sin(a), s=38, color=color, edgecolor="white", lw=1.0, zorder=3)


figm, axs = plt.subplots(1, 3, figsize=(7.4, 2.8))
am = np.deg2rad(c0)

ax = axs[0]; circ(ax, "(a) text-prototype init")
ax.add_patch(Wedge((0, 0), 1.16, c0 - 13, c0 + 13, color=GRAY, alpha=0.13, zorder=1))
dots(ax, head_init, BLUE); dots(ax, tail_init, ORANGE)
ax.annotate("", xy=(0.6 * np.cos(am), 0.6 * np.sin(am)), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))
ax.text(0.6 * np.cos(am) + 0.14, 0.6 * np.sin(am), r"$\mu$", color="#555", fontsize=12, va="center")
ax.text(0, -1.16, "anisotropic cone: all prototypes\n$\\approx\\mu$ (head & tail alike)", ha="center", va="top", fontsize=7.5, color="#333")

ax = axs[1]; circ(ax, "(b) after fine-tuning")
ax.add_patch(Wedge((0, 0), 1.16, c0 - 13, c0 + 13, color=VERM, alpha=0.12, zorder=1))
dots(ax, head_ft, BLUE); dots(ax, tail_froz, ORANGE)
ax.text(0, -1.16, "head escapes the cone;\ntail frozen at init (collinear)", ha="center", va="top", fontsize=7.5, color="#333")

ax = axs[2]; circ(ax, "(c) + prototype centering")
dots(ax, head_ft, BLUE); dots(ax, tail_spr, ORANGE)
ax.text(0, -1.16, "centering spreads the init\nthe tail cannot move from", ha="center", va="top", fontsize=7.5, color="#333")

handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUE, markersize=8, label='head (many-shot)'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=ORANGE, markersize=8, label='tail (few-shot)')]
figm.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
figm.tight_layout(rect=[0, 0, 1, 0.92])
figm.savefig(f"{OUT}/fig_mechanism.pdf"); figm.savefig(f"{OUT}/fig_mechanism.png")
plt.close(figm)

# ============ Fig 4: freeze intervention -- the init advantage centering provides ============
# center - baseline, trainable vs frozen classifier. Freezing (init cannot be overwritten)
# turns the small trainable gain into a large one on every dataset, incl. iNat.
ds = ["ImageNet-LT", "Places-LT", "iNat2018"]
panels = {
    r"$\Delta$ Overall": ([0.18, 0.09, -0.11], [3.49, 3.85, 12.43]),
    r"$\Delta$ Few":     ([1.59, 1.62, -0.23], [11.62, 6.62, 10.64]),
}
figf, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
x = np.arange(len(ds)); w = 0.38
for ax, (title, (tr, fr)) in zip(axs, panels.items()):
    style_ax(ax)
    b1 = ax.bar(x - w / 2, tr, w, color=GRAY, edgecolor="white", lw=0.6, label="trainable classifier", zorder=3)
    b2 = ax.bar(x + w / 2, fr, w, color=BLUE, edgecolor="white", lw=0.6, label="frozen classifier", zorder=3)
    ax.axhline(0, color="#9aa0a6", lw=0.8, zorder=2)
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.annotate(f"{h:+.1f}", (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 2 if h >= 0 else -9), textcoords="offset points",
                    ha="center", fontsize=6.8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(ds, fontsize=7.3)
    ax.set_title(title, fontsize=9.5)
    ax.set_ylabel("centering gain (pp)")
    ax.margins(y=0.18)
axs[0].legend(loc="upper left", fontsize=7.3, handlelength=1.2)
figf.tight_layout(pad=0.5)
figf.savefig(f"{OUT}/fig_freeze.pdf"); figf.savefig(f"{OUT}/fig_freeze.png")
plt.close(figf)

print("wrote:")
for f in ["fig_pca_ucurve", "fig_breadth_predictor", "fig_mechanism", "fig_freeze"]:
    print(f"  {OUT}/{f}.pdf  {OUT}/{f}.png")
