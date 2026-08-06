#!/usr/bin/env python
"""
Paper figures for the prototype-centering study. Numbers are the verified aggregates
from output/ (see comments for provenance). Colorblind-safe (Okabe-Ito), IEEE-column size.

    /opt/anaconda3/bin/python scripts/make_paper_figs.py
Outputs: output/_paper/fig_{pca_ucurve,mechanism,freeze}.{pdf,png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "output/_paper"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito (CVD-safe). blue = ImageNet / helps ; orange = Places ; vermillion = hurts
BLUE, ORANGE, VERM, GRAY = "#0072B2", "#E69F00", "#D55E00", "#8a8a8a"
INK, MUTE, SUB = "#1a1a1a", "#7a828c", "#3f4650"   # schematic (Fig 3) text tones

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

sd_in, sd_pl = 0.23, 0.33          # seed std on Few (Table 1), used as the noise band

fig, ax = plt.subplots(figsize=(3.5, 2.25))
ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=3, width=0.7, labelsize=8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#b8bcc2")

ax.axhline(0, color="#9aa0a6", lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.text(-0.28, -0.08, "uncentered baseline", color="#9aa0a6", fontsize=7,
        ha="left", va="top")

for y, sd, c, name in [(in_few, sd_in, BLUE, "ImageNet-LT"), (pl_few, sd_pl, ORANGE, "Places-LT")]:
    ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.13, lw=0, zorder=2)
    ax.plot(x, y, "-", color=c, lw=1.4, zorder=3)
    ax.plot(x, y, "o", color=c, ms=3.6, mec="white", mew=0.8, zorder=4, label=name)

ax.annotate("", xy=(-0.1, 2.02), xytext=(3.1, 2.02),
            arrowprops=dict(arrowstyle="-", color="#c2c7ce", lw=0.8))
for xb in (-0.1, 3.1):
    ax.plot([xb, xb], [1.94, 2.02], color="#c2c7ce", lw=0.8)
ax.text(1.5, 2.08, "ImageNet-LT flat within seed noise", fontsize=7.0,
        color="#7a828c", ha="center", va="bottom")

ax.legend(loc="upper right", fontsize=7.6, handlelength=1.1, handletextpad=0.5,
          borderaxespad=0.1, labelspacing=0.3)
ax.set_xticks(x); ax.set_xticklabels(k_lab)
ax.set_xlabel(r"principal components removed  $k$", fontsize=8.5)
ax.set_ylabel(r"$\Delta$ Few accuracy (pp)", fontsize=8.5)
ax.set_xlim(-0.35, len(x) - 0.65)
ax.set_ylim(-1.0, 2.35)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig_pca_ucurve.pdf"); fig.savefig(f"{OUT}/fig_pca_ucurve.png")
plt.close(fig)

# ============ Fig 3 (schematic): the mechanism ============
# uniform anisotropy at init -> head fine-tunes out of the cone, tail stays ->
# centering spreads the init the tail is stuck with.
from matplotlib.patches import Wedge, FancyArrowPatch
from matplotlib.lines import Line2D

C0, HALF = 90.0, 11.0

head_init = C0 + np.array([-7.5, -3.5, 0.5, 4.5, 8.0])
tail_init = C0 + np.array([-5.5, -1.5, 2.5, 6.5, 9.5])
head_ft = np.array([26.0, 54, 82, 124, 150])
tail_ft = tail_init + np.array([0.8, -0.6, 0.5, -0.9, 0.4])
head_ctr = head_ft
tail_ctr = np.array([44.0, 70, 96, 134, 156])


def polar(ang, r=1.0):
    a = np.deg2rad(ang)
    return r * np.cos(a), r * np.sin(a)


def panel(ax, tag, title, sub):
    t = np.linspace(np.deg2rad(16), np.deg2rad(164), 200)
    ax.plot(np.cos(t), np.sin(t), color="#c5cad2", lw=1.1, solid_capstyle="round", zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(0.22, 1.30)
    ax.axis("off")
    ax.text(0.0, 1.05, f"{tag}  {title}", transform=ax.transAxes,
            fontsize=9.0, color=INK, va="bottom", ha="left")
    ax.text(0.5, -0.13, sub, transform=ax.transAxes, fontsize=8.2, color=SUB,
            ha="center", va="top", linespacing=1.4)


def cone(ax, dashed=False):
    if dashed:
        for s in (-1, 1):
            a = np.deg2rad(C0 + s * HALF)
            ax.plot([0.45 * np.cos(a), 1.10 * np.cos(a)], [0.45 * np.sin(a), 1.10 * np.sin(a)],
                    color=GRAY, lw=0.7, ls=(0, (2.5, 2.5)), alpha=0.6, zorder=1)
    else:
        ax.add_patch(Wedge((0, 0), 1.13, C0 - HALF, C0 + HALF,
                           facecolor=GRAY, alpha=0.15, edgecolor="none", zorder=1))


def dots(ax, ang, color, marker="o", s=36):
    x, y = polar(np.asarray(ang, dtype=float))
    ax.scatter(x, y, s=s, marker=marker, color=color, edgecolor="white", lw=1.0, zorder=4)


def move(ax, a0, a1, color):
    sgn = 1 if a1 > a0 else -1
    p0, p1 = polar(a0 + 3 * sgn, 1.13), polar(a1 - 4 * sgn, 1.13)
    rad = 0.10 if a1 > a0 else -0.10
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=7.5,
                                 lw=0.95, color=color, alpha=0.8, zorder=3))


fig, axs = plt.subplots(1, 3, figsize=(7.3, 1.85))

# ---- (a) ---------------------------------------------------------------
ax = axs[0]
panel(ax, "(a)", "text-prototype init", "every class starts inside\nthe same narrow cone")
cone(ax)
ax.annotate("", xy=polar(C0, 0.90), xytext=polar(C0, 0.50),
            arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.0, shrinkA=0, shrinkB=0))
ax.text(0.08, 0.68, r"$\hat\mu$", color=MUTE, fontsize=10)
dots(ax, head_init, BLUE)
dots(ax, tail_init, ORANGE, marker="^", s=40)
h_head = Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                markeredgecolor="white", markersize=6, label="head class")
h_tail = Line2D([0], [0], marker="^", color="none", markerfacecolor=ORANGE,
                markeredgecolor="white", markersize=6.5, label="tail class")
ax.legend(handles=[h_head, h_tail], loc="lower left", fontsize=7.8, frameon=False,
          handletextpad=0.35, borderpad=0.0, borderaxespad=0.1, labelspacing=0.35)

# ---- (b) ---------------------------------------------------------------
ax = axs[1]
panel(ax, "(b)", "after fine-tuning", "head classes move out,\nthe tail does not")
cone(ax)
move(ax, head_init[0], head_ft[0], BLUE)
move(ax, head_init[-1], head_ft[-1], BLUE)
dots(ax, head_ft, BLUE)
dots(ax, tail_ft, ORANGE, marker="^", s=40)
ax.annotate("tail unmoved", polar(C0, 1.0), xytext=(0, 20), textcoords="offset points",
            fontsize=8, color=ORANGE, ha="center",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7, alpha=0.55))

# ---- (c) ---------------------------------------------------------------
ax = axs[2]
panel(ax, "(c)", "with centering", "the tail starts outside\nthe cone instead")
cone(ax, dashed=True)
move(ax, tail_init[0], tail_ctr[0], ORANGE)
move(ax, tail_init[-1], tail_ctr[-1], ORANGE)
dots(ax, head_ctr, BLUE)
dots(ax, tail_ctr, ORANGE, marker="^", s=40)

fig.subplots_adjust(left=0.004, right=0.996, top=0.86, bottom=0.30, wspace=0.03)
fig.savefig(f"{OUT}/fig_mechanism.pdf"); fig.savefig(f"{OUT}/fig_mechanism.png")
plt.close(fig)

# ============ Fig 4: freeze intervention -- the init advantage centering provides ============
# center - baseline, trainable vs frozen classifier. Freezing (init cannot be overwritten)
# turns the small trainable gain into a large one on every dataset, incl. iNat.
ds = ["ImageNet-LT", "Places-LT", "iNat2018"]
panels = {
    r"$\Delta$ Overall": ([0.18, 0.09, -0.11], [3.49, 3.85, 12.43]),
    r"$\Delta$ Few":     ([1.59, 1.61, -0.23], [11.62, 6.62, 10.64]),
}

figf, axs = plt.subplots(1, 2, figsize=(7.2, 2.0), sharey=True)
y = np.arange(len(ds))[::-1]
for ax, (title, (tr, fr)) in zip(axs, panels.items()):
    ax.grid(axis="x", color="#ececec", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#b8bcc2")
    ax.tick_params(length=3, width=0.7, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="#9aa0a6", lw=0.8, zorder=1)
    for yi, a, b in zip(y, tr, fr):
        ax.annotate("", xy=(b, yi), xytext=(a, yi),
                    arrowprops=dict(arrowstyle="-|>", color="#c3ccd6", lw=2.4,
                                    shrinkA=3, shrinkB=4, mutation_scale=9), zorder=2)
        ax.plot([a], [yi], "o", ms=5, color=GRAY, mec="white", mew=0.9, zorder=3)
        ax.plot([b], [yi], "o", ms=6, color=BLUE, mec="white", mew=0.9, zorder=3)
        ax.annotate(f"{b:+.1f}", (b, yi), xytext=(7, 0), textcoords="offset points",
                    fontsize=7.6, color=BLUE, va="center")
        ax.annotate(f"{a:+.1f}", (a, yi), xytext=(-7, 0), textcoords="offset points",
                    fontsize=7.2, color="#7a828c", va="center", ha="right")
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("centering gain (pp)", fontsize=8.5)
    ax.set_xlim(-3.4, 14.6)
axs[0].set_yticks(y); axs[0].set_yticklabels(ds, fontsize=8.5)
axs[0].set_ylim(-0.5, len(ds) - 0.25)

axs[0].annotate("trainable", (0.18, y[0]), xytext=(-2, 13), textcoords="offset points",
                fontsize=7.4, color="#7a828c", ha="center")
axs[0].annotate("frozen", (3.49, y[0]), xytext=(2, 13), textcoords="offset points",
                fontsize=7.4, color=BLUE, ha="center")

figf.tight_layout(pad=0.4)
figf.savefig(f"{OUT}/fig_freeze.pdf"); figf.savefig(f"{OUT}/fig_freeze.png")
plt.close(figf)

print("wrote:")
for f in ["fig_pca_ucurve", "fig_mechanism", "fig_freeze"]:
    print(f"  {OUT}/{f}.pdf  {OUT}/{f}.png")
