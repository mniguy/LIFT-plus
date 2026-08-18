#!/usr/bin/env python
"""Figure 1 (teaser). Class prototypes on the unit circle and the decision region each one
owns. Without centering the tail prototypes crowd together and their regions are slivers, so
a tail image falls on a boundary. Removing the shared direction widens the same regions.
    /opt/anaconda3/bin/python scripts/make_teaser_fig.py
"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

OUT = sys.argv[1] if len(sys.argv) > 1 else "output/_paper"
BLUE, ORANGE, INK, SOFT = "#0072B2", "#E69F00", "#1a1a1a", "#f6e2be"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 400,
                     "font.family": "sans-serif", "legend.frameon": False})

RAW_HEAD = np.array([52., 70, 112, 130])
RAW_TAIL = np.array([86., 93, 100])
CEN_HEAD = np.array([26., 64, 118, 156])
CEN_TAIL = np.array([46., 92, 138])
Z_RAW, Z_CEN = 96.5, 88.0      # the same tail image, before and after
R = 1.0


def polar(a, r=R):
    a = np.deg2rad(np.asarray(a, float))
    return r * np.cos(a), r * np.sin(a)


def panel(ax, head, tail, z, title):
    order = np.sort(tail)
    bounds = [(order[i] + order[i + 1]) / 2 for i in range(len(order) - 1)]
    edges = [order[0] - (bounds[0] - order[0])] + bounds + \
            [order[-1] + (order[-1] - bounds[-1])]

    for i in range(len(order)):                       # decision region of each tail class
        ax.add_patch(Wedge((0, 0), R, edges[i], edges[i + 1], facecolor=SOFT,
                           edgecolor=ORANGE, lw=.7, alpha=.85, zorder=1))

    t = np.linspace(np.deg2rad(6), np.deg2rad(174), 300)
    ax.plot(np.cos(t), np.sin(t), color="#ccd1d8", lw=1.2, zorder=2)

    zx, zy = polar(z)
    ax.add_patch(FancyArrowPatch((0, 0), (zx, zy), arrowstyle="-|>", mutation_scale=12,
                                 lw=1.8, color=INK, zorder=5))
    ax.add_patch(Circle((zx, zy), 0.058, facecolor="white", edgecolor=INK, lw=1.5, zorder=6))
    ax.add_patch(Circle((zx, zy), 0.024, facecolor=INK, lw=0, zorder=7))

    ax.scatter(*polar(head), s=54, color=BLUE, ec="white", lw=1.2, zorder=4)
    ax.scatter(*polar(tail), s=62, marker="^", color=ORANGE, ec="white", lw=1.2, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-1.32, 1.32); ax.set_ylim(-0.34, 1.34)
    ax.axis("off")
    ax.text(0, -0.30, title, ha="center", va="top", fontsize=11, color=INK)


fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.45))
panel(axs[0], RAW_HEAD, RAW_TAIL, Z_RAW, "without centering")
panel(axs[1], CEN_HEAD, CEN_TAIL, Z_CEN, "with centering")

axs[0].text(0, 1.24, "tail classes share one narrow region", ha="center",
            fontsize=8.8, color=ORANGE)
axs[1].text(0, 1.24, "each tail class owns a wide one", ha="center",
            fontsize=8.8, color=ORANGE)

fig.legend(handles=[
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
           markeredgecolor="white", markersize=7, label="head class"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor=ORANGE,
           markeredgecolor="white", markersize=7.5, label="tail class"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
           markeredgecolor=INK, markeredgewidth=1.5, markersize=7, label="image feature"),
], loc="lower center", ncol=3, fontsize=8.6, columnspacing=1.9,
   handletextpad=.45, bbox_to_anchor=(0.5, -0.03))

fig.subplots_adjust(left=.01, right=.99, top=1.0, bottom=.19, wspace=.02)
fig.savefig(os.path.join(OUT, "fig_teaser.pdf"))
fig.savefig(os.path.join(OUT, "fig_teaser.png"))
print("wrote", OUT)
