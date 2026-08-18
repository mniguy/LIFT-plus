#!/usr/bin/env python
"""Figure 2: the operation, drawn as a pipeline. Prototypes are shown as colour strips so
that the shared pattern, and its removal, are visible without reading any formula.
    /opt/anaconda3/bin/python scripts/make_method_fig.py
"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

OUT = sys.argv[1] if len(sys.argv) > 1 else "output/_paper"
INK, MUTE, LINE = "#1a1a1a", "#7a828c", "#c2c7ce"
BLUE, ORANGE = "#0072B2", "#E69F00"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 400,
                     "font.family": "sans-serif", "font.size": 9})

rng = np.random.default_rng(4)
D = 26
shared = rng.normal(0, 1, D) * 1.35                 # the direction every class carries
delta = rng.normal(0, 1, (3, D)) * 0.55             # what actually separates the classes
raw = shared + delta
mu = raw.mean(0)
cen = raw - mu
NAMES = ["goldfish", "barn", "volcano"]
CMAP = "RdBu_r"
V_RAW = np.abs(raw).max()
V_CEN = np.abs(cen).max()   # renormalised, so each set uses its own scale

fig, ax = plt.subplots(figsize=(7.2, 2.35))
ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")


def strip(x, y, vec, vlim, w=13.5, h=2.0, lw=0.6):
    ax.imshow(vec[None, :], cmap=CMAP, vmin=-vlim, vmax=vlim, aspect="auto",
              extent=(x, x + w, y, y + h), zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="white", lw=lw, zorder=4))


def box(x, y, w, h, label, sub=None, fc="#f4f6f8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=LINE, lw=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (1.1 if sub else 0), label, ha="center", va="center",
            fontsize=8.8, color=INK, zorder=5)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 1.6, sub, ha="center", va="center",
                fontsize=7.4, color=MUTE, zorder=5)


def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10,
                                 lw=1.2, color=MUTE, zorder=2))


ys = [22.5, 17.6, 12.7]

# --- 1. class names -----------------------------------------------------
for n, y in zip(NAMES, ys):
    ax.text(7.6, y + 1.0, f'“a photo of a {n}.”', ha="center", va="center",
            fontsize=7.3, color=INK)
    ax.add_patch(FancyBboxPatch((1.2, y - 0.2), 12.8, 2.4,
                                boxstyle="round,pad=0.2,rounding_size=0.8",
                                fc="white", ec=LINE, lw=.8, zorder=1))
arrow(14.8, 18.0, 18.6)

# --- 2. text encoder ----------------------------------------------------
ax.add_patch(Polygon([[18.5, 11.6], [18.5, 25.6], [30.8, 22.7], [30.8, 14.5]],
                     closed=True, fc="#f4f6f8", ec=LINE, lw=1.0, zorder=2))
ax.text(24.6, 19.6, "CLIP text\nencoder", ha="center", va="center", fontsize=8.8,
        color=INK, zorder=5, linespacing=1.25)
ax.text(24.6, 16.2, "frozen", ha="center", va="center", fontsize=7.4, color=MUTE, zorder=5)
arrow(31.6, 35.5, 18.6)

# --- 3. prototypes ------------------------------------------------------
for v, y in zip(raw, ys):
    strip(36, y, v, V_RAW)
ax.text(42.7, 26.4, "one vector per class", ha="center", fontsize=7.6, color=MUTE)

# highlight one entry across all classes, and where it goes
J, W = 13, 13.5 / D
xc = 36 + J * W
ax.add_patch(Rectangle((xc - 0.12, ys[-1] - 0.15), W + 0.24, (ys[0] + 2.0) - ys[-1] + 0.3,
                       fill=False, ec=INK, lw=1.1, ls=(0, (2.2, 1.6)), zorder=6))
ax.annotate("", xy=(xc + W / 2, 9.4), xytext=(xc + W / 2, 12.3),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
strip(36, 6.6, mu, V_RAW)
ax.add_patch(Rectangle((xc - 0.12, 6.45), W + 0.24, 2.3, fill=False, ec=INK, lw=1.1,
                       ls=(0, (2.2, 1.6)), zorder=6))
ax.text(42.7, 5.2, "their average, entry by entry", ha="center", va="top",
        fontsize=7.6, color=MUTE)

# --- 4. the subtraction -------------------------------------------------
ax.text(54.3, 18.6, "$-$", ha="center", va="center", fontsize=15, color=INK)
ax.text(54.3, 8.6, "subtract\nthe average", ha="center", va="top", fontsize=7.4, color=MUTE)
for v, y in zip(cen, ys):
    strip(59, y, v, V_CEN)
xc2 = 59 + J * W
ax.add_patch(Rectangle((xc2 - 0.12, ys[-1] - 0.15), W + 0.24, (ys[0] + 2.0) - ys[-1] + 0.3,
                       fill=False, ec=INK, lw=1.1, ls=(0, (2.2, 1.6)), zorder=6))
ax.text(65.7, 26.4, "what is left differs\nby class", ha="center", fontsize=7.6, color=ORANGE)
arrow(73.0, 76.5, 18.6)

# --- 5. classifier ------------------------------------------------------
box(76.5, 12.0, 22, 13.5, "", fc="#f4f6f8")
ax.text(87.5, 27.0, "classifier weights", ha="center", fontsize=8.8, color=INK)
for y, v in zip(ys, cen):
    strip(79.5, y, v, V_CEN, w=16, h=1.8)
ax.text(87.5, 10.6, "trained as usual from here on,\nwith no parameters added",
        ha="center", va="top", fontsize=7.4, color=MUTE)

fig.subplots_adjust(left=.005, right=.995, top=.99, bottom=.01)
fig.savefig(os.path.join(OUT, "fig_method.pdf"))
fig.savefig(os.path.join(OUT, "fig_method.png"))
print("wrote", OUT)
