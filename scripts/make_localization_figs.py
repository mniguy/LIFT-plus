#!/usr/bin/env python
"""
Localization-axis figures for the prototype-centering study (Sec. discussion/localization).
Numbers verified from output/center_hier25, output/center_cluster25, output/center_tree25,
output/center_local25 (2026-08-02/03) -- see tables_cascade.tex for the source tables.
Matches make_paper_figs.py's style conventions (Okabe-Ito CVD-safe, IEEE column size).

    /opt/anaconda3/bin/python scripts/make_localization_figs.py
Outputs: output/paper/fig_localization_axis.{pdf,png}, output/paper/fig_residual_shape.{pdf,png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "output/_paper"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito (CVD-safe), extending make_paper_figs.py's BLUE=ImageNet-LT, ORANGE=Places-LT
# convention with GREEN=iNaturalist (unused elsewhere in the palette so far).
BLUE, ORANGE, GREEN, VERM, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8a8a8a"

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


# ============ Fig: localization axis -- cos-to-global predicts accuracy, opposite sign ============
# provenance: Table tab:in_local / tab:inat_hier (Few acc.), y indexed to each dataset's own
# global-centering point so both series share one axis (r is unchanged by this shift).
in_cos  = np.array([1.000, 0.887, 0.847, 0.845, 0.836, 0.813, 0.805])
in_few  = np.array([75.12, 74.60, 73.71, 74.25, 73.88, 73.25, 73.09])
in_lab  = ["global", "knn-50", "cluster\n(size 32)", "cascade\n(coarse)", "knn-20", "cascade", "cluster\n(size 16)"]

inat_cos = np.array([1.000, 0.915, 0.868, 0.819, 0.817, 0.751, 0.739, 0.729, 0.719])
inat_few = np.array([82.13, 82.31, 82.15, 82.46, 82.34, 82.46, 82.50, 82.52, 82.60])
inat_lab = ["global", "order", "family", "cluster\n$k$=50", "genus", "cluster\n$k$=254", "cascade",
            "cascade\n(full)", "cluster\n$k$=500"]

in_delta = in_few - in_few[0]
inat_delta = inat_few - inat_few[0]

r_in = np.corrcoef(in_cos, in_few)[0, 1]
r_inat = np.corrcoef(inat_cos, inat_few)[0, 1]

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=3, width=0.7, labelsize=8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#b8bcc2")
ax.axhline(0, color="#9aa0a6", lw=0.8, ls=(0, (4, 3)), zorder=1)

for x, y, c, name, r in [
    (in_cos, in_delta, BLUE, "ImageNet-LT", r_in),
    (inat_cos, inat_delta, GREEN, "iNaturalist 2018", r_inat),
]:
    b_, a_ = np.polyfit(x, y, 1)
    xx = np.linspace(min(x) - 0.01, 1.01, 20)
    ax.plot(xx, a_ + b_ * xx, "-", color=c, lw=1.1, alpha=0.45, zorder=2)
    ax.scatter(x, y, s=20, color=c, edgecolor="white", lw=0.7, zorder=3)
    ax.scatter([1.0], [0.0], s=46, facecolor=c, edgecolor="white", lw=1.1, zorder=4)

ax.annotate(f"iNaturalist 2018   $r={r_inat:+.2f}$", xy=(0.88, 0.56), fontsize=7.6,
            color=GREEN, ha="center", va="center")
ax.annotate(f"ImageNet-LT   $r={r_in:+.2f}$", xy=(0.995, -1.75), fontsize=7.6,
            color=BLUE, ha="left", va="center")
ax.annotate("global centering", xy=(1.0, 0.0), xytext=(0, -26), textcoords="offset points",
            fontsize=7.2, color="#7a828c", ha="center", va="top",
            arrowprops=dict(arrowstyle="-", color="#b8bcc2", lw=0.7))

ax.set_xlabel("cosine to global-centered init  (more local $\\rightarrow$)", fontsize=8.5)
ax.set_ylabel(r"$\Delta$ Few accuracy (pp)", fontsize=8.5)
ax.invert_xaxis()
ax.set_xlim(1.035, 0.695)
ax.set_ylim(-2.3, 0.9)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig_localization_axis.pdf")
fig.savefig(f"{OUT}/fig_localization_axis.png")
plt.close(fig)

# ============ Fig: residual shape after global centering -- spike vs. continuum ============
# provenance: Table tab:why_locality. Within-group cosine among classes sharing a group, AFTER
# global centering has already been applied -- what a local mean would additionally remove.
levels_inat = ["genus", "family", "order"]
resid_inat = [0.835, 0.164, 0.108]
levels_in = [r"$h_1$", r"$h_2$", r"$h_3$"]
resid_in = [0.452, 0.312, 0.219]

fig, ax = plt.subplots(figsize=(3.5, 2.2))
ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=3, width=0.7, labelsize=8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#b8bcc2")

x_inat = np.arange(len(levels_inat))
x_in = np.arange(len(levels_in)) + len(levels_inat) + 0.7
bars = list(ax.bar(x_inat, resid_inat, 0.52, color=GREEN, lw=0, zorder=3)) + \
       list(ax.bar(x_in, resid_in, 0.52, color=BLUE, lw=0, zorder=3))
for bb in bars:
    h = bb.get_height()
    ax.annotate(f"{h:.2f}", (bb.get_x() + bb.get_width() / 2, h), xytext=(0, 2),
                textcoords="offset points", ha="center", fontsize=7.2, color="#3f4650")
ax.set_xticks(np.concatenate([x_inat, x_in]))
ax.set_xticklabels(levels_inat + levels_in, fontsize=8)
for xs, name, c in [(x_inat, "iNaturalist", GREEN), (x_in, "ImageNet-LT", BLUE)]:
    ax.text(xs.mean(), -0.115, f"{name}\n(fine $\\rightarrow$ coarse)", ha="center", va="top",
            fontsize=7.4, color=c, linespacing=1.3)
ax.set_ylabel("within-group cosine\nafter global centering", fontsize=8.5)
ax.set_ylim(0, 0.98)
ax.set_xlim(-0.6, x_in[-1] + 0.6)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig_residual_shape.pdf")
fig.savefig(f"{OUT}/fig_residual_shape.png")
plt.close(fig)

print(f"r_in={r_in:.3f}  r_inat={r_inat:.3f}")
print("wrote:")
for f in ["fig_localization_axis", "fig_residual_shape"]:
    print(f"  {OUT}/{f}.pdf  {OUT}/{f}.png")
