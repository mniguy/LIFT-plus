#!/usr/bin/env python
"""Q8 -- head/med/few accuracy vs #epochs, baseline vs centering (offline, no GPU).

Reads the runs produced by run_center_epochs.sh:
    output/center_epochs25/<data>/<variant>_ep<E>/{cls_accs.npy, cls_num_list.npy}
and draws one panel per dataset: x = epochs, three group curves (head/med/few),
baseline solid / center dashed. Shows the epoch overfit axis (head up / few down as
epochs grow) and whether centering shifts it.

  python scripts/plot_epoch_groups.py --root output/center_epochs25 \
      --datasets imagenet_lt places_lt --out output/paper/epoch_groups
"""
import argparse, glob, os, re

import numpy as np

COLORS = {"head": "crimson", "med": "goldenrod", "few": "steelblue"}


def group_accs(run):
    a = np.load(os.path.join(run, "cls_accs.npy")).astype(float)   # 0-100, per class
    cn = np.load(os.path.join(run, "cls_num_list.npy"))
    h, m, f = cn > 100, (cn >= 20) & (cn <= 100), cn < 20
    return {"head": a[h].mean(), "med": a[m].mean(), "few": a[f].mean(), "all": a.mean()}


def series(root, data, variant):
    """-> sorted (epochs[], {group: accs[]}) over existing <variant>_ep<E> dirs."""
    pts = []
    for d in glob.glob(os.path.join(root, data, f"{variant}_ep*")):
        m = re.search(r"_ep(\d+)$", d)
        if m and os.path.exists(os.path.join(d, "cls_accs.npy")):
            pts.append((int(m.group(1)), group_accs(d)))
    pts.sort()
    eps = [e for e, _ in pts]
    return eps, {g: [v[g] for _, v in pts] for g in ["head", "med", "few", "all"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/center_epochs25")
    ap.add_argument("--datasets", nargs="+", default=["imagenet_lt", "places_lt"])
    ap.add_argument("--out", default="output/paper/epoch_groups")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.datasets), figsize=(5.4 * len(args.datasets), 4.4))
    if len(args.datasets) == 1:
        axes = [axes]

    any_data = False
    for ax, data in zip(axes, args.datasets):
        drawn = False
        for variant, ls in [("baseline", "-"), ("center", "--")]:
            eps, ys = series(args.root, data, variant)
            if not eps:
                continue
            any_data = drawn = True
            for g in ["head", "med", "few"]:
                ax.plot(eps, ys[g], ls, color=COLORS[g], marker="o", ms=4,
                        label=f"{g} {variant}")
                print(f"{data:12s} {variant:8s} {g:4s} " +
                      " ".join(f"ep{e}:{v:.2f}" for e, v in zip(eps, ys[g])))
        ax.set_title(data); ax.set_xlabel("num_epochs"); ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        if drawn:
            ax.legend(fontsize=7, ncol=2)

    if not any_data:
        print(f"[warn] no runs found under {args.root} -- run scripts/run_center_epochs.sh first")
        return
    fig.suptitle("Head/Med/Few vs epochs: baseline (solid) vs center (dashed)")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\n[save] {args.out}.png")


if __name__ == "__main__":
    main()
