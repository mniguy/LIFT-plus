#!/usr/bin/env python
"""Per-group TRAIN-accuracy trend, baseline vs centering, for each dataset.

Parses the running per-group train acc printed each batch:
    epoch [e/E] batch [b/B] ... (mean <All> many <Head> med <Med> few <Few>) ...
and draws a datasets x {All,Head,Med,Few} grid, baseline (solid) vs center (dashed).
x-axis = training progress (epoch fraction) so 5-ep and 15-ep runs line up.
These are TRAIN accuracies from the log (test acc is only evaluated once, at the end).
No GPU needed.

  python scripts/plot_train_acc.py \
      --pairs imagenet_lt=output/loss_robust25/imagenet_lt/LA_baseline,output/loss_robust25/imagenet_lt/LA_center \
              places_lt=output/loss_robust25/places_lt/LA_baseline,output/loss_robust25/places_lt/LA_center \
              inat2018=output/breadth25/inat2018/baseline_15ep,output/breadth25/inat2018/center_15ep \
      --out output/paper/train_acc
"""
import argparse, glob, os, re

import numpy as np

LINE = re.compile(r"epoch \[(\d+)/(\d+)\] batch \[(\d+)/(\d+)\].*?"
                  r"\(mean ([\d.]+) many ([\d.]+) med ([\d.]+) few ([\d.]+)\)")
GROUPS = ["All", "Head", "Med", "Few"]   # log order: mean, many, med, few


def parse(run):
    logs = sorted(glob.glob(os.path.join(run, "log-*.txt")))
    if not logs:
        raise FileNotFoundError(f"no log-*.txt in {run}")
    prog, vals = [], []
    with open(logs[-1]) as f:
        for ln in f:
            m = LINE.search(ln)
            if not m:
                continue
            e, E, b, B = map(int, m.groups()[:4])
            prog.append((e - 1 + b / B) / E)
            vals.append([float(x) for x in m.groups()[4:]])   # All, Head, Med, Few
    return np.array(prog), np.array(vals)


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")            # edge-pad so ends don't dip
    return np.convolve(yp, np.ones(k) / k, mode="valid")[:len(y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="ds=BASELINE_RUN,CENTER_RUN")
    ap.add_argument("--out", default="output/paper/train_acc")
    ap.add_argument("--smooth", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nds = len(args.pairs)
    fig, axes = plt.subplots(nds, 4, figsize=(4 * 4, 3.1 * nds), squeeze=False)

    for row, spec in enumerate(args.pairs):
        ds, runs = spec.split("=", 1)
        base_run, cent_run = runs.split(",")
        xb, yb = parse(base_run)
        xc, yc = parse(cent_run)
        for col, g in enumerate(GROUPS):
            ax = axes[row][col]
            ax.plot(xb, smooth(yb[:, col], args.smooth), color="crimson", lw=1.5, label="baseline")
            ax.plot(xc, smooth(yc[:, col], args.smooth), color="steelblue", lw=1.5, label="center")
            print(f"{ds:12s} {g:4s} baseline_end={yb[-50:,col].mean():.2f}  center_end={yc[-50:,col].mean():.2f}")
            if row == 0:
                ax.set_title(g)
            if col == 0:
                ax.set_ylabel(f"{ds}\ntrain acc")
            if row == nds - 1:
                ax.set_xlabel("training progress")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Per-group TRAIN accuracy: baseline (red) vs center (blue)")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\n[save] {args.out}.png")


if __name__ == "__main__":
    main()
