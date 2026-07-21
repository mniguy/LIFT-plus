#!/usr/bin/env python
"""Training-loss trend, baseline vs centering, per dataset (Q7).

Parses the per-batch loss printed in each run's text log
    epoch [e/E] batch [b/B] time ... loss <inst> (<runavg>) acc ...
and overlays baseline vs center as one panel per dataset. x-axis is training
progress (epoch-fraction) so datasets with different epoch counts line up.
No GPU / model needed -- works off the saved logs.

  python scripts/plot_train_loss.py \
      --pairs imagenet_lt=output/loss_robust25/imagenet_lt/LA_baseline,output/loss_robust25/imagenet_lt/LA_center \
              places_lt=output/loss_robust25/places_lt/LA_baseline,output/loss_robust25/places_lt/LA_center \
              inat2018=output/breadth25/inat2018/baseline_15ep,output/breadth25/inat2018/center_15ep \
      --out output/paper/train_loss
"""
import argparse, glob, os, re

import numpy as np

LINE = re.compile(r"epoch \[(\d+)/(\d+)\] batch \[(\d+)/(\d+)\].*?loss\s+([\d.]+)\s+\(([\d.]+)\)")


def parse(run, use_runavg=False):
    logs = sorted(glob.glob(os.path.join(run, "log-*.txt")))
    if not logs:
        raise FileNotFoundError(f"no log-*.txt in {run}")
    prog, loss = [], []
    with open(logs[-1]) as f:
        for ln in f:
            m = LINE.search(ln)
            if not m:
                continue
            e, E, b, B, inst, avg = m.groups()
            e, E, b, B = int(e), int(E), int(b), int(B)
            prog.append((e - 1 + b / B) / E)          # fraction of total training in [0,1]
            loss.append(float(avg if use_runavg else inst))
    return np.array(prog), np.array(loss)


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")            # edge-pad so ends don't dip
    return np.convolve(yp, np.ones(k) / k, mode="valid")[:len(y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="ds=BASELINE_RUN,CENTER_RUN triples")
    ap.add_argument("--out", default="output/paper/train_loss")
    ap.add_argument("--runavg", action="store_true", help="plot log's running-avg loss instead of smoothed instantaneous")
    ap.add_argument("--smooth", type=int, default=25, help="rolling-mean window for instantaneous loss")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.pairs), figsize=(5.2 * len(args.pairs), 4.2))
    if len(args.pairs) == 1:
        axes = [axes]

    for ax, spec in zip(axes, args.pairs):
        ds, runs = spec.split("=", 1)
        base_run, cent_run = runs.split(",")
        for run, label, color in [(base_run, "baseline", "crimson"),
                                  (cent_run, "center", "steelblue")]:
            x, y = parse(run, args.runavg)
            ys = smooth(y, args.smooth)
            ax.plot(x, ys, color=color, label=label, lw=1.6)
            print(f"{ds:12s} {label:9s} start={y[:50].mean():.3f} end={y[-50:].mean():.3f} "
                  f"min={y.min():.3f} n={len(y)}")
        ax.set_title(ds)
        ax.set_xlabel("training progress (epoch fraction)")
        ax.set_ylabel("running-avg loss" if args.runavg else f"train loss (smooth {args.smooth})")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Training-loss trend: baseline vs centering")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\n[save] {args.out}.png")


if __name__ == "__main__":
    main()
