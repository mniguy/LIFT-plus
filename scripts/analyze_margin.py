#!/usr/bin/env python
"""H_D -- per-group DECISION MARGIN, baseline vs center (offline, from dumped logits).

margin_i = logit[true_i] - max_{c != true_i} logit[c]   (>0 iff correctly classified)
Uses logits.npy + y_true.npy + cls_num_list.npy written by SAVE_LOGITS (run_margin_dump.sh).

H_D (effective-margin hypothesis): centering enlarges the margin, most for Few, and the margin
gain tracks the accuracy gain. Reported in cosine units (margin / classifier_scale) so it's
scale-agnostic and interpretable as an angular separation.

  python scripts/analyze_margin.py output/margin25/imagenet_lt/baseline output/margin25/imagenet_lt/center
"""
import argparse, os
import numpy as np


def load(run):
    lo = np.load(os.path.join(run, "logits.npy")).astype(np.float32)   # [N, C]
    yt = np.load(os.path.join(run, "y_true.npy")).astype(int)          # [N]
    cn = np.load(os.path.join(run, "cls_num_list.npy"))                # [C]
    return lo, yt, cn


def margins(lo, yt):
    N = len(yt)
    true = lo[np.arange(N), yt]
    tmp = lo.copy(); tmp[np.arange(N), yt] = -np.inf
    runner = tmp.max(1)
    return true - runner                                              # >0 iff correct


def group_stats(m, yt, cn, scale):
    g = {"Many": cn > 100, "Med": (cn >= 20) & (cn <= 100), "Few": cn < 20, "All": np.ones_like(cn, bool)}
    out = {}
    for name, cmask in g.items():
        sel = cmask[yt]                                               # images whose true class is in group
        mm = m[sel] / scale                                          # cosine-unit margin
        if len(mm) == 0:
            continue
        out[name] = dict(margin=mm.mean(), pos=100 * (mm > 0).mean(),  # pos% = recall
                         corr=mm[mm > 0].mean() if (mm > 0).any() else float("nan"),
                         near=100 * ((mm > 0) & (mm < 0.05)).mean())   # barely-correct (<0.05 cos)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline"); ap.add_argument("center")
    ap.add_argument("--scale", type=float, default=25.0)
    args = ap.parse_args()

    res = {}
    for tag, run in [("baseline", args.baseline), ("center", args.center)]:
        lo, yt, cn = load(run)
        res[tag] = group_stats(margins(lo, yt), yt, cn, args.scale)

    print(f"\nDecision margin (cosine units = margin / scale {args.scale:g}), baseline vs center")
    print(f"{'group':5} | {'mean margin':>22} | {'recall %(margin>0)':>22} | {'margin|correct':>22}")
    print(f"{'':5} | {'base':>7} {'cent':>7} {'Δ':>6} | {'base':>7} {'cent':>7} {'Δ':>6} | {'base':>7} {'cent':>7} {'Δ':>6}")
    for name in ["Many", "Med", "Few", "All"]:
        b, c = res["baseline"].get(name), res["center"].get(name)
        if not b or not c:
            continue
        print(f"{name:5} | {b['margin']:7.4f} {c['margin']:7.4f} {c['margin']-b['margin']:+6.4f} | "
              f"{b['pos']:7.2f} {c['pos']:7.2f} {c['pos']-b['pos']:+6.2f} | "
              f"{b['corr']:7.4f} {c['corr']:7.4f} {c['corr']-b['corr']:+6.4f}")
    print("\nreading: H_D wants center 'mean margin' Δ > 0, largest for Few; 'recall' Δ = the accuracy gain.")


if __name__ == "__main__":
    main()
