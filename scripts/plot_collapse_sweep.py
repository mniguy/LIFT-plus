#!/usr/bin/env python
"""F2: collapse-recovery figure.

Reads output/collapse_sweep/<ds>/<variant>_lam<lambda>/cls_accs.npy and plots
accuracy vs regularization strength (InfoNCE lambda) for each gate variant,
split into all / head / med / few panels. Shows that 'fixed' collapses head/med
as lambda grows while gating stays robust.

  python scripts/plot_collapse_sweep.py --root output/collapse_sweep/places_lt --out .../F2
"""
import argparse
import glob
import os
import re
import sys

import numpy as np


def splits(c):
    c = np.asarray(c)
    return (c > 100), ((c >= 20) & (c <= 100)), (c < 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--order", nargs="*", default=["fixed", "agreement", "freq_inv"])
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "F2")

    pat = re.compile(r"^(.*)_lam([0-9.]+)$")
    data = {}  # variant -> {lambda: [all,head,med,few]}
    cls_num = None
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        f = os.path.join(d, "cls_accs.npy")
        if not os.path.isdir(d) or not os.path.exists(f):
            continue
        mobj = pat.match(os.path.basename(d))
        if not mobj:
            continue
        v, lam = mobj.group(1), float(mobj.group(2))
        if cls_num is None:
            cls_num = np.load(os.path.join(d, "cls_num_list.npy"))
        h, m, fw = splits(cls_num)
        a = np.load(f).astype(float)
        data.setdefault(v, {})[lam] = [a.mean(), a[h].mean(), a[m].mean(), a[fw].mean()]
    if not data:
        sys.exit(f"no <variant>_lam<lambda> runs with cls_accs.npy under {args.root}")

    variants = [v for v in args.order if v in data] + [v for v in data if v not in args.order]
    panels = ["all", "head", "med", "few"]
    style = {"fixed": ("--", "crimson"), "agreement": ("-", "steelblue"), "freq_inv": ("-", "seagreen")}

    # text summary (always)
    print(f"\n=== {args.root} ===")
    for v in variants:
        lams = sorted(data[v])
        print(f"  {v}")
        for lam in lams:
            a, hh, mm, ff = data[v][lam]
            print(f"    lam={lam:<6g} all {a:6.2f}  head {hh:6.2f}  med {mm:6.2f}  few {ff:6.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), sharex=True)
        for j, p in enumerate(panels):
            for v in variants:
                lams = sorted(data[v])
                ys = [data[v][lam][j] for lam in lams]
                ls, col = style.get(v, ("-", None))
                ax[j].plot(lams, ys, ls, color=col, marker="o", lw=2, label=v)
            ax[j].set_xscale("log")
            ax[j].set_xlabel("InfoNCE reg strength  λ (log)")
            ax[j].set_title(p)
            ax[j].grid(alpha=0.3)
        ax[0].set_ylabel("accuracy")
        ax[0].legend(title="gate")
        fig.suptitle("Collapse–recovery: strong text-prior reg collapses head/med (fixed); "
                     "agreement gating stays robust")
        fig.tight_layout()
        fig.savefig(out + ".png", dpi=150)
        print(f"\n[save] {out}.png")
    except Exception as e:
        print(f"[warn] figure skipped ({e}); text summary above is still valid")


if __name__ == "__main__":
    main()
