#!/usr/bin/env python
"""E6 -- practicality: wall-clock + memory of each post-hoc estimator, and why the
confusion-matrix baselines (BBSE/RLLS/GS-B3SE) do NOT scale to iNat.

For each dataset we resample ONE test batch (total samples) and time the prior estimate
(median of --repeats). Confusion methods need a C x C matrix + O(C^3) solve, so for large
C we do NOT run them (would OOM/hang) -- instead we REPORT the projected memory (3*C^2*8 B)
and solve cost (~2/3 C^3 flops). GPA is O(N*C) for the E-step + a K-dim projection.

Closes C7: "cheap / post-hoc" is quantified, and "confusion methods can't scale" is a number.

  python scripts/cost_table.py \
    ImageNet-LT:output/test_agnostic_ms/imagenet_lt/seed0 \
    Places-LT:output/test_agnostic_ms/places_lt/seed0 \
    iNat2018:output/test_agnostic/inat2018/lift+ \
    --out output/paper/tables_cost.tex
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured_prior_adapt import (softmax, make_priors, resample_idx, em,
                                     shrink_to_uniform, build_groups, bbse_C, bbse_pi)
from compare_baselines import rlls_pi, gsb3se_pi, sade_predict
from lsc_baseline import lsc_pi

BIG_C = 3000
# order: scalable estimators first, then confusion-matrix (C x C) ones
SCALABLE = ["per-class EM", "LSC", "SADE*", "GPA", "GPA-S"]
CONFUSION = ["BBSE", "RLLS", "GS-B3SE"]


def timeit(fn, repeats):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1e3   # ms


def human_bytes(b):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024 or u == "TB":
            return f"{b:.1f}{u}"
        b /= 1024


def per_dataset(root, total, repeats, gmode, K, cal_frac=0.4):
    logits = np.load(os.path.join(root, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(root, "y_true.npy")).astype(int)
    cn = np.load(os.path.join(root, "cls_num_list.npy")).astype(np.float64)
    C = logits.shape[1]
    small = C <= BIG_C
    priors = make_priors(cn); groups = build_groups(cn, gmode, K)
    pi_train = cn / cn.sum()
    rng = np.random.default_rng(0)

    perm = rng.permutation(len(y))
    ncal = int(cal_frac * len(y)); cal, pool = perm[:ncal], perm[ncal:]
    by = [pool[y[pool] == c] for c in range(C)]
    by = [b if len(b) else np.array([0]) for b in by]
    sel = resample_idx(by, priors["forward"], total, rng)
    zl = logits[sel]; probs = softmax(zl)

    ms = {}
    ms["per-class EM"] = timeit(lambda: em(probs), repeats)
    ms["LSC"] = timeit(lambda: lsc_pi(zl), repeats)
    ms["SADE*"] = timeit(lambda: sade_predict(zl, pi_train), repeats)
    ms["GPA"] = timeit(lambda: em(probs, groups=groups), repeats)
    ms["GPA-S"] = timeit(lambda: shrink_to_uniform(em(probs, groups=groups), 1.0), repeats)

    conf_mem = 3 * C * C * 8            # Cmat + AtA + temporaries, float64
    conf_flops = (2.0 / 3.0) * C ** 3   # dense solve
    if small:
        pcal = softmax(logits[cal]); yc = y[cal]
        # end-to-end from logits: each is an ALTERNATIVE, so it pays the O(N*C) C x C build
        # itself (not shared) + the O(C^3) solve -- fair vs GPA's from-logits cost.
        ms["BBSE"] = timeit(lambda: bbse_pi(bbse_C(pcal, yc, C), probs), repeats)
        ms["RLLS"] = timeit(lambda: rlls_pi(bbse_C(pcal, yc, C), pcal, probs, C), repeats)
        ms["GS-B3SE"] = timeit(lambda: gsb3se_pi(bbse_C(pcal, yc, C), pcal, probs, C), repeats)
    return C, small, ms, conf_mem, conf_flops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="Name:root per dataset (single root)")
    ap.add_argument("--out", default="output/paper/tables_cost.tex")
    ap.add_argument("--total", type=int, default=15000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--group-mode", default="quantile")
    ap.add_argument("--n-groups", type=int, default=5)
    args = ap.parse_args()

    data = []
    for spec in args.specs:
        name, root = spec.split(":", 1)
        C, small, ms, cmem, cflop = per_dataset(root.split(",")[0], args.total, args.repeats,
                                                 args.group_mode, args.n_groups)
        data.append((name, C, small, ms, cmem, cflop))
        print(f"\n=== {name} (C={C}, N={args.total}) wall-clock ms (median/{args.repeats}) ===")
        for m in SCALABLE + CONFUSION:
            if m in ms:
                print(f"  {m:14s} {ms[m]:8.2f} ms")
            elif m in CONFUSION:
                print(f"  {m:14s}  INFEASIBLE  (needs {C}x{C} confusion = "
                      f"{human_bytes(cmem)} + {cflop:.1e} flop solve)")

    # LaTeX: methods x datasets, ms (or 'infeasible: mem') ; confusion block separated
    names = [d[0] for d in data]
    L = [f"% E6 cost  N={args.total}  median/{args.repeats}  (ms; confusion methods project mem/flops at large C)",
         "\\begin{tabular}{l" + "c" * len(names) + "}\\toprule",
         "Estimator & " + " & ".join(f"{n} (C={C})" for (n, C, *_ ) in data) + "\\\\\\midrule"]
    for m in SCALABLE:
        L.append(f"{m} & " + " & ".join(f"{d[3][m]:.1f} ms" for d in data) + "\\\\")
    L.append("\\midrule\\multicolumn{" + str(len(names) + 1) +
             "}{l}{\\emph{confusion-matrix ($C\\times C$, $O(C^3)$ solve):}}\\\\")
    for m in CONFUSION:
        cells = []
        for (n, C, small, ms, cmem, cflop) in data:
            cells.append(f"{ms[m]:.1f} ms" if small else f"\\ding{{55}} ({human_bytes(cmem)})")
        L.append(f"{m} & " + " & ".join(cells) + "\\\\")
    L.append("\\bottomrule\\end{tabular}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
