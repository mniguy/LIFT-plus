#!/usr/bin/env python
"""E8 -- honest failure-mode probe: where does frequency-GROUPING break?

GPA assumes the test prior is (approximately) piecewise-constant within data-driven
FREQUENCY groups. That holds for the standard LT shifts (forward/uniform/backward are
monotone functions of train frequency), but NOT for shifts that vary WITHIN a frequency
band. We stress it with non-monotone / within-group-heterogeneous test priors:

  forward / backward : monotone-in-frequency (grouping-friendly reference)
  dirichlet(a)       : per-class independent mass ~ Dir(a) -> heterogeneous WITHIN a group
  spike              : ~90% mass on a random 1% of classes -> a single group can't represent it
  ushape             : U-shaped in frequency rank (head+tail heavy) -> STILL a function of
                       frequency, so grouping should survive (control that isolates the cause)

Same decision rule / estimators as the main comparison. Honest expectation: on dirichlet /
spike, per-class EM (given enough samples) catches up to or beats GPA, because the true prior
is not piecewise-constant-in-frequency; on ushape, GPA still wins. This DEFINES the method's
regime and gives the paper's limitation paragraph.

  python scripts/failure_modes.py \
    ImageNet-LT:output/test_agnostic_ms/imagenet_lt/seed0 \
    Places-LT:output/test_agnostic_ms/places_lt/seed0 \
    --out output/paper/tables_failure.tex
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured_prior_adapt import (softmax, resample_idx, em, shrink_to_uniform,
                                     split_masks, build_groups, logadj)

METHODS = ["no-adapt", "per-class EM", "GPA", "GPA-S", "oracle"]


def make_test_priors(cn, rng, n_dir=3):
    """Non-monotone / within-group-heterogeneous test priors (beyond forward/backward)."""
    C = len(cn); order = np.argsort(cn)
    fwd = cn / cn.sum()
    bwd = np.empty(C); bwd[order] = fwd[order[::-1]]; bwd /= bwd.sum()
    pr = {"forward": [fwd], "backward": [bwd]}
    pr["dirichlet"] = [rng.dirichlet(np.full(C, 0.5)) for _ in range(n_dir)]  # heterogeneous within group
    sp = []
    for _ in range(n_dir):
        s = np.full(C, 0.1 / C); k = max(1, C // 100)
        s[rng.choice(C, k, replace=False)] += 0.9 / k; sp.append(s / s.sum())
    pr["spike"] = sp
    rank = np.empty(C); rank[order] = np.arange(C)                # 0=rarest .. C-1=most frequent
    u = np.abs(rank - (C - 1) / 2.0) + 1.0; pr["ushape"] = [u / u.sum()]   # head+tail heavy (still freq-fn)
    return pr


def eval_prior(logits, y, cn, pi, groups, masks, total, gamma, rng, by):
    sel = resample_idx(by, pi, total, rng); zl, yl = logits[sel], y[sel]
    C = logits.shape[1]; probs = softmax(zl)
    pi_grp = em(probs, groups=groups)
    adj = {"no-adapt": np.zeros(C), "per-class EM": logadj(em(probs), C),
           "GPA": logadj(pi_grp, C), "GPA-S": logadj(shrink_to_uniform(pi_grp, gamma), C),
           "oracle": logadj(pi, C)}
    out = {}
    for m in METHODS:
        out[m] = float(((zl + adj[m]).argmax(1) == yl).mean() * 100)
    return out


def per_root(root, gmode, K, total, gamma, n_dir):
    logits = np.load(os.path.join(root, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(root, "y_true.npy")).astype(int)
    cn = np.load(os.path.join(root, "cls_num_list.npy")).astype(np.float64)
    C = logits.shape[1]
    groups = build_groups(cn, gmode, K); masks = split_masks(cn)
    rng = np.random.default_rng(0)
    by = [np.where(y == c)[0] for c in range(C)]
    by = [b if len(b) else np.array([0]) for b in by]
    pr = make_test_priors(cn, rng, n_dir)
    res = {}   # cond -> {method: All acc mean over draws}
    for cond, pis in pr.items():
        runs = [eval_prior(logits, y, cn, pi, groups, masks, total, gamma, rng, by) for pi in pis]
        res[cond] = {m: float(np.mean([r[m] for r in runs])) for m in METHODS}
    return C, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--out", default="output/paper/tables_failure.tex")
    ap.add_argument("--group-mode", default="quantile")
    ap.add_argument("--n-groups", type=int, default=5)
    ap.add_argument("--total", type=int, default=15000)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--n-dir", type=int, default=3)
    args = ap.parse_args()

    conds = ["forward", "backward", "ushape", "dirichlet", "spike"]
    data = []
    for spec in args.specs:
        name, root = spec.split(":", 1)
        C, res = per_root(root.split(",")[0], args.group_mode, args.n_groups,
                          args.total, args.gamma, args.n_dir)
        data.append((name, C, res))
        print(f"\n=== {name} (C={C})  All acc; GPA vs per-class EM (Δ = GPA - EM) ===")
        print(f"  {'condition':12s} {'no-adapt':>9s} {'EM':>8s} {'GPA':>8s} {'GPA-S':>8s} "
              f"{'oracle':>8s} {'GPA-EM':>8s}  regime")
        for c in conds:
            r = res[c]; d = r["GPA"] - r["per-class EM"]
            tag = "grouping-OK" if d >= 0 else "GROUPING LOSES"
            print(f"  {c:12s} {r['no-adapt']:9.2f} {r['per-class EM']:8.2f} {r['GPA']:8.2f} "
                  f"{r['GPA-S']:8.2f} {r['oracle']:8.2f} {d:+8.2f}  {tag}")

    # LaTeX: condition x (method acc) for the first dataset, plus GPA-EM gap
    L = [f"% E8 failure modes  group={args.group_mode}/{args.n_groups} total={args.total} "
         f"(All acc, mean over draws; last col = GPA - per-class EM)",
         "\\begin{tabular}{ll" + "c" * (len(METHODS) + 1) + "}\\toprule",
         "Dataset & Shift & " + " & ".join(METHODS) + " & GPA$-$EM\\\\\\midrule"]
    for name, C, res in data:
        for i, c in enumerate(conds):
            r = res[c]; d = r["GPA"] - r["per-class EM"]
            row = [name if i == 0 else "", c] + [f"{r[m]:.2f}" for m in METHODS] + [f"{d:+.2f}"]
            L.append(" & ".join(row) + "\\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule\\end{tabular}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
