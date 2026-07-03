#!/usr/bin/env python
"""Definitive baseline comparison for the test-agnostic prior-adaptation paper.

Methods (all share the SAME decision rule argmax[z + tau*log(pi/uniform)], differ only
in the estimated prior pi):
  no-adapt      : pi = uniform (LIFT+)
  per-class EM  : Saerens/MLLS full categorical
  BBSE          : confusion-matrix inversion (needs labeled cal set)      [small C only]
  RLLS          : BBSE + ridge shrinkage of the weight-change (Azizzadenesheli'19) [small C]
  GS-B3SE       : graph-smoothed BBSE (structure via Laplacian; needs C x C)      [small C]
  LSC (ICML'24) : self-trained confidence-filtered per-class marginal (Wei et al.)
  SADE*         : SADE-style 3-expert test-time aggregation, entropy-min (single-model approx)
  GPA (ours)    : constrained grouped EM (K frequency groups)
  GPA-S (ours)  : GPA + KL-gated adaptive shrinkage
  oracle        : true test prior (upper bound)

BBSE/RLLS need a C x C confusion + O(C^3) solve, so they are skipped for large C
(iNat, 8142) -- itself evidence that confusion-matrix methods do not scale.

  python scripts/compare_baselines.py \
    ImageNet-LT:output/test_agnostic_ms/imagenet_lt/seed0,...seed1,...seed2 \
    Places-LT:...  iNat2018:output/test_agnostic/inat2018/lift+ \
    --out output/paper/tables_baselines.tex
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured_prior_adapt import (softmax, make_priors, resample_idx, em,
                                     shrink_to_uniform, split_masks, build_groups,
                                     bbse_C, bbse_pi)
from lsc_baseline import lsc_pi

SPLITS = ["Many", "Med", "Few", "All"]
BIG_C = 3000  # above this, skip BBSE/RLLS (C x C solve infeasible)


def rlls_pi(Cmat, probs_cal, probs_tgt, C, lam_rel=0.05):
    """RLLS: ridge-regularized weight-change on top of BBSE. pi_source = uniform (balanced cal).
    lam is set RELATIVE to the operator scale so it regularizes without collapsing to no-adapt."""
    pi_s = np.full(C, 1.0 / C)
    q_s = probs_cal.mean(0)
    q_t = probs_tgt.mean(0)
    A = Cmat * pi_s[None, :]                 # joint confusion; A@1 = q_s
    AtA = A.T @ A
    lam = lam_rel * np.trace(AtA) / C        # ~ fraction of mean operator eigenvalue
    dw = np.linalg.solve(AtA + lam * np.eye(C), A.T @ (q_t - q_s))
    w = np.clip(1.0 + dw, 0, None)
    pi = w * pi_s
    s = pi.sum()
    return pi / s if s > 0 else pi_s


def gsb3se_pi(Cmat, probs_cal, probs_tgt, C, mu_rel=0.5, lam_rel=0.05):
    """E3 -- GS-B3SE-style: Graph-Smoothed BBSE. BBSE weight-change with a graph-Laplacian
    smoothness penalty mu * dw^T L dw, so structurally-similar classes share estimated shift
    (variance reduction via STRUCTURE, like our grouping). Graph W = symmetrized calibration
    confusion (classes confused together are 'similar'), zero diagonal. KEY: the smoothing is
    over a C-dim weight and still needs the C x C confusion + O(C^3) solve -> like BBSE it does
    NOT scale to iNat (auto-skipped), whereas frequency-grouping reduces to K dims and does.

    mu is SCALE-MATCHED to the operator (like RLLS's lam_rel) so the Laplacian regularizes
    rather than dominating the solve and collapsing dw->0 (== a crippled no-adapt baseline).
    mu_rel=0.5 gives a genuine, competitive graph-smoothed estimator (a fair strong baseline)."""
    pi_s = np.full(C, 1.0 / C)
    q_s = probs_cal.mean(0); q_t = probs_tgt.mean(0)
    A = Cmat * pi_s[None, :]
    AtA = A.T @ A
    W = 0.5 * (Cmat + Cmat.T); np.fill_diagonal(W, 0.0)
    L = np.diag(W.sum(1)) - W                 # graph Laplacian
    sA = np.trace(AtA) / C
    lam = lam_rel * sA
    mu = mu_rel * sA / (np.trace(L) / C + 1e-12)   # match Laplacian scale to the operator
    dw = np.linalg.solve(AtA + lam * np.eye(C) + mu * L, A.T @ (q_t - q_s))
    w = np.clip(1.0 + dw, 0, None)
    pi = w * pi_s
    s = pi.sum()
    return pi / s if s > 0 else pi_s


def _simplex_grid(step):
    """All 3-expert convex weights on a lattice of the 2-simplex."""
    n = int(round(1.0 / step))
    return [np.array([i, j, n - i - j], float) / n
            for i in range(n + 1) for j in range(n + 1 - i)]


def sade_predict(zl, pi_train, subset=4000, seed=0):
    """E1a -- SADE-style test-time aggregation (single-model approximation, NOT full SADE).

    Build 3 skill-diverse virtual experts from the SAME logits by re-injecting the train
    prior (exactly how SADE's experts differ, by the LA exponent): forward = z + log pi_train
    (head-favoring), uniform = z, backward = z - log pi_train (tail-favoring). Then learn the
    convex aggregation weights label-free by MINIMIZING mean prediction entropy over the test
    batch -- SADE's transductive self-supervision spirit without the 2-view consistency term
    (we only have the TTE-averaged logit). Report as 'SADE*' (approximation)."""
    logpt = np.log(np.clip(pi_train, 1e-12, None))
    P = [softmax(zl + logpt), softmax(zl), softmax(zl - logpt)]   # fwd / uni / bwd
    N = zl.shape[0]
    rng = np.random.default_rng(seed)
    sub = rng.choice(N, size=min(subset, N), replace=False)
    Ps = [p[sub] for p in P]
    best_w, best_h = None, np.inf
    for w in _simplex_grid(0.1):
        p = w[0] * Ps[0] + w[1] * Ps[1] + w[2] * Ps[2]
        h = float(-(p * np.log(np.clip(p, 1e-12, None))).sum(1).mean())
        if h < best_h:
            best_h, best_w = h, w
    p = best_w[0] * P[0] + best_w[1] * P[1] + best_w[2] * P[2]
    return p.argmax(1)


def logadj(pi, C):
    return np.log(np.clip(pi, 1e-12, None)) - np.log(1.0 / C)


def per_root(root, tau, gamma, gmode, K, trials, total, seed, cal_frac=0.4):
    logits = np.load(f"{root}/logits.npy").astype(np.float64)
    y = np.load(f"{root}/y_true.npy").astype(int)
    cn = np.load(f"{root}/cls_num_list.npy").astype(np.float64)
    C = logits.shape[1]
    small = C <= BIG_C
    methods = (["no-adapt", "per-class EM"] + (["BBSE", "RLLS", "GS-B3SE"] if small else []) +
               ["LSC (ICML'24)", "SADE*", "GPA", "GPA-S", "oracle"])
    masks = split_masks(cn); groups = build_groups(cn, gmode, K); priors = make_priors(cn)
    pi_train = cn / cn.sum()
    cols = ["forward", "uniform", "backward"]
    rng = np.random.default_rng(seed)
    acc = {m: {c: {s: [] for s in SPLITS} for c in cols} for m in methods}
    for _ in range(trials):
        N = len(y); perm = rng.permutation(N)
        if small:
            ncal = int(cal_frac * N); cal, pool = perm[:ncal], perm[ncal:]
            Cmat = bbse_C(softmax(logits[cal]), y[cal], C); probs_cal = softmax(logits[cal])
        else:
            pool, Cmat, probs_cal = perm, None, None
        by = [pool[y[pool] == c] for c in range(C)]
        by = [b if len(b) else np.array([0]) for b in by]
        for c in cols:
            sel = resample_idx(by, priors[c], total, rng); zl, yl = logits[sel], y[sel]
            probs = softmax(zl); pi_grp = em(probs, groups=groups)
            adj = {"no-adapt": np.zeros(C),
                   "per-class EM": logadj(em(probs), C),
                   "LSC (ICML'24)": logadj(lsc_pi(zl, tau), C),
                   "GPA": logadj(pi_grp, C),
                   "GPA-S": logadj(shrink_to_uniform(pi_grp, gamma), C),
                   "oracle": logadj(priors[c], C)}
            if small:
                adj["BBSE"] = logadj(bbse_pi(Cmat, probs), C)
                adj["RLLS"] = logadj(rlls_pi(Cmat, probs_cal, probs, C), C)
                adj["GS-B3SE"] = logadj(gsb3se_pi(Cmat, probs_cal, probs, C), C)
            preds = {m: (zl + tau * adj[m]).argmax(1) for m in adj}
            preds["SADE*"] = sade_predict(zl, pi_train)
            for m in methods:
                ok = (preds[m] == yl)
                for s, msk in masks.items():
                    idx = msk[yl]
                    acc[m][c][s].append(ok[idx].mean() * 100 if idx.sum() else np.nan)
                acc[m][c]["All"].append(ok.mean() * 100)
    allmean = {m: float(np.mean([np.nanmean(acc[m][c]["All"]) for c in cols])) for m in methods}
    base = {s: np.mean([np.nanmean(acc["no-adapt"][c][s]) for c in cols]) for s in SPLITS}
    delta = {m: {s: float(np.mean([np.nanmean(acc[m][c][s]) for c in cols]) - base[s]) for s in SPLITS}
             for m in methods}
    return methods, allmean, delta


ORDER = ["no-adapt", "per-class EM", "BBSE", "RLLS", "GS-B3SE", "LSC (ICML'24)", "SADE*",
         "GPA", "GPA-S", "oracle"]


def fmt(vals):
    v = np.array(vals)
    return f"{v.mean():.2f}" if len(v) < 2 else f"{v.mean():.2f}\\tiny$\\pm${v.std(ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--out", default="output/paper/tables_baselines.tex")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--group-mode", default="quantile")
    ap.add_argument("--n-groups", type=int, default=5)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--inat-trials", type=int, default=2)
    ap.add_argument("--total", type=int, default=15000)
    args = ap.parse_args()

    results = {}   # name -> (allmean_by_root list, delta_by_root list, methods)
    for spec in args.specs:
        name, roots = spec.split(":", 1); roots = roots.split(",")
        ams, dls, meth = [], [], None
        for r in roots:
            C = np.load(f"{r}/logits.npy", mmap_mode="r").shape[1]
            tr = args.inat_trials if C > BIG_C else args.trials
            meth, am, dl = per_root(r, args.tau, args.gamma, args.group_mode, args.n_groups,
                                    tr, args.total, 0)
            ams.append(am); dls.append(dl)
        results[name] = (ams, dls, meth)
        print(f"\n=== {name}  (C={C}, seeds={len(roots)}) ===")
        for m in ORDER:
            if m not in meth: continue
            av = [a[m] for a in ams]; d = np.mean([x[m]["All"] for x in dls])
            print(f"  {m:16s} All {np.mean(av):6.2f}  (Δ {d:+.2f}) | "
                  f"Many/Med/Few Δ " + "/".join(f"{np.mean([x[m][s] for x in dls]):+.1f}"
                                                 for s in ["Many", "Med", "Few"]))

    # LaTeX T1: All accuracy per method x dataset (mean±std over seeds)
    names = list(results.keys())
    L = [f"% baselines  group={args.group_mode}/{args.n_groups} tau={args.tau} gamma={args.gamma}",
         "\\begin{tabular}{l" + "c" * len(names) + "}\\toprule",
         "Method & " + " & ".join(names) + "\\\\\\midrule"]
    for m in ORDER:
        cells = []
        for n in names:
            ams, _, meth = results[n]
            cells.append(fmt([a[m] for a in ams]) if m in meth else "--")
        if m == "oracle":
            L.append("\\midrule")
        L.append(f"{m} & " + " & ".join(cells) + "\\\\")
    L.append("\\bottomrule\\end{tabular}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
