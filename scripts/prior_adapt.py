#!/usr/bin/env python
"""Method zoo for test-agnostic prior adaptation (post-hoc on saved logits).

Decision rule: argmax_c [ logit_c + tau * adj_c ], adj from each estimator:
  no-adapt  : adj = 0                                  (LIFT+; optimal only for uniform)
  oracle    : adj = log pi_target                      (knows the test prior; upper bound)
  em        : adj = log pi_EM   (vanilla Saerens EM)
  em_conf   : EM with samples weighted by confidence (max prob)  -> less estimation noise
  rpa(ours) : em_conf + ADAPTIVE shrinkage toward uniform:
                pi_final = (1-b)*pi + b*uniform,  b = exp(-gamma * KL(pi || uniform))
              data looks uniform -> b~1 (protect uniform); shifted -> b~0 (keep gains)

Evaluated by resampling the (balanced) test set to forward/uniform/backward priors.
Logits-only, so it iterates without re-running the model.

  python scripts/prior_adapt.py --root output/test_agnostic/imagenet_lt/lift+ --tau 1 --gamma 3
"""
import argparse
import os

import numpy as np


def softmax(z, T=1.0):
    z = (z / T) - (z / T).max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_priors(cls_num):
    C = len(cls_num)
    fwd = cls_num / cls_num.sum()
    uni = np.full(C, 1.0 / C)
    order = np.argsort(cls_num)
    bwd = np.empty(C); bwd[order] = fwd[order[::-1]]; bwd /= bwd.sum()
    return {"forward": fwd, "uniform": uni, "backward": bwd}


def resample(y, pi, total, rng):
    by = [np.where(y == c)[0] for c in range(len(pi))]
    cnt = rng.multinomial(total, pi)
    return np.concatenate([rng.choice(by[c], size=k, replace=True)
                           for c, k in enumerate(cnt) if k > 0 and len(by[c]) > 0])


def em(probs, weights=None, iters=200, tol=1e-7):
    """Weighted Saerens EM. probs = posterior under uniform ref. Returns estimated prior."""
    N, C = probs.shape
    ref = np.full(C, 1.0 / C)
    pi = np.full(C, 1.0 / C)
    w = np.ones(N) if weights is None else weights
    wsum = w.sum()
    for _ in range(iters):
        q = probs * (pi / ref)[None, :]
        q /= q.sum(axis=1, keepdims=True)
        new = (w[:, None] * q).sum(0) / wsum
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    return pi


def kl_to_uniform(pi):
    C = len(pi); u = 1.0 / C
    p = np.clip(pi, 1e-12, None)
    return float((p * np.log(p / u)).sum())


def estimate_logadj(zl, method, gamma):
    """Return the per-class log-adjustment vector for a method, given subset logits."""
    C = zl.shape[1]
    logu = np.log(np.full(C, 1.0 / C))
    if method == "no-adapt":
        return np.zeros(C)
    probs = softmax(zl)
    if method == "em":
        pi = em(probs)
    elif method == "em_conf":
        pi = em(probs, weights=probs.max(1))
    elif method == "em_shrink":
        pi = em(probs)
        b = np.exp(-gamma * kl_to_uniform(pi))
        pi = (1 - b) * pi + b / C
    elif method == "rpa":
        pi = em(probs, weights=probs.max(1))
        b = np.exp(-gamma * kl_to_uniform(pi))
        pi = (1 - b) * pi + b / C
    else:
        raise ValueError(method)
    return np.log(np.clip(pi, 1e-12, None)) - logu  # subtract logu so "uniform" => 0 adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cls-num", default=None)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=3.0, help="rpa adaptive-shrink strength")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--total", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+",
                    default=["no-adapt", "em", "em_conf", "rpa", "oracle"])
    args = ap.parse_args()

    logits = np.load(os.path.join(args.root, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(args.root, "y_true.npy")).astype(int)
    cls_num = np.load(args.cls_num or os.path.join(args.root, "cls_num_list.npy")).astype(np.float64)
    N, C = logits.shape
    priors = make_priors(cls_num)
    logp = {k: np.log(np.clip(p, 1e-12, None)) - np.log(1.0 / C) for k, p in priors.items()}
    cols = ["forward", "uniform", "backward"]
    rng = np.random.default_rng(args.seed)

    acc = {mth: {c: [] for c in cols} for mth in args.methods}
    for _ in range(args.trials):
        for c in cols:
            sel = resample(y, priors[c], args.total, rng)
            zl, yl = logits[sel], y[sel]
            for mth in args.methods:
                adj = logp[c] if mth == "oracle" else estimate_logadj(zl, mth, args.gamma)
                pred = (zl + args.tau * adj).argmax(1)
                acc[mth][c].append((pred == yl).mean() * 100)

    print(f"\n=== prior-adapt  {args.root}  (C={C}, tau={args.tau}, gamma={args.gamma}, "
          f"trials={args.trials}x{args.total}) ===")
    print(f"  {'method':10s} " + " ".join(f"{c:>9s}" for c in cols) + f" {'mean':>9s}")
    for mth in args.methods:
        means = [np.mean(acc[mth][c]) for c in cols]
        print(f"  {mth:10s} " + " ".join(f"{m:9.3f}" for m in means) + f" {np.mean(means):9.3f}")


if __name__ == "__main__":
    main()
