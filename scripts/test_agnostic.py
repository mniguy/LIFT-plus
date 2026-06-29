#!/usr/bin/env python
"""Test-agnostic (prior-shift) evaluation.

Given a model's raw test logits + true labels (balanced test set), build resampled
test sets with three class-priors -- forward (train-like / head-heavy), uniform,
backward (tail-heavy) -- and report top-1 accuracy under three decision rules:

  no-adapt : argmax(raw logit)                     what LIFT+ does; optimal only for uniform
  EM       : estimate the test prior unsupervised (Saerens EM) -> argmax(logit + tau*log pi_est)
  oracle   : argmax(logit + tau*log pi_target)     upper bound (knows the target prior)

A method that adapts to the (unknown) test prior (EM) should beat no-adapt under
forward/backward; the mean over the three priors is the test-agnostic headline.

Requires logits.npy (run with SAVE_LOGITS True) + y_true.npy + cls_num_list.npy.

Usage:
  python scripts/test_agnostic.py --root output/test_agnostic/imagenet_lt/lift+
"""
import argparse
import os

import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def estimate_prior_em(probs, iters=200, tol=1e-7):
    """Saerens-Latinne-Decaestecker EM. probs = model posterior under a uniform
    reference prior (LA-trained logits -> uniform ref). Returns estimated prior."""
    C = probs.shape[1]
    ref = np.full(C, 1.0 / C)
    pi = np.full(C, 1.0 / C)
    for _ in range(iters):
        q = probs * (pi / ref)[None, :]
        q /= q.sum(axis=1, keepdims=True)
        new = q.mean(axis=0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return pi


def make_priors(cls_num):
    C = len(cls_num)
    fwd = cls_num / cls_num.sum()
    uni = np.full(C, 1.0 / C)
    order = np.argsort(cls_num)            # ascending freq
    bwd = np.empty(C)
    bwd[order] = fwd[order[::-1]]          # mirror: rank i gets freq of rank C-1-i
    bwd /= bwd.sum()
    return {"forward": fwd, "uniform": uni, "backward": bwd}


def resample(y, pi, total, rng):
    """Indices drawn so class proportions ~ pi (with replacement within a class)."""
    C = len(pi)
    by_class = [np.where(y == c)[0] for c in range(C)]
    counts = rng.multinomial(total, pi)
    sel = [rng.choice(by_class[c], size=k, replace=True)
           for c, k in enumerate(counts) if k > 0 and len(by_class[c]) > 0]
    return np.concatenate(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with logits.npy + y_true.npy")
    ap.add_argument("--cls-num", default=None, help="cls_num_list.npy (default <root>/cls_num_list.npy)")
    ap.add_argument("--tau", type=float, nargs="+", default=[1.0],
                    help="logit-adjustment strength(s); pass several to sweep (e.g. --tau 0.25 0.5 1 2)")
    ap.add_argument("--trials", type=int, default=5, help="resampling trials to average")
    ap.add_argument("--total", type=int, default=20000, help="samples per resampled test set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logits = np.load(os.path.join(args.root, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(args.root, "y_true.npy")).astype(int)
    cls_num = np.load(args.cls_num or os.path.join(args.root, "cls_num_list.npy")).astype(np.float64)
    N, C = logits.shape
    assert len(y) == N and len(cls_num) == C, f"shape mismatch N={N} C={C} y={len(y)} cls={len(cls_num)}"

    priors = make_priors(cls_num)
    logp = {k: np.log(np.clip(p, 1e-12, None)) for k, p in priors.items()}
    cols = ["forward", "uniform", "backward"]
    rng = np.random.default_rng(args.seed)

    rows = ["no-adapt"]
    for t in args.tau:
        rows += [f"EM tau={t:g}", f"oracle tau={t:g}"]
    acc = {r: {c: [] for c in cols} for r in rows}

    for _ in range(args.trials):
        for c in cols:
            sel = resample(y, priors[c], args.total, rng)
            zl, yl = logits[sel], y[sel]
            acc["no-adapt"][c].append((zl.argmax(1) == yl).mean() * 100)
            logpi_em = np.log(np.clip(estimate_prior_em(softmax(zl)), 1e-12, None))  # tau-independent
            for t in args.tau:
                acc[f"EM tau={t:g}"][c].append(((zl + t * logpi_em).argmax(1) == yl).mean() * 100)
                acc[f"oracle tau={t:g}"][c].append(((zl + t * logp[c]).argmax(1) == yl).mean() * 100)

    print(f"\n=== test-agnostic  {args.root}  (N={N}, C={C}, "
          f"trials={args.trials}x{args.total}) ===")
    print(f"  {'rule':14s} " + " ".join(f"{c:>9s}" for c in cols) + f" {'mean':>9s}")
    for r in rows:
        means = [np.mean(acc[r][c]) for c in cols]
        print(f"  {r:14s} " + " ".join(f"{m:9.3f}" for m in means) + f" {np.mean(means):9.3f}")
    print("\n  >> WIN if EM (≈oracle) >> no-adapt on forward/backward; 'mean' is the test-agnostic headline.")
    print("     (tau interacts with the logit scale -- sweep it; uniform col should ≈ no-adapt.)")


if __name__ == "__main__":
    main()
