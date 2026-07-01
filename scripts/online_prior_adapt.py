#!/usr/bin/env python
"""Direction alpha: ONLINE / streaming ESS-gated prior adaptation (post-hoc on logits).

Hypothesis (A->D->B->M->P->I->J->R):
  A: test-time prior adaptation assumes the whole test set (transductive) is available.
  D: streaming / small-batch deployment -- few samples per estimate.
  B: small-batch EM has high variance -> pi_hat is spiky -> over-correction hurts the
     small / no-shift batches (a small batch always looks "shifted" by chance).
  M: All-accuracy as a function of batch size B (regret grows as B shrinks).
  I: shrink toward uniform gated by BOTH KL-to-uniform AND effective sample size n,
     so a small batch automatically falls back to no-adapt.
  J: per-batch em_shrink with a KL-ONLY gate (no n awareness) -- over-corrects at small B.
  R: scale prior correction by distributional distance AND sample budget.

Same decision rule / EM as scripts/prior_adapt.py, so numbers are comparable.
The transductive em_shrink of prior_adapt.py == this script at B = full test (mode=cumulative).

  python scripts/online_prior_adapt.py --root output/test_agnostic/imagenet_lt/lift+ \
      --batch-sizes 50 200 1000 5000 --mode batch --tau 1 --gamma 1 --kappa 500
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


def em(probs, iters=200, tol=1e-7):
    """Saerens EM. probs = posterior under uniform ref. Returns estimated prior."""
    N, C = probs.shape
    ref = np.full(C, 1.0 / C)
    pi = np.full(C, 1.0 / C)
    for _ in range(iters):
        q = probs * (pi / ref)[None, :]
        q /= q.sum(axis=1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    return pi


def kl_to_uniform(pi):
    C = len(pi)
    p = np.clip(pi, 1e-12, None)
    return float((p * np.log(p * C)).sum())


def shrink(pi, n, method, gamma, kappa):
    """Return prior shrunk toward uniform. n = effective sample count behind pi."""
    C = len(pi)
    kl = kl_to_uniform(pi)
    if method == "naive":                 # no shrinkage at all
        b = 0.0
    elif method == "kl":                  # KL-only gate (J control: no n awareness)
        b = np.exp(-gamma * kl)
    elif method == "ess":                 # ours: multiplicative ESS gate n/(n+kappa)
        b = np.exp(-gamma * kl * n / (n + kappa))
    elif method == "floor":               # ours-alt: subtract chi-square noise floor of KL
        kl_eff = max(0.0, kl - (C - 1) / (2.0 * max(n, 1)))
        b = np.exp(-gamma * kl_eff)
    else:
        raise ValueError(method)
    return (1 - b) * pi + b / C


def logadj_from_pi(pi):
    C = len(pi)
    return np.log(np.clip(pi, 1e-12, None)) - np.log(1.0 / C)


def stream_eval(logits, y, methods, mode, batch, tau, gamma, kappa, rng):
    """Replay (logits, y) as a shuffled stream of `batch`-sized chunks; online predict."""
    N, C = logits.shape
    perm = rng.permutation(N)
    correct = {m: 0 for m in methods}
    seen = 0
    pooled_probs = None  # for cumulative mode
    for s in range(0, N, batch):
        idx = perm[s:s + batch]
        zl, yl = logits[idx], y[idx]
        probs = softmax(zl)
        if mode == "cumulative":
            pooled_probs = probs if pooled_probs is None else np.vstack([pooled_probs, probs])
            est_probs, n = pooled_probs, pooled_probs.shape[0]
        else:                              # batch: estimate from this batch only
            est_probs, n = probs, probs.shape[0]
        pi_raw = em(est_probs)
        for m in methods:
            if m == "no-adapt":
                pred = zl.argmax(1)
            else:
                pi = shrink(pi_raw, n, m, gamma, kappa)
                pred = (zl + tau * logadj_from_pi(pi)).argmax(1)
            correct[m] += (pred == yl).sum()
        seen += len(idx)
    return {m: correct[m] / seen * 100 for m in methods}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cls-num", default=None)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[50, 200, 1000, 5000])
    ap.add_argument("--mode", choices=["batch", "cumulative"], default="batch")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=500.0, help="ESS budget for the 'ess' gate")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--total", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+",
                    default=["no-adapt", "naive", "kl", "ess", "floor"])
    args = ap.parse_args()

    logits = np.load(os.path.join(args.root, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(args.root, "y_true.npy")).astype(int)
    cls_num = np.load(args.cls_num or os.path.join(args.root, "cls_num_list.npy")).astype(np.float64)
    C = logits.shape[1]
    priors = make_priors(cls_num)
    cols = ["forward", "uniform", "backward"]
    rng = np.random.default_rng(args.seed)

    print(f"\n=== ONLINE prior-adapt  {args.root}  (C={C}, mode={args.mode}, "
          f"tau={args.tau}, gamma={args.gamma}, kappa={args.kappa}) ===")
    print("legend: naive=no shrink | kl=KL-only gate (J) | ess=ESS gate (ours) | floor=noise-floor (ours-alt)")
    for B in args.batch_sizes:
        acc = {m: {c: [] for c in cols} for m in args.methods}
        for _ in range(args.trials):
            for c in cols:
                sel = resample(y, priors[c], args.total, rng)
                res = stream_eval(logits[sel], y[sel], args.methods, args.mode,
                                  B, args.tau, args.gamma, args.kappa, rng)
                for m in args.methods:
                    acc[m][c].append(res[m])
        print(f"\n-- batch size B={B} --")
        print(f"  {'method':9s} " + " ".join(f"{c:>9s}" for c in cols) + f" {'mean':>9s}")
        for m in args.methods:
            means = [np.mean(acc[m][c]) for c in cols]
            print(f"  {m:9s} " + " ".join(f"{v:9.3f}" for v in means) + f" {np.mean(means):9.3f}")


if __name__ == "__main__":
    main()
