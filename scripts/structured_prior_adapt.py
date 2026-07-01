#!/usr/bin/env python
"""Direction beta: STRUCTURED / sample-aware prior estimation (post-hoc on logits).

Hypothesis (A->D->B->M->P->I->J->R):
  A: the test prior must be estimated as a free per-class categorical.
  D: many classes, few test samples per class (esp. tail).
  B: full-categorical EM variance ~ C/N -> tail pi_hat dominated by noise (B1) ->
     wrong prior injected on rare classes (B2) -> the forward-Few / backward-Many
     accuracy crater + the residual oracle gap (B3).
  M: per-class prior-estimation L1 error, decomposed head/med/few.
  I: estimate the prior at a LOWER granularity the sample budget supports --
     piecewise-constant within data-driven frequency groups, via constrained EM.
  J: full categorical EM (current #4, em_shrink); plus BBSE (standard label-shift est.).
  R: estimate test priors at the granularity your test sample budget supports.

De-circularization: the ESTIMATION groups (build_groups) are data-driven from TRAIN
counts only (quantile / log-width / kmeans), independent of the fixed Many/Med/Few
thresholds used for EVALUATION (split_masks). Sweep --n-groups to show robustness.

Same EM / decision rule as scripts/prior_adapt.py. Importable: see scripts/sweep_beta.py.

  python scripts/structured_prior_adapt.py --root output/test_agnostic/imagenet_lt/lift+ \
      --group-mode quantile --n-groups 5 --tau 1 --gamma 1
"""
import argparse
import os

import numpy as np


# ---------- shared estimation machinery (mirrors scripts/prior_adapt.py) ----------
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


def resample_idx(by, pi, total, rng):
    cnt = rng.multinomial(total, pi)
    return np.concatenate([rng.choice(by[c], size=k, replace=True)
                           for c, k in enumerate(cnt) if k > 0 and len(by[c]) > 0])


def em(probs, groups=None, iters=200, tol=1e-7):
    """Saerens EM; if `groups` given, constrain pi piecewise-constant within each group."""
    N, C = probs.shape
    ref = np.full(C, 1.0 / C)
    pi = np.full(C, 1.0 / C)
    for _ in range(iters):
        q = probs * (pi / ref)[None, :]
        q /= q.sum(axis=1, keepdims=True)
        new = q.mean(0)
        if groups is not None:                 # project onto piecewise-constant
            proj = new.copy()
            for g in groups:
                proj[g] = new[g].mean()
            new = proj                          # averaging within a partition keeps sum=1
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    return pi


def kl_to_uniform(pi):
    C = len(pi)
    p = np.clip(pi, 1e-12, None)
    return float((p * np.log(p * C)).sum())


def bbse_C(probs_cal, y_cal, C):
    """Soft confusion C[i,j] = P(yhat=i | y=j) from a labeled calibration set."""
    Cmat = np.eye(C)
    for j in range(C):
        idx = np.where(y_cal == j)[0]
        if len(idx):
            Cmat[:, j] = probs_cal[idx].mean(0)
    return Cmat


def bbse_pi(Cmat, probs_tgt, ridge=1e-3):
    """Black-Box Shift Estimation: solve C @ pi = q for the target prior, project to simplex."""
    q = probs_tgt.mean(0)
    A = Cmat
    pi = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ q)
    pi = np.clip(pi, 0, None)
    s = pi.sum()
    return pi / s if s > 0 else np.full(len(q), 1.0 / len(q))


def shrink_to_uniform(pi, gamma):
    C = len(pi)
    b = np.exp(-gamma * kl_to_uniform(pi))
    return (1 - b) * pi + b / C


# ---------- grouping ----------
def split_masks(cls_num):
    """Fixed standard LT eval split (NEVER used for estimation grouping)."""
    many = cls_num > 100; few = cls_num < 20; med = ~many & ~few
    return {"Many": many, "Med": med, "Few": few}


def build_groups(cls_num, mode, K):
    """Data-driven ESTIMATION groups from TRAIN counts only (de-circularized)."""
    C = len(cls_num)
    if mode == "fixed":                        # == eval split (circular baseline)
        return [np.where(m)[0] for m in split_masks(cls_num).values()]
    order = np.argsort(cls_num)                # ascending count
    if mode == "quantile":                     # equal #classes per group
        return [np.sort(s) for s in np.array_split(order, K)]
    if mode == "logwidth":                     # equal-width bins in log-count
        lc = np.log(cls_num)
        edges = np.linspace(lc.min(), lc.max(), K + 1); edges[-1] += 1e-9
        gid = np.clip(np.digitize(lc, edges[1:-1]), 0, K - 1)
        return [np.where(gid == k)[0] for k in range(K) if (gid == k).any()]
    raise ValueError(mode)


# ---------- per-method prior estimate ----------
def estimate_pi(probs, method, groups, gamma, Cmat=None):
    if method == "em":
        return em(probs)
    if method == "em_shrink":
        return shrink_to_uniform(em(probs), gamma)
    if method == "em_group":
        return em(probs, groups=groups)
    if method == "em_group_shrink":
        return shrink_to_uniform(em(probs, groups=groups), gamma)
    if method == "bbse":
        return bbse_pi(Cmat, probs)
    if method == "bbse_shrink":
        return shrink_to_uniform(bbse_pi(Cmat, probs), gamma)
    raise ValueError(method)


def logadj(pi, C):
    return np.log(np.clip(pi, 1e-12, None)) - np.log(1.0 / C)


# ---------- evaluation (importable) ----------
def run_eval(logits, y, cls_num, methods, tau, gamma, group_mode, n_groups,
             trials, total, seed, cal_frac=0.4, compute_l1=True):
    """Returns acc[method][col][split] and l1[estimator][col][split] (lists over trials)."""
    C = logits.shape[1]
    eval_masks = split_masks(cls_num)
    groups = build_groups(cls_num, group_mode, n_groups)
    priors = make_priors(cls_num)
    logp_oracle = {k: logadj(p, C) for k, p in priors.items()}
    cols = ["forward", "uniform", "backward"]
    splits = ["Many", "Med", "Few", "All"]
    rng = np.random.default_rng(seed)
    need_cal = any(m.startswith("bbse") for m in methods)
    l1_ests = (["em", "em_group"] + (["bbse"] if need_cal else [])) if compute_l1 else []

    acc = {m: {c: {s: [] for s in splits} for c in cols} for m in methods}
    l1 = {e: {c: {s: [] for s in ["Many", "Med", "Few"]} for c in cols} for e in l1_ests}

    for _ in range(trials):
        N = len(y); perm = rng.permutation(N)
        if need_cal:
            ncal = int(cal_frac * N)
            cal, pool = perm[:ncal], perm[ncal:]
            Cmat = bbse_C(softmax(logits[cal]), y[cal], C)
        else:
            pool, Cmat = perm, None
        by = [pool[y[pool] == c] for c in range(C)]
        by = [b if len(b) else np.array([0]) for b in by]   # guard empty class in pool

        for c in cols:
            sel = resample_idx(by, priors[c], total, rng)
            zl, yl = logits[sel], y[sel]
            probs = softmax(zl)
            for m in methods:
                if m == "no-adapt":
                    adj = np.zeros(C)
                elif m == "oracle":
                    adj = logp_oracle[c]
                else:
                    adj = logadj(estimate_pi(probs, m, groups, gamma, Cmat), C)
                pred = (zl + tau * adj).argmax(1)
                correct = (pred == yl)
                for s, msk in eval_masks.items():
                    idx = msk[yl]
                    acc[m][c][s].append(correct[idx].mean() * 100 if idx.sum() else np.nan)
                acc[m][c]["All"].append(correct.mean() * 100)
            for e in l1_ests:
                pi_hat = estimate_pi(probs, e, groups, gamma, Cmat)
                for s, msk in eval_masks.items():
                    l1[e][c][s].append(np.abs(pi_hat[msk] - priors[c][msk]).sum())
    return acc, l1, cols, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cls-num", default=None)
    ap.add_argument("--group-mode", choices=["fixed", "quantile", "logwidth"], default="quantile")
    ap.add_argument("--n-groups", type=int, default=5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--total", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+",
                    default=["no-adapt", "em", "em_shrink", "bbse",
                             "em_group", "em_group_shrink", "oracle"])
    args = ap.parse_args()

    logits = np.load(os.path.join(args.root, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(args.root, "y_true.npy")).astype(int)
    cls_num = np.load(args.cls_num or os.path.join(args.root, "cls_num_list.npy")).astype(np.float64)

    acc, l1, cols, splits = run_eval(logits, y, cls_num, args.methods, args.tau, args.gamma,
                                     args.group_mode, args.n_groups, args.trials, args.total, args.seed)
    cell = np.nanmean
    print(f"\n=== STRUCTURED prior-adapt  {args.root}  (C={logits.shape[1]}, "
          f"group={args.group_mode}/{args.n_groups}, tau={args.tau}, gamma={args.gamma}) ===")

    print("\n[1] All accuracy (mean over the 3 shift conditions)")
    print(f"  {'method':16s} " + " ".join(f"{c:>9s}" for c in cols) + f" {'mean':>9s}")
    for m in args.methods:
        v = [cell(acc[m][c]["All"]) for c in cols]
        print(f"  {m:16s} " + " ".join(f"{x:9.3f}" for x in v) + f" {np.mean(v):9.3f}")

    print("\n[2] Head/Med/Few split (delta vs no-adapt, mean/3) -- crater?")
    print(f"  {'method':16s} {'Many':>8s} {'Med':>8s} {'Few':>8s} {'All':>8s}")
    base = {s: np.mean([cell(acc['no-adapt'][c][s]) for c in cols]) for s in splits}
    for m in args.methods:
        ds = [np.mean([cell(acc[m][c][s]) for c in cols]) - base[s] for s in splits]
        print(f"  {m:16s} " + " ".join(f"{x:+8.2f}" for x in ds))

    print("\n[3] prior-estimation L1 error by split (lower=better; M metric, mean/3)")
    print(f"  {'estimator':16s} {'Many':>8s} {'Med':>8s} {'Few':>8s}")
    for e in l1:
        v = [np.mean([cell(l1[e][c][s]) for c in cols]) for s in ["Many", "Med", "Few"]]
        print(f"  {e:16s} " + " ".join(f"{x:8.4f}" for x in v))


if __name__ == "__main__":
    main()
