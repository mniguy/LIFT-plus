#!/usr/bin/env python
"""E2 -- validate our LSC reimplementation against the OFFICIAL code (C2/C8).

The paper's only prior SOTA (LSC, Wei et al. ICML'24) is currently a reimplementation
(scripts/lsc_baseline.py::lsc_pi). A reviewer will not trust "we beat SOTA" off a reimpl.
This script closes that by running the OFFICIAL estimator on the SAME saved logits and
asserting equivalence (and it produces the reference numbers to match even before you clone).

Official repo (has NO iNaturalist -> running it on our iNat logits is itself a new result):
  git clone https://github.com/Stomach-ache/label-shift-correction /path/to/LSC

The official per-class estimator lives in methods/pda.py (confidence-filtered self-trained
marginal). Its exact function name may differ across commits, so point at it explicitly:
  --lsc-repo /path/to/LSC  --lsc-entry methods.pda:estimate_prior
The entry must be callable as f(logits_or_probs, ...)->pi (C,). If the official API needs a
model/loader instead of logits, port its inner estimator loop into a thin wrapper of that
signature (a few lines) -- the point is to feed it OUR logits, not retrain.

Without --lsc-repo the script still prints our lsc_pi reference (All acc + group-mean prior)
per dataset; the official run must reproduce these within --tol.

  python scripts/validate_lsc.py \
    ImageNet-LT:output/test_agnostic_ms/imagenet_lt/seed0 \
    Places-LT:output/test_agnostic_ms/places_lt/seed0 \
    iNat2018:output/test_agnostic/inat2018/lift+ \
    [--lsc-repo /path/to/LSC --lsc-entry methods.pda:estimate_prior]
"""
import argparse
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured_prior_adapt import softmax, make_priors, resample_idx, split_masks, logadj
from lsc_baseline import lsc_pi

COLS = ["forward", "uniform", "backward"]


def load_entry(repo, entry):
    """Best-effort import of the official estimator: 'pkg.mod:callable'."""
    sys.path.insert(0, repo)
    mod_name, _, fn = entry.partition(":")
    mod = importlib.import_module(mod_name)
    if not fn:
        raise ValueError("give --lsc-entry as module:callable")
    return getattr(mod, fn)


def group_means(pi, cn):
    """Summarize a prior by its Many/Med/Few group means (stable fingerprint to compare)."""
    m = split_masks(cn)
    return {s: float(pi[msk].mean()) for s, msk in m.items()}


def acc_of(logits, y, pi_fn, cn, tau, total, trials, seed=0):
    """All acc (mean over forward/uniform/backward) using the given estimator pi_fn(zl)->pi."""
    C = logits.shape[1]; priors = make_priors(cn); rng = np.random.default_rng(seed)
    by = [np.where(y == c)[0] for c in range(C)]
    by = [b if len(b) else np.array([0]) for b in by]
    per_col = []
    fp = None  # fingerprint prior from the first forward trial
    for _ in range(trials):
        col_acc = []
        for c in COLS:
            sel = resample_idx(by, priors[c], total, rng); zl, yl = logits[sel], y[sel]
            pi = pi_fn(zl)
            if c == "forward" and fp is None:
                fp = pi
            ok = ((zl + tau * logadj(pi, C)).argmax(1) == yl)
            col_acc.append(ok.mean() * 100)
        per_col.append(np.mean(col_acc))
    return float(np.mean(per_col)), fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--lsc-repo", default=None)
    ap.add_argument("--lsc-entry", default="methods.pda:estimate_prior")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--total", type=int, default=15000)
    ap.add_argument("--tol", type=float, default=0.15, help="max |official-reimpl| All-acc gap to PASS")
    args = ap.parse_args()

    official = None
    if args.lsc_repo:
        try:
            fn = load_entry(args.lsc_repo, args.lsc_entry)
            official = lambda zl: np.asarray(fn(softmax(zl)), float)   # official on OUR logits
            print(f"[official] loaded {args.lsc_entry} from {args.lsc_repo}")
        except Exception as e:
            print(f"[official] FAILED to load ({e!r}); printing reimpl reference only.\n"
                  f"           clone the repo and pass a matching --lsc-entry (see header).")
    else:
        print("[official] no --lsc-repo given; printing reimpl reference numbers to match.\n")

    print(f"{'dataset':12s} {'C':>6s} {'reimpl All':>11s} {'official All':>13s} {'|gap|':>7s}  verdict")
    for spec in args.specs:
        name, root = spec.split(":", 1); root = root.split(",")[0]
        logits = np.load(f"{root}/logits.npy").astype(np.float64)
        y = np.load(f"{root}/y_true.npy").astype(int)
        cn = np.load(f"{root}/cls_num_list.npy").astype(np.float64)
        C = logits.shape[1]

        re_acc, re_fp = acc_of(logits, y, lambda zl: lsc_pi(zl, args.tau),
                               cn, args.tau, args.total, args.trials)
        if official is not None:
            of_acc, of_fp = acc_of(logits, y, official, cn, args.tau, args.total, args.trials)
            gap = abs(of_acc - re_acc)
            verdict = "PASS" if gap <= args.tol else "FAIL -> use official in table"
            print(f"{name:12s} {C:6d} {re_acc:11.2f} {of_acc:13.2f} {gap:7.2f}  {verdict}")
            gm_r, gm_o = group_means(re_fp, cn), group_means(of_fp, cn)
            print(f"             prior group-mean Many/Med/Few  reimpl "
                  f"{gm_r['Many']:.2e}/{gm_r['Med']:.2e}/{gm_r['Few']:.2e}  official "
                  f"{gm_o['Many']:.2e}/{gm_o['Med']:.2e}/{gm_o['Few']:.2e}")
        else:
            gm_r = group_means(re_fp, cn)
            print(f"{name:12s} {C:6d} {re_acc:11.2f} {'--':>13s} {'--':>7s}  reference "
                  f"(prior grp-mean {gm_r['Many']:.2e}/{gm_r['Med']:.2e}/{gm_r['Few']:.2e})")

    if official is None:
        print("\nNext: clone the official repo (see header), re-run with --lsc-repo/--lsc-entry;\n"
              "PASS = our reimpl is faithful (removes the 'reimpl not trusted' attack); a large\n"
              "gap => swap the official estimator into scripts/compare_baselines.py's LSC row.")


if __name__ == "__main__":
    main()
