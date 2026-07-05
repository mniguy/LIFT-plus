#!/usr/bin/env python
"""Aggregate the seed 0-10 ablation for the warmup recipe (= final_tte/verify_best).

Reads the logs produced by scripts/run_seed_ablation.sh:
    <root>/<dataset>/method_seed<s>/log-*.txt      (warmup recipe)
    <root>/<dataset>/baseline_seed<s>/log-*.txt     (semantic no-warmup, optional)

Reports per-seed All/Head/Med/Few, mean +/- std, and (if baselines exist) the
PAIRED difference method-baseline per seed. The paired mean +/- std on Few is the
statistic that decides whether the +0.5 tail gain survives the run-noise band.

    python scripts/agg_seed_ablation.py --root output/final_tte/seed_ablation \
        --datasets imagenet_lt places_lt
"""
import argparse
import glob
import os
import re

SPLITS = ["All", "Head", "Med", "Few"]


def read_result(run_dir):
    """Latest completed log in run_dir -> dict(All, Head, Med, Few) or None."""
    logs = sorted(glob.glob(os.path.join(run_dir, "log-*.txt")))
    for f in reversed(logs):  # newest first
        txt = open(f, errors="ignore").read()
        m = re.findall(r"\* Many:\s*([\d.]+)%\s*Med:\s*([\d.]+)%\s*Few:\s*([\d.]+)%", txt)
        if not m:
            continue
        ov = re.findall(r"\* Overall accuracy:\s*([\d.]+)%", txt)
        head, med, few = map(float, m[-1])
        allv = float(ov[-1]) if ov else float("nan")
        return {"All": allv, "Head": head, "Med": med, "Few": few}
    return None


def collect(root, dataset, prefix):
    """seed -> result dict, over <root>/<dataset>/<prefix>_seed<s>."""
    out = {}
    for d in glob.glob(os.path.join(root, dataset, prefix + "_seed*")):
        m = re.search(r"_seed(\d+)$", d)
        if not m:
            continue
        r = read_result(d)
        if r is not None:
            out[int(m.group(1))] = r
    return out


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n  # population std (fixed set of seeds)
    return mu, var ** 0.5


def fmt_row(label, r):
    return "%-8s | %6s %6s %6s %6s" % (
        label,
        *("%.2f" % r[s] if r and s in r else "  -  " for s in SPLITS),
    )


def report(root, dataset, baseline_root=None):
    method = collect(root, dataset, "method")
    base = collect(baseline_root or root, dataset, "baseline")
    seeds = sorted(method)
    print("\n" + "=" * 78)
    print("%s   method seeds=%s%s" % (
        dataset,
        seeds if seeds else "(none found)",
        "   baseline seeds=%s" % sorted(base) if base else "",
    ))
    print("=" * 78)
    print("%-8s | %6s %6s %6s %6s" % ("", *SPLITS))
    for s in seeds:
        print(fmt_row("m/seed%d" % s, method[s]))
    # method mean/std
    for stat, fn in (("mean", lambda v: mean_std(v)[0]), ("std", lambda v: mean_std(v)[1])):
        r = {sp: fn([method[s][sp] for s in seeds]) for sp in SPLITS}
        print(fmt_row("METHOD " + stat, r))

    if not base:
        print("(no baseline runs found -- run with RUN_BASELINE=1 for a paired test)")
        return

    print("-" * 78)
    paired = sorted(set(method) & set(base))
    r = {sp: mean_std([base[s][sp] for s in paired])[0] for sp in SPLITS}
    print(fmt_row("BASE mean", r))

    # paired difference method - baseline, per split
    print("-" * 78)
    print("Paired  method - baseline  (over %d shared seeds: %s)" % (len(paired), paired))
    for sp in SPLITS:
        diffs = [method[s][sp] - base[s][sp] for s in paired]
        mu, sd = mean_std(diffs)
        verdict = ""
        if sp == "Few":
            verdict = "  <- SIGNAL" if abs(mu) > sd and mu > 0 else "  <- within noise"
        print("  d%-5s : mean %+0.2f  std %0.2f  (min %+0.2f, max %+0.2f)%s"
              % (sp, mu, sd, min(diffs), max(diffs), verdict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/final_tte/seed_ablation")
    ap.add_argument("--baseline-root", default=None,
                    help="read baseline_seed* from here instead of --root "
                         "(e.g. reuse output/seed_ablation baselines for the gate variant)")
    ap.add_argument("--datasets", nargs="+", default=["imagenet_lt", "places_lt"])
    args = ap.parse_args()
    for ds in args.datasets:
        report(args.root, ds, baseline_root=args.baseline_root)


if __name__ == "__main__":
    main()
