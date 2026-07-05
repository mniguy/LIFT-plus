#!/usr/bin/env python
"""Aggregate the 2x2 (warmup x classifier_scale) freq_inv-gating grid on seeds 8,9.

Reads <root>/<dataset>/<warmup|nowarmup>_scale{25,30}_seed{8,9}/log-*.txt and prints,
per dataset, each of the 4 cells at seed 8 and 9 plus their mean (All/Head/Med/Few).

    python scripts/agg_gate_2x2.py --root output/gate_2x2 --datasets imagenet_lt places_lt
"""
import argparse
import glob
import os
import re

SPLITS = ["All", "Head", "Med", "Few"]


def read_result(run_dir):
    for f in reversed(sorted(glob.glob(os.path.join(run_dir, "log-*.txt")))):
        txt = open(f, errors="ignore").read()
        m = re.findall(r"\* Many:\s*([\d.]+)%\s*Med:\s*([\d.]+)%\s*Few:\s*([\d.]+)%", txt)
        if not m:
            continue
        ov = re.findall(r"\* Overall accuracy:\s*([\d.]+)%", txt)
        head, med, few = map(float, m[-1])
        return {"All": float(ov[-1]) if ov else float("nan"),
                "Head": head, "Med": med, "Few": few}
    return None


def fmt(label, seed, r):
    if r is None:
        return "%-18s %-5s | %s" % (label, seed, "(missing)")
    return "%-18s %-5s | %6.2f %6.2f %6.2f %6.2f" % (
        label, seed, r["All"], r["Head"], r["Med"], r["Few"])


def report(root, ds, seeds):
    print("\n" + "=" * 62)
    print(ds)
    print("=" * 62)
    print("%-18s %-5s | %6s %6s %6s %6s" % ("cell", "seed", *SPLITS))
    for wu in ["warmup", "nowarmup"]:
        for sc in ["25", "30"]:
            cell = "%s_scale%s" % (wu, sc)
            got = []
            for s in seeds:
                r = read_result(os.path.join(root, ds, "%s_seed%s" % (cell, s)))
                print(fmt(cell, s, r))
                if r:
                    got.append(r)
            if len(got) == len(seeds) and got:
                mean = {k: sum(g[k] for g in got) / len(got) for k in SPLITS}
                print(fmt(cell, "mean", mean))
            print("-" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/gate_2x2")
    ap.add_argument("--datasets", nargs="+", default=["imagenet_lt", "places_lt"])
    ap.add_argument("--seeds", nargs="+", default=["8", "9"])
    args = ap.parse_args()
    for ds in args.datasets:
        report(args.root, ds, args.seeds)


if __name__ == "__main__":
    main()
