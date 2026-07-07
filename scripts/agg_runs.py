#!/usr/bin/env python
"""Generic tabulator: print All/Head/Med/Few for every completed run under a root.

Reusable across experiments (group-scale, caption-geom, caption-apply, ...).
    python scripts/agg_runs.py output/group_scale
    python scripts/agg_runs.py output/caption_geom --sort few
"""
import argparse
import glob
import os
import re


def read_result(run_dir):
    for f in reversed(sorted(glob.glob(os.path.join(run_dir, "log-*.txt")))):
        txt = open(f, errors="ignore").read()
        m = re.findall(r"\* Many:\s*([\d.]+)%\s*Med:\s*([\d.]+)%\s*Few:\s*([\d.]+)%", txt)
        if not m:
            continue
        ov = re.findall(r"\* Overall accuracy:\s*([\d.]+)%", txt)
        head, med, few = map(float, m[-1])
        return {"All": float(ov[-1]) if ov else float("nan"), "Head": head, "Med": med, "Few": few}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sort", default="path", choices=["path", "few", "all"])
    args = ap.parse_args()

    rows = []
    for f in glob.glob(os.path.join(args.root, "**", "log-*.txt"), recursive=True):
        d = os.path.dirname(f)
        r = read_result(d)
        if r:
            rows.append((os.path.relpath(d, args.root), r))
    rows = {lbl: r for lbl, r in rows}  # dedupe dirs
    items = list(rows.items())
    if args.sort == "few":
        items.sort(key=lambda kv: -kv[1]["Few"])
    elif args.sort == "all":
        items.sort(key=lambda kv: -kv[1]["All"])
    else:
        items.sort()

    print(f"# {args.root}   ({len(items)} runs)")
    print(f"{'run':<44} {'All':>6} {'Head':>6} {'Med':>6} {'Few':>6}")
    for lbl, r in items:
        print(f"{lbl:<44} {r['All']:>6.2f} {r['Head']:>6.2f} {r['Med']:>6.2f} {r['Few']:>6.2f}")


if __name__ == "__main__":
    main()
