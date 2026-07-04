#!/usr/bin/env python
"""Aggregate the LR x num_epochs grid (scripts/run_lr_epoch_sweep.sh).

Reads <root>/<dataset>/lr<lr>_ep<ep>/log-*.txt and prints, per dataset:
  - a full table (All/Head/Med/Few) sorted by Few,
  - a Few pivot matrix (rows = lr, cols = num_epochs),
  - the best cell by Few and by All, with delta vs a single-seed reference baseline.

The reference baselines are the 11-seed LIFT+ means from output/seed_ablation
(ImageNet Few 74.00, Places Few 52.10). A single-seed sweep cell only beats them
if it clears the ~0.4 run-noise band -- then re-run that cell across seeds.

    python scripts/agg_lr_epoch_sweep.py --root output/lr_epoch_sweep \
        --datasets imagenet_lt places_lt
"""
import argparse
import glob
import os
import re

SPLITS = ["All", "Head", "Med", "Few"]
REF_BASELINE = {  # 11-seed LIFT+ mean Few (from output/seed_ablation)
    "imagenet_lt": {"Few": 74.00, "All": 78.42},
    "places_lt": {"Few": 52.10, "All": 51.99},
}


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


def collect(root, dataset):
    """(lr_str, ep_int) -> result dict."""
    cells = {}
    for d in glob.glob(os.path.join(root, dataset, "lr*_ep*")):
        m = re.search(r"lr([0-9.eE+\-]+)_ep(\d+)$", os.path.basename(d))
        if not m:
            continue
        r = read_result(d)
        if r is not None:
            cells[(m.group(1), int(m.group(2)))] = r
    return cells


def report(root, dataset):
    cells = collect(root, dataset)
    print("\n" + "=" * 72)
    print("%s   %d cells" % (dataset, len(cells)))
    print("=" * 72)
    if not cells:
        print("(no runs found)")
        return
    ref = REF_BASELINE.get(dataset, {})
    ref_few = ref.get("Few")

    # full table sorted by Few desc
    print("%-8s %-4s | %6s %6s %6s %6s | %s" %
          ("lr", "ep", "All", "Head", "Med", "Few",
           "dFew vs base" if ref_few else ""))
    for (lr, ep), r in sorted(cells.items(), key=lambda kv: -kv[1]["Few"]):
        dtxt = "%+0.2f" % (r["Few"] - ref_few) if ref_few else ""
        print("%-8s %-4d | %6.2f %6.2f %6.2f %6.2f | %s" %
              (lr, ep, r["All"], r["Head"], r["Med"], r["Few"], dtxt))

    # Few pivot matrix
    lrs = sorted({k[0] for k in cells}, key=float)
    eps = sorted({k[1] for k in cells})
    print("\nFew pivot (rows=lr, cols=num_epochs):")
    print("  lr\\ep  | " + " ".join("%6d" % e for e in eps))
    for lr in lrs:
        row = " ".join(("%6.2f" % cells[(lr, e)]["Few"]) if (lr, e) in cells else "   -  "
                        for e in eps)
        print("  %-6s | %s" % (lr, row))

    # best cells
    best_few = max(cells.items(), key=lambda kv: kv[1]["Few"])
    best_all = max(cells.items(), key=lambda kv: kv[1]["All"])
    print("\nbest Few : lr=%s ep=%d -> Few %.2f (All %.2f)%s"
          % (best_few[0][0], best_few[0][1], best_few[1]["Few"], best_few[1]["All"],
             "  d=%+0.2f vs base %.2f" % (best_few[1]["Few"] - ref_few, ref_few) if ref_few else ""))
    print("best All : lr=%s ep=%d -> All %.2f (Few %.2f)"
          % (best_all[0][0], best_all[0][1], best_all[1]["All"], best_all[1]["Few"]))
    if ref_few:
        gain = best_few[1]["Few"] - ref_few
        print("note: single-seed; run-noise band ~0.4 on Few. best gain %+0.2f is %s -- "
              "re-run that cell across seeds before trusting."
              % (gain, "worth checking" if gain > 0.4 else "within noise"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/lr_epoch_sweep")
    ap.add_argument("--datasets", nargs="+", default=["imagenet_lt"])
    args = ap.parse_args()
    for ds in args.datasets:
        report(args.root, ds)


if __name__ == "__main__":
    main()
