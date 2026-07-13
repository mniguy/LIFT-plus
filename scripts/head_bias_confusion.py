#!/usr/bin/env python
"""
Experiment 2 -- head-bias confusion analysis (the mechanism's "why", made direct).

Claim: the shared prototype direction mu makes cosine scores favor head classes, so
tail test images LEAK into Many. Centering removes mu -> the leak shrinks and those
images return to their own (Few) classes. This turns "mu creates a head bias" from a
verbal claim into a measured one, and shows centering behaves like a geometric
logit-adjustment.

Pure offline analysis of artifacts test() already dumps -- no GPU / training:
    <run>/y_true.npy       per-image ground-truth class
    <run>/y_pred.npy       per-image predicted class
    <run>/cls_num_list.npy per-class train counts (for Many/Med/Few grouping)

    python scripts/head_bias_confusion.py \
        --baseline "output/seed_ablation 25/imagenet_lt/baseline_seed0" \
        --center   "output/prompt_center25/imagenet_lt/center" \
        --name ImageNet-LT
"""
import argparse
import os
import numpy as np

GROUPS = ["Many", "Med", "Few"]


def group_of_class(cn):
    """Map each class id -> shot group, matching trainer.groups()."""
    cn = np.asarray(cn)
    g = np.empty(len(cn), dtype=object)
    g[cn > 100] = "Many"
    g[(cn >= 20) & (cn <= 100)] = "Med"
    g[cn < 20] = "Few"
    return g


def load(run):
    yt = np.load(os.path.join(run, "y_true.npy"))
    yp = np.load(os.path.join(run, "y_pred.npy"))
    cn = np.load(os.path.join(run, "cls_num_list.npy"))
    return yt, yp, cn


def group_confusion(yt, yp, g):
    """Row-normalized group->group confusion: for each TRUE group, the fraction of its
    images whose PREDICTED class falls in each group. Diagonal = 'stayed in own group'."""
    tg, pg = g[yt], g[yp]
    M = np.zeros((3, 3))
    for i, tr in enumerate(GROUPS):
        rows = tg == tr
        n = rows.sum()
        if n == 0:
            M[i] = np.nan
            continue
        for j, pr in enumerate(GROUPS):
            M[i, j] = (pg[rows] == pr).mean()
    return M


def per_group_acc(yt, yp, g):
    tg = g[yt]
    out = {}
    for tr in GROUPS:
        rows = tg == tr
        out[tr] = (yp[rows] == yt[rows]).mean() * 100 if rows.sum() else float("nan")
    return out


def fmt_row(name, vals, suffix="%"):
    cells = "  ".join(f"{v:6.2f}" for v in vals)
    return f"  {name:9} | {cells} {suffix}"


def report(name, base_run, center_run):
    ytb, ypb, cn = load(base_run)
    ytc, ypc, cnc = load(center_run)
    assert np.array_equal(cn, cnc), "cls_num_list differs between runs (different dataset?)"
    g = group_of_class(cn)

    Mb = group_confusion(ytb, ypb, g)
    Mc = group_confusion(ytc, ypc, g)
    accb = per_group_acc(ytb, ypb, g)
    accc = per_group_acc(ytc, ypc, g)

    print(f"\n================ {name} ================")
    print(f"  baseline: {base_run}")
    print(f"  center:   {center_run}")

    for title, M in [("BASELINE", Mb), ("CENTER", Mc), ("DELTA (center - baseline)", Mc - Mb)]:
        hdr = "true\\pred"
        print(f"\n  {title}  -- row=true group, col=predicted group (row-normalized %)")
        print(f"  {hdr:9} | {'Many':>6}  {'Med':>6}  {'Few':>6}")
        for i, tr in enumerate(GROUPS):
            print(fmt_row(tr, M[i] * 100, suffix=""))

    # headline: the tail->head leak
    leak_b = Mb[GROUPS.index("Few"), GROUPS.index("Many")] * 100
    leak_c = Mc[GROUPS.index("Few"), GROUPS.index("Many")] * 100
    corr_b = Mb[GROUPS.index("Few"), GROUPS.index("Few")] * 100
    corr_c = Mc[GROUPS.index("Few"), GROUPS.index("Few")] * 100
    print("\n  --- headline (tail behaviour) ---")
    print(f"  Few -> Many leak : {leak_b:5.2f}%  ->  {leak_c:5.2f}%   (delta {leak_c - leak_b:+.2f} pp)")
    print(f"  Few -> Few group : {corr_b:5.2f}%  ->  {corr_c:5.2f}%   (delta {corr_c - corr_b:+.2f} pp)")
    print(f"  Few class acc    : {accb['Few']:5.2f}%  ->  {accc['Few']:5.2f}%   (delta {accc['Few'] - accb['Few']:+.2f} pp)")
    print(f"  Many class acc   : {accb['Many']:5.2f}%  ->  {accc['Many']:5.2f}%   (delta {accc['Many'] - accb['Many']:+.2f} pp)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="run dir with y_true/y_pred/cls_num_list.npy")
    ap.add_argument("--center", required=True, help="run dir with y_true/y_pred/cls_num_list.npy")
    ap.add_argument("--name", default="dataset")
    a = ap.parse_args()
    report(a.name, a.baseline, a.center)
