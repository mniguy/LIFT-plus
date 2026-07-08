#!/usr/bin/env python
"""
Diagnostic metrics (M) for the prototype-centering study. Operates purely on saved
artifacts (no GPU / model / dataset needed):
    <run>/ckpts/init/checkpoint.pth.tar  -> W_init  (classifier weight at init)
    <run>/checkpoint.pth.tar             -> W_final (trained classifier weight)
    <run>/cls_num_list.npy               -> per-class train counts

M1  prototype anisotropy by frequency group (supports B1):
      - coll     = mean pairwise cosine within a group (collinearity)
      - muAlign  = mean cos(prototype, global centroid mu)
    reported for the init prototypes as-loaded AND after one mean-centering step.
    Prediction: Few > Many for both; centering collapses them toward 0.

M2  weight drift vs class frequency (supports B2 -- "tail can't fine-tune away init"):
      - drift_i = 1 - cos(W_final_i, W_init_i)   (angular move from init during training)
    Prediction: head drifts MORE, tail stays near init (positive corr with log count).

    python scripts/analyze_anisotropy.py "output/seed_ablation 25/imagenet_lt/baseline_seed0" \
                                         output/prompt_center25/imagenet_lt/center
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F


def _clf_weight(ckpt_path):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)["tuner"]
    return d["classifier.weight"].float()


def load(run):
    W_final = _clf_weight(os.path.join(run, "checkpoint.pth.tar"))
    init_p = os.path.join(run, "ckpts", "init", "checkpoint.pth.tar")
    W_init = _clf_weight(init_p) if os.path.exists(init_p) else None
    cn = np.load(os.path.join(run, "cls_num_list.npy"))
    return W_init, W_final, cn


def groups(cn):
    cn = np.asarray(cn)
    return {"Many": cn > 100, "Med": (cn >= 20) & (cn <= 100), "Few": cn < 20, "All": np.ones_like(cn, bool)}


def collinearity(P, mask):
    X = F.normalize(P[mask], dim=-1)
    n = X.shape[0]
    if n < 2:
        return float("nan")
    S = X @ X.T
    return ((S.sum() - n) / (n * (n - 1))).item()   # mean off-diagonal cosine


def mu_align(P, mask, mu):
    return (F.normalize(P[mask], dim=-1) @ F.normalize(mu, dim=0)).mean().item()


def analyze(run):
    W_init, W_final, cn = load(run)
    g = groups(cn)
    if W_init is None:
        print(f"\n=== {run} ===  (no ckpts/init -> skipping M1/M2, need W_init)")
        return
    P = F.normalize(W_init, dim=-1)
    mu = P.mean(0)
    Pc = F.normalize(P - mu, dim=-1)
    print(f"\n=== {run} ===   |mu(init)| = {mu.norm():.4f}")

    print("M1  init-prototype anisotropy by group:")
    print(f"  {'group':5} {'n':>5} | {'coll_raw':>9} {'coll_cent':>9} | {'muAlign_raw':>11} {'muAlign_cent':>12}")
    for name, mask in g.items():
        n = int(mask.sum())
        if n < 2:
            continue
        print(f"  {name:5} {n:5d} | {collinearity(P, mask):9.4f} {collinearity(Pc, mask):9.4f} | "
              f"{mu_align(P, mask, mu):11.4f} {mu_align(Pc, mask, mu):12.4f}")

    drift = (1 - F.cosine_similarity(F.normalize(W_final, dim=-1),
                                     F.normalize(W_init, dim=-1), dim=-1)).numpy()
    logc = np.log(cn.astype(float))
    r = np.corrcoef(logc, drift)[0, 1]
    print(f"M2  weight drift (1 - cos(W_final,W_init)) vs log(count):  pearson r = {r:+.3f}")
    print(f"  {'group':5} {'n':>5} | {'mean_drift':>10}")
    for name, mask in g.items():
        n = int(mask.sum())
        if n < 1:
            continue
        print(f"  {name:5} {n:5d} | {drift[mask].mean():10.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs (each with checkpoint.pth.tar, ckpts/init/, cls_num_list.npy)")
    for run in ap.parse_args().runs:
        analyze(run)
