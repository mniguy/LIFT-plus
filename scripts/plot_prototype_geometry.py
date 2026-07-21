#!/usr/bin/env python
"""Visualize CLIP-text prototype geometry (semantic init) per dataset.

(1) nearest-neighbor cosine histogram: how many classes have a near-twin, per dataset.
(2) iNat prototype PCA colored by the largest genera: same-genus species collapse to a blob.

Prototypes = frozen-baseline classifier.weight (= "a photo of a {classname}." through CLIP
text encoder). No GPU needed.

  python scripts/plot_prototype_geometry.py --out output/paper
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

RUNS = {"ImageNet": "output/freeze_center25/imagenet_lt/baseline",
        "Places":   "output/freeze_center25/places_lt/baseline",
        "iNat":     "output/freeze_center25/inat2018/baseline"}
COLORS = {"ImageNet": "steelblue", "Places": "seagreen", "iNat": "crimson"}


def protos(run):
    W = torch.load(os.path.join(run, "checkpoint.pth.tar"),
                   map_location="cpu", weights_only=False)["tuner"]["classifier.weight"].float()
    return F.normalize(W, dim=1)


def nn_cos(X, chunk=512):
    n = X.shape[0]; out = np.empty(n)
    for i in range(0, n, chunk):
        S = X[i:i + chunk] @ X.T
        for r in range(S.shape[0]):
            S[r, i + r] = -1
        out[i:i + chunk] = S.max(1).values.numpy()
    return out


def fig_nn_hist(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(0.4, 1.0, 61)
    for ds, run in RUNS.items():
        c = nn_cos(protos(run))
        ax.hist(c, bins=bins, density=True, histtype="step", lw=2, color=COLORS[ds],
                label=f"{ds} (median {np.median(c):.2f}, >0.9 {100*(c>0.9).mean():.0f}%)")
    ax.axvline(0.9, color="gray", ls="--", lw=1)
    ax.text(0.9, ax.get_ylim()[1] * 0.95, " cos=0.9 (near-twin)", color="gray", fontsize=8, va="top")
    ax.set_xlabel("nearest-neighbor cosine  (per class, to its most-similar other class)")
    ax.set_ylabel("density")
    ax.set_title("How close is each class to its nearest neighbor?  (CLIP-text prototypes)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out + "/proto_nn_hist.png", dpi=150)
    print(f"[save] {out}/proto_nn_hist.png")


def fig_inat_pca(out, topk=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter
    cat = json.load(open("datasets/iNaturalist2018/categories.json"))
    name2gen = {c["name"]: c["genus"] for c in cat}
    names = sorted(name2gen)                        # class index order
    gen = np.array([name2gen[n] for n in names])
    X = protos(RUNS["iNat"]).numpy()
    Xc = X - X.mean(0)                              # center (remove shared mu)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T                               # top-2 PC projection

    top_genera = [g for g, _ in Counter(gen).most_common(topk)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # --- left: global PCA, top genera colored ---
    ax.scatter(Z[:, 0], Z[:, 1], s=3, c="lightgray", alpha=0.4, linewidths=0)  # all species
    cmap = plt.cm.tab10
    for k, g in enumerate(top_genera):
        m = gen == g
        ax.scatter(Z[m, 0], Z[m, 1], s=46, color=cmap(k), edgecolor="k", linewidths=0.4,
                   label=f"{g} ({m.sum()} spp.)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("iNat prototypes (PCA), colored by genus")
    ax.legend(fontsize=8, title="largest genera")

    # --- right: within-genus vs between-genus pairwise cosine ---
    Xt = torch.tensor(X)
    within = []
    for g in set(gen):
        idx = np.where(gen == g)[0]
        if len(idx) < 2:
            continue
        S = Xt[idx] @ Xt[idx].T
        iu = np.triu_indices(len(idx), 1)
        within.append(S.numpy()[iu])
    within = np.concatenate(within)
    rng = np.random.default_rng(0)
    a = rng.integers(0, len(names), 200000); b = rng.integers(0, len(names), 200000)
    ok = gen[a] != gen[b]
    between = (Xt[a[ok]] * Xt[b[ok]]).sum(1).numpy()
    bins = np.linspace(0.2, 1.0, 65)
    ax2.hist(between, bins=bins, density=True, histtype="step", lw=2, color="gray",
             label=f"different genus (mean {between.mean():.2f})")
    ax2.hist(within, bins=bins, density=True, histtype="stepfilled", lw=2, color="crimson",
             alpha=0.55, label=f"same genus (mean {within.mean():.2f})")
    ax2.set_xlabel("pairwise cosine between two species' prototypes")
    ax2.set_ylabel("density")
    ax2.set_title("Same-genus species are near-duplicates")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle("CLIP can't read Latin epithets: same-genus prototypes collapse together")
    fig.tight_layout(); fig.savefig(out + "/proto_inat_pca.png", dpi=150)
    print(f"[save] {out}/proto_inat_pca.png")
    print(f"  same-genus mean cos = {within.mean():.3f} (n={len(within)})  "
          f"different-genus mean cos = {between.mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/paper")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_nn_hist(args.out)
    fig_inat_pca(args.out)
