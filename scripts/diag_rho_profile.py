#!/usr/bin/env python
"""Within-group cosine profile of the prompt prototypes, AFTER global centering. No training.

This is the quantity in fig_residual_shape / tab:why_locality, computed straight from the text
encoder so it can be read BEFORE committing GPU time to a centering arm. It answers "is there
anything left to subtract at this level, and how many classes would actually receive it".

Three blocks:
  A  rho_within per taxonomy level, with the cliff ratio to its parent. A CLIFF (iNat genus->family
     is 4.98x) marks a nuisance cluster living at one specific scale; a smooth decay (ImageNet's
     1.45x per hop) is the semantic hierarchy itself, i.e. signal, and localizing into it costs
     accuracy (fig_localization_axis, ImageNet r=+0.90 over a 2.03pp spread).
  B  COVERAGE -- how many classes a level can actually serve, at m>=2 and at PROMPT_CENTER_GENUS_MIN.
     On iNat this is what motivates mode=cohesion: genus is the only level worth using and the size gate
     admits 28.0% of classes to it, while m>=2 would admit 63.2%.
  C  the taxonomy-free (k-means) profile at matched granularity, as the control for "could this
     structure have been found geometrically". On iNat it could not: genus groups with m=5-9 measure
     rho 0.825 while k-means clusters of median size 12 measure 0.479.

    python scripts/diag_rho_profile.py --dataset inat2018
    python scripts/diag_rho_profile.py --dataset imagenet_lt --levels h1,h2,h3
"""
import argparse
import json
import os

import numpy as np
import torch

# (categories.json, classname source, default level chain fine->coarse)
DATASETS = {
    "inat2018": ("datasets/iNaturalist2018/categories.json", None,
                 ["genus", "family", "order", "class", "phylum", "kingdom"]),
    "imagenet_lt": ("datasets/ImageNet_LT/categories.json", "datasets/ImageNet_LT/classnames.txt",
                    ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"]),
}


def load_classnames(cats_path, names_txt):
    cats = json.load(open(cats_path))
    taxo = {c["name"]: c for c in cats if "name" in c}
    if names_txt is None:                       # iNat: the trainer sorts the unique category names
        names = sorted(set(c["name"] for c in cats))
    else:                                       # ImageNet-LT: classnames.txt fixes the order
        with open(names_txt) as f:                # one bare name per line, spaces included
            names = [ln.strip() for ln in f if ln.strip()]
    return names, taxo


def encode(names, cache, template="a photo of a {}."):
    """RAW (unnormalized) CLIP text features -- what _center_prototypes actually receives."""
    if os.path.exists(cache):
        return np.load(cache)
    import clip
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = clip.load("ViT-B/16", device=dev)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(names), 256):
            tok = clip.tokenize([template.format(n.replace("_", " ")) for n in names[i:i + 256]]).to(dev)
            out.append(model.encode_text(tok).float().cpu())
    X = torch.cat(out).numpy().astype(np.float64)
    np.save(cache, X)
    return X


def rho_of(Zn, groups, min_size=2):
    """Size-weighted mean pairwise within-group cosine, over groups with at least min_size members."""
    num = den = 0.0
    for idxs in groups.values():
        m = len(idxs)
        if m < min_size:
            continue
        s = Zn[idxs].sum(0)
        num += (s @ s - m) / 2.0                # sum over the C(m,2) pairs
        den += m * (m - 1) / 2.0
    return num / den if den else float("nan")


def spherical_kmeans(Y, k, iters=25, seed=0):
    """Small dependency-free k-means on unit rows (this repo's sklearn is not always importable)."""
    rng = np.random.default_rng(seed)
    C = Y[rng.choice(len(Y), k, replace=False)].copy()
    for _ in range(iters):
        lab = (Y @ C.T).argmax(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = Y[m].mean(0)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    return (Y @ C.T).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="inat2018", choices=sorted(DATASETS))
    ap.add_argument("--levels", default=None, help="comma list, fine->coarse (default: per dataset)")
    ap.add_argument("--genus-min", type=int, default=5, help="the gate whose coverage cost to report")
    ap.add_argument("--kmeans-sizes", default="16,32,50,254,500", help="avg cluster sizes for block C")
    ap.add_argument("--cache-dir", default="output/_diag")
    args = ap.parse_args()

    cats_path, names_txt, default_levels = DATASETS[args.dataset]
    levels = [s.strip() for s in args.levels.split(",")] if args.levels else default_levels
    names, taxo = load_classnames(cats_path, names_txt)
    os.makedirs(args.cache_dir, exist_ok=True)
    X = encode(names, os.path.join(args.cache_dir, f"{args.dataset}_text_raw.npy"))
    C = len(names)
    print(f"[{args.dataset}] {C} classes, mean row norm {np.linalg.norm(X, axis=1).mean():.3f}, "
          f"|global centroid| {np.linalg.norm(X.mean(0)):.4f}")

    Z = X - X.mean(0)                            # global centering: the w=0 reference
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)

    def groups_at(lv):
        g = {}
        for i, n in enumerate(names):
            k = taxo.get(n, {}).get(lv)
            if k is not None:
                g.setdefault(k, []).append(i)
        return g

    print("\n=== A. rho_within per level (fine -> coarse), and the cliff to its parent ===")
    print(f"{'level':10s} {'rho':>7s} {'#groups':>8s} {'mean m':>7s} {'cliff':>7s}")
    rhos, gs = {}, {}
    for lv in levels:
        gs[lv] = groups_at(lv)
        rhos[lv] = rho_of(Zn, gs[lv])
    for i, lv in enumerate(levels):
        sizes = np.array([len(v) for v in gs[lv].values()])
        par = levels[i + 1] if i + 1 < len(levels) else None
        cliff = f"{rhos[lv] / rhos[par]:.2f}x" if par and rhos[par] > 0 else "-"
        print(f"{lv:10s} {rhos[lv]:7.3f} {len(gs[lv]):8d} {sizes.mean():7.1f} {cliff:>7s}")
    print("  a cliff (>~3x) marks a nuisance cluster at one scale; a flat 1.0-1.6x continuum is the")
    print("  semantic hierarchy itself, and localizing into it removes signal, not nuisance.")

    print(f"\n=== B. COVERAGE: classes a level can serve (gate = m >= {args.genus_min}) ===")
    print(f"{'level':10s} {'m>=2':>15s} {'m>=gate':>15s} {'singletons':>15s}")
    for lv in levels:
        g = gs[lv]
        c2 = sum(len(v) for v in g.values() if len(v) >= 2)
        cg = sum(len(v) for v in g.values() if len(v) >= args.genus_min)
        c1 = sum(len(v) for v in g.values() if len(v) == 1)
        print(f"{lv:10s} {c2:6d} ({100*c2/C:5.1f}%) {cg:6d} ({100*cg/C:5.1f}%) {c1:6d} ({100*c1/C:5.1f}%)")

    print("\n--- B2. is the finest level's rho an artifact of small groups? ---")
    fine = levels[0]
    for lo, hi in [(2, 2), (3, 4), (5, 9), (10, 10**9)]:
        sub = {k: v for k, v in gs[fine].items() if lo <= len(v) <= hi}
        n = sum(len(v) for v in sub.values())
        if n:
            hi_s = "inf" if hi > 10**8 else str(hi)
            print(f"  {fine} groups with {lo}<=m<={hi_s:>3s}: {len(sub):5d} groups, {n:5d} classes, "
                  f"rho={rho_of(Zn, sub):.3f}")
    print("  rho RISING with m means the gate is excluding real structure, not protecting against noise.")

    print("\n=== C. taxonomy-free k-means control at matched granularity ===")
    print(f"{'avg size':>9s} {'k':>6s} {'rho':>7s} {'median m':>9s}")
    prev = None
    for avg in [int(s) for s in args.kmeans_sizes.split(",")]:
        k = max(2, round(C / avg))
        lab = spherical_kmeans(Zn, k)
        g = {}
        for i, l in enumerate(lab):
            g.setdefault(int(l), []).append(i)
        sizes = np.array([len(v) for v in g.values()])
        r = rho_of(Zn, g)
        cliff = f"{prev/r:.2f}x" if prev else ""   # fine/coarse, same orientation as block A
        print(f"{avg:9d} {k:6d} {r:7.3f} {np.median(sizes):9.0f}  {cliff}")
        prev = r
    print("  if the taxonomy levels show a cliff and k-means does not, the structure is lexical/")
    print("  taxonomic rather than geometric, and no unsupervised grouping will recover it.")


if __name__ == "__main__":
    main()
