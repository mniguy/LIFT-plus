#!/usr/bin/env python
"""Is the shared-direction defect a property of CLIP ViT-B/16, or of CLIP text encoders generally?

Closes the cheapest part of the backbone-generalization gap: the paper's problem statement
(|mu| ~ 0.8, rho ~ 0.54, so 1/rho ~ 1.9x of inter-class logit gap is lost) is currently measured
on ONE text encoder. Re-measuring it on other encoders needs no training, no images and no
labels -- only the class-name list -- so it can be run before committing GPU hours to a full
ViT-L/14 training arm.

Reads classnames straight off disk (no dataset root / no image files needed):
    imagenet_lt  datasets/ImageNet_LT/classnames.txt
    places_lt    datasets/Places_LT/classnames.txt
    inat2018     datasets/iNaturalist2018/categories.json   (sorted(set(name)), matching
                                                             iNaturalist2018.get_classnames())

Reported per (backbone, dataset):
    D            text embedding dimension
    |mu|         norm of the mean unit prototype = size of the shared component
    coll         mean pairwise cosine between prototypes
    a            mean cos(prototype, mu_hat)          (should be ~= |mu| when coll ~= a^2)
    a_std        spread of a across classes           (small => the shared term is class-constant
                                                       and cancels in argmax)
    rho          mean discriminative norm fraction = sqrt(1 - a^2)
    1/rho        predicted inter-class logit-gap loss factor
    coll vs a^2  if coll ~= a^2 then ALL apparent inter-class similarity is the shared component,
                 i.e. the class residuals were already mutually orthogonal (the paper's key claim)

    python scripts/measure_anisotropy_backbone.py
    python scripts/measure_anisotropy_backbone.py --backbones ViT-B/16 ViT-L/14 --datasets imagenet_lt
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

import clip

CLASSNAME_SRC = {
    "imagenet_lt": ("txt", "datasets/ImageNet_LT/classnames.txt"),
    "places_lt": ("txt", "datasets/Places_LT/classnames.txt"),
    "inat2018": ("inat", "datasets/iNaturalist2018/categories.json"),
}


def classnames(dataset):
    kind, path = CLASSNAME_SRC[dataset]
    if not os.path.exists(path):
        return None
    if kind == "txt":
        return [l.strip() for l in open(path) if l.strip()]
    cats = json.load(open(path))
    return sorted({c["name"] for c in cats if "name" in c})


def encode(model, names, template, device, batch=256):
    feats = []
    with torch.no_grad():
        for i in range(0, len(names), batch):
            toks = clip.tokenize([template.format(n.replace("_", " ")) for n in names[i:i + batch]])
            feats.append(model.encode_text(toks.to(device)).float().cpu())
    return F.normalize(torch.cat(feats), dim=-1)


def stats(X):
    mu = X.mean(0)
    mu_hat = F.normalize(mu, dim=0)
    a = X @ mu_hat
    rho = (X - a.unsqueeze(1) * mu_hat).norm(dim=1)
    n = X.shape[0]
    S = X @ X.T
    coll = ((S.sum() - n) / (n * (n - 1))).item()
    return dict(D=X.shape[1], mu=mu.norm().item(), coll=coll, a=a.mean().item(),
                a_std=a.std().item(), rho=rho.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=["ViT-B/16", "ViT-L/14"])
    ap.add_argument("--datasets", nargs="*", default=["imagenet_lt", "places_lt", "inat2018"])
    ap.add_argument("--template", default="a photo of a {}.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--subsample", type=int, default=0, help="use only every Nth class (0 = all)")
    a = ap.parse_args()

    print(f"template = {a.template!r}   device = {a.device}")
    hdr = (f"{'backbone':16} {'dataset':12} {'C':>6} {'D':>5} {'|mu|':>7} {'coll':>7} "
           f"{'a':>7} {'a_std':>7} {'rho':>7} {'1/rho':>7} {'a^2':>7} {'coll-a^2':>9}")
    print(hdr)
    print("-" * len(hdr))
    for bb in a.backbones:
        try:
            model, _ = clip.load(bb, device=a.device)
        except Exception as e:                                  # download / SSL / OOM
            print(f"{bb:16} !! could not load: {type(e).__name__}: {e}")
            continue
        model.eval()
        for ds in a.datasets:
            names = classnames(ds)
            if names is None:
                print(f"{bb:16} {ds:12} !! classname file missing -- skipped")
                continue
            if a.subsample > 1:
                names = names[::a.subsample]
            s = stats(encode(model, names, a.template, a.device))
            print(f"{bb:16} {ds:12} {len(names):6d} {s['D']:5d} {s['mu']:7.4f} {s['coll']:7.4f} "
                  f"{s['a']:7.4f} {s['a_std']:7.4f} {s['rho']:7.4f} {1/s['rho']:7.2f} "
                  f"{s['a']**2:7.4f} {s['coll']-s['a']**2:+9.4f}")
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()

    print("\nreading:")
    print("  |mu| ~ 0.8 and rho ~ 0.55 on a NEW encoder => the defect is a property of CLIP text")
    print("    encoders, not of one checkpoint, and the problem statement generalizes.")
    print("  coll - a^2 ~ 0  => class residuals are already mutually orthogonal, so centering")
    print("    promotes existing structure rather than creating separability (the paper's claim).")
    print("  a_std << a  => the shared term is near class-constant and cancels in argmax, which is")
    print("    why raw prototypes still classify at all and why zero-shot centering does not help.")


if __name__ == "__main__":
    main()
