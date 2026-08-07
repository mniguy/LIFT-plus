#!/usr/bin/env python
"""Offline probe (GPU-free, needs /tmp/inat_diff_feats.pt from the CPU CLIP encode) for the question
"instead of centering with the lexical diff, what if the diff IS the initialization?"

Two readings of that question, both measured here against the inits we already understand:
  A) W0 = normalize(diff)            where diff_c = embed("Genus species") - embed("species")
  B) W0 = normalize(X - diff)        which is algebraically just embed("species"), the epithet-only init

Metrics (all on unit-normalized rows, matching how the classifier actually uses them):
  within-genus / within-family / all-pairs mean cosine  -- lower within-genus = congeneric species are
      easier to tell apart, which is iNat's actual bottleneck
  top5-conf  -- mean cosine to each class's 5 nearest OTHER classes, i.e. the fine-grained confusion
      pressure; the same statistic run_center_tree.sh pre-registered its level ladder on
  |mu|       -- anisotropy of the init
"""
import json
import torch
import torch.nn.functional as F

d = torch.load("/tmp/inat_diff_feats.pt", map_location="cpu", weights_only=False)
names, X_full, X_epi = d["names"], d["X_full"], d["X_epi"]
cats = {c["name"]: c for c in json.load(open("datasets/iNaturalist2018/categories.json")) if "name" in c}

genus_of = [n.split()[0] for n in names]
family_of = [cats.get(n, {}).get("family") for n in names]


def group_idx(keys):
    g = {}
    for i, k in enumerate(keys):
        if k is not None:
            g.setdefault(k, []).append(i)
    return {k: v for k, v in g.items() if len(v) >= 2}


GEN, FAM = group_idx(genus_of), group_idx(family_of)


def within(Xn, groups):
    tot, n = 0.0, 0
    for idxs in groups.values():
        Z = Xn[idxs]
        S = Z @ Z.T
        m = len(idxs)
        tot += (S.sum() - m).item()
        n += m * (m - 1)
    return tot / max(n, 1)


def allpairs(Xn, chunk=1024):
    C = Xn.shape[0]
    tot = 0.0
    for i in range(0, C, chunk):
        tot += (Xn[i:i + chunk] @ Xn.T).sum().item()
    return (tot - C) / (C * (C - 1))


def top5conf(Xn, chunk=1024, k=5):
    vals = []
    for i in range(0, Xn.shape[0], chunk):
        S = Xn[i:i + chunk] @ Xn.T
        for r in range(S.shape[0]):
            S[r, i + r] = -2.0
        vals.append(S.topk(k, dim=1).values.mean(1))
    return torch.cat(vals).mean().item()


def report(name, X, ref=None):
    Xn = F.normalize(X.float(), dim=-1)
    row = (f"{name:22s} {within(Xn, GEN):13.4f} {within(Xn, FAM):14.4f} "
           f"{allpairs(Xn):10.4f} {top5conf(Xn):10.4f} {Xn.mean(0).norm().item():8.4f}")
    if ref is not None:
        row += f" {(Xn * ref).sum(-1).mean().item():9.4f}"
    print(row)
    return Xn


diff = X_full - X_epi
# match trainer.py's genus_lex exactly: genera below min_size fall back to the GLOBAL mean of X,
# they do not go unsubtracted (an earlier version of this probe left them at zero, which made the
# genus_lex row look less centered than the raw prototypes).
gmean_diff = X_full.mean(0).unsqueeze(0).repeat(X_full.shape[0], 1)
_all_genus = {}
for i, k in enumerate(genus_of):
    _all_genus.setdefault(k, []).append(i)
for idxs in _all_genus.values():
    if len(idxs) >= 5:
        t = torch.as_tensor(idxs)
        gmean_diff[t] = diff[t].mean(0)

print(f"{'init':22s} {'within-genus':>13s} {'within-family':>14s} {'all-pairs':>10s} "
      f"{'top5-conf':>10s} {'|mu|':>8s} {'cos-to-glo':>9s}")
glo = F.normalize(X_full - X_full.mean(0), dim=-1)
report("raw (baseline)", X_full, glo)
report("global centering", X_full - X_full.mean(0), glo)
print("-" * 92)
report("A: diff as init", diff, glo)
report("A': diff, centered", diff - diff.mean(0), glo)
report("B: X - diff (=epithet)", X_epi, glo)
print("-" * 92)
report("genus_lex (X-gmean)", X_full - gmean_diff, glo)

print()
print("Is diff genus-shared? (high within-genus on diff itself => reading A cannot discriminate "
      "congeneric species)")
dn = F.normalize(diff, dim=-1)
print(f"  diff within-genus={within(dn, GEN):.4f}  within-family={within(dn, FAM):.4f}  "
      f"all-pairs={allpairs(dn):.4f}")

# ---------------------------------------------------------------------------------------------
# Follow-up: every plausible reading of "diff + global centering", since A' above is only one of
# them. mu is always the mean of whatever tensor is being centered.
# ---------------------------------------------------------------------------------------------
print()
print("=== combinations of the global-centered prototype and the global-centered diff ===")
print(f"{'init':30s} {'within-genus':>13s} {'within-family':>14s} {'all-pairs':>10s} "
      f"{'top5-conf':>10s} {'|mu|':>8s} {'cos-to-glo':>9s}")
Xc = X_full - X_full.mean(0)                    # global-centered prototype  (= 'global centering')
Dc = diff - diff.mean(0)                        # global-centered diff       (= A')
Ec = X_epi - X_epi.mean(0)                      # global-centered epithet    (= B + centering)
report("global X            (ref)", Xc, glo)
report("A': global diff", Dc, glo)
report("B': global epithet", Ec, glo)
report("Xc - Dc", Xc - Dc, glo)
report("Xc + Dc", Xc + Dc, glo)
for a in (0.25, 0.5, 1.0, 2.0):
    report(f"norm(Xc) + {a:.2f}*norm(Dc)",
           F.normalize(Xc, dim=-1) + a * F.normalize(Dc, dim=-1), glo)
