#!/usr/bin/env python
"""The two diagnostics the paper proposes, consolidated (fix #1) and applied to the
negative controls as well as the intervention (fix #3). No GPU / dataset / model needed --
reads only <run>/ckpts/init/checkpoint.pth.tar, <run>/checkpoint.pth.tar, <run>/cls_num_list.npy.

  rho  -- DISCRIMINATIVE NORM FRACTION  (init-side; the cause, B1)
      rho_c = || p_c - (p_c . mu_ref) mu_ref ||        p_c = unit-normalized init prototype
      Fraction of the cosine classifier's unit-norm budget that does class-discriminative
      work. mu_ref is held FIXED at the shared direction of the RAW (baseline) prototypes so
      that every run is scored against the same reference: "did this intervention remove the
      shared component the raw prototypes had?" Raw ~0.54 -> centered ~1.0, and the resulting
      inter-class logit gap scales as 1/rho (measured 1.76-1.88x, predicted 1.86x).

  SCG  -- SELF-CORRECTION GAP  (training-side; why the failure is tail-specific, B2)
      SCG(g)  = coll_init(g) - coll_final(g)     per frequency group, coll = mean pairwise
                cosine within the group -> how much shared-direction bias that group removed
                on its own during training.
      SCG_c   = muAlign_init(c) - muAlign_final(c)   per class, so it can be correlated with
                log(count). r(SCG_c, log n_c) > 0 is the frequency gradient that defines D;
                it should vanish in a balanced regime (P3).

Reading: rho separates I from J at INIT time (before any training); SCG says whether training
could repair what rho reports. The intervention should raise rho AND flatten SCG; a negative
control that only looks like the intervention should move neither.

!! WHICH METRIC TO ACTUALLY PUT IN THE PAPER (measured, ImageNet Few, I vs the three J controls,
   margin expressed as gap-to-nearest-J divided by the J's own spread):
       coll_final 4.7x    SCG 8.7x    drift 0.7x (fails)    rho_init 1.9x
   Despite SCG winning that margin, report COLLINEARITY AT TWO TIME POINTS, BY GROUP -- not SCG:
     * coll_final orders the runs consistently with accuracy where SCG does not. fewonly has a
       LARGER |SCG| (-0.0723) than centering (-0.0334) but lower accuracy (74.31 vs 75.12);
       coll_final gets it right (0.0103 vs 0.0023).
     * SCG = coll_init - coll_final is a lossy compression: it cannot say whether the run ENDED
       in a repaired state, which is the thing job (3) and (4) need.
     * coll_init by group (Many .645 / Med .632 / Few .635) shows the defect is UNIFORM, and
       coll_final by group (.237 / .343 / .441) shows only the repair capacity is graded --
       jobs (1) and (2) in one table, with no new metric invented.
   Keep rho for job (1) only: it supplies the budget reading (54% of the unit norm is wasted ->
   1/rho = 1.9x logit-gap loss). Do NOT use rho for job (4): randdir scores 0.72, above the
   baseline's 0.59, which reads as a partial repair that never happened.
   Drop drift entirely: it fails job (4) at 0.7x and already failed as a per-class predictor.
   This function still computes SCG because it is the cheapest way to see the two time points
   summarized, but treat it as an intermediate quantity, not a proposed metric.

SCOPE -- both are STATE variables, not per-class outcome predictors. Measured on ImageNet-LT,
per-class Delta-acc vs SCG_c gives simple r=-0.124 but partial r=+0.065 once log count is
controlled (drift_c: -0.118 -> +0.071); r(SCG_c, log n_c)=0.823. Per class, SCG is frequency in
disguise exactly as weight-drift was. What SCG does that drift cannot is discriminate the
intervention from its controls: on ImageNet Few it is -0.033 for centering vs +0.106..+0.122 for
every J and +0.134 for baseline -- a sign flip, ~10x the controls' own spread -- while drift puts
centering at 0.048 against a J range of 0.063..0.084, a margin narrower than that spread.
rho is likewise a reparameterization of mu-alignment (rho = sqrt(1 - muAlign^2)), not new
information; it adds the fixed raw reference (so centered runs stay comparable) and the 1/rho
logit-gap reading. rho is necessary but not sufficient -- randdir raises rho to 0.72 yet scores
below baseline, because it leaves SCG untouched.

    python scripts/diag_rho_scg.py --preset controls
    python scripts/diag_rho_scg.py --baseline "output/seed_ablation 25/imagenet_lt/baseline_seed0" \
        --runs output/prompt_center25/imagenet_lt/center output/center_control25/imagenet_lt/randdir
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

GROUPS = ("Many", "Med", "Few")


def clf_weight(ckpt):
    return torch.load(ckpt, map_location="cpu", weights_only=False)["tuner"]["classifier.weight"].float()


def load(run):
    init_p = os.path.join(run, "ckpts", "init", "checkpoint.pth.tar")
    fin_p = os.path.join(run, "checkpoint.pth.tar")
    cn_p = os.path.join(run, "cls_num_list.npy")
    if not (os.path.exists(init_p) and os.path.exists(fin_p) and os.path.exists(cn_p)):
        return None
    return F.normalize(clf_weight(init_p), dim=-1), F.normalize(clf_weight(fin_p), dim=-1), np.load(cn_p)


def groups_of(cn):
    cn = np.asarray(cn)
    return {"Many": cn > 100, "Med": (cn >= 20) & (cn <= 100), "Few": cn < 20}


def coll(P, mask):
    """Mean off-diagonal cosine inside a group."""
    X = P[mask]
    n = X.shape[0]
    if n < 2:
        return float("nan")
    S = X @ X.T
    return ((S.sum() - n) / (n * (n - 1))).item()


def rho(P, mu_ref):
    """Discriminative norm fraction, per class, against a FIXED reference direction."""
    return (P - (P @ mu_ref).unsqueeze(1) * mu_ref).norm(dim=1)


def mu_align(P, mu_ref):
    return P @ mu_ref


def analyze(runs, baseline_run, label):
    ref = load(baseline_run)
    if ref is None:
        print(f"\n{label}: baseline run missing artifacts ({baseline_run}) -- skipping")
        return
    W0_raw, _, cn = ref
    mu_ref = F.normalize(W0_raw.mean(0), dim=0)      # shared direction of the RAW prototypes
    g = groups_of(cn)

    print(f"\n{'='*100}\n{label}   (mu_ref from raw baseline init, |mu_raw| = {W0_raw.mean(0).norm():.4f})\n{'='*100}")
    hdr = (f"  {'run':26} {'group':5} | {'coll_init':>10} {'coll_final':>11} | {'rho_init':>9} | "
           f"{'SCG':>8} | {'r(SCG,logn)':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for run in runs:
        r = load(run)
        parts = os.path.relpath(run).split(os.sep)
        name = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        if r is None:
            print(f"  {name:26} {'':5} | (missing ckpts/init or checkpoint -- skipped)")
            continue
        W0, W1, cn_r = r
        if not np.array_equal(cn_r, cn):
            print(f"  {name:26} {'':5} | (cls_num_list mismatch vs baseline -- skipped)")
            continue

        rh = rho(W0, mu_ref)
        # per-class SCG from mu-alignment, correlated with log count (one number per run)
        scg_c = (mu_align(W0, mu_ref) - mu_align(W1, mu_ref)).numpy()
        rr = np.corrcoef(np.log(cn.astype(float)), scg_c)[0, 1]

        for gi, (gname, m) in enumerate(g.items()):
            if m.sum() < 2:
                continue
            ci, cf = coll(W0, m), coll(W1, m)
            lbl = name if gi == 0 else ""
            tail = f"{rr:+11.3f}" if gi == 0 else " " * 11
            print(f"  {lbl:26} {gname:5} | {ci:10.4f} {cf:11.4f} | {rh[m].mean():9.4f} | "
                  f"{ci - cf:8.4f} | {tail}")

    print("\n  PRIMARY columns are coll_init / coll_final by group -- that pair does all four jobs:")
    print("    (1) defect exists & is UNIFORM across groups   -> coll_init roughly equal Many/Med/Few")
    print("    (2) only the REPAIR capacity is frequency-graded -> coll_final Many << Few")
    print("    (3) the intervention repairs it                 -> coll_final ~0 for centering")
    print("    (4) look-alike controls do not                  -> coll_final stays at baseline level")
    print("  rho_init supports (1) only: 1/rho is the inter-class logit-gap loss (0.54 -> 1.9x).")
    print("  SCG is shown as the derived difference; it wins the raw (4) margin but mis-orders runs")
    print("  (fewonly: larger |SCG| than centering, lower accuracy), so do not report it as a metric.")


PRESETS = {
    # (label, baseline run, [runs...])
    "controls": [
        ("ImageNet-LT: I vs J at init and after training",
         "output/seed_ablation 25/imagenet_lt/baseline_seed0",
         ["output/seed_ablation 25/imagenet_lt/baseline_seed0",
          "output/prompt_center25/imagenet_lt/center",
          "output/center_control25/imagenet_lt/randdir",
          "output/center_control25/imagenet_lt/headonly",
          "output/center_control25/imagenet_lt/perclass_rand",
          "output/center_control25/imagenet_lt/fewonly"]),
        ("Places-LT: I vs J at init and after training",
         "output/seed_ablation 25/places_lt/baseline_seed0",
         ["output/seed_ablation 25/places_lt/baseline_seed0",
          "output/prompt_center25/places_lt/center",
          "output/center_control25/places_lt/randdir",
          "output/center_control25/places_lt/headonly",
          "output/center_control25/places_lt/perclass_rand",
          "output/center_control25/places_lt/fewonly"]),
    ],
    # severity is built lazily by _severity_preset(): each CIFAR severity has its own class
    # count vector, so each one needs its own mu_ref / baseline rather than a shared anchor.
    "severity": None,
    "inat": [
        ("iNaturalist 2018: grouping-axis variants",
         "output/breadth25/inat2018/baseline_15ep",
         ["output/breadth25/inat2018/baseline_15ep",
          "output/breadth25/inat2018/center_15ep",
          "output/center_local25/inat2018/genus",
          "output/center_local25/inat2018/cascade"]),
    ],
}


def _severity_preset():
    """One block per CIFAR severity dir; the baseline inside that dir defines its own mu_ref.

    Ordered from most to least imbalanced so the P3 endpoint (balanced, IR1) reads last:
    r(SCG, log n) should shrink toward 0 as the severity drops, because a balanced regime has
    no frequency axis for the self-correction gap to be graded along.
    """
    blocks = []
    for d in ["output/ir_extremes25/cifar100_ir200", "output/breadth25/cifar100_ir100",
              "output/breadth25/cifar100_ir50", "output/ir_extremes25/cifar100_ir40",
              "output/breadth25/cifar100_ir10", "output/balanced25/cifar100"]:
        if not os.path.isdir(d):
            continue
        runs = sorted(p for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p))
        base = next((p for p in runs if "baseline" in os.path.basename(p)), None)
        if base and runs:
            blocks.append((f"{os.path.basename(d)} (severity sweep, P3 endpoint = balanced)", base, runs))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--baseline", help="run whose init defines mu_ref (the raw prototypes)")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--label", default="custom")
    a = ap.parse_args()

    if a.preset:
        blocks = _severity_preset() if a.preset == "severity" else PRESETS[a.preset]
        for label, base, runs in blocks:
            analyze(runs, base, label)
    elif a.baseline and a.runs:
        analyze(a.runs, a.baseline, a.label)
    else:
        ap.error("give --preset, or both --baseline and --runs")


if __name__ == "__main__":
    main()
