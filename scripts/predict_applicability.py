#!/usr/bin/env python
"""
Composite applicability predictor (M') for prototype centering -- formalizes the two-factor
story behind fig_breadth_predictor.py (drift alone is NOT sufficient: CIFAR-IR50 has LOWER
Few weight-drift than ImageNet-LT yet centering does not help there, because its tail is
already near its accuracy ceiling). This combines the two factors into one pre-registered,
round-number rule and CHECKS it out of sample, instead of citing drift alone.

Two ingredients, both computable from a single BASELINE (no-centering) run -- i.e. usable
BEFORE ever training a centered model:
    drift_Few    = mean_{c in Few} 1 - cos(W_final_c, W_init_c)   (self-correction proxy, B2)
    headroom_Few = 100 - mean_{c in Few} cls_acc_c                (room left to repair, from cls_accs.npy)

Rule (round numbers fixed BEFORE looking at IR20/IR200; NOT fit by regression on 5 points --
n=5 is too few to fit >1 free parameter without overfitting):
    predict "centering helps Few"  iff  drift_Few < 0.15   AND   headroom_Few > 18.0
    (0.15 sits an order of magnitude below iNat's 0.853 and above every LT benchmark's ~0.04-0.08;
     18.0 sits between CIFAR-IR50's 15.56 -- the one calibration point where drift is low but
     centering still failed -- and CIFAR-IR100's 20.37, the weakest point where it still won.)

Usage:
    python scripts/predict_applicability.py --calibrate
        -> reproduces the 5 known points (ImageNet-LT, Places-LT, CIFAR-IR100/IR50, iNat2018)
           and checks the rule's predicted sign against the actually observed ΔFew.

    python scripts/predict_applicability.py "output/ir_extremes25/cifar100_ir40/baseline_seed0" --name "CIFAR-IR40"
        -> prints drift_Few / headroom_Few / predicted sign for a NEW run, with NO ground truth
           baked in (use once run_ir_extremes.sh baselines finish, BEFORE looking at the +center arm).
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

DRIFT_THRESH = 0.15
HEADROOM_THRESH = 18.0

# provenance: Table `tab:diag` / `tab:main` / `tab:controls` in output/paper/tables_centering.tex
# (baseline Few acc from tables_centering.tex tab:main / breadth25; ΔFew = centering - baseline)
CALIBRATION = [
    # name,           run dir,                                                    baseline_Few_acc, observed_DeltaFew
    ("ImageNet-LT",   "output/seed_ablation 25/imagenet_lt/baseline_seed0",        73.58,  +1.59),
    ("Places-LT",     "output/seed_ablation 25/places_lt/baseline_seed0",          51.62,  +1.62),
    ("CIFAR-IR100",   "output/breadth25/cifar100_ir100/baseline",                  79.63,  +2.24),
    ("CIFAR-IR50",    "output/breadth25/cifar100_ir50/baseline",                   84.44,  -0.27),
    ("iNat2018",      "output/breadth25/inat2018/baseline_15ep",                   82.36,  -0.23),
]

# OUT-OF-SAMPLE (2026-07-14): predicted from CALIBRATION alone, BEFORE these were trained
# (see draft_intro_method.tex Sec. "A composite, pre-registered applicability rule").
# ΔFew here is the 5-seed mean from output/ir_extremes25 (run_ir_extremes.sh); kept separate
# from CALIBRATION -- folding these in after the fact would erase the pre-registration.
VALIDATION = [
    ("CIFAR-IR40",    "output/ir_extremes25/cifar100_ir40/baseline_seed0",         83.60,  -0.14),
    ("CIFAR-IR200",   "output/ir_extremes25/cifar100_ir200/baseline_seed0",        73.73,  +5.88),
]


def _clf_weight(ckpt_path):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)["tuner"]
    return d["classifier.weight"].float()


def few_drift_headroom(run):
    """Compute (drift_Few, headroom_Few) for a baseline run from saved artifacts only."""
    W_final = _clf_weight(os.path.join(run, "checkpoint.pth.tar"))
    W_init = _clf_weight(os.path.join(run, "ckpts", "init", "checkpoint.pth.tar"))
    cn = np.load(os.path.join(run, "cls_num_list.npy"))
    cls_accs = np.load(os.path.join(run, "cls_accs.npy"))

    few_mask = cn < 20
    drift = (1 - F.cosine_similarity(F.normalize(W_final, dim=-1),
                                     F.normalize(W_init, dim=-1), dim=-1)).numpy()
    drift_few = float(drift[few_mask].mean())
    headroom_few = float(100.0 - cls_accs[few_mask].mean())
    return drift_few, headroom_few


def predict(drift_few, headroom_few):
    helps = (drift_few < DRIFT_THRESH) and (headroom_few > HEADROOM_THRESH)
    return helps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="baseline (no-centering) run dirs to predict on")
    ap.add_argument("--name", action="append", default=[], help="display name per run (order-matched)")
    ap.add_argument("--calibrate", action="store_true", help="reproduce the 5 known calibration points")
    args = ap.parse_args()

    if args.calibrate:
        print(f"rule: predict HELPS iff drift_Few < {DRIFT_THRESH} AND headroom_Few > {HEADROOM_THRESH}\n")

        def run_set(rows):
            n_match = 0
            for name, run, base_few, obs_dfew in rows:
                drift_few, headroom_few = few_drift_headroom(run)
                pred_helps = predict(drift_few, headroom_few)
                obs_helps = obs_dfew > 0
                match = pred_helps == obs_helps
                n_match += match
                print(f"{name:14} {drift_few:10.3f} {headroom_few:13.2f} {str(pred_helps):>10} "
                      f"{obs_dfew:+14.2f} {'OK' if match else 'MISS':>6}")
            return n_match

        print("-- calibration set (fixed the 0.15 / 18.0 thresholds) --")
        print(f"{'dataset':14} {'drift_Few':>10} {'headroom_Few':>13} {'predicted':>10} {'observed ΔFew':>14} {'match':>6}")
        n_cal = run_set(CALIBRATION)
        print(f"{n_cal}/{len(CALIBRATION)} signs match (expected: this is what the thresholds were set to fit).\n")

        print("-- out-of-sample validation (predicted BEFORE training; see VALIDATION provenance comment) --")
        print(f"{'dataset':14} {'drift_Few':>10} {'headroom_Few':>13} {'predicted':>10} {'observed ΔFew':>14} {'match':>6}")
        n_val = run_set(VALIDATION)
        print(f"{n_val}/{len(VALIDATION)} signs match -- this is the number that actually tests the rule.")
        return

    for i, run in enumerate(args.runs):
        name = args.name[i] if i < len(args.name) else run
        drift_few, headroom_few = few_drift_headroom(run)
        pred_helps = predict(drift_few, headroom_few)
        print(f"{name}: drift_Few={drift_few:.3f}  headroom_Few={headroom_few:.2f}  "
              f"predicted={'HELPS' if pred_helps else 'neutral/hurts'}")


if __name__ == "__main__":
    main()
