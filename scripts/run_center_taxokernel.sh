#!/bin/bash
#
# PROMPT_CENTER_MODE=taxo_kernel -- SOFT taxonomic neighbourhood centering (2026-08-21).
#
#   mu_i = sum_{j != i} gamma^d(i,j) O_j / sum_{j != i} gamma^d(i,j)      out_i = O_i - mu_i
#
# d(i,j) = the level at which i and j first share an ancestor: same genus 1, family 2, order 3,
# class 4, phylum 5, kingdom 6, unrelated 7.
#
# ================================ WHY THIS MODE EXISTS ================================
# Every other taxonomy mode makes a BINARY membership test ("is j in my genus?"), which creates the
# degenerate state "my group has 0 other members" and then patches it after the fact:
#     genus/cascade  -> PROMPT_CENTER_GENUS_MIN, an arbitrary cutoff of 5
#     cascade        -> a hand-written fallback chain genus->family->order->global
#     level          -> no patch at all; produced exact zero rows and destroyed the run
#                       (see scripts/run_center_res0.sh: All 0.01 on every arm)
#     level_keep     -> +O applied to ALL 8142 classes to repair the few singletons, i.e. a uniform
#                       alpha=0.5 tax on the 63% of classes that never had a problem
#                       (see scripts/run_center_res1.sh: 4 arms within 0.07, the alpha axis is dead)
# Here the d=1 term simply DROPS OUT of numerator and denominator when a class has no genus-mates,
# and the nearest non-empty level takes over. No branch, no cutoff, no fallback chain, one knob.
#
# THE IDENTITY THAT JUSTIFIES EXCLUDING SELF: for any group of size k >= 2,
#     O_i - mu^(-i) = k/(k-1) * (O_i - mu)
# a POSITIVE scalar multiple, so after row-normalization leave-one-out centering is EXACTLY plain
# centering. Verified on all 5142 non-singleton iNat classes: per-class cos 1.0000000000 (min
# 0.9999996), length ratio matches k/(k-1) to 5e-7. k = 1 is therefore the ONLY real singularity,
# and excluding self is what keeps this formulation out of it: mu_i is a mean of OTHER classes, so a
# zero row is structurally impossible. Measured min pre-norm row norm 1.96-10.5 vs a raw row of ~23.
#
# ================================ OFFLINE GEOMETRY (real 8142 iNat prototypes) ================================
#   gamma   zero rows   cos-to-global   top5conf     note
#   0.00        0          0.5440        0.4167      limit: mean of the NEAREST non-empty relatives;
#                                                    zero hyperparameters. Outside the measured band.
#   0.01        0          0.6589        0.4233
#   0.02        0          0.7094        0.4306
#   0.03        0          0.7459        0.4376      <- dead centre of the 0.72-0.75 winning band
#   0.05        0          0.8015        0.4506
#   0.10        0          0.8865        0.4918
#   0.30        0          0.9696        0.5950      near-duplicate of mode=global; NOT worth a run
#   0.90        0          0.9997        0.6361      confirms gamma -> 1 converges to mode=global
#   reference: raw 0.5668/0.9050 | global 1.0000/0.6536 | cascade ~0.743/~0.60
# top5conf (mean cosine to each class's 5 nearest OTHER classes -- iNat's actual confusion
# bottleneck) is 0.42-0.44 here, LOWER than anything else measured in this project.
#
# gamma=0 census (which level each class actually ends up using, with NO gate and 100% coverage):
#   genus 5142 (63.2%) | family 2539 (31.2%) | order 398 (4.9%) | class 54 | phylum 4 | kingdom 4 | all 1
# Worked examples at gamma=0.03: Quercus agrifolia (28-species genus) puts 99.3% of its weight on its
# 27 genus-mates; Abaeis nicippe (singleton genus) automatically splits 55.9% family / 43.3% order.
#
# ================================ PRE-REGISTERED PREDICTIONS (written before running) ================================
# BASE RATE FIRST: 59 iNat centering arms measured in this project land in 80.46-81.02, a spread of
# 0.56. Do not expect a large move. Note also that top5conf has NOT predicted accuracy on this dataset
# before, so the 0.42-0.44 figure is a reason to test this, not a reason to expect a win.
#   g003 (cos-to-global 0.7459): the in-band bet. Sits where cascade (0.743 -> 80.84) and
#     g_bottomup_fo (-> 81.02) sit, but with a much lower top5conf. PREDICT 80.7 - 81.0, i.e. matching
#     cascade. Beating 81.02 would be the first genuinely new result on iNat in this line of work.
#   g0   (cos-to-global 0.5440): OUT of the measured band (0.719-1.000) and close to diff_init's
#     0.56, which came back 80.57 -- a tie with baseline. PREDICT 80.4 - 80.8, a tie. Run it anyway
#     because it is the ZERO-hyperparameter arm: if it ties cascade, the method needs no constant at
#     all, which is the strongest possible version of the "unified criterion" claim.
#   FALSIFIER for the whole idea: if g003 lands below 80.5 despite sitting in the band with the best
#   confusion geometry measured here, then cos-to-global is not the axis that matters on iNat and the
#   localization story needs rethinking.
#   WHAT COUNTS AS SUCCESS EVEN ON A TIE: this mode removes PROMPT_CENTER_CASCADE (level list),
#   PROMPT_CENTER_GENUS_MIN (the arbitrary 5) and PROMPT_CENTER_CASCADE_MEAN (residual/full) -- three
#   knobs -> one, or zero at gamma=0. A tie with cascade is a methodological result on its own.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50
#   g_bottomup_fo (center_nested25)              81.02  75.73  80.79  82.69   <- the number to beat
#   iNat seed noise (5-ep/scale-30 proxy): All ~0.06, Head ~0.74, Med ~0.16, Few ~0.23.
#
# NOTE: yacs is type-strict, so gamma MUST be written with a decimal point ("0.0", not "0").
#
#   bash scripts/run_center_taxokernel.sh                     # g0 + g003 (default)
#   ARMS="0.0 0.01 0.03 0.1" bash scripts/run_center_taxokernel.sh    # full gamma axis
#   python scripts/agg_runs.py output/center_taxokernel --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
ARMS=${ARMS:-"0.0 0.03"}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_taxokernel"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for gam in ${ARMS}; do
  case "${gam}" in *.*) ;; *) echo "ERROR: gamma must have a decimal point (yacs is type-strict): '${gam}' -> '${gam}.0'"; exit 1;; esac
  tag="g$(echo "${gam}" | tr -d '.')"
  out="${OUT_ROOT}/inat2018/${tag}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  echo "=== [inat2018] taxo_kernel gamma=${gam} (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE taxo_kernel PROMPT_CENTER_GAMMA "${gam}" \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    check each log's '[PROMPT_CENTER taxo_kernel] ... rows are ZERO' line FIRST -- it must read"
echo "    0/8142; a nonzero count would mean the leave-one-out construction is broken."
echo "    Headline comparison: g003 vs cascade 80.84/75.81/80.57/82.50 and g_bottomup_fo 81.02."
echo "    g0 is the zero-hyperparameter arm; a tie with cascade is already a methodological result."
