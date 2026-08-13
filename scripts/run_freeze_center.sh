#!/bin/bash
#
# Experiment C -- the causal test of init-persistence, by INTERVENTION.
# Correlational analysis (per-class Delta-acc vs weight-drift, controlling frequency) does
# NOT support "frozen tail -> centering helps": frequency dominates, drift's coefficient is
# small and wrong-signed. But correlation cannot separate drift from frequency (r=0.82).
#
# Here we manipulate init-persistence directly: FREEZE_CLASSIFIER=True keeps the classifier
# pinned at its (centered or raw) init, so it CANNOT fine-tune away from the init -- holding
# dataset, frequency, headroom, #classes, epochs all fixed. Only PEFT (image encoder) trains.
#
#   Predictions if init-persistence IS causal:
#     - iNat (frozen):                    centering HELPS (Delta > 0)   [vs neutral when trainable]
#     - ImageNet-LT / Places-LT (frozen): centering helps ALL groups incl. HEAD
#                                         (head Delta flips + on BOTH)  [vs head Delta ~ 0 when trainable]
#   Two "helps" datasets (ImageNet-LT, Places-LT) make the head-flips-positive prediction robust.
#   If iNat-frozen stays neutral / head-frozen stays flat -> init-persistence is NOT the
#   mechanism; the frequency(+headroom) account stands.
#
#   NOTE: overall accuracy drops under a frozen classifier (weaker config). The test is the
#   center-vs-baseline gap WITHIN the frozen setting, and how it differs from the trainable
#   setting (breadth25 / prompt_center25), NOT the absolute number.
#
# ---------------------------------------------------------------------------------------------
# 2026-07-25 ADDITION: frozen `cascade` on iNat (variant added; IN/PL unchanged and already done).
#
# Why: trainable cascade is the first variant to BEAT baseline on iNat (80.84/75.81/80.57/82.50 vs
# baseline 80.63/74.62/80.50/82.36, i.e. +0.21/+1.19/+0.07/+0.14; global center was -0.11/+0.24/
# -0.09/-0.23). Its drift split is the tell: Many drift FELL (-0.021) and Many gained +1.19, while
# Med/Few drift ROSE (+0.068/+0.079) and stayed flat -- i.e. the Med/Few init advantage was
# overwritten, exactly what freezing the classifier blocks.
#
# Reference (this same folder, already run, seed 0, 15 ep):
#   inat2018/baseline  22.91 / 67.66 / 23.37 / 10.64
#   inat2018/center    35.34 / 68.33 / 39.65 / 21.28     (= +12.43 / +0.67 / +16.28 / +10.64)
#
#   Prediction: cascade's init is geometrically cleaner than global's on the full 8142 classes
#   (overall prototype collinearity 0.00154 vs 0.00740; within-genus 0.042 vs 0.828), so if the
#   frozen gain tracks init quality, frozen cascade should EXCEED frozen center -- most in Med/Few,
#   whose advantage the trainable run demonstrably threw away.
#   If frozen cascade <= frozen center, then cascade's trainable Head win is NOT about a better
#   init and needs another explanation (e.g. it merely suits what the head encoder already does).
#
#   DATASETS="inat2018" bash scripts/run_freeze_center.sh     # runs cascade, skips finished ones
# ---------------------------------------------------------------------------------------------
# 2026-08-08 ADDITION: frozen `g_bottomup_gf` / `g_bottomup` (nested mode) on iNat.
#
# Why now: the trainable sweep is SATURATED. 9 nested/cascade arms all land in 80.80-80.95, a spread
# of 0.15, while pairs whose inits are 99% identical swing by far more than that -- g_bottomup_gf vs
# g_bottomup (cos 0.9906) differ by +0.87 on Many, g_topdown_fg vs g_topdown (cos 0.9864) by -0.83,
# cascade vs cascade_lex (cos 0.9896) by 0.22 on Overall. Nothing in the trainable regime can resolve
# these arms, because iNat's trainable classifier drifts 0.85 from init and overwrites the very thing
# being compared. The frozen regime is where the signal is 50-200x larger (center - baseline = +12.43
# Overall / +16.28 Med / +10.64 Few), so it is the only place these inits can actually be ranked.
#
# g_bottomup_gf is the arm worth freezing: bottom-up is the ONLY structurally novel family here
# (top-down provably telescopes to cascade -- per-class cos(topdown3, g_topdown) = 1.0000 on all 7897
# covered classes, cos(topdown3, cascade) = 1.000 on the 2278 fully-covered ones), and _gf is
# g_bottomup with its measurably dead `order` level removed (|mu| 0.033 after global, vs global 0.828
# / genus 0.576 / family 0.159), which cost nothing when trainable (+0.02 Overall).
#
# PRE-REGISTERED, written before running:
#   frozen cascade / g_bottomup_gf  >  frozen center (35.34)  by a margin >> 1 pt
#     => the hierarchical init IS genuinely better and the trainable regime simply overwrites it;
#        the 0.1-level trainable differences stay uninterpretable but the mechanism claim is made.
#   frozen cascade / g_bottomup_gf  ~=  frozen center (35.34)
#     => hierarchy adds NOTHING on top of plain global centering. Then every trainable difference
#        chased above was noise, and the paper's simpler claim ("remove the one shared direction")
#        is the correct and stronger one. This outcome is just as publishable -- it retires the
#        localization branch with evidence rather than by omission.
#   Read Med/Few first: those are the groups whose init advantage the trainable run demonstrably
#   threw away (Med/Few drift ROSE +0.068/+0.079 under cascade while Many's FELL -0.021).
#
#   VARIANTS="cascade g_bottomup_gf" bash scripts/run_freeze_center.sh
# ---------------------------------------------------------------------------------------------
# 2026-08-08 RESULT of the above, and the follow-up it selects.
#
#   frozen iNat:   baseline 22.91 / 67.66 / 23.37 / 10.64
#                  center   35.34 / 68.33 / 39.65 / 21.28
#                  cascade  31.99 / 72.53 / 36.75 / 15.37   (= -3.35 / +4.20 / -2.90 / -5.91 vs center)
#
# The ordering FLIPS between regimes: trainable cascade BEATS global (+0.32) while frozen cascade
# LOSES to it (-3.35). So the hierarchical init is not a better init; it is a worse one, and its
# trainable win is a training-dynamics artifact. The group split says where: local centering helps
# Head (+4.20) and damages the tail (Few -5.91, Med -2.90), matching the coverage census (39.4% of
# Many sit in a genus >= 5 vs 26.4% Med / 27.0% Few, so Head gets the most and most stable local
# treatment while tail groups get means estimated from ~5 members). It also explains the drift split
# that was previously read as "overwriting": cascade's Many drift FELL (-0.021) because its Head init
# was good and needed no repair, while Med/Few drift ROSE (+0.068/+0.079) because the optimizer was
# REPAIRING a damaged tail init, not discarding a good one.
#
# FOLLOW-UP: g_bottomup_fo (global,family,order). Chosen over g_bottomup_gf because gf sits at
# cos-to-global 0.719, right next to cascade's 0.743 and containing the same genus level, so it would
# mostly re-measure cascade. fo sits at 0.851, a NEW point between global (1.000) and cascade (0.743),
# and it drops genus entirely -- the weakest localization we have. It is also the trainable #1
# (81.02), which makes it the decisive test of "the trainable winners are dynamics, not init quality".
#
# PRE-REGISTERED, written before running. Linear interpolation in cos-to-global through the two known
# frozen points (global 1.000 -> 35.34, cascade 0.743 -> 31.99, slope 13.0 pts per unit cos-g) puts
# fo at 0.851 -> ~33.4 Overall.
#   fo lands near 33.4, i.e. between global and cascade  => dose-response: the more local the removal,
#     the worse the initialization. Strongest possible form of the "globally" clause, and it retires
#     the iNaturalist counterexample with a graded causal curve rather than a single contrast.
#   fo lands at or above global's 35.34  => the damage is specific to FINE levels (genus), and coarse
#     localization is harmless. The design law then needs a granularity boundary, not a blanket ban.
#   fo lands at or below cascade's 31.99 => cos-to-global does not order frozen init quality either,
#     and the mechanism is something other than distance-from-global.
#   Read Few first: it carries the largest frozen effect (cascade -5.91) and is the group the paper
#   is about.
#
#   VARIANTS="g_bottomup_fo" bash scripts/run_freeze_center.sh
# ---------------------------------------------------------------------------------------------
#
#   bash scripts/run_freeze_center.sh
#   python scripts/agg_runs.py output/freeze_center25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
VARIANTS=${VARIANTS:-"g_bottomup_fo"}          # empty -> per-dataset default_variants() below
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}              # default (ImageNet-LT / Places-LT)
INAT_EPOCHS=${INAT_EPOCHS:-15}  # iNat native protocol
OUT_ROOT=${OUT_ROOT:-"freeze_center25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}" FREEZE_CLASSIFIER True
  mda True tte True
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  cascade)  echo "PROMPT_CENTER True PROMPT_CENTER_MODE cascade PROMPT_CENTER_CASCADE genus,family,order" ;;
  genus)    echo "PROMPT_CENTER True PROMPT_CENTER_MODE genus" ;;
  # nested arms (2026-08-08). FREEZE_CLASSIFIER only drops the classifier's gradients after the
  # init is built, so it composes with any PROMPT_CENTER_MODE.
  g_bottomup_gf) echo "PROMPT_CENTER True PROMPT_CENTER_MODE nested PROMPT_CENTER_NESTED_LEVELS global,genus,family" ;;
  g_bottomup)    echo "PROMPT_CENTER True PROMPT_CENTER_MODE nested PROMPT_CENTER_NESTED_LEVELS global,genus,family,order" ;;
  g_bottomup_fo) echo "PROMPT_CENTER True PROMPT_CENTER_MODE nested PROMPT_CENTER_NESTED_LEVELS global,family,order" ;;
  *) return 1 ;; esac; }

# cascade/genus need categories.json -> iNat only; IN/PL keep the original baseline+center pair
# (already finished, so a bare run of this script does the iNat cascade work and skips the rest).
default_variants(){ case "$1" in
  inat2018) echo "cascade" ;;
  *)        echo "baseline center" ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "$data" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for v in ${VARIANTS:-$(default_variants "${data}")}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] freeze-clf ${v} (scale ${SCALE}, ${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
