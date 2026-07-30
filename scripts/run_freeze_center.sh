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
#
#   bash scripts/run_freeze_center.sh
#   python scripts/agg_runs.py output/freeze_center25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
VARIANTS=${VARIANTS:-}          # empty -> per-dataset default_variants() below
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
