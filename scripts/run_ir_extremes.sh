#!/bin/bash
#
# Out-of-sample test of the drift+headroom applicability rule (scripts/predict_applicability.py).
# The rule was calibrated on 5 points (ImageNet-LT, Places-LT, CIFAR-IR100/IR50, iNat2018) and
# gives OPPOSITE predictions for two new CIFAR-100-LT severities:
#
#   CIFAR-IR200 (imb_factor 0.005, more severe than IR100): predicted to HELP Few, plausibly by
#     MORE than IR100's +2.24pp (broader/more extreme tail -> more headroom, still-low drift).
#   CIFAR-IR40  (imb_factor 0.025, milder than IR50): predicted NEUTRAL/NO benefit, like IR50
#     (continuing the IR100->IR50 headroom trend as severity decreases toward IR10, where the
#     Few (<20-sample) bucket vanishes entirely).
#
# NOTE on IR20: an earlier draft of this test used CIFAR-IR20, but IR20's imb_factor (0.05) gives
# a rarest-class count of 25 samples -- ABOVE the Few (<20) cutoff, so its Few group is EMPTY
# (same degenerate case as IR10, which is why breadth25 only reports Overall for IR10). IR40 was
# chosen instead specifically because it is the mildest severity with a non-trivial Few group
# (13 classes, min count 12) -- see scripts/predict_applicability.py docstring / class-count check.
#
# Run baseline + centering (5-seed each, matching center_seeds25 protocol) on both:
#   bash scripts/run_ir_extremes.sh
#   SEEDS="0 1 2 3 4" bash scripts/run_ir_extremes.sh
#
# Then check the rule's prediction against the observed sign:
#   python scripts/predict_applicability.py "output/ir_extremes25/cifar100_ir200/baseline_seed0" --name "CIFAR-IR200"
#   python scripts/predict_applicability.py "output/ir_extremes25/cifar100_ir40/baseline_seed0"  --name "CIFAR-IR40"
#   python scripts/agg_runs.py output/ir_extremes25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"cifar100_ir40 cifar100_ir200"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
VARIANTS=${VARIANTS:-"baseline center"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"ir_extremes25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}"
  mda True tte True num_epochs "${EPOCHS}"
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    for s in ${SEEDS}; do
      out="${OUT_ROOT}/${data}/${v}_seed${s}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] ${v} seed=${s} (scale ${SCALE}, ${EPOCHS} ep) ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ \
        "${BASE_ARGS[@]}" ${va} seed "${s}" \
        output_dir "${out}"
    done
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
