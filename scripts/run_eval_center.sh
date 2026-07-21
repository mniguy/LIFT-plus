#!/bin/bash
#
# H_B test -- is centering's Few gain a DECISION-TIME debias, or does it need training?
# Train a RAW baseline, then RE-EVALUATE the SAME trained model with EVAL_CENTER=True
# (mu removed from the TRAINED classifier weight at test time only).
#
#   step 1: train raw baseline           -> its own log holds the CONTROL eval (no centering)
#   step 2: test_only + EVAL_CENTER True  -> same weights, decision-time centering
#
#   Prediction:
#     H_B (geometric logit-adjustment): eval-center RECOVERS much of the Few gain -> no training needed.
#     H_A (training-time target geometry): eval-center does ~nothing (or hurts, like zero-shot -23)
#                                          -> the benefit is realized during PEFT co-adaptation.
#   Compare against full centering (train w/ PROMPT_CENTER): the training-time version (+1.6 Few).
#
#   bash scripts/run_eval_center.sh
#   python scripts/agg_runs.py output/eval_center25 --sort path   # base(control) vs evalcenter
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
SCALE=${SCALE:-25}; EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"eval_center25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}" PROMPT_CENTER False
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True PEFT_WARMUP False num_epochs "${EPOCHS}"
)
completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  base="${OUT_ROOT}/${data}/base"
  if completed "${base}"; then echo "  [skip] ${base}"; else
    echo "=== [${data}] train RAW baseline (control eval in its log) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" seed "${SEED}" output_dir "${base}"
  fi

  ec="${OUT_ROOT}/${data}/evalcenter"
  if completed "${ec}"; then echo "  [skip] ${ec}"; else
    echo "=== [${data}] EVAL-ONLY centering (test_only on the trained baseline) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" seed "${SEED}" \
      test_only True model_dir "output/${base}" \
      EVAL_CENTER True PROMPT_CENTER_MODE global \
      output_dir "${ec}"
  fi
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    base = control (raw eval) ; evalcenter = decision-time centering of the SAME model"
