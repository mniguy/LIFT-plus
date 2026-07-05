#!/bin/bash
#
# Recommended lr/epoch CANDIDATES (not a full product) on imagenet_lt + places_lt,
# seed 0. Writes into output/lr_epoch_sweep/ so scripts/agg_lr_epoch_sweep.py
# aggregates them together with the earlier grid.
#
# Same warmup recipe as run_lr_epoch_sweep.sh: hybrid init + KD 0.001 + InfoNCE 0.005
# + PEFT image-warmup (ep1, lr 5e-4), TTE, MDA, classifier_scale 30. Only lr & num_epochs vary.
#
# Chosen from the seed-0 sweep insight (Few peaks at dose = lr*num_epochs ~ 0.08-0.10;
# above -> tail collapses). We probe specific pairs, NOT a product (a product would
# run catastrophic cells like lr0.02 x ep20):
#   Tier1 gentle corner : 0.0075x11, 0.0075x13, 0.005x16, 0.005x20  (low lr, same dose)
#   Tier2 peak refine    : 0.02x3, 0.02x4, 0.015x6                   (earlier/between peak)
#
#   bash scripts/run_lr_epoch_candidates.sh
#   DATASETS=imagenet_lt bash scripts/run_lr_epoch_candidates.sh
#   python scripts/agg_lr_epoch_sweep.py --root output/lr_epoch_sweep \
#       --datasets imagenet_lt places_lt
set -euo pipefail

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
OUT_ROOT=${OUT_ROOT:-"lr_epoch_sweep"}
# lr:num_epochs candidate pairs
PAIRS=${PAIRS:-"0.02:4 0.015:6 0.01:8"}

[ -f main.py ] || { echo "ERROR: run from the repo root (main.py not found)"; exit 1; }

# fixed warmup recipe (lr & num_epochs injected per candidate)
BASE_ARGS=(
  classifier_init hybrid classifier_scale 30
  TEXT_REG_LAMBDA 0.001 INFONCE_LAMBDA 0.005
  PRIOR_REG_MODE fixed
  HYBRID_CAPTION_SOURCE wiki HYBRID_TOPK 8 SIM_THRESHOLD 0.6
  mda True tte True
  PEFT_WARMUP True PEFT_WARMUP_EPOCHS 1 PEFT_WARMUP_LR 5e-4 PEFT_WARMUP_IMAGE True
)

completed () { grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for pair in ${PAIRS}; do
    lr="${pair%%:*}"; ep="${pair##*:}"
    out="${OUT_ROOT}/${data}/lr${lr}_ep${ep}"
    if completed "${out}"; then echo "  [skip] ${out}"; continue; fi
    echo "=== [${data}] lr=${lr} num_epochs=${ep} seed=${SEED} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" lr "${lr}" num_epochs "${ep}" seed "${SEED}" \
      output_dir "${out}"
  done
done

echo ""
echo "=== aggregate ==="
echo "  ${PYTHON} scripts/agg_lr_epoch_sweep.py --root output/${OUT_ROOT} --datasets ${DATASETS}"
