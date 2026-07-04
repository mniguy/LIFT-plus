#!/bin/bash
#
# LR x num_epochs grid sweep for THE warmup recipe.
#
# Fixed method  : hybrid init + KD 0.001 + InfoNCE 0.005, NO gating,
#                 PEFT image-warmup (ep1, lr 5e-4), TTE, MDA.
# Swept axes    : main `lr` and main `num_epochs`.
#
# WHY a full grid (not a single long run evaluated per-epoch): both the LR schedule
# (CosineAnnealingLR over num_epochs) and the MDA crop schedule are tied to
# num_epochs, and train() never evaluates mid-run. So "epoch k of an N-epoch run"
# is NOT the "optimal k-epoch config" -- each (lr, num_epochs) must be trained fresh.
#
# Single seed by design (this finds the best region cheaply). Validate the winner
# across seeds afterwards with scripts/run_seed_ablation.sh (paired vs baseline).
#
#   bash scripts/run_lr_epoch_sweep.sh
#   LRS="0.01 0.02" EPOCHS="5 10" DATASETS=imagenet_lt bash scripts/run_lr_epoch_sweep.sh
#   python scripts/agg_lr_epoch_sweep.py            # aggregate when done
set -euo pipefail

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt"}
LRS=${LRS:-"0.01 0.02 0.04"}
EPOCHS=${EPOCHS:-"5 8 10 15"}
OUT_ROOT=${OUT_ROOT:-"lr_epoch_sweep"}

[ -f main.py ] || { echo "ERROR: run from the repo root (main.py not found)"; exit 1; }

# recipe knobs that stay fixed across the grid (lr / num_epochs are injected per cell)
METHOD_ARGS=(
  classifier_init hybrid
  TEXT_REG_LAMBDA 0.001
  INFONCE_LAMBDA 0.005
  PRIOR_REG_MODE fixed
  HYBRID_CAPTION_SOURCE wiki
  HYBRID_TOPK 8
  SIM_THRESHOLD 0.6
  mda True
  tte True
  PEFT_WARMUP True
  PEFT_WARMUP_EPOCHS 1
  PEFT_WARMUP_LR 5e-4
  PEFT_WARMUP_IMAGE True
)

completed () { grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for ep in ${EPOCHS}; do
    for lr in ${LRS}; do
      out="${OUT_ROOT}/${data}/lr${lr}_ep${ep}"
      if completed "${out}"; then echo "  [skip] ${out}"; continue; fi
      echo "=== [${data}] lr=${lr} num_epochs=${ep} seed=${SEED} ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ \
        "${METHOD_ARGS[@]}" \
        lr "${lr}" num_epochs "${ep}" seed "${SEED}" \
        output_dir "${out}"
    done
  done
done

echo ""
echo "=== aggregate ==="
echo "  ${PYTHON} scripts/agg_lr_epoch_sweep.py --root output/${OUT_ROOT} --datasets ${DATASETS}"
