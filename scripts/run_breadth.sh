#!/bin/bash
#
# Dataset breadth for prototype centering (I = global). Shows the training-free centering
# generalizes beyond ImageNet-LT / Places-LT. Semantic init + aux OFF, scale 25, matched
# baseline vs center per dataset. iNat is the key case: its wiki captions are broken, but
# semantic (prompt) prototypes are fine, so centering still applies -> generalization payoff.
#
#   ImageNet-LT / Places-LT already done: baseline = "seed_ablation 25", center = prompt_center25.
#   This script covers iNat2018 + CIFAR-100-LT (IR100/50/10).
#
#   bash scripts/run_breadth.sh
#   SCALE=30 DATASETS=inat2018 bash scripts/run_breadth.sh   # match a dataset's native scale
#   python scripts/agg_runs.py output/breadth25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"cifar100_ir100 cifar100_ir50 cifar100_ir10 inat2018"}
VARIANTS=${VARIANTS:-"baseline center"}
SCALE=${SCALE:-25}
OUT_ROOT=${OUT_ROOT:-"breadth25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}"
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True num_epochs 5 PEFT_WARMUP False
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] breadth ${v} (scale ${SCALE}) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
