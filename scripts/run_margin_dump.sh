#!/bin/bash
#
# H_D (effective-margin test) -- dump test logits from the MATCHED baseline vs center pair
# (loss_robust25, both scale 25, only PROMPT_CENTER differs) so the per-class DECISION MARGIN
# (logit_true - best competitor) is measured on identical scale. No training: test_only on the
# existing trained checkpoints, SAVE_LOGITS writes logits.npy for offline analysis.
#
#   H_D predicts: centering ENLARGES the margin, most for Few (rare classes cross the threshold),
#   and the margin gain tracks the accuracy gain. If the tail margin barely moves, H_D is weak.
#
#   bash scripts/run_margin_dump.sh
#   python scripts/analyze_margin.py output/margin25/imagenet_lt/baseline output/margin25/imagenet_lt/center
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
OUT_ROOT=${OUT_ROOT:-"margin25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON="classifier_init semantic classifier_scale 25 tte True SAVE_LOGITS True"
completed(){ [ -f "./output/$1/logits.npy" ]; }

for data in ${DATASETS}; do
  for v in baseline center; do
    if [ "$v" = "baseline" ]; then src="loss_robust25/${data}/LA_baseline"; else src="loss_robust25/${data}/LA_center"; fi
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    [ -f "./output/${src}/checkpoint.pth.tar" ] || { echo "  MISSING ckpt: output/${src}"; continue; }
    echo "=== [${data}] dump logits ${v} <- ${src} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      ${COMMON} test_only True model_dir "output/${src}" output_dir "${out}"
  done
done
echo; echo "=== analyze:"
for data in ${DATASETS}; do
  echo "  ${PYTHON} scripts/analyze_margin.py output/${OUT_ROOT}/${data}/baseline output/${OUT_ROOT}/${data}/center"
done
