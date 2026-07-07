#!/bin/bash
#
# #4: WHERE to apply the hybrid-caption blend (need vs reliability).
# Base: hybrid init + KD 0.001 + InfoNCE 0.005, scale 30, TTE, no warmup.
#   all      : every class uses the caption blend (current)
#   tail     : only few-shot classes use captions (original "need" hypothesis)
#   headmed  : only head+med use captions (reverse -- captions are more reliable there)
#   reliable : only classes with >= CAPTION_RELIABLE_MIN selected captions
#
# NOTE iNat: swap in the aligned corpus first (scripts/realign_inat_wiki.py).
#
#   bash scripts/run_caption_apply.sh
#   python scripts/agg_runs.py output/caption_apply --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
MODES=${MODES:-"all tail headmed reliable"}
RELIABLE_MIN=${RELIABLE_MIN:-2}
OUT_ROOT=${OUT_ROOT:-"caption_apply"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init hybrid classifier_scale 30
  TEXT_REG_LAMBDA 0.001 INFONCE_LAMBDA 0.005 PRIOR_REG_MODE fixed
  HYBRID_CAPTION_SOURCE wiki HYBRID_TOPK 8 SIM_THRESHOLD 0.6
  mda True tte True num_epochs 5 PEFT_WARMUP False
)
completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for m in ${MODES}; do
    out="${OUT_ROOT}/${data}/${m}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] caption-apply ${m} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" CAPTION_APPLY "${m}" CAPTION_RELIABLE_MIN "${RELIABLE_MIN}" seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
