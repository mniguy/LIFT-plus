#!/bin/bash
#
# #3: hybrid-caption geometry -- reduce averaging dilution / tail noise.
# Base: hybrid init + KD 0.001 + InfoNCE 0.005, scale 30, TTE, no warmup (isolate the
# caption geometry). Variants toggle CAPTION_CENTER / CAPTION_SHRINK / CAPTION_BLEND.
#   convex   : current behavior (baseline)
#   center   : subtract global prompt centroid before averaging
#   shrink   : down-weight caption when few selected (tail noise)
#   residual : add only the caption component orthogonal to the prompt
#   all      : center + shrink + residual
#
# NOTE iNat: swap in the aligned corpus first (scripts/realign_inat_wiki.py).
#
#   bash scripts/run_caption_geom.sh
#   python scripts/agg_runs.py output/caption_geom --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"convex center shrink residual all"}
OUT_ROOT=${OUT_ROOT:-"caption_geom"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init hybrid classifier_scale 30
  TEXT_REG_LAMBDA 0.001 INFONCE_LAMBDA 0.005 PRIOR_REG_MODE fixed
  HYBRID_CAPTION_SOURCE wiki HYBRID_TOPK 8 SIM_THRESHOLD 0.6
  mda True tte True num_epochs 5 PEFT_WARMUP False
)
variant_args(){ case "$1" in
  convex)   echo "CAPTION_CENTER False CAPTION_SHRINK False CAPTION_BLEND convex" ;;
  center)   echo "CAPTION_CENTER True  CAPTION_SHRINK False CAPTION_BLEND convex" ;;
  shrink)   echo "CAPTION_CENTER False CAPTION_SHRINK True  CAPTION_BLEND convex" ;;
  residual) echo "CAPTION_CENTER False CAPTION_SHRINK False CAPTION_BLEND residual" ;;
  all)      echo "CAPTION_CENTER True  CAPTION_SHRINK True  CAPTION_BLEND residual" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] caption-geom ${v} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
