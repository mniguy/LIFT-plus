#!/bin/bash
#
# CONTROL for #3: is the caption-geometry gain from the CAPTIONS, or just from
# de-anisotropizing the prototype (subtracting the global prompt centroid)?
#
# Same pipeline as run_caption_geom.sh (KD 0.001 + InfoNCE 0.005, scale 25, TTE,
# no warmup) but classifier_init=semantic (prompt-only, NO captions). Only knob is
# PROMPT_CENTER. This fills the "no caption" row of the 2x2:
#
#                    no center            center
#   no caption   ->  plain (this)         center (this)
#   caption      ->  caption_geom25/convex  caption_geom25/center
#
#   center ~= caption_geom25/center  -> caption content is a red herring; the win is centering
#   center ~= plain                  -> centering alone does nothing; the caption residual matters
#
#   bash scripts/run_prompt_center.sh
#   python scripts/agg_runs.py output/prompt_center25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"plain center"}
OUT_ROOT=${OUT_ROOT:-"prompt_center25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# Matches caption_geom25 base (scale 25), minus the caption knobs.
BASE_ARGS=(
  classifier_init semantic classifier_scale 25
  TEXT_REG_LAMBDA 0.001 INFONCE_LAMBDA 0.005 PRIOR_REG_MODE fixed
  mda True tte True num_epochs 5 PEFT_WARMUP False
)
variant_args(){ case "$1" in
  plain)  echo "PROMPT_CENTER False" ;;
  center) echo "PROMPT_CENTER True"  ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] prompt-center ${v} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
