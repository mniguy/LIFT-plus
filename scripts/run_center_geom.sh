#!/bin/bash
#
# #3: generalize the prototype de-anisotropization. The CAPTION_CENTER win came from
# removing the global prompt centroid, and the gain concentrated in Few -- so sweep the
# centering geometry from a single-vector subtraction up to full whitening. Caption-free
# (semantic init + PROMPT_CENTER), same pipeline as run_prompt_center.sh.
#
#   global : p - mu_global                          (Direction-1 center; baseline)
#   group  : p - mu_head          (head/many-group centroid)
#   tail   : p - rarity_i * mu_global   (per-class strength ~ inverse freq; strong for tail)
#   std    : (p - mu) / std_dim          (diagonal whitening / standardization)
#   whiten : ZCA whitening               (decorrelate + unit variance)
#
#   bash scripts/run_center_geom.sh
#   python scripts/agg_runs.py output/center_geom25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"global group tail std whiten"}
OUT_ROOT=${OUT_ROOT:-"center_geom25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# Caption-free centering pipeline (= run_prompt_center.sh center cell), only the mode varies.
BASE_ARGS=(
  classifier_init semantic classifier_scale 25 PROMPT_CENTER True
  TEXT_REG_LAMBDA 0.001 INFONCE_LAMBDA 0.005 PRIOR_REG_MODE fixed
  mda True tte True num_epochs 5 PEFT_WARMUP False
)
variant_args(){ case "$1" in
  global|group|tail|std|whiten) echo "PROMPT_CENTER_MODE $1" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] center-geom ${v} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
