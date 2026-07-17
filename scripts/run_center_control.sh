#!/bin/bash
#
# J negative controls for the centering intervention (I = PROMPT_CENTER_MODE=global).
# Same pipeline as run_center_geom.sh (semantic, aux OFF, scale 25), each control looks
# like centering but does NOT remove the shared anisotropic direction:
#
#   randdir       : subtract a RANDOM direction of matched norm   -> is it specifically mu?
#   headonly      : center head+med only, leave Few RAW           -> must we touch the frozen tail?
#   fewonly       : center Few only (global mu), leave head+med RAW -> is the tail the WHOLE gain?
#   perclass_rand : independent random dir per class (init noise)  -> is any init perturbation enough?
#
# Prediction (mechanism = "centering fixes the frozen tail init"): NONE of these recover Few.
# Compare against:  I  = prompt_center25/*/center  (Few 75.12 / 53.58)
#                   baseline (11-seed) Few 73.58 / 51.62
#
#   bash scripts/run_center_control.sh
#   python scripts/agg_runs.py output/center_control25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"fewonly"}
OUT_ROOT=${OUT_ROOT:-"center_control25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale 25 PROMPT_CENTER True
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True num_epochs 5 PEFT_WARMUP False
)
variant_args(){ case "$1" in
  randdir|headonly|fewonly|perclass_rand) echo "PROMPT_CENTER_MODE $1" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] center-control ${v} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
