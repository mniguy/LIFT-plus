#!/bin/bash
#
# CONTROL for #3: is the caption-geometry gain from the CAPTIONS, or just from
# de-anisotropizing the prototype (subtracting the global prompt centroid)?
#
# LA loss only (aux OFF), classifier_init=semantic (prompt-only, NO captions). Only knob
# is PROMPT_CENTER. The no-center cell is ALREADY on record as the 11-seed baseline in
# "output/seed_ablation 25/" (identical config: aux off, scale 25, mda/tte/5ep/no-warmup):
#     ImageNet 78.33 +/- 0.06    Places 52.15 +/- 0.10    (n=11)
# so we only run `center` and test it against that anchor.
#
#   center >> anchor (>2 sigma: >78.46 IN / >52.35 PL)  -> centering alone helps (Direction 3 worth it)
#   center ~= anchor                                    -> centering alone is a no-op
#
#   bash scripts/run_prompt_center.sh
#   VARIANTS="plain center" bash scripts/run_prompt_center.sh   # also re-run no-center in-session
#   python scripts/agg_runs.py output/prompt_center25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"center"}
OUT_ROOT=${OUT_ROOT:-"prompt_center25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# Matches the "seed_ablation 25" baseline (semantic, scale 25, aux OFF), only PROMPT_CENTER varies.
BASE_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True num_epochs 5
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
