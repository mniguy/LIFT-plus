#!/bin/bash
#
# Multi-seed the headline (I = global centering) so it can be paired against the 11-seed
# baseline in "output/seed_ablation 25/" (IN 78.33+/-0.06, PL 52.15+/-0.10) for a proper
# paired test. seed 0 already exists as prompt_center25/*/center; this adds seeds 1..N.
# Semantic init, aux OFF, scale 25 (identical to the baseline config).
#
#   bash scripts/run_center_seeds.sh
#   SEEDS="1 2 3 4 5 6 7 8 9 10" bash scripts/run_center_seeds.sh
#   python scripts/agg_runs.py output/center_seeds25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
SEEDS=${SEEDS:-"1 2 3 4"}
OUT_ROOT=${OUT_ROOT:-"center_seeds25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale 25 PROMPT_CENTER True PROMPT_CENTER_MODE global
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True num_epochs 5 PEFT_WARMUP False
)
completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for s in ${SEEDS}; do
    out="${OUT_ROOT}/${data}/center_seed${s}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] center seed=${s} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" seed "${s}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
