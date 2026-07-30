#!/bin/bash
#
# Multi-seed confirmation for the RARITY-WEIGHTED centering variants (tail / kappa50 / logcount)
# on ImageNet-LT and Places-LT. Settles the one open question about the paper's main-method
# choice: single-seed runs put tail/kappa/logcount slightly ABOVE global on Overall (PL 52.43 /
# 52.38 / 52.41 vs global 52.24) while clearly BELOW it on Few -- but the Overall gap is only
# ~2 sigma on one seed (IN's whole spread, 0.03, equals the 5-seed std), so it may be noise.
#
#   Headline metric of the paper is Few, where global wins at matched seed 0:
#     IN  global +1.63  vs tail +1.13        PL  global +2.21  vs tail +1.71   (Delta vs baseline_seed0)
#   Overall is where the rarity variants edge ahead, and that is what needs multiple seeds.
#
#   Reading when done: if the rarity variants stay ahead on Overall at 5 seeds, the ablation must
#   say "Overall -> rarity-weighted, Few -> uniform"; if it washes out, uniform global centering
#   stands as the single main method and rarity weighting becomes a clean negative ablation
#   ("weighting trades Few away for Head and buys nothing overall").
#
# Settings are byte-identical to run_center_kappa.sh / run_center_geom.sh so seed 0 (already run,
# see below) can be pooled with the seeds produced here.
#   seed 0 already exists at:
#     tail      output/center_geom25/{imagenet_lt,places_lt}/tail
#     kappa50   output/center_kappa25/{imagenet_lt,places_lt}/kappa50
#     logcount  output/center_kappa25/{imagenet_lt,places_lt}/logcount
#   baseline seeds 0-4   output/seed_ablation 25/{imagenet_lt,places_lt}/baseline_seed{0..4}
#   global   seed 0      output/prompt_center25/{imagenet_lt,places_lt}/center
#   global   seeds 1-4   output/center_seeds25/{imagenet_lt,places_lt}/center_seed{1..4}
#
# Cost: ImageNet ~15 min/run, Places ~5 min/run -> 4 seeds x 3 variants = 3h (IN) + 1h (PL).
# Split across two GPUs by dataset to halve it:
#   GPU_ID=0 DATASETS="imagenet_lt" bash scripts/run_rarity_seeds.sh &
#   GPU_ID=1 DATASETS="places_lt"   bash scripts/run_rarity_seeds.sh &
#
#   bash scripts/run_rarity_seeds.sh
#   python scripts/agg_rarity_seeds.py            # pools seed 0 + these, mean/std/paired-t vs baseline
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"tail kappa50 logcount"}
SEEDS=${SEEDS:-"1 2 3 4"}
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"rarity_seeds25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale 25 PROMPT_CENTER True
  mda True tte True num_epochs "${EPOCHS}"
)
variant_args(){ case "$1" in
  global|tail|logcount) echo "PROMPT_CENTER_MODE $1" ;;
  kappa[0-9]*)          echo "PROMPT_CENTER_MODE kappa PROMPT_CENTER_KAPPA ${1#kappa}" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    for s in ${SEEDS}; do
      out="${OUT_ROOT}/${data}/${v}_seed${s}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] ${v} seed ${s} (${EPOCHS} ep) ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
        "${BASE_ARGS[@]}" ${va} seed "${s}" output_dir "${out}"
    done
  done
done
echo; echo "=== analyze: ${PYTHON} scripts/agg_rarity_seeds.py ==="
