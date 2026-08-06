#!/bin/bash
#
# MUST-ADD (E) -- 5 seeds for the two middle imbalance ratios.
#
# WHY. draft_results.tex Sec. "The gain scales with imbalance severity" and tab:irextremes
# both state "four imbalance ratios, 5 seeds each". That is not what the artifacts contain:
#     IR40, IR200  -> output/ir_extremes25/, 5 seeds (baseline_seed0..4, center_seed0..4)
#     IR50, IR100  -> output/breadth25/,     SINGLE seed (baseline, center)
# The severity trend (-0.14, -0.27, +2.24, +5.88) therefore has n=1 at both interior points,
# and the two neutral cases that the paper uses to derive its applicability conditions
# (IR50 "already at ceiling") rest on one run. This script fills IR50/IR100 to 5 seeds in the
# SAME root as IR40/IR200 so the whole table comes from one place.
#
# Protocol copied from run_ir_extremes.sh / run_breadth.sh: scale 25, semantic, MDA, TTE, 5 ep.
# Seed 0 is re-run here for a uniform root; it should reproduce breadth25 exactly
# (cfg.deterministic=True), which doubles as a determinism check -- if it does not, that is
# itself a finding and the seed-noise estimates in tab:main need revisiting.
#
#   bash scripts/run_ir_mid_seeds.sh
#   SEEDS="1 2 3 4" bash scripts/run_ir_mid_seeds.sh          # skip the seed-0 duplicate
#   DATASETS=cifar100_ir10 bash scripts/run_ir_mid_seeds.sh   # optional 4th neutral point
#   python scripts/agg_runs.py output/ir_extremes25 --sort path
#
# Cost: 2 ratios x 2 arms x 5 seeds = 20 runs @ 5 ep on CIFAR-100 (cheap).
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"cifar100_ir100 cifar100_ir50"}
VARIANTS=${VARIANTS:-"baseline center"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"ir_extremes25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=( classifier_init semantic classifier_scale "${SCALE}" mda True tte True )
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    for s in ${SEEDS}; do
      out="${OUT_ROOT}/${data}/${v}_seed${s}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] ${v} seed=${s} ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ \
        "${BASE_ARGS[@]}" num_epochs "${EPOCHS}" ${va} \
        seed "${s}" output_dir "${out}"
    done
  done
done
echo
echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    Then rewrite tab:irextremes with per-ratio mean+/-std and the correct seed counts."
