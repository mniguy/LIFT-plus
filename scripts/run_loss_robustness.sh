#!/bin/bash
#
# Experiment 3 -- is centering an INDEPENDENT lever, or does it just duplicate what a
# long-tail loss (logit adjustment) already does?
#
# For each loss we run centering ON vs OFF. If the Few gain survives across every loss --
# including plain CE (no prior correction at all) and BalancedSoftmax/LDAM/LA (different
# prior-correction schemes) -- then centering is doing something losses do NOT: a
# geometric fix, stackable on top of any of them. A large gain under CE is the strongest
# form of this argument.
#
#   losses: CE (none) | Focal | CB (class-balanced) | GRW (generalized reweight) | BS (balanced
#           softmax) | LDAM | LA (logit-adjusted, our default) | LADE | VS (vector-scaling,
#           Kini 2021 -- the modern additive+multiplicative generalization of LA/CDT)
#   center: baseline (PROMPT_CENTER False) vs center (global)
#   semantic init, aux off, scale 25, 5 ep, mda+tte -- matches the main-result setup.
#
#   bash scripts/run_loss_robustness.sh
#   python scripts/agg_runs.py output/loss_robust25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
LOSSES=${LOSSES:-"CE Focal CB GRW BS LDAM LA LADE VS"}
VARIANTS=${VARIANTS:-"baseline center"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"loss_robust25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}"
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True PEFT_WARMUP False num_epochs "${EPOCHS}"
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for loss in ${LOSSES}; do
    for v in ${VARIANTS}; do
      va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
      out="${OUT_ROOT}/${data}/${loss}_${v}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] loss=${loss} ${v} (scale ${SCALE}, ${EPOCHS} ep) ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ \
        "${BASE_ARGS[@]}" loss_type "${loss}" ${va} seed "${SEED}" \
        output_dir "${out}"
    done
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
