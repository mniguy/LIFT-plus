#!/bin/bash
#
# Gating experiment on ImageNet-LT (hybrid init = the main method).
# ImageNet counterpart of scripts/run_places_gate.sh.
# The class gate weights the per-class KD/InfoNCE reg by sim(image-mean, prototype).
# PRIOR_GATE_NORM rescales the raw cosine; PRIOR_GATE_INVERT flips the direction.
#
# vs Places: just the dataset. ImageNet uses the default num_epochs (5) and the
# default SIM_THRESHOLD (0.6), same as Places. No PROMPT_MODE — the places_*
# prompt templates are scene-specific and don't apply to ImageNet object labels.
#
# IMPORTANT: run at a lambda where the reg ACTUALLY moves splits. At the old
# default (0.001) every variant collapses to baseline. Tune LAMBDA_KD / LAMBDA_NCE
# for ImageNet before trusting these results.
#
# Variants:
#   fixed         : no gate (uniform reg)            -- baseline at this lambda
#   minmax        : high-sim classes get stronger reg (trust prior where it aligns)
#   minmax_inv    : low-sim  classes get stronger reg (inject prior where it disagrees)
#   rank          : minmax's rank-based, distribution-robust variant
#   rank_inv      : rank + inverted direction
#
# Analyze with:
#   PYTHON scripts/analyze_places.py output/imagenet_gate --baseline fixed

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}

# Set these from a lambda sweep (where tail gains / head-med drop appear).
LAMBDA_KD=${LAMBDA_KD:-0.001}
LAMBDA_NCE=${LAMBDA_NCE:-0.005}   # turn on (e.g. 0.05) to also gate the InfoNCE term

COMMON_ARGS="-d imagenet_lt -b clip_vit_b16 -m lift+ tte True PEFT_WARMUP False \
    classifier_init hybrid \
    TEXT_REG_LAMBDA ${LAMBDA_KD} INFONCE_LAMBDA ${LAMBDA_NCE}"

wait_for_gpu () {
    while true; do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
        [ "${free:-0}" -ge "${MIN_FREE_MIB}" ] && break
        echo "    [wait] GPU ${GPU_ID} free=${free}MiB < ${MIN_FREE_MIB}MiB; retry in 60s..."
        sleep 60
    done
}

run_gate () {
    local name="$1"; shift
    wait_for_gpu
    echo "=== [ImageNet gate kd=${LAMBDA_KD} nce=${LAMBDA_NCE}] ${name} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        ${COMMON_ARGS} \
        "$@" \
        output_dir "imagenet_gate/${name}"
}

run_gate fixed       PRIOR_REG_MODE fixed
run_gate minmax      PRIOR_REG_MODE class_gate PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT False
run_gate minmax_inv  PRIOR_REG_MODE class_gate PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT True
run_gate rank        PRIOR_REG_MODE class_gate PRIOR_GATE_NORM rank   PRIOR_GATE_INVERT False
run_gate rank_inv    PRIOR_REG_MODE class_gate PRIOR_GATE_NORM rank   PRIOR_GATE_INVERT True

echo "=== [ImageNet gate] done. Analyze with: ==="
echo "    ${PYTHON} scripts/analyze_places.py output/imagenet_gate --baseline fixed"
