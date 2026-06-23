#!/bin/bash
#
# Sweep PROMPT_MODE (the template used to build text prototypes) while holding
# the gate fixed at the rank_inv setting (rank normalization + inverted
# direction: low-sim classes get the stronger reg). hybrid init.
#
# The prototype produced by PROMPT_MODE feeds (1) classifier init, (2) the reg
# target, and (3) the gate's similarity reference -- so changing it shifts both
# prior quality AND the gate distribution.
#
# NOTE: at LAMBDA_KD=0.001 the reg barely moves splits, so differences here may
# be noise. Set LAMBDA_KD/LAMBDA_NCE from the lambda sweep for a real comparison.
#
# Analyze with:
#   PYTHON scripts/analyze_places.py output/places_promptmode --baseline default

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}

LAMBDA_KD=${LAMBDA_KD:-0.001}
LAMBDA_NCE=${LAMBDA_NCE:-0.005}

# rank_inv gate, held fixed across the sweep
GATE_ARGS="PRIOR_REG_MODE class_gate PRIOR_GATE_NORM rank PRIOR_GATE_INVERT True"

COMMON_ARGS="-d places_lt -b clip_vit_b16 -m lift+ tte True PEFT_WARMUP False \
    classifier_init hybrid \
    TEXT_REG_LAMBDA ${LAMBDA_KD} INFONCE_LAMBDA ${LAMBDA_NCE} ${GATE_ARGS}"

PROMPT_MODES=${PROMPT_MODES:-"default places_scene places_place places_ensemble"}

wait_for_gpu () {
    while true; do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
        [ "${free:-0}" -ge "${MIN_FREE_MIB}" ] && break
        echo "    [wait] GPU ${GPU_ID} free=${free}MiB < ${MIN_FREE_MIB}MiB; retry in 60s..."
        sleep 60
    done
}

for mode in ${PROMPT_MODES}; do
    wait_for_gpu
    echo "=== [Places promptmode | rank_inv gate] ${mode} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        ${COMMON_ARGS} \
        PROMPT_MODE ${mode} \
        output_dir "places_promptmode/${mode}"
done

echo "=== [Places promptmode] done. Analyze with: ==="
echo "    ${PYTHON} scripts/analyze_places.py output/places_promptmode --baseline default"
