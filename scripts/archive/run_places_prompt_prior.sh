#!/bin/bash

GPU_ID=${GPU_ID:-0}

COMMON_ARGS="-d places_lt -b clip_vit_b16 -m lift+ seed 0 tte True PEFT_WARMUP False"

run_exp () {
    local name="$1"
    shift

    echo "=== [Places Prompt/Prior] ${name} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        ${COMMON_ARGS} \
        "$@" \
        output_dir "places_prompt_prior/${name}"
}

run_exp semantic_default_base \
    classifier_init semantic \
    PROMPT_MODE default \
    TEXT_REG_LAMBDA 0 \
    INFONCE_LAMBDA 0

run_exp hybrid_default_base \
    classifier_init hybrid \
    PROMPT_MODE default \
    TEXT_REG_LAMBDA 0 \
    INFONCE_LAMBDA 0

run_exp semantic_places_ensemble \
    classifier_init semantic \
    PROMPT_MODE places_ensemble \
    TEXT_REG_LAMBDA 0 \
    INFONCE_LAMBDA 0

run_exp hybrid_places_ensemble \
    classifier_init hybrid \
    PROMPT_MODE places_ensemble \
    TEXT_REG_LAMBDA 0 \
    INFONCE_LAMBDA 0

run_exp hybrid_places_fixed_kd \
    classifier_init hybrid \
    PROMPT_MODE places_ensemble \
    TEXT_REG_LAMBDA 0.001 \
    INFONCE_LAMBDA 0 \
    PRIOR_REG_MODE fixed

run_exp hybrid_places_fixed_nce \
    classifier_init hybrid \
    PROMPT_MODE places_ensemble \
    TEXT_REG_LAMBDA 0 \
    INFONCE_LAMBDA 0.005 \
    PRIOR_REG_MODE fixed

run_exp hybrid_places_gated_kd \
    classifier_init hybrid \
    PROMPT_MODE places_ensemble \
    TEXT_REG_LAMBDA 0.001 \
    INFONCE_LAMBDA 0 \
    PRIOR_REG_MODE class_gate

run_exp hybrid_places_gated_kd_nce \
    classifier_init hybrid \
    PROMPT_MODE places_ensemble \
    TEXT_REG_LAMBDA 0.001 \
    INFONCE_LAMBDA 0.005 \
    PRIOR_REG_MODE class_gate
