#!/bin/bash

GPU_ID=0

echo "=== [0] LIFT+ baseline ==="

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d places_lt -b clip_vit_b16 -m lift+ \
    classifier_init semantic \
    tte True \
    TEXT_REG_LAMBDA 0.0 \
    INFONCE_LAMBDA 0.0 \
    output_dir baseline/places_lt

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 \
    classifier_init semantic \
    tte True \
    TEXT_REG_LAMBDA 0.0 \
    INFONCE_LAMBDA 0.0 \
    output_dir baseline/inat2018


echo "=== [1] Proposed Method ==="

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d places_lt -b clip_vit_b16 -m lift+ \
    tte True \
    PEFT_WARMUP True \
    output_dir final_tte/places_lt

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 \
    tte True \
    PEFT_WARMUP True \
    output_dir final_tte/inat2018