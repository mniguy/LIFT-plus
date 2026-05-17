#!/bin/bash

GPU_ID=0

# -----------------------------------------------------------------------
# 0. Verify best result (74.60% few) on this GPU before main experiments
# -----------------------------------------------------------------------
echo "=== [0] Verify best ==="

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d imagenet_lt -b clip_vit_b16 -m lift+ \
    tte True \
    v.lora True \
    PEFT_WARMUP True \
    WARMUP_TEXT_REG_T 0.001 \
    output_dir final_tte/verify_best

# Warmup hypers
WARMUP_LR=5e-4

# -----------------------------------------------------------------------
# A. Re-run warmup ablation winners with TTE
#    AdaptFormer-only, no scaling — just warmup variation
# -----------------------------------------------------------------------
echo "=== [1] Warmup ablation + TTE ==="

# ep2: warmup epochs=2, lr=5e-4
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d imagenet_lt -b clip_vit_b16 -m lift+ \
    tte True \
    PEFT_WARMUP True \
    PEFT_WARMUP_EPOCHS 2 \
    PEFT_WARMUP_LR 5e-4 \
    output_dir final_tte/warmup/ep2

# lr_1e-4: warmup epochs=1, lr=1e-4
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d imagenet_lt -b clip_vit_b16 -m lift+ \
    tte True \
    PEFT_WARMUP True \
    PEFT_WARMUP_EPOCHS 1 \
    PEFT_WARMUP_LR 1e-4 \
    output_dir final_tte/warmup/lr_1e-4
