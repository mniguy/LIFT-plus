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

# -----------------------------------------------------------------------
# B. Warmup with InfoNCE (text+nce ablation, verify_best base + TTE)
#    Previous best: text0.005_nce0.0001 (few=74.07, AdaptFormer-only)
#    Now layered on verify_best setup (lora True, WARMUP_TEXT_REG_T 0.001)
# -----------------------------------------------------------------------
echo "=== [2] Warmup InfoNCE + TTE ==="

for TEXT in 0.0001 0.005; do
    for NCE in 0.0001 0.001; do
        TAG="text${TEXT}_nce${NCE}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
            -d imagenet_lt -b clip_vit_b16 -m lift+ \
            tte True \
            v.lora True \
            PEFT_WARMUP True \
            WARMUP_TEXT_REG_LAMBDA ${TEXT} \
            WARMUP_INFONCE_LAMBDA ${NCE} \
            output_dir final_tte/warmup_nce/${TAG}
    done
done
