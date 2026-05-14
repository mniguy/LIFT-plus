#!/bin/bash
# Single-module + scaling + warmup (all 5 methods)
#
# Motivation: best result so far (74.60% few) used dual-module (AdaptFormer + LoRA)
# with last-2-layer scaling + warmup. Goal: match or close that gap with one module.
#
# Unexplored combinations:
#   A) AdaptFormer + last-layer scaling + warmup (1+2+3+4+5)
#   B) LoRA        + last-layer scaling + warmup (1+2+3+4+5)
#
# Reference baselines:
#   scale/layer2_scale8.0          : adaptformer-only, 1+2+4        → overall=78.67  few=73.71
#   scale_lora/layer2_scale16.0    : lora-only,        1+2+4        → overall=78.51  few=74.15
#   warmup/ep3                     : adaptformer-only, 1+2+3+5      → overall=77.23  few=72.69
#   hybrid_AdaptFormer_scaling/    : dual-module,      1+2+3+4+5    → overall=78.58  few=74.26 (best)
#
# Output: output/single_module/adaptformer_* and output/single_module/lora_*

GPU_ID=0

# Warmup hypers
WARMUP_LR=5e-4

# -----------------------------------------------------------------------
# A. AdaptFormer-only + last-layer scaling + warmup
#    Best single-module scale: layer2_scale8.0 (few=73.71)
#    Try ep=1 (same as best dual-module) and ep=3 (best from warmup ablation)
# -----------------------------------------------------------------------
echo "=== [A] AdaptFormer + scaling + warmup ==="

for LAYERS in 1 2; do
    for SCALE in 8.0 16.0; do
        for EP in 1 3; do
            TAG="adaptformer_layer${LAYERS}_scale${SCALE}_ep${EP}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
                -d imagenet_lt -b clip_vit_b16 -m lift+ \
                v.adaptformer_layers_last ${LAYERS} \
                v.adaptformer_dim_last_scale ${SCALE} \
                PEFT_WARMUP True \
                PEFT_WARMUP_EPOCHS ${EP} \
                PEFT_WARMUP_LR ${WARMUP_LR} \
                output_dir final/${TAG}
        done
    done
done

# -----------------------------------------------------------------------
# B. LoRA-only + last-layer scaling + warmup
#    Best single-module scale: layer2_scale16.0 (few=74.15)
#    lift+.yaml has adaptformer=True by default — must disable it
# -----------------------------------------------------------------------
echo "=== [B] LoRA + scaling + warmup ==="

for LAYERS in 1 2; do
    for SCALE in 8.0 16.0; do
        for EP in 1 3; do
            TAG="lora_layer${LAYERS}_scale${SCALE}_ep${EP}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
                -d imagenet_lt -b clip_vit_b16 -m lift+ \
                v.adaptformer False \
                v.lora True \
                v.lora_layers_last ${LAYERS} \
                v.lora_dim_last_scale ${SCALE} \
                PEFT_WARMUP True \
                PEFT_WARMUP_EPOCHS ${EP} \
                PEFT_WARMUP_LR ${WARMUP_LR} \
                output_dir final/${TAG}
        done
    done
done
