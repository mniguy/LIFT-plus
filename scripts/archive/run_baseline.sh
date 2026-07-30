#!/bin/bash
# Baseline: original LIFT (no hybrid caption, no TEXT_REG, no InfoNCE, no warmup, no TTE)
# Use this checkpoint + cls_accs.npy for visualization comparison.

GPU_ID=0

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d imagenet_lt -b clip_vit_b16 -m lift+ \
    classifier_init semantic \
    output_dir baseline_lift
