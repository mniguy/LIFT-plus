#!/bin/bash
# AdaptFormer + Text-guided Warmup ablation
# Warmup uses text regularization (KD) only — no InfoNCE
#
# Output structure under ./output/:
#   warmup/ep{N}/          — epoch ablation
#   warmup/lr_{LR}/        — LR ablation
#   init/svd/              — text-SVD init (uncentered)
#   init/pca/              — text-PCA init (centered)
#   init/svd_warm/         — text-SVD + warmup
#   init/pca_warm/         — text-PCA + warmup
#   progressive/s{N}/      — layer-progressive, start=N
#   progressive/s3_ep6/    — layer-progressive, start=3, 6 epochs
#
# Run order:
#   1. Warmup epoch ablation   — how long to warm
#   2. Warmup LR ablation      — how fast to warm
#   3. Init ablation           — SVD vs PCA, with/without warmup
#   4. Progressive warmup      — layer unlock schedule

# -----------------------------------------------------------------------
# Shared warmup hypers — update BEST_EP / BEST_LR / BEST_TEXT after group 1-2
# -----------------------------------------------------------------------
GPU_ID=0

BEST_EP=1
BEST_LR=5e-4
BEST_TEXT=0.0001

# -----------------------------------------------------------------------
# 1. Warmup epoch ablation  (ImageNet-LT)
# -----------------------------------------------------------------------
echo "=== [1] Warmup epoch ablation ==="

for EP in 1 2 3 5; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
        PEFT_WARMUP True \
        PEFT_WARMUP_EPOCHS ${EP} \
        PEFT_WARMUP_LR     5e-4 \
        WARMUP_TEXT_REG_LAMBDA 0.0001 \
        output_dir warmup/ep${EP}
done

# -----------------------------------------------------------------------
# 2. Warmup LR ablation  (ImageNet-LT)
# -----------------------------------------------------------------------
echo "=== [2] Warmup LR ablation ==="

for LR in 1e-4 5e-4 1e-3 2e-3; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
        PEFT_WARMUP True \
        PEFT_WARMUP_EPOCHS 1 \
        PEFT_WARMUP_LR     ${LR} \
        WARMUP_TEXT_REG_LAMBDA 0.0001 \
        output_dir warmup/lr_${LR}
done

# -----------------------------------------------------------------------
# 3. Init ablation: text-SVD / text-PCA  (ImageNet-LT)
#    SVD  — uncentered, dominant directions in text embedding space
#    PCA  — centered (W - mean), class-discriminative directions
# -----------------------------------------------------------------------
echo "=== [3] Init ablation (SVD / PCA) ==="

# SVD init only (no warmup)
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
    PEFT_WARMUP False \
    v.adaptformer_init text_svd \
    output_dir init/svd

# PCA init only (no warmup)
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
    PEFT_WARMUP False \
    v.adaptformer_init text_pca \
    output_dir init/pca

# SVD init + warmup
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
    PEFT_WARMUP True \
    PEFT_WARMUP_EPOCHS ${BEST_EP}  PEFT_WARMUP_LR ${BEST_LR} \
    WARMUP_TEXT_REG_LAMBDA ${BEST_TEXT} \
    v.adaptformer_init text_svd \
    output_dir init/svd_warm

# PCA init + warmup
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
    PEFT_WARMUP True \
    PEFT_WARMUP_EPOCHS ${BEST_EP}  PEFT_WARMUP_LR ${BEST_LR} \
    WARMUP_TEXT_REG_LAMBDA ${BEST_TEXT} \
    v.adaptformer_init text_pca \
    output_dir init/pca_warm

# -----------------------------------------------------------------------
# 4. Layer-progressive warmup ablation  (ImageNet-LT)
#    Starts with last N layers, adds one layer earlier each epoch.
# -----------------------------------------------------------------------
echo "=== [4] Layer-progressive warmup ==="

for START in 2 3 6; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
        PEFT_WARMUP True \
        PEFT_WARMUP_EPOCHS ${BEST_EP} \
        PEFT_WARMUP_LR ${BEST_LR} \
        WARMUP_TEXT_REG_LAMBDA ${BEST_TEXT} \
        PEFT_WARMUP_PROGRESSIVE True \
        PEFT_WARMUP_PROGRESSIVE_START ${START} \
        output_dir progressive/s${START}
done

# more epochs to fully traverse all layers
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
    PEFT_WARMUP True \
    PEFT_WARMUP_EPOCHS 6 \
    PEFT_WARMUP_LR ${BEST_LR} \
    WARMUP_TEXT_REG_LAMBDA ${BEST_TEXT} \
    PEFT_WARMUP_PROGRESSIVE True \
    PEFT_WARMUP_PROGRESSIVE_START 3 \
    output_dir progressive/s3_ep6

# -----------------------------------------------------------------------
# 5. Tail-weighted InfoNCE warmup ablation  (ImageNet-LT)
#    Per-sample InfoNCE weighted by 1/n_c^power — tail classes get stronger
#    text alignment signal during warmup.
#    NOTE: InfoNCE is enabled only for this group.
# -----------------------------------------------------------------------
echo "=== [5] Tail-weighted InfoNCE warmup ==="

for POWER in 0.25 0.5 1.0; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
        PEFT_WARMUP True \
        PEFT_WARMUP_EPOCHS ${BEST_EP}  PEFT_WARMUP_LR ${BEST_LR} \
        WARMUP_TEXT_REG_LAMBDA ${BEST_TEXT} \
        WARMUP_INFONCE_LAMBDA  0.005 \
        WARMUP_TAIL_WEIGHTED True \
        WARMUP_TAIL_WEIGHT_POWER ${POWER} \
        output_dir tail_weighted/power_${POWER}
done
