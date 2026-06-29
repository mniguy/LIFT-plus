#!/bin/bash
#
# #3: tail feature distribution calibration (text-free, on top of LIFT+ features).
# Stage 1: train plain LIFT+ (saves checkpoint).
# Stage 2: feature_calibration.py -- extract features, calibrate tail Gaussians from
#          head covariance, synthesize balanced features, RE-TRAIN the cosine head,
#          eval ORIG head vs CALIBRATED head on the same single-crop test features.
#
# Controlled comparison (same features, only the head differs). Single seed for a
# first signal; if the Δfew/Δall is positive and clears noise, run more seeds.

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
TARGET_N=${TARGET_N:-200}
K=${K:-2}
ALPHA=${ALPHA:-0.21}

for ds in ${DATASETS}; do
  ckpt="calib/${ds}/lift+_train"
  echo "=== [${ds}] stage1: train LIFT+ ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
    -d "${ds}" -b clip_vit_b16 -m lift+ tte True PEFT_WARMUP False seed "${SEED}" \
    classifier_init semantic TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0 PRIOR_REG_MODE fixed \
    output_dir "${ckpt}"

  echo "=== [${ds}] stage2: feature calibration ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} scripts/feature_calibration.py \
    -d "${ds}" -b clip_vit_b16 -m lift+ \
    --ckpt "output/${ckpt}" --out "output/calib/${ds}" \
    --target-n "${TARGET_N}" --k "${K}" --alpha "${ALPHA}" --seed "${SEED}"
done

echo ""
echo "=== done. Each output/calib/<ds> prints LIFT+(orig head) vs calibrated + Δ. ==="
