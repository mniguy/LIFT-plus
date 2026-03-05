#!/bin/bash

GPU_ID=2
DATA_ARG="imagenet_lt"
BACKBONE_ARG="clip_vit_b16"
METHOD_ARG="lift+"

SIM_THRESHOLD=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
LAMBDAS=(0.001)
TS=(0.01)

BASE_OUTPUT_DIR="simthres"

echo "--- Starting sweep ---"
echo "GPU: ${GPU_ID}"
echo "SIM_THRESHOLD: ${SIM_THRESHOLD[@]}"
echo "LAMBDAS: ${LAMBDAS[@]}"
echo "TS: ${TS[@]}"
echo ""

for S in "${SIM_THRESHOLD[@]}"; do
  for L in "${LAMBDAS[@]}"; do
    for T in "${TS[@]}"; do
      TAG="S${S}"
      OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TAG}"

      echo "=============================================================="
      echo ">> SIM_THRESHOLD=${S}, TEXT_REG_LAMBDA=${L}, TEXT_REG_T=${T}"
      echo ">> Output: ${OUTPUT_DIR}"
      echo "=============================================================="

      CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        -d ${DATA_ARG} \
        -b ${BACKBONE_ARG} \
        -m ${METHOD_ARG} \
        tte True \
        SIM_THRESHOLD ${S} \
        TEXT_REG_LAMBDA ${L} \
        TEXT_REG_T ${T} \
        output_dir ${OUTPUT_DIR}

      echo ">> Finished: ${TAG}"
      echo ""
    done
  done
done

echo "--- All experiments finished! ---"