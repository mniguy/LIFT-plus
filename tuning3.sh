#!/bin/bash

GPU_ID=1
DATA_ARG="imagenet_lt"
BACKBONE_ARG="clip_vit_b16"
METHOD_ARG="lift+"

TEXT_REG_LAMBDA=0.001
TEXT_REG_T=0.01

INFONCE_LAMBDAS=(0.005)
INFONCE_TS=(0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.5)

BASE_OUTPUT_DIR="infonce"

echo "--- Starting sweep ---"
echo "GPU: ${GPU_ID}"
echo "TEXT_REG_LAMBDA: ${TEXT_REG_LAMBDA}"
echo "TEXT_REG_T: ${TEXT_REG_T}"
echo "INFONCE_LAMBDAS: ${INFONCE_LAMBDAS[@]}"
echo "INFONCE_TS: ${INFONCE_TS[@]}"
echo ""

for L in "${INFONCE_LAMBDAS[@]}"; do
  for T in "${INFONCE_TS[@]}"; do
    TAG="infonceL${L}_T${T}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TAG}"

    echo "=============================================================="
    echo ">> INFONCE_LAMBDA=${L}, INFONCE_T=${T}"
    echo ">> Output: ${OUTPUT_DIR}"
    echo "=============================================================="

    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
      -d ${DATA_ARG} \
      -b ${BACKBONE_ARG} \
      -m ${METHOD_ARG} \
      tte True \
      TEXT_REG_LAMBDA ${TEXT_REG_LAMBDA} \
      TEXT_REG_T ${TEXT_REG_T} \
      INFONCE_LAMBDA ${L} \
      INFONCE_T ${T} \
      output_dir ${OUTPUT_DIR}

    echo ">> Finished: ${TAG}"
    echo ""
  done
done

echo "--- All experiments finished! ---"