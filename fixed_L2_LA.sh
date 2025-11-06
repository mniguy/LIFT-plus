#!/bin/bash

GPU_ID=3
DATA_ARG="imagenet_lt"
BACKBONE_ARG="clip_vit_b16"
METHOD_ARG="lift+"

# --- alpha 고정값 설정 ---
FIXED_ALPHA="0.3"

BASE_OUTPUT_DIR="FIXED/L2_LA"

echo "--- Starting HYBRID_BETA sweep (0.0 to 1.0) ---"
echo "Using GPU: ${GPU_ID}"

for i in $(seq 0 10)
do
  if [ $i -eq 10 ]; then
    BETA_VAL="1.0"
  else
    BETA_VAL="0.$i"
  fi

  OUTPUT_DIR="${BASE_OUTPUT_DIR}/beta_${BETA_VAL}"
  
  echo ""
  echo "=============================================================="
  echo ">> Running experiment for HYBRID_BETA = ${BETA_VAL}"
  echo ">> Output will be saved to: ${OUTPUT_DIR}"
  echo "=============================================================="

  CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d ${DATA_ARG} \
    -b ${BACKBONE_ARG} \
    -m ${METHOD_ARG} \
    classifier L2NormClassifier \
    HYBRID_BETA ${BETA_VAL} \
    HYBRID_ALPHA ${FIXED_ALPHA} \
    ADAPTIVE_ALPHA False \
    output_dir ${OUTPUT_DIR}
         
  echo "Experiment for HYBRID_BETA = ${BETA_VAL} finished."
  echo "--------------------------------------------------------------"
done

echo ""
echo "--- All beta sweep experiments finished! ---"