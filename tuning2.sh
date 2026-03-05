#!/bin/bash

# =================================================================
#               실험 환경 설정 (사용자 수정 영역)
# =================================================================

GPU_ID=2

DATA_ARG="imagenet_lt"
BACKBONE_ARG="clip_vit_b16"
METHOD_ARG="lift+"

# sweep할 값들
LAM_TAILS=(0.005 0.01 0.02 0.05 0.1)
LAM_OTHERS=(0.0 0.001 0.002 0.005 0.01)
TS=(0.5)

BASE_OUTPUT_DIR="textreg_tail"

echo "--- Starting TextReg sweep (tail/other/T) ---"
echo "Using GPU: ${GPU_ID}"
echo "Base output dir: ${BASE_OUTPUT_DIR}"
echo "Tail Lambdas:  ${LAM_TAILS[@]}"
echo "Other Lambdas: ${LAM_OTHERS[@]}"
echo "Temps:         ${TS[@]}"
echo ""

# =================================================================
#                       실험 자동 실행 루프
# =================================================================

for LTAIL in "${LAM_TAILS[@]}"; do
  for LOTHER in "${LAM_OTHERS[@]}"; do
    for T in "${TS[@]}"; do

      TAG="tail${LTAIL}_other${LOTHER}_T${T}"
      OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TAG}"

      echo "=============================================================="
      echo ">> Running: TAIL=${LTAIL}, OTHER=${LOTHER}, T=${T}"
      echo ">> Output: ${OUTPUT_DIR}"
      echo "=============================================================="

      mkdir -p "${OUTPUT_DIR}"

      CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        -d ${DATA_ARG} \
        -b ${BACKBONE_ARG} \
        -m ${METHOD_ARG} \
        tte True \
        TEXT_REG_LAMBDA_TAIL ${LTAIL} \
        TEXT_REG_LAMBDA_OTHER ${LOTHER} \
        TEXT_REG_T ${T} \
        output_dir ${OUTPUT_DIR}

      echo ">> Finished: ${TAG}"
      echo ""
    done
  done
done

echo "--- All TextReg sweep experiments finished! ---"