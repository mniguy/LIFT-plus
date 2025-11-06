#!/bin/bash

# =================================================================
#               실험 환경 설정 (사용자 수정 영역)
# =================================================================

# 사용할 GPU ID 설정
GPU_ID=1

# 공통적으로 사용할 설정 인자
DATA_ARG="imagenet_lt"
BACKBONE_ARG="clip_vit_b16"
METHOD_ARG="lift+"

# 모든 실험 결과가 저장될 기본 폴더명
BASE_OUTPUT_DIR="ADAPT/L2_LA"

echo "--- Starting HYBRID_BETA sweep (0.0 to 1.0) ---"
echo "Using GPU: ${GPU_ID}"

# =================================================================
#                       실험 자동 실행 루프
# =================================================================

# 0부터 10까지 1씩 증가하는 루프 (HYBRID_BETA = 0.0, 0.1, ..., 1.0)
for i in $(seq 0 10)
do
  # HYBRID_BETA 값 생성 (0.0, 0.1, ..., 1.0)
  if [ $i -eq 10 ]; then
    BETA_VAL="1.0"
  else
    BETA_VAL="0.$i"
  fi
  
  # 각 실험 결과를 저장할 고유한 출력 폴더 이름 설정
  # 예: beta_sweep/beta_0.7
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/beta_${BETA_VAL}"
  
  echo ""
  echo "=============================================================="
  echo ">> Running experiment for HYBRID_BETA = ${BETA_VAL}"
  echo ">> Output will be saved to: ${OUTPUT_DIR}"
  echo "=============================================================="
  
  # main.py 실행
  # 💡 HYBRID_BETA 뒤에 ${BETA_VAL} 값을 명시적으로 전달
  # 💡 다른 고정 파라미터(예: FIXED_ALPHA)가 있다면 아래에 추가하세요.
  CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d ${DATA_ARG} \
    -b ${BACKBONE_ARG} \
    -m ${METHOD_ARG} \
    classifier L2NormClassifier \
    HYBRID_BETA ${BETA_VAL} \
    output_dir ${OUTPUT_DIR}
         
  echo "Experiment for HYBRID_BETA = ${BETA_VAL} finished."
  echo "--------------------------------------------------------------"
done

echo ""
echo "--- All beta sweep experiments finished! ---"