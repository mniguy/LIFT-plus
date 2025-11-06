#!/bin/bash

# =================================================================
#               실험 환경 설정 (사용자 수정 영역)
# =================================================================

# 사용할 GPU ID 설정
GPU_ID=0

# 공통적으로 사용할 설정 인자
DATA_ARG="imagenet_lt"
BACKBONE_ARG="clip_vit_b16"
METHOD_ARG="lift+"

# 💡 --- 고정할 파라미터가 있다면 여기에 설정 --- 💡
# 예: 위키 캡션 Top-K, 고정된 Beta (필요하다면)
FIXED_TOPK=8
# FIXED_BETA="0.0" 

# 모든 실험 결과가 저장될 기본 폴더명
BASE_OUTPUT_DIR="dynamic_alpha_sweep"

echo "--- Starting Dynamic Alpha Sweep (Many, Med, Few) ---"
echo "Using GPU: ${GPU_ID}"


# =================================================================
#           💡 테스트할 Alpha 조합 목록 (사용자 수정 영역) 💡
# =================================================================
#
# 여기에 테스트하고 싶은 조합을 추가하세요.
# 각 배열의 동일한 인덱스(i)가 하나의 실험 조합이 됩니다.
# (예: 1번째 실험 = 0.3, 0.6, 0.9 / 2번째 실험 = 0.5, 0.5, 0.5)

# ALPHA_MANY (Many-shot)에 적용할 값 목록
ALPHA_MANY_LIST=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# ALPHA_MED (Medium-shot)에 적용할 값 목록
ALPHA_MED_LIST=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# ALPHA_FEW (Few-shot)에 적용할 값 목록
ALPHA_FEW_LIST=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# --- 조합 설명 ---
# Combo 1 (0.3, 0.6, 0.9): Few일수록 캡션 비중(1-alpha)을 높게 (사용자님 예시)
# Combo 2 (0.5, 0.5, 0.5): 모든 클래스 동일한 비율 (베이스라인)
# Combo 3 (0.9, 0.6, 0.3): Many일수록 캡션 비중을 높게
# Combo 4 (1.0, 1.0, 1.0): 모든 클래스 프롬프트만 사용 (캡션x)
# Combo 5 (0.0, 0.0, 0.0): 모든 클래스 캡션만 사용 (프롬프트x)


# =================================================================
#                       실험 자동 실행 루프
# =================================================================

# 정의된 조합의 총 개수 확인 (배열 길이)
num_experiments=${#ALPHA_MANY_LIST[@]}

echo "Total experiments to run: ${num_experiments}"

# 0부터 (조합 개수 - 1)까지 반복
for i in $(seq 0 $(($num_experiments - 1)))
do
  # 1. 현재 인덱스(i)에 해당하는 Alpha 값들을 추출
  alpha_m=${ALPHA_MANY_LIST[$i]}
  alpha_e=${ALPHA_MED_LIST[$i]} # (mEdium)
  alpha_f=${ALPHA_FEW_LIST[$i]}

  # 2. 이번 실험 결과를 저장할 고유한 폴더명 생성
  # 예: dynamic_alpha_sweep/alpha_0.3_0.6_0.9
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/alpha_${alpha_m}_${alpha_e}_${alpha_f}"
  
  echo ""
  echo "=============================================================="
  echo ">> Running Experiment #$(($i + 1)) / ${num_experiments}"
  echo "   - ALPHA_MANY: ${alpha_m}"
  echo "   - ALPHA_MED : ${alpha_e}"
  echo "   - ALPHA_FEW : ${alpha_f}"
  echo "   - Output dir: ${OUTPUT_DIR}"
  echo "--------------------------------------------------------------"
  
  # 3. main.py 실행
  # 💡 3개의 Alpha 값을 모두 커맨드 라인 인자로 전달
  CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    -d ${DATA_ARG} \
    -b ${BACKBONE_ARG} \
    -m ${METHOD_ARG} \
    HYBRID_TOPK ${FIXED_TOPK} \
    ALPHA_MANY ${alpha_m} \
    ALPHA_MED ${alpha_e} \
    ALPHA_FEW ${alpha_f} \
    output_dir ${OUTPUT_DIR}
    # HYBRID_BETA ${FIXED_BETA} \ # <--- Beta 값도 고정해야 한다면 주석 해제
         
  echo "Experiment for combo (${alpha_m}, ${alpha_e}, ${alpha_f}) finished."
  echo "=============================================================="
  sleep 2 # 다음 실험 전 잠시 대기 (선택 사항)
done

echo ""
echo "--- All dynamic alpha sweep experiments finished! ---"