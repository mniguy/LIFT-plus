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

# 💡 --- 고정할 파라미터가 있다면 여기에 설정 --- 💡
FIXED_TOPK=8

# 모든 실험 결과가 저장될 기본 폴더명
BASE_OUTPUT_DIR="alpha_grid_search"

echo "--- Starting Dynamic Alpha Grid Search (All Combinations) ---"
echo "Using GPU: ${GPU_ID}"


# =================================================================
#           💡 테스트할 Alpha 후보 목록 (사용자 수정 영역) 💡
# =================================================================
#
# 여기에 각 그룹별로 테스트하고 싶은 모든 후보 값을 공백으로 띄어 나열하세요.
# 스크립트가 이 3개 리스트의 모든 조합(경우의 수)을 만들어 실행합니다.

# ALPHA_MANY (Many-shot) 후보
ALPHA_MANY_CANDIDATES=(0.1 0.3 0.5)

# ALPHA_MED (Medium-shot) 후보
ALPHA_MED_CANDIDATES=(0.3 0.5 0.7)

# ALPHA_FEW (Few-shot) 후보
ALPHA_FEW_CANDIDATES=(0.5 0.7 0.9)

# --- 예시 ---
# 위와 같이 설정하면:
# 3 (Many) * 3 (Med) * 3 (Few) = 총 27개의 실험이 실행됩니다.
# (0.3, 0.5, 0.7), (0.3, 0.5, 0.9), (0.3, 0.5, 1.0), (0.3, 0.7, 0.7), ...

# =================================================================
#                       실험 자동 실행 루프
# =================================================================

# 총 실험 횟수 계산
num_many=${#ALPHA_MANY_CANDIDATES[@]}
num_med=${#ALPHA_MED_CANDIDATES[@]}
num_few=${#ALPHA_FEW_CANDIDATES[@]}
total_experiments=$(($num_many * $num_med * $num_few))

echo "Total experiments to run (Grid Search): $num_many(Many) * $num_med(Med) * $num_few(Few) = ${total_experiments}"
echo ""

current_experiment=1

# 3중 중첩 루프를 사용하여 모든 경우의 수 실행
for alpha_m in "${ALPHA_MANY_CANDIDATES[@]}"
do
  for alpha_e in "${ALPHA_MED_CANDIDATES[@]}" # (mEdium)
  do
    for alpha_f in "${ALPHA_FEW_CANDIDATES[@]}"
    do
      # 1. 이번 실험 결과를 저장할 고유한 폴더명 생성
      # 예: alpha_grid_search/m_0.3_e_0.5_f_0.7
      OUTPUT_DIR="${BASE_OUTPUT_DIR}/m_${alpha_m}_e_${alpha_e}_f_${alpha_f}"
      
      echo ""
      echo "=============================================================="
      echo ">> Running Experiment ${current_experiment} / ${total_experiments}"
      echo "   - ALPHA_MANY: ${alpha_m}"
      echo "   - ALPHA_MED : ${alpha_e}"
      echo "   - ALPHA_FEW : ${alpha_f}"
      echo "   - Output dir: ${OUTPUT_DIR}"
      echo "--------------------------------------------------------------"

    # 2. main.py 실행
      CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        -d ${DATA_ARG} \
        -b ${BACKBONE_ARG} \
        -m ${METHOD_ARG} \
        HYBRID_TOPK ${FIXED_TOPK} \
        ALPHA_MANY ${alpha_m} \
        ALPHA_MED ${alpha_e} \
        ALPHA_FEW ${alpha_f} \
        output_dir ${OUTPUT_DIR}
             
      echo "Experiment for combo (M:${alpha_m}, E:${alpha_e}, F:${alpha_f}) finished."
      echo "=============================================================="
      
      # 실험 카운터 증가
      ((current_experiment++))
      sleep 2 # 다음 실험 전 잠시 대기 (선택 사항)
    done
  done
done

echo ""
echo "--- All grid search experiments finished! ---"