#!/bin/bash
#
# PERFORMANCE VERDICT: does the full method beat plain LIFT+ across seeds?
#
# Controlled ablation ladder, all IDENTICAL except init + KD/InfoNCE (+ gating).
# MULTI-SEED on purpose: the run-noise band is ~0.8, so a single seed cannot tell
# a real gain from noise. Paired (same-seed) comparison; focus on the Few split.
#
#   lift+      : LIFT+ baseline   (semantic init, no KD, no InfoNCE, no gate)
#   hybrid     : + hybrid caption init
#   hybrid_kd  : + KD (text-prior logit reg)
#   full       : + InfoNCE        (hybrid + KD + InfoNCE, no gate)
#   full_gate  : + agreement gating (= the complete proposed method)
#
# Fastest verdict = endpoints only:  RUNGS="lift+ full_gate"
# Verdict computed by scripts/analyze_method_ablation.py (paired Δ vs lift+).

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"0 1 2"}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
LAMBDA_KD=${LAMBDA_KD:-0.001}
LAMBDA_NCE=${LAMBDA_NCE:-0.005}
RUNGS=${RUNGS:-"lift+ hybrid hybrid_kd full full_gate"}

rung_args () {
    case "$1" in
        lift+)     echo "classifier_init semantic TEXT_REG_LAMBDA 0.0        INFONCE_LAMBDA 0.0        PRIOR_REG_MODE fixed" ;;
        hybrid)    echo "classifier_init hybrid   TEXT_REG_LAMBDA 0.0        INFONCE_LAMBDA 0.0        PRIOR_REG_MODE fixed" ;;
        hybrid_kd) echo "classifier_init hybrid   TEXT_REG_LAMBDA ${LAMBDA_KD} INFONCE_LAMBDA 0.0        PRIOR_REG_MODE fixed" ;;
        full)      echo "classifier_init hybrid   TEXT_REG_LAMBDA ${LAMBDA_KD} INFONCE_LAMBDA ${LAMBDA_NCE} PRIOR_REG_MODE fixed" ;;
        full_gate) echo "classifier_init hybrid   TEXT_REG_LAMBDA ${LAMBDA_KD} INFONCE_LAMBDA ${LAMBDA_NCE} PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE image_text PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT False" ;;
        *)         return 1 ;;
    esac
}

for data in ${DATASETS}; do
  for rung in ${RUNGS}; do
    args=$(rung_args "${rung}") || { echo "unknown rung: ${rung}"; exit 1; }
    for s in ${SEEDS}; do
      echo "=== [${data}] ${rung} seed=${s} ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ tte True PEFT_WARMUP False seed "${s}" \
        ${args} \
        output_dir "method_ablation/${data}/${rung}_seed${s}"
    done
  done
done

echo ""
echo "=== analyze (per dataset): ==="
for data in ${DATASETS}; do
  echo "  ${PYTHON} scripts/analyze_method_ablation.py --root output/method_ablation/${data} --baseline lift+ --target full_gate"
  echo "  ${PYTHON} scripts/analyze_method_ablation.py --root output/method_ablation/${data} --baseline lift+ --target full"
done
