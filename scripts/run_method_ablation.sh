#!/bin/bash
#
# Does the method survive WITHOUT gating? Controlled ablation ladder vs plain LIFT+.
#
# All rungs are IDENTICAL except classifier init + KD/InfoNCE (NO gating anywhere,
# PRIOR_REG_MODE fixed). MULTI-SEED on purpose: the run-noise band is ~0.8, so a
# single seed cannot tell a real gain from noise. Paired (same-seed) comparison.
#
#   lift+      : LIFT+ baseline   (semantic init, no KD, no InfoNCE)
#   hybrid     : + hybrid caption init
#   hybrid_kd  : + KD (text-prior logit reg)
#   full       : + InfoNCE        (= hybrid + KD + InfoNCE = the method minus gating)
#
# Fastest verdict = just the two endpoints:  RUNGS="lift+ full"
# Verdict computed by scripts/analyze_method_ablation.py (paired Δ full - lift+).

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"0 1 2"}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
LAMBDA_KD=${LAMBDA_KD:-0.001}
LAMBDA_NCE=${LAMBDA_NCE:-0.005}
RUNGS=${RUNGS:-"lift+ hybrid hybrid_kd full"}

rung_args () {
    case "$1" in
        lift+)     echo "classifier_init semantic TEXT_REG_LAMBDA 0.0       INFONCE_LAMBDA 0.0" ;;
        hybrid)    echo "classifier_init hybrid   TEXT_REG_LAMBDA 0.0       INFONCE_LAMBDA 0.0" ;;
        hybrid_kd) echo "classifier_init hybrid   TEXT_REG_LAMBDA ${LAMBDA_KD} INFONCE_LAMBDA 0.0" ;;
        full)      echo "classifier_init hybrid   TEXT_REG_LAMBDA ${LAMBDA_KD} INFONCE_LAMBDA ${LAMBDA_NCE}" ;;
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
        ${args} PRIOR_REG_MODE fixed \
        output_dir "method_ablation/${data}/${rung}_seed${s}"
    done
  done
done

echo ""
echo "=== analyze (per dataset): ==="
for data in ${DATASETS}; do
  echo "  ${PYTHON} scripts/analyze_method_ablation.py --root output/method_ablation/${data} --baseline lift+ --target full"
done
