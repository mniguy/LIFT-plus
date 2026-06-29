#!/bin/bash
#
# #1: per-class learnable temperature (CosineClassifierPCT) vs plain LIFT+.
# NO text-prior anything (semantic init, no KD/InfoNCE/gate) -- isolates the
# classifier-geometry lever that our scale=25->30 finding pointed to.
# MULTI-SEED (the gain must clear the ~0.8 noise band to count).
#
#   lift     : LIFT+ baseline      (CosineClassifier, single global scale)
#   lift_pct : LIFT+ + per-class learnable scale (CosineClassifierPCT)
#
# Compare:
#   python scripts/analyze_method_ablation.py --root output/pct/<ds> --baseline lift --target lift_pct
#   python scripts/summarize_splits.py        --root output/pct/<ds> --seed 0 --baseline lift --order lift lift_pct

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"0 1 2"}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
RUNGS=${RUNGS:-"lift lift_pct"}

rung_args () {
    case "$1" in
        lift)     echo "classifier CosineClassifier" ;;
        lift_pct) echo "classifier CosineClassifierPCT" ;;
        *)        return 1 ;;
    esac
}

for data in ${DATASETS}; do
  for rung in ${RUNGS}; do
    args=$(rung_args "${rung}") || { echo "unknown rung: ${rung}"; exit 1; }
    for s in ${SEEDS}; do
      echo "=== [${data}] ${rung} seed=${s} ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ tte True PEFT_WARMUP False seed "${s}" \
        classifier_init semantic TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0 PRIOR_REG_MODE fixed \
        ${args} \
        output_dir "pct/${data}/${rung}_seed${s}"
    done
  done
done

echo ""
echo "=== verdict (paired Δ, noise-aware): ==="
for data in ${DATASETS}; do
  echo "  ${PYTHON} scripts/analyze_method_ablation.py --root output/pct/${data} --baseline lift --target lift_pct"
done
