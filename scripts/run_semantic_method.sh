#!/bin/bash
#
# Is the wiki-caption (hybrid) init the problem? Keep KD + InfoNCE + gating, but
# use SEMANTIC init (plain CLIP text prototype). Note: init=semantic also makes
# the KD/InfoNCE target AND the gate use the semantic prototype (text_prior_weight),
# i.e. "plain CLIP text everywhere instead of wiki-caption blend".
#
#   sem_gate : semantic init + KD + InfoNCE + agreement gating  (the requested variant)
#   sem_full : semantic init + KD + InfoNCE, no gate            (to isolate gating)
#
# Written into output/method_ablation/<ds>/ so it sits next to lift+, full, full_gate.
# Compare (seed 0):
#   python scripts/summarize_splits.py --root output/method_ablation/<ds> --seed 0 \
#       --baseline lift+ --order lift+ full full_gate sem_full sem_gate

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"0"}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
LAMBDA_KD=${LAMBDA_KD:-0.001}
LAMBDA_NCE=${LAMBDA_NCE:-0.005}
RUNGS=${RUNGS:-"sem_gate sem_full"}

rung_args () {
    case "$1" in
        sem_gate) echo "classifier_init semantic TEXT_REG_LAMBDA ${LAMBDA_KD} INFONCE_LAMBDA ${LAMBDA_NCE} PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE image_text PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT False" ;;
        sem_full) echo "classifier_init semantic TEXT_REG_LAMBDA ${LAMBDA_KD} INFONCE_LAMBDA ${LAMBDA_NCE} PRIOR_REG_MODE fixed" ;;
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
        ${args} \
        output_dir "method_ablation/${data}/${rung}_seed${s}"
    done
  done
done

echo ""
echo "=== compare (seed 0): ==="
for data in ${DATASETS}; do
  echo "  ${PYTHON} scripts/summarize_splits.py --root output/method_ablation/${data} --seed 0 --baseline lift+ --order lift+ full full_gate sem_full sem_gate"
done
