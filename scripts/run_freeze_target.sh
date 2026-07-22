#!/bin/bash
#
# H_A target-quality diagnostic: under a FROZEN classifier (only PEFT/image encoder trains),
# which INIT TARGET is best? This isolates "how good is the target we hand the frozen tail"
# from any training-time self-correction.
#
#   baseline : semantic (raw CLIP text prototypes)                 -> the anisotropic target
#   center   : semantic + PROMPT_CENTER global (mu removed)        -> our current method
#   imgmean  : class_mean (per-class MEAN of CLIP image features)  -> data-aligned target
#
# H_A says the tail needs a well-separated target it can't self-reach. centering de-anisotropizes
# the TEXT target; imgmean instead points the target at where the class's images actually cluster.
# If frozen imgmean > frozen center (esp. iNat, where text targets are near-duplicate), then a
# BETTER target than centering exists -> motivates the image-mean + shrinkage blend (which
# subsumes centering at the extreme tail). If imgmean <= center, centering's text target is
# already near-optimal for the frozen regime.
#
# Lands imgmean next to the existing freeze_center25/*/{baseline,center} so agg_runs compares all.
#   VARIANTS="baseline center imgmean" bash scripts/run_freeze_target.sh   # full matched set
#   bash scripts/run_freeze_target.sh                                      # imgmean only (reuse existing)
#   python scripts/agg_runs.py output/freeze_center25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"imgmean"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}              # ImageNet-LT / Places-LT
INAT_EPOCHS=${INAT_EPOCHS:-15}  # iNat native protocol
OUT_ROOT=${OUT_ROOT:-"freeze_center25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_scale "${SCALE}" FREEZE_CLASSIFIER True
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True PEFT_WARMUP False
)
variant_args(){ case "$1" in
  baseline) echo "classifier_init semantic PROMPT_CENTER False" ;;
  center)   echo "classifier_init semantic PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  imgmean)  echo "classifier_init class_mean PROMPT_CENTER False" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "$data" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] freeze-clf ${v} (scale ${SCALE}, ${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "=== target quality: ${PYTHON} scripts/analyze_anisotropy.py output/${OUT_ROOT}/inat2018/imgmean ==="
