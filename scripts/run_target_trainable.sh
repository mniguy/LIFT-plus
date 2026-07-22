#!/bin/bash
#
# TRAINABLE target comparison -- does a data-aligned target beat centered-text once the
# classifier can also train? (frozen diagnostic freeze_target25 showed image-mean BEATS center
# on Places but loses on ImageNet, with a Head cost from raw-CLIP staleness. Trainable should
# let the classifier chase the drifting features -> the staleness Head cost may vanish.)
#
#   baseline : semantic (raw text)
#   center   : semantic + PROMPT_CENTER global (current method)
#   imgmean  : class_mean (per-class image-feature mean)
#   blend    : img_shrink (count-adaptive imagemean<->centered-text, IMG_SHRINK_KAPPA)
#
# NOTE the frozen finding: count-based lambda looked BACKWARDS on Places (imgmean helped Few,
# hurt Head, opposite to head->image / tail->text). So read the blend critically; if imgmean
# wins but blend doesn't, the lambda axis (count) is wrong and should be redesigned.
#
#   bash scripts/run_target_trainable.sh
#   python scripts/agg_runs.py output/target_trainable25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"baseline center imgmean blend"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
KAPPA=${KAPPA:-20}
OUT_ROOT=${OUT_ROOT:-"target_trainable25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_scale "${SCALE}"
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True PEFT_WARMUP False
)
variant_args(){ case "$1" in
  baseline) echo "classifier_init semantic PROMPT_CENTER False" ;;
  center)   echo "classifier_init semantic PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  imgmean)  echo "classifier_init class_mean PROMPT_CENTER False" ;;
  blend)    echo "classifier_init img_shrink PROMPT_CENTER False IMG_SHRINK_KAPPA ${KAPPA}" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "$data" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] trainable target=${v} (scale ${SCALE}, ${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
