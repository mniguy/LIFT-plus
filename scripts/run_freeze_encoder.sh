#!/bin/bash
#
# H_E test -- is centering's benefit on the CLASSIFIER/target side, or does it need the
# ENCODER to learn better features? Inverse of run_freeze_center.sh: here we FREEZE the
# PEFT/encoder and train ONLY the classifier, from raw vs centered init.
#
#   FREEZE_ENCODER=True: image features are fixed (~raw CLIP); only the classifier moves.
#
#   Prediction:
#     H_A (target geometry, classifier-side): centering STILL helps Few even with a frozen encoder
#         (the tail classifier can't self-correct -> centered init persists).
#     H_E (feature quality, encoder-side): centering's gain VANISHES/hurts when the encoder can't
#         adapt (like zero-shot -23) -> the benefit needed encoder co-adaptation.
#   Read alongside FREEZE_CLASSIFIER (encoder-only trains, center +11.6) and trainable (both, +1.6).
#
# ALSO includes the head/tail COUPLING test (fewonly + frozen encoder): trainable fewonly drops
# Head ~0.4 even though Head's init is untouched -> hypothesis: tail centering reshapes the SHARED
# encoder, which propagates to Head. With the encoder FROZEN, tail centering can't touch the
# features, so:
#   Head(fewonly) ~= Head(baseline)  -> coupling was ENCODER-mediated (confirmed)
#   Head(fewonly) <  Head(baseline)  -> classifier/softmax-side coupling instead
# The targeted run is ImageNet:  VARIANTS="baseline fewonly" DATASETS="imagenet_lt" bash scripts/run_freeze_encoder.sh
#
#   bash scripts/run_freeze_encoder.sh
#   python scripts/agg_runs.py output/freeze_encoder25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt inat2018"}
VARIANTS=${VARIANTS:-"baseline center fewonly"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"freeze_encoder25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}" FREEZE_ENCODER True
  mda True tte True
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  fewonly)  echo "PROMPT_CENTER True PROMPT_CENTER_MODE fewonly" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "$data" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] freeze-encoder ${v} (scale ${SCALE}, ${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
