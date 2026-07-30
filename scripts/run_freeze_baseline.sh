#!/bin/bash
#
# Experiment C -- the causal test of init-persistence, by INTERVENTION.
# Correlational analysis (per-class Delta-acc vs weight-drift, controlling frequency) does
# NOT support "frozen tail -> centering helps": frequency dominates, drift's coefficient is
# small and wrong-signed. But correlation cannot separate drift from frequency (r=0.82).
#
# Here we manipulate init-persistence directly: FREEZE_CLASSIFIER=True keeps the classifier
# pinned at its (centered or raw) init, so it CANNOT fine-tune away from the init -- holding
# dataset, frequency, headroom, #classes, epochs all fixed. Only PEFT (image encoder) trains.
#
#   Predictions if init-persistence IS causal:
#     - iNat (frozen):                    centering HELPS (Delta > 0)   [vs neutral when trainable]
#     - ImageNet-LT / Places-LT (frozen): centering helps ALL groups incl. HEAD
#                                         (head Delta flips + on BOTH)  [vs head Delta ~ 0 when trainable]
#   Two "helps" datasets (ImageNet-LT, Places-LT) make the head-flips-positive prediction robust.
#   If iNat-frozen stays neutral / head-frozen stays flat -> init-persistence is NOT the
#   mechanism; the frequency(+headroom) account stands.
#
#   NOTE: overall accuracy drops under a frozen classifier (weaker config). The test is the
#   center-vs-baseline gap WITHIN the frozen setting, and how it differs from the trainable
#   setting (breadth25 / prompt_center25), NOT the absolute number.
#
#   bash scripts/run_freeze_center.sh
#   python scripts/agg_runs.py output/freeze_center25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt inat2018"}
VARIANTS=${VARIANTS:-"baseline"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}              # default (ImageNet-LT / Places-LT)
INAT_EPOCHS=${INAT_EPOCHS:-15}  # iNat native protocol
OUT_ROOT=${OUT_ROOT:-"freeze_center25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}" FREEZE_CLASSIFIER True
  mda True tte True
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
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
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
