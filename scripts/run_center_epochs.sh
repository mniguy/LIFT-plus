#!/bin/bash
#
# Q8 -- epoch axis vs centering. The real overfit lever in this codebase is EPOCHS
# (more epochs -> Head up, Few down; see lr_epoch_sweep), NOT centering (which reduces
# head drift, i.e. fits head LESS). This runs baseline vs center across a few epoch
# budgets so we can see (a) whether the head/med "cost" of longer training is real, and
# (b) whether fewer epochs + centering COMPOUNDS the Few gain.
#
# Everything except num_epochs is identical to the headline config (semantic, aux OFF,
# scale 25, mda+tte) so the only moving parts are {epochs} x {center on/off}.
# iNat is excluded (15-ep native protocol, different regime) -- add it separately if needed.
#
#   bash scripts/run_center_epochs.sh
#   EPOCHS="3 4 5 6 8" bash scripts/run_center_epochs.sh
#   python scripts/agg_runs.py output/center_epochs25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
EPOCHS=${EPOCHS:-"4 6"}
VARIANTS=${VARIANTS:-"baseline center"}
SCALE=${SCALE:-25}
OUT_ROOT=${OUT_ROOT:-"center_epochs25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}"
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True PEFT_WARMUP False
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for ep in ${EPOCHS}; do
    for v in ${VARIANTS}; do
      va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
      out="${OUT_ROOT}/${data}/${v}_ep${ep}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] ${v} ep=${ep} (scale ${SCALE}) ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ \
        "${BASE_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" \
        output_dir "${out}"
    done
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
