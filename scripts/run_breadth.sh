#!/bin/bash
#
# Dataset breadth for prototype centering (I = global). Shows the training-free centering
# generalizes beyond ImageNet-LT / Places-LT. Semantic init + aux OFF, scale 25, matched
# baseline vs center per dataset. iNat is the key case: its wiki captions are broken, but
# semantic (prompt) prototypes are fine, so centering still applies.
#
# Per-dataset epochs (match each dataset's native protocol): iNat2018 = 15 (LIFT+ trains
# iNat for 15 epochs at scale 25), CIFAR-100-LT = 5. iNat outputs go to <data>/<variant>_<ep>ep
# so a 15-ep run does not collide with / silently skip an earlier 5-ep run.
#
#   ImageNet-LT / Places-LT already done: baseline = "seed_ablation 25", center = prompt_center25.
#
#   bash scripts/run_breadth.sh
#   DATASETS=inat2018 bash scripts/run_breadth.sh          # iNat only, 15 ep
#   INAT_EPOCHS=15 EPOCHS=5 bash scripts/run_breadth.sh    # override epochs
#   python scripts/agg_runs.py output/breadth25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"cifar100_ir100 cifar100_ir50 cifar100_ir10 inat2018"}
VARIANTS=${VARIANTS:-"baseline center"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}              # default (CIFAR)
INAT_EPOCHS=${INAT_EPOCHS:-15}  # iNat native protocol (LIFT+)
OUT_ROOT=${OUT_ROOT:-"breadth25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}"
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
    sub="${v}"
    if [ "$data" = "inat2018" ]; then sub="${v}_${ep}ep"; fi   # keep 5-ep and 15-ep iNat separate
    out="${OUT_ROOT}/${data}/${sub}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] breadth ${v} (scale ${SCALE}, ${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
