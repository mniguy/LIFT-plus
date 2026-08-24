#!/bin/bash

set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
LEVEL=${LEVEL:-genus}
LV3=${LV3:-"genus,family,order"}                       # cascade's own levels
LV6=${LV6:-"genus,family,order,class,phylum,kingdom"}  # the entire taxonomy
ANCHORS=${ANCHORS:-0}          # 1 = also re-run cascade/global (needed if SEED/EPOCHS differ)
ARMS=${ARMS:-"w_one shrink_group shrink_level shrink_lv3 shrink_lv6 cliff_group"}
OUT_ROOT=${OUT_ROOT:-"center_cohesion25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
  PROMPT_CENTER True
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

run(){   # run <name> <extra cfg opts...>
  local data="$1"; local name="$2"; shift 2
  local out="${OUT_ROOT}/${data}/${name}"
  completed "${out}" && { echo "  [skip] ${out}"; return 0; }
  echo "=== [${data}] ${name} (${ep} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
    "${COMMON_ARGS[@]}" num_epochs "${ep}" "$@" \
    seed "${SEED}" output_dir "${out}"
}

# arm -> the cfg opts that follow "PROMPT_CENTER_MODE cohesion"
arm_spec(){ case "$1" in
  w_one)         echo "PROMPT_CENTER_COHESION_LEVELS ${LEVEL} PROMPT_CENTER_COHESION_W one" ;;
  shrink_group)  echo "PROMPT_CENTER_COHESION_LEVELS ${LEVEL} PROMPT_CENTER_COHESION_W shrink PROMPT_CENTER_COHESION_RHO group" ;;
  shrink_level)  echo "PROMPT_CENTER_COHESION_LEVELS ${LEVEL} PROMPT_CENTER_COHESION_W shrink PROMPT_CENTER_COHESION_RHO level" ;;
  shrink_lv3)    echo "PROMPT_CENTER_COHESION_LEVELS ${LV3} PROMPT_CENTER_COHESION_W shrink PROMPT_CENTER_COHESION_RHO group" ;;
  shrink_lv6)    echo "PROMPT_CENTER_COHESION_LEVELS ${LV6} PROMPT_CENTER_COHESION_W shrink PROMPT_CENTER_COHESION_RHO group" ;;
  cliff_group)   echo "PROMPT_CENTER_COHESION_LEVELS ${LEVEL} PROMPT_CENTER_COHESION_W cliff PROMPT_CENTER_COHESION_RHO group" ;;
  *) return 1 ;; esac; }

for data in ${DATASETS}; do
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi

  if [ "${ANCHORS}" != "0" ]; then
    run "${data}" cascade PROMPT_CENTER_MODE cascade
    run "${data}" global  PROMPT_CENTER_MODE global
  fi
  for arm in ${ARMS}; do
    spec=$(arm_spec "${arm}") || { echo "unknown arm ${arm}"; exit 1; }
    run "${data}" "${arm}" PROMPT_CENTER_MODE cohesion ${spec}
  done
done

echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    check each cohesion log's '[PROMPT_CENTER cohesion] ... w>0 for N/C classes' line FIRST:"
echo "    N should be 5142/8142 on iNat at level=genus. A much smaller N means the weight"
echo "    collapsed and the run is really just global centering wearing a different name."
echo "    then compare Few against the existing anchors at the SAME config:"
echo "      cascade 82.50  output/center_local25/inat2018/cascade"
echo "      global  82.13  output/breadth25/inat2018/center_15ep"
echo "    does w_one beat cascade? that is the coverage claim (28.0% -> 63.2% of classes)."
