#!/bin/bash
#
# H_D axis + goal(a): stack the TWO tail-margin levers.
#   centering   -> enlarges the tail margin GEOMETRICALLY (orthogonal targets; H_D confirmed).
#   group-scale -> enlarges the tail margin via a larger per-class SCALE (smaller required margin
#                  vs the LA freq penalty). Rarest -> S_TAIL, most-frequent -> S_HEAD.
# Both act on the same axis (H_D). Question: do they STACK (extra Few) and can tuning S_HEAD
# lift/protect Head (goal a)?
#
# References (already run):
#   uniform center (loss_robust25 LA_center):  IN 78.51/81.0/77.5/75.16 , PL 52.41/51.3/52.8/53.70
#   group-scale h25/t30 (no center):           IN 78.42/81.04/77.4/74.60, PL 52.14/51.4/52.6/52.28
#   uniform baseline:                          IN 78.30/81.06/77.5/73.46, PL 52.21/51.7/53.0/51.45
#
#   PAIRS are S_HEAD_S_TAIL. 25_30 = tail lever (head at baseline scale); 30_35 = head also sharpened.
#   bash scripts/run_group_center.sh
#   python scripts/agg_runs.py output/group_center25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
PAIRS=${PAIRS:-"25_30 25_35 30_35"}     # S_HEAD_S_TAIL
CENTER=${CENTER:-"True"}                 # set False to get the no-center group-scale sweep
OUT_ROOT=${OUT_ROOT:-"group_center25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

center_args(){ [ "${CENTER}" = "True" ] && echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" || echo "PROMPT_CENTER False"; }
completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for p in ${PAIRS}; do
    sh=${p%_*}; st=${p#*_}
    tag="h${sh}_t${st}"; [ "${CENTER}" = "True" ] && tag="${tag}_center"
    out="${OUT_ROOT}/${data}/${tag}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] group-scale head=${sh} tail=${st} center=${CENTER} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      classifier CosineClassifierGroupScale classifier_init semantic classifier_scale 25 \
      TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0 PRIOR_REG_MODE fixed \
      mda True tte True num_epochs 5 PEFT_WARMUP False \
      GROUP_SCALE_HEAD "${sh}" GROUP_SCALE_TAIL "${st}" $(center_args) seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
