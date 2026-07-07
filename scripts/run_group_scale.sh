#!/bin/bash
#
# #1: per-frequency-group FIXED cosine scale. Tests the analysis that tail classes
# need a larger effective scale (smaller required margin vs the LA freq penalty).
# Clean isolation: plain LIFT+ (semantic init, no KD/InfoNCE, no warmup), TTE.
# head=25 (softer for the abundant head classes), tail=30 (sharper -> smaller required
# margin for rare classes). One config per dataset. Override with S_HEAD / TAILS.
#
#   bash scripts/run_group_scale.sh
#   python scripts/agg_runs.py output/group_scale --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
S_HEAD=${S_HEAD:-25}
TAILS=${TAILS:-"30"}
OUT_ROOT=${OUT_ROOT:-"group_scale"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for st in ${TAILS}; do
    out="${OUT_ROOT}/${data}/head${S_HEAD}_tail${st}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] group-scale head=${S_HEAD} tail=${st} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      classifier CosineClassifierGroupScale classifier_init semantic \
      TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0 PRIOR_REG_MODE fixed \
      mda True tte True num_epochs 5 PEFT_WARMUP False \
      GROUP_SCALE_HEAD "${S_HEAD}" GROUP_SCALE_TAIL "${st}" seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
