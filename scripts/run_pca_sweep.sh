#!/bin/bash
#
# How MUCH de-anisotropization is optimal? Sweep top-k principal-component removal
# (mode=pca, All-but-the-Top style). k=0 == global (mean only); large k -> whiten-like
# collapse. Traces the U-curve that explains why `global` wins and `whiten` fails, and
# gives the paper's quantitative design law (R). Semantic init, aux OFF, scale 25.
#
#   k=0  : mean-center only            (== center_geom25/global)
#   k=1+ : also remove top-k PCs        (k~1 ~ removing the dominant shared direction)
#   k>>  : approaches whiten (collapse)
#
#   bash scripts/run_pca_sweep.sh
#   KS="0 1 2 5 10 20 50" bash scripts/run_pca_sweep.sh
#   python scripts/agg_runs.py output/pca_sweep25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
KS=${KS:-"0 1 2 5 10 20"}
OUT_ROOT=${OUT_ROOT:-"pca_sweep25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale 25 PROMPT_CENTER True PROMPT_CENTER_MODE pca
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True num_epochs 5 PEFT_WARMUP False
)
completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for k in ${KS}; do
    out="${OUT_ROOT}/${data}/k${k}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] pca k=${k} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" PROMPT_CENTER_PCA_K "${k}" seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
