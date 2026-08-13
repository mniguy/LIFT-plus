#!/bin/bash

set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
CODES=${CODES:-"51 3 5 024"}
MIN_SIZE=${MIN_SIZE:-5}
RENORM=${RENORM:-False}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_final25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# digit -> level name
digit_level(){ case "$1" in
  0) echo "global" ;; 1) echo "kingdom" ;; 2) echo "phylum" ;; 3) echo "class" ;;
  4) echo "order"  ;; 5) echo "family"  ;; 6) echo "genus"  ;;
  *) return 1 ;; esac; }

# "135" -> "kingdom,class,family"
code_to_levels(){
  local code="$1" out="" d lv
  while [ -n "${code}" ]; do
    d="${code:0:1}"; code="${code:1}"
    lv=$(digit_level "${d}") || { echo "bad digit '${d}' in code '$1'" >&2; return 1; }
    out="${out}${out:+,}${lv}"
  done
  echo "${out}"
}

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for code in ${CODES}; do
  lv=$(code_to_levels "${code}") || exit 1
  out="${OUT_ROOT}/inat2018/c${code}"
  completed "${out}" && { echo "  [skip] ${out} (${lv})"; continue; }
  echo "=== [inat2018] code ${code} -> ${lv}  (min_size=${MIN_SIZE} renorm=${RENORM}, ${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE nested \
    PROMPT_CENTER_NESTED_LEVELS "${lv}" PROMPT_CENTER_NESTED_MEAN recompute \
    PROMPT_CENTER_NESTED_RENORM "${RENORM}" PROMPT_CENTER_GENUS_MIN "${MIN_SIZE}" \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    output dirs are named c<code>, e.g. c135 = kingdom,class,family."
echo "    check each log's '[PROMPT_CENTER nested] ... -> lvl(n=..,|mu|=..)' line against the header"
echo "    table before trusting a run; a trailing level with |mu| near 0 contributed nothing."
echo "    Read as a TREND across depth and ordering. Single-arm gaps under ~0.3 Overall are within"
echo "    the run-to-run variation already demonstrated on this dataset."
