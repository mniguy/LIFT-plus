#!/bin/bash
#
# MUST-fix #2: is the centering gain an artifact of the augmentation pipeline?
#
# Every number in the paper is measured with mda=True AND tte=True. Those are the two pipeline
# components the method never varies, so the most obvious reviewer question is unanswered:
#   "TTE is FiveCrop test-time ensembling. Is the +1.7 Few gain an interaction with that, rather
#    than a property of the initialization?"
# MDA (progressive crop-scale schedule) is the training-side counterpart of the same worry.
#
# 2x2x2 design: {baseline, center} x {mda on/off} x {tte on/off}, on ImageNet-LT and Places-LT.
# The mda=True,tte=True cells already exist (seed_ablation 25 / prompt_center25, identical
# settings) and are skipped here; they are printed by the analyzer for reference.
#
#   Reference for the mda+tte cell (seed 0):
#     IN  baseline 78.28 / 81.03 / 77.43 / 73.49    center 78.51 / 81.01 / 77.46 / 75.12  (+1.63 Few)
#     PL  baseline 52.17 / 51.67 / 52.93 / 51.37    center 52.32 / 51.23 / 52.64 / 53.58  (+2.21 Few)
#
#   WHAT CLOSES THE CONFOUND: the Few gain must survive tte=False. It does not need to be the same
#   size -- absolute accuracy drops without test-time ensembling -- but the SIGN and rough
#   magnitude must hold. Judge against the 5-seed Few sigma = 0.32 (diff sigma = 0.45).
#   If the gain collapses at tte=False, the paper's claim has to be restated as being about the
#   full LIFT+ pipeline including TTE, which would be a materially weaker contribution.
#
#   Cost: 12 new runs (4 of 16 cells already exist). IN 6x15min + PL 6x5min = 2.0 h.
#   Cheapest useful subset -- just the TTE question, 4 runs / 40 min:
#     AUGS="mdaT_tteF" bash scripts/run_aug_confound.sh
#
#   bash scripts/run_aug_confound.sh
#   python scripts/agg_runs.py output/aug_confound25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"baseline center"}
AUGS=${AUGS:-"mdaT_tteF mdaF_tteT mdaF_tteF"}   # mdaT_tteT already exists elsewhere
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"aug_confound25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale 25
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  num_epochs "${EPOCHS}" PEFT_WARMUP False
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

aug_args(){ case "$1" in
  mdaT_tteT) echo "mda True  tte True"  ;;
  mdaT_tteF) echo "mda True  tte False" ;;
  mdaF_tteT) echo "mda False tte True"  ;;
  mdaF_tteF) echo "mda False tte False" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for a in ${AUGS}; do
    aa=$(aug_args "$a") || { echo "unknown aug setting $a"; exit 1; }
    for v in ${VARIANTS}; do
      va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
      out="${OUT_ROOT}/${data}/${a}_${v}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] ${a} ${v} ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
        "${BASE_ARGS[@]}" ${aa} ${va} seed "${SEED}" output_dir "${out}"
    done
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    the cell that matters is mdaT_tteF: does the Few gain survive without TTE?"
