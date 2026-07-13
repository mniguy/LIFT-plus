#!/bin/bash
#
# Experiment 1 -- the PUREST init test: zero-shot centering, NO training at all.
#
# freeze_center25 froze the classifier but still trained the PEFT image encoder.
# Here we remove even that: method `zs` builds raw CLIP (no PEFT, no classifier) and
# classifies test images by cosine to the text prototypes. We center those prototypes
# (subtract mu) BEFORE the cosine step -- centered-vs-raw prototypes is the ONLY
# difference, with zero training dynamics of any kind.
#
#   Claim tested: the centered init geometry itself classifies the tail better.
#   If Few rises here (no training), the "better init" claim holds in its strongest form.
#   whiten is included as the design-law contrast: it should NOT help (isotropy != the fix).
#
#   Requires the 2-line addition in trainer.test() (zero-shot PROMPT_CENTER path).
#   Cost: ~one test pass per run (no training) -> cheap.
#
#   bash scripts/run_zeroshot_center.sh
#   python scripts/agg_runs.py output/zeroshot_center25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"imagenet_lt places_lt inat2018"}
VARIANTS=${VARIANTS:-"raw center whiten"}
OUT_ROOT=${OUT_ROOT:-"zeroshot_center25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

variant_args(){ case "$1" in
  raw)    echo "PROMPT_CENTER False" ;;
  center) echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  whiten) echo "PROMPT_CENTER True PROMPT_CENTER_MODE whiten" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] zero-shot ${v} (no training) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m zs \
      ${va} output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
