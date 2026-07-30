#!/bin/bash
#
# MUST-fix #3: does the gain depend on the logit scale we happened to pick?
#
# The paper's quantitative mechanism is about DECISION MARGINS: removing the shared direction
# raises rho from ~0.54 to 1.0, which widens inter-class logit gaps by 1/rho (1.6x on ImageNet-LT,
# 1.95x on Places-LT, measured). The cosine classifier then multiplies every logit by
# classifier_scale. So scale sits directly on top of the quantity the mechanism is about, and
# every reported number uses a single value (25) -- which earlier runs in this project did not
# (they used 30). A reviewer will ask whether 25 is where the effect happens to live.
#
#   scale 25 is included on purpose as a REPRODUCTION CHECK: it should land on the known
#   seed-0 numbers, confirming this sweep's pipeline matches the main table.
#     IN  baseline 78.28 / 81.03 / 77.43 / 73.49    center 78.51 / 81.01 / 77.46 / 75.12
#     PL  baseline 52.17 / 51.67 / 52.93 / 51.37    center 52.32 / 51.23 / 52.64 / 53.58
#
#   WHAT CLOSES IT: Delta-Few > 0 at every scale, with no scale where the sign flips. The size may
#   vary -- a very small scale saturates the softmax and a very large one sharpens it, so both ends
#   should compress the gain somewhat. A single-peaked curve centred exactly on 25 would be bad
#   news and must be reported honestly if that is what appears.
#   Judge against the 5-seed Few sigma = 0.32 (diff sigma = 0.45).
#
#   Cost: 16 runs. IN 8x15min + PL 8x5min = 2.7 h.
#   Drop the reproduction check to save 40 min:  SCALES="15 30 40" bash scripts/run_scale_sweep.sh
#
#   bash scripts/run_scale_sweep.sh
#   python scripts/agg_runs.py output/scale_sweep25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"baseline center"}
SCALES=${SCALES:-"15 25 40"}
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"scale_sweep25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic
  mda True tte True num_epochs "${EPOCHS}"
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for s in ${SCALES}; do
    for v in ${VARIANTS}; do
      va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
      out="${OUT_ROOT}/${data}/scale${s}_${v}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] scale ${s} ${v} ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
        "${BASE_ARGS[@]}" classifier_scale "${s}" ${va} seed "${SEED}" output_dir "${out}"
    done
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    required: Delta-Few > 0 at EVERY scale. scale25 should reproduce the main-table numbers."
