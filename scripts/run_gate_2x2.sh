#!/bin/bash
#
# 2x2 grid on seeds 8,9 x {imagenet_lt, places_lt} = 16 runs.
#
# Fixed method : hybrid caption init + KD 0.001 (loss1) + InfoNCE 0.005 (loss2)
#                + freq_inv gating (class_gate, source=frequency, rank, INVERT=True),
#                TTE, MDA, 5 epochs.
# Swept 2x2    : PEFT image-warmup {on(ep1,lr5e-4) | off}  x  classifier_scale {25 | 30}.
#
#   bash scripts/run_gate_2x2.sh
#   DATASETS=imagenet_lt SEEDS=8 bash scripts/run_gate_2x2.sh
#   python scripts/agg_gate_2x2.py            # aggregate when done
set -euo pipefail

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"8 9"}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
WARMUPS=${WARMUPS:-"on off"}
SCALES=${SCALES:-"25 30"}
OUT_ROOT=${OUT_ROOT:-"gate_2x2"}

[ -f main.py ] || { echo "ERROR: run from the repo root (main.py not found)"; exit 1; }

# fixed base: hybrid + KD + InfoNCE + freq_inv gate
BASE_ARGS=(
  classifier_init hybrid
  TEXT_REG_LAMBDA 0.001 INFONCE_LAMBDA 0.005
  HYBRID_CAPTION_SOURCE wiki HYBRID_TOPK 8 SIM_THRESHOLD 0.6
  mda True tte True num_epochs 5
  PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE frequency
  PRIOR_GATE_NORM rank PRIOR_GATE_INVERT True PRIOR_GATE_POWER 1.0
)

completed () { grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for s in ${SEEDS}; do
    for wu in ${WARMUPS}; do
      for sc in ${SCALES}; do
        if [ "${wu}" = "on" ]; then
          wu_args=(PEFT_WARMUP True PEFT_WARMUP_EPOCHS 1 PEFT_WARMUP_LR 5e-4 PEFT_WARMUP_IMAGE True)
          wu_tag=warmup
        else
          wu_args=(PEFT_WARMUP False)
          wu_tag=nowarmup
        fi
        out="${OUT_ROOT}/${data}/${wu_tag}_scale${sc}_seed${s}"
        if completed "${out}"; then echo "  [skip] ${out}"; continue; fi
        echo "=== [${data}] seed=${s} warmup=${wu} scale=${sc} ==="
        CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
          -d "${data}" -b clip_vit_b16 -m lift+ \
          "${BASE_ARGS[@]}" classifier_scale "${sc}" "${wu_args[@]}" seed "${s}" \
          output_dir "${out}"
      done
    done
  done
done

echo ""
echo "=== aggregate ==="
echo "  ${PYTHON} scripts/agg_gate_2x2.py --root output/${OUT_ROOT} --datasets ${DATASETS}"
