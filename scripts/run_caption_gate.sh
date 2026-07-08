#!/bin/bash
#
# #2: the caption blend already IS an implicit agreement gate --
#   cap_w = clamp(cos(prompt, caption_centered), 0, 1)  (trainer.py)
# so classes whose caption residual disagrees with the prompt get cap_w=0 (no blend).
# Expose the gate FORM explicitly and ablate it, holding the winning geometry fixed
# (hybrid init + CAPTION_CENTER + convex, scale 25, KD/InfoNCE, TTE, no warmup).
#
#   soft : cap_w = clamp(cos,0,1)                 (continuous, = current caption_geom25/center)
#   hard : cap_w = 1 if cos > GATE_TAU else 0     (binary agreement gate)
#   freq : cap_w = clamp(cos,0,1) * rarity        (tail-scaled: rarest -> full, head -> off)
#
#   bash scripts/run_caption_gate.sh
#   GATE_TAU=0.15 VARIANTS=hard bash scripts/run_caption_gate.sh   # sweep the hard threshold
#   python scripts/agg_runs.py output/caption_gate25 --sort few
set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"soft hard freq"}
GATE_TAU=${GATE_TAU:-0.1}
OUT_ROOT=${OUT_ROOT:-"caption_gate25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# Winning caption geometry (= caption_geom25/center), only CAPTION_GATE varies.
BASE_ARGS=(
  classifier_init hybrid classifier_scale 25
  CAPTION_CENTER True CAPTION_BLEND convex CAPTION_SHRINK False
  TEXT_REG_LAMBDA 0.001 INFONCE_LAMBDA 0.005 PRIOR_REG_MODE fixed
  HYBRID_CAPTION_SOURCE wiki HYBRID_TOPK 8 SIM_THRESHOLD 0.6
  mda True tte True num_epochs 5 PEFT_WARMUP False
)
variant_args(){ case "$1" in
  soft) echo "CAPTION_GATE soft" ;;
  hard) echo "CAPTION_GATE hard CAPTION_GATE_TAU ${GATE_TAU}" ;;
  freq) echo "CAPTION_GATE freq" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    tag="$v"; [ "$v" = "hard" ] && tag="hard_tau${GATE_TAU}"
    out="${OUT_ROOT}/${data}/${tag}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] caption-gate ${tag} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort few ==="
