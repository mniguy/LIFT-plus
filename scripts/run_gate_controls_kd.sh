#!/bin/bash
#
# gate_controls on hybrid + KD ONLY (InfoNCE off). Isolates gating's effect on the
# KD (text-prior logit) term. Same gate variants as run_gate_controls.sh.
#
#   fixed / agreement / shuffled / freq / freq_inv
#
# At the default LAMBDA_KD=0.001 the KD term is weak (expect ~baseline, like
# gate_controls). To probe a KD-driven collapse + recovery, crank LAMBDA_KD
# (e.g. 0.05) and lower TEXT_REG_T (e.g. 0.001), analogous to the InfoNCE collapse.
#
# Output: output/gate_controls_kd/<ds>/<variant>/
# Compare: python scripts/summarize_splits.py --root output/gate_controls_kd/<ds> --baseline fixed

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
LAMBDA_KD=${LAMBDA_KD:-0.001}
KD_T=${KD_T:-0.01}                      # TEXT_REG_T; lower (e.g. 0.001) to sharpen / induce collapse

run_gate () {
    local data="$1" name="$2"; shift 2
    echo "=== [${data} KD-only kd=${LAMBDA_KD} T=${KD_T}] ${name} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ tte True PEFT_WARMUP False seed "${SEED}" \
        classifier_init hybrid TEXT_REG_LAMBDA "${LAMBDA_KD}" INFONCE_LAMBDA 0.0 TEXT_REG_T "${KD_T}" \
        "$@" \
        output_dir "gate_controls_kd/${data}/${name}"
}

for data in ${DATASETS}; do
    run_gate "${data}" fixed      PRIOR_REG_MODE fixed
    run_gate "${data}" agreement  PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE image_text PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT False
    run_gate "${data}" shuffled   PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE shuffled   PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT False
    run_gate "${data}" freq       PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE frequency  PRIOR_GATE_NORM rank   PRIOR_GATE_INVERT False
    run_gate "${data}" freq_inv   PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE frequency  PRIOR_GATE_NORM rank   PRIOR_GATE_INVERT True
done

echo ""
echo "=== compare: ==="
for data in ${DATASETS}; do
  echo "  ${PYTHON} scripts/summarize_splits.py --root output/gate_controls_kd/${data} --baseline fixed --order fixed agreement shuffled freq freq_inv"
done
