#!/bin/bash
#
# DECISIVE TEST: does gating RECOVER the split collapse that STRONG uniform reg causes?
#
# At low lambda (KD 0.001 / NCE 0.005) the reg is inert -> gating vs no-gating is
# within noise (why gate_controls looked flat). The reg only HURTS splits at HIGH
# lambda. ImageNet InfoNCE evidence (gating off):
#     NCE=0.005 T=0.08 : Many 81.35  Med 77.45  Few 74.06   (fine)
#     NCE=0.05  T=0.001: Many 80.14  Med 76.46  Few 73.00   (collapse)
#     NCE=0.1   T=0.001: Many 79.11  Med 75.64  Few 70.69   (severe collapse)
#
# Here we run the COLLAPSE operating point with each gate and check who recovers:
#   fixed     : uniform reg            -> the collapse (baseline)
#   agreement : cos(image-mean,proto)  -> the proposed method
#   shuffled  : agreement values, wrong classes  (negative control)
#   freq      : gate by class count                (alt axis)
#   freq_inv  : gate by class count, tail stronger
#
# WIN condition for the method: agreement recovers Many/Med (and Few) that fixed
# lost, and does so MORE than shuffled (signal matters) and than freq* (not just
# frequency). If agreement<=fixed even here, the gating idea is genuinely dead.
#
# Collapse point (override via env). Default = moderate, clearly recoverable.
COLLAPSE_KD=${COLLAPSE_KD:-0.0}        # KD off -> isolate the InfoNCE collapse
COLLAPSE_NCE=${COLLAPSE_NCE:-0.05}
COLLAPSE_NCE_T=${COLLAPSE_NCE_T:-0.001}
# Severe alternative: COLLAPSE_NCE=0.1 COLLAPSE_NCE_T=0.001

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}

run_gate () {
    local data="$1" name="$2"; shift 2
    echo "=== [${data} collapse kd=${COLLAPSE_KD} nce=${COLLAPSE_NCE} T=${COLLAPSE_NCE_T}] ${name} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ tte True PEFT_WARMUP False seed "${SEED}" \
        classifier_init hybrid \
        TEXT_REG_LAMBDA "${COLLAPSE_KD}" INFONCE_LAMBDA "${COLLAPSE_NCE}" INFONCE_T "${COLLAPSE_NCE_T}" \
        "$@" \
        output_dir "gate_collapse/${data}/${name}"
}

for data in ${DATASETS}; do
    run_gate "${data}" fixed      PRIOR_REG_MODE fixed
    run_gate "${data}" agreement  PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE image_text PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT False
    run_gate "${data}" shuffled   PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE shuffled   PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT False
    run_gate "${data}" freq       PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE frequency  PRIOR_GATE_NORM rank   PRIOR_GATE_INVERT False
    run_gate "${data}" freq_inv   PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE frequency  PRIOR_GATE_NORM rank   PRIOR_GATE_INVERT True
done

echo ""
echo "=== compare (does gating recover the collapse?): ==="
for data in ${DATASETS}; do
  echo "  ${PYTHON} scripts/summarize_splits.py --root output/gate_collapse/${data} --baseline fixed"
done
