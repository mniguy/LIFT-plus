#!/bin/bash
#
# Gating controls / variants on ImageNet-LT and Places-LT (seed=0 fixed).
#
# Full method context everywhere: hybrid init + KD + InfoNCE, gate on.
# Only the GATE SIGNAL changes between runs:
#
#   fixed      : no gate (uniform reg)                       -- baseline
#   agreement  : gate by cos(image-mean, prototype), minmax  -- the proposed method
#   shuffled   : agreement gate values, permuted to wrong classes
#                -> negative control: tests whether the agreement SIGNAL matters,
#                   or just having a non-uniform gate. Should NOT recover the gain.
#   freq       : gate by class train frequency (rank), high-freq -> stronger reg
#   freq_inv   : gate by class train frequency (rank), low-freq (tail) -> stronger reg
#                -> alternative axis: "is the agreement gate just frequency in disguise?"
#
# Output: output/gate_controls/<dataset>/<name>/  (each has cls_accs.npy etc.)

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEED=${SEED:-0}
LAMBDA_KD=${LAMBDA_KD:-0.001}
LAMBDA_NCE=${LAMBDA_NCE:-0.005}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}

run_gate () {
    local data="$1" name="$2"; shift 2
    echo "=== [${data} gate seed=${SEED} kd=${LAMBDA_KD} nce=${LAMBDA_NCE}] ${name} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ tte True PEFT_WARMUP False seed "${SEED}" \
        classifier_init hybrid TEXT_REG_LAMBDA "${LAMBDA_KD}" INFONCE_LAMBDA "${LAMBDA_NCE}" \
        "$@" \
        output_dir "gate_controls/${data}/${name}"
}

for data in ${DATASETS}; do
    # references (for direct comparison in the same folder)
    run_gate "${data}" fixed      PRIOR_REG_MODE fixed
    run_gate "${data}" agreement  PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE image_text PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT False

    # NEW: negative control -- shuffled agreement gate
    run_gate "${data}" shuffled   PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE shuffled   PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT False

    # NEW: alternative axis -- frequency gate (both directions)
    run_gate "${data}" freq       PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE frequency  PRIOR_GATE_NORM rank   PRIOR_GATE_INVERT False
    run_gate "${data}" freq_inv   PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE frequency  PRIOR_GATE_NORM rank   PRIOR_GATE_INVERT True
done

echo "=== done. results under output/gate_controls/{imagenet_lt,places_lt}/<name>/cls_accs.npy ==="
