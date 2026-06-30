#!/bin/bash
#
# F2 data: regularization-strength sweep, fixed (no gate) vs gated.
# Sweeps InfoNCE lambda at the collapse-inducing temperature (T=0.001), KD off,
# hybrid init, for each gate variant. As lambda grows, fixed collapses head/med;
# the figure shows agreement/freq gating staying robust.
#
# Output: output/collapse_sweep/<ds>/<variant>_lam<lambda>/cls_accs.npy
# Plot:   python scripts/plot_collapse_sweep.py --root output/collapse_sweep/<ds>

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEED=${SEED:-0}
NCE_T=${NCE_T:-0.001}
DATASETS=${DATASETS:-"places_lt"}                 # add imagenet_lt if you want both
LAMBDAS=${LAMBDAS:-"0.005 0.01 0.02 0.05 0.1"}
VARIANTS=${VARIANTS:-"fixed agreement"}           # add freq_inv for the alt axis

variant_args () {
    case "$1" in
        fixed)     echo "PRIOR_REG_MODE fixed" ;;
        agreement) echo "PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE image_text PRIOR_GATE_NORM minmax PRIOR_GATE_INVERT False" ;;
        freq_inv)  echo "PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE frequency  PRIOR_GATE_NORM rank   PRIOR_GATE_INVERT True" ;;
        *)         return 1 ;;
    esac
}

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    vargs=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    for lam in ${LAMBDAS}; do
      echo "=== [${data}] ${v} lambda=${lam} T=${NCE_T} ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
        -d "${data}" -b clip_vit_b16 -m lift+ tte True PEFT_WARMUP False seed "${SEED}" \
        classifier_init hybrid TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA "${lam}" INFONCE_T "${NCE_T}" \
        ${vargs} \
        output_dir "collapse_sweep/${data}/${v}_lam${lam}"
    done
  done
done

echo ""
echo "=== plot: ==="
for data in ${DATASETS}; do
  echo "  ${PYTHON} scripts/plot_collapse_sweep.py --root output/collapse_sweep/${data} --out output/collapse_sweep/${data}/F2"
done
