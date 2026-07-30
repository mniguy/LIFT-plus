#!/bin/bash
#
# P3 CLOSURE: the missing no-D control. Every other centering result lives in an imbalanced
# regime, so "the effect is specific to data-sparse classes" was asserted rather than shown.
# Balanced CIFAR-100 (imb_factor=None -> all 100 classes at 500 images) is the D=0 endpoint of
# the severity sweep we already have.
#
#   severity sweep, Delta Overall (center - baseline), existing single/5-seed results:
#     CIFAR-IR200  +1.67   (Few +5.88, t~27)
#     CIFAR-IR100  +0.50   (Few +2.24)
#     CIFAR-IR50   -0.09   (Few -0.27)
#     CIFAR-IR40   -0.16   (Few -0.14)
#     CIFAR-IR1    ????    <- THIS RUN
#
#   PREDICTION (pre-registered here, before the run): Delta Overall ~ 0 (|Delta| < 0.3), and the
#   per-class SCG frequency gradient r(SCG, log n) should collapse toward 0 because there is no
#   frequency axis left. If centering still helps materially at IR1, the whole D framing ("the
#   tail cannot self-correct") is wrong and the mechanism section must be rewritten -- this run
#   is a genuine falsification test, not a formality.
#
#   NOTE on group metrics: at IR1 every class has 500 images, so Many holds all 100 classes and
#   Med/Few are EMPTY. Read Overall only; the Head/Med/Few columns are degenerate by construction.
#
#   bash scripts/run_balanced_control.sh
#   python scripts/agg_runs.py output/balanced25 --sort path
#   python scripts/diag_rho_scg.py --baseline output/balanced25/cifar100/baseline \
#          --runs output/balanced25/cifar100/baseline output/balanced25/cifar100/center --label "CIFAR-100 balanced (D=0)"
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"cifar100"}
VARIANTS=${VARIANTS:-"baseline center"}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"balanced25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# identical to the CIFAR cells of run_breadth.sh / run_center_seeds.sh
BASE_ARGS=(
  classifier_init semantic classifier_scale 25
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True num_epochs "${EPOCHS}" PEFT_WARMUP False
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    for s in ${SEEDS}; do
      out="${OUT_ROOT}/${data}/${v}_seed${s}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] ${v} seed ${s} (${EPOCHS} ep, BALANCED / D=0 control) ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
        "${BASE_ARGS[@]}" ${va} seed "${s}" output_dir "${out}"
    done
  done
done
echo; echo "=== read: Delta Overall should be ~0. Compare against the severity sweep in the header. ==="
