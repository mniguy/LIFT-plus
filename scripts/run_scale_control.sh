#!/bin/bash
#
# MUST-ADD (B) -- the logit-scale confound control.
#
# WHY. draft_results.tex Sec. "Why the centered init is better" argues centering raises each
# prototype's discriminative fraction from rho_bar to 1 and therefore "amplifies inter-class
# logit gaps by 1/rho_bar: measured 1.60x on ImageNet-LT, 1.95x on Places-LT". The cosine
# classifier's scale s is fixed at 25 in every run reported in the paper. So centering
# changes the EFFECTIVE logit temperature by ~1.6-2.0x, and no experiment currently separates
# "better direction geometry" from "better temperature".
#
#   Reviewer's question: is +1.59 Few just s=25 -> s_eff=40?
#   Pass condition: no baseline scale reproduces the centered arm's (Many, Few) point, and
#                   the centered arm still wins at its OWN best scale.
#   Fail condition: baseline at s ~ 25/rho_bar matches centering. Then the mechanism section
#                   must be rewritten around temperature, not geometry.
#
# The rho_bar-matched points (from mean_c(rho_c^2) reported in draft_intro_method.tex):
#     ImageNet-LT  rho_bar = sqrt(0.402) = 0.634  ->  baseline-matched s = 25/0.634 = 39.4
#                                                 ->  center-matched   s = 25*0.634 = 15.8
#     Places-LT    rho_bar = sqrt(0.266) = 0.516  ->  baseline-matched s = 25/0.516 = 48.5
#                                                 ->  center-matched   s = 25*0.516 = 12.9
# The default grid {15,20,25,32,40,50} brackets all four of them, so no extra runs are needed;
# report the grid and mark the matched points in the table.
#
# Everything except classifier_scale is identical to tab:main (semantic init, LA, MDA, TTE, 5 ep).
#
#   bash scripts/run_scale_control.sh
#   SCALES="25 40" SEEDS="0" bash scripts/run_scale_control.sh    # minimal decisive pair
#   python scripts/agg_runs.py output/scale_control25 --sort path
#
# Cost: 6 scales x 2 arms x 3 seeds x 2 datasets = 72 runs @ 5 ep.
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
SCALES=${SCALES:-"15 20 25 32 40 50"}
VARIANTS=${VARIANTS:-"baseline center"}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"scale_control25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=( classifier_init semantic mda True tte True )
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for sc in ${SCALES}; do
    for v in ${VARIANTS}; do
      va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
      for s in ${SEEDS}; do
        out="${OUT_ROOT}/${data}/${v}_s${sc}_seed${s}"
        completed "${out}" && { echo "  [skip] ${out}"; continue; }
        echo "=== [${data}] ${v} scale=${sc} seed=${s} ==="
        CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
          -d "${data}" -b clip_vit_b16 -m lift+ \
          "${BASE_ARGS[@]}" classifier_scale "${sc}" num_epochs "${EPOCHS}" ${va} \
          seed "${s}" output_dir "${out}"
      done
    done
  done
done
echo
echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    Compare center@s=25 against baseline@s=40 (IN) / baseline@s=50 (PL): those are the"
echo "    1/rho_bar-matched temperatures. If they tie, the geometry story does not hold."
