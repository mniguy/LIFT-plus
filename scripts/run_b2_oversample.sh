#!/bin/bash
#
# SHOULD-fix #4: close the SECOND link of the causal chain by intervention.
#
# The chain the paper argues is
#   B1  the prototypes waste ~69% of the cosine classifier's unit-norm budget on one shared
#       direction, compressing inter-class logit gaps by 1/rho
#   B2  training can remove that bias itself, but only in proportion to the gradient signal a
#       class receives -- so head self-corrects (collinearity .65 -> .24) and tail does not
#       (.63 -> .44)
#   B3  the tail's margins stay narrow and Few images leak into Many/Med
#
# B1 is closed by intervention (centering removes it; freeze shows the pinned target is better;
# three negative controls show it must be mu specifically). **B2 is currently CORRELATIONAL ONLY** --
# the evidence is that repair size tracks class frequency (r(SCG, log n) = +0.82). A reviewer can
# answer "frequency correlates with everything; you have not manipulated gradient signal."
#
# THIS IS THE MANIPULATION. ClassBalancedSampler (utils/samplers.py) draws the SAME number of
# samples per epoch as the dataset, so total gradient budget and steps/epoch are unchanged, but
# allocates them equally across classes. Verified on ImageNet-LT: 115846 draws either way, per-class
# draws 115-116 (std 0.36), so Few classes receive 9.6x their usual gradient signal and Many 0.50x.
#
#   PREDICTION (pre-registered, write the result down only after reading this):
#     If B2 is right, giving the tail head-level gradient signal lets it self-correct, so there is
#     less left for centering to contribute and **Delta-Few must SHRINK substantially** under
#     balanced sampling. Direction and mechanism check, both required:
#       (a) Delta-Few(balanced) << Delta-Few(default)                        [accuracy]
#       (b) SCG(Few) under balanced-baseline << SCG(Few) under default-baseline, and
#           r(SCG, log n) collapses toward 0                                 [the mechanism itself]
#     If Delta-Few is unchanged under balanced sampling, B2 is WRONG: the tail's failure to repair
#     is not about gradient signal, and the mechanism section must be rewritten around whatever
#     else distinguishes rare classes (e.g. intrinsic sample diversity, not count).
#
# !! LOSS CHOICE IS NOT OPTIONAL: this uses loss_type=CE. Under balanced sampling the effective
#    training prior is uniform, so LA / BS / LDAM would apply a prior correction that is no longer
#    warranted and double-correct. CE has no prior term, so the sampler is the only thing that
#    changes between the two arms.
#
#   The default-sampling CE arm ALREADY EXISTS in output/loss_robust25 (identical settings):
#     IN  CE_baseline 72.18 / 86.07 / 69.17 / 43.46    CE_center 72.89 / 86.11 / 69.71 / 46.63
#         -> Delta-Few = +3.17
#     PL  CE_baseline 41.92 / 56.47 / 37.67 / 24.83    CE_center 42.69 / 56.05 / 38.42 / 27.86
#         -> Delta-Few = +3.03
#   so only the balanced arm is run here: 4 runs, IN 2x15min + PL 2x5min = 40 min.
#
#   bash scripts/run_b2_oversample.sh
#   python scripts/agg_runs.py output/b2_oversample25 --sort path
#   # then the mechanism half of the prediction:
#   python scripts/diag_rho_scg.py \
#       --baseline output/b2_oversample25/imagenet_lt/balanced_baseline \
#       --runs output/b2_oversample25/imagenet_lt/balanced_baseline \
#              output/b2_oversample25/imagenet_lt/balanced_center \
#       --label "ImageNet-LT, class-balanced sampling (B2 intervention)"
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"baseline center"}
SAMPLERS=${SAMPLERS:-"balanced"}      # add "default" to reproduce the loss_robust25 CE cells here
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"b2_oversample25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# CE on purpose -- see the loss note above. Everything else matches loss_robust25's CE cells.
BASE_ARGS=(
  classifier_init semantic classifier_scale 25 loss_type CE
  mda True tte True num_epochs "${EPOCHS}"
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for smp in ${SAMPLERS}; do
    for v in ${VARIANTS}; do
      va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
      out="${OUT_ROOT}/${data}/${smp}_${v}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] sampler=${smp} ${v} (CE) ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
        "${BASE_ARGS[@]}" train_sampler "${smp}" ${va} seed "${SEED}" output_dir "${out}"
    done
  done
done
echo
echo "=== read against the pre-registered prediction in this script's header ==="
echo "    (a) Delta-Few(balanced) must be much smaller than +3.17 (IN) / +3.03 (PL)"
echo "    (b) SCG(Few) must drop and r(SCG,log n) must collapse -- run diag_rho_scg.py"
