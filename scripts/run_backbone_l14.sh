#!/bin/bash
#
# Backbone generalization: does centering still help on CLIP ViT-L/14?
#
# WHY THIS IS THE HIGHEST-VALUE REMAINING EXPERIMENT
# Every result in the paper is on CLIP ViT-B/16 (configs/backbone/ held exactly one file until
# now). The problem statement -- "text prototypes share a dominant direction, so only ~54% of the
# cosine classifier's unit-norm budget is discriminative" -- is a claim about a text encoder, and
# a reviewer will ask whether it is a property of one checkpoint. A second backbone converts
# "single-backbone limitation" into a two-point generalization result.
#
# DIAGNOSIS STEP: ALREADY DONE (2026-07-30, measure_anisotropy_backbone.py, full class sets)
#   backbone  dataset   |mu|     coll     a_std    rho      1/rho    coll - a^2
#   ViT-B/16  IN       0.7676   0.5889   0.0953   0.6259    1.60     -0.0004
#   ViT-B/16  PL       0.8558   0.7317   0.0384   0.5122    1.95     -0.0007
#   ViT-B/16  iNat     0.8306   0.6899   0.0966   0.5306    1.88     -0.0000
#   ViT-L/14  IN       0.7226   0.5216   0.1111   0.6752    1.48     -0.0005
#   ViT-L/14  PL       0.8139   0.6615   0.0463   0.5758    1.74     -0.0009
#   ViT-L/14  iNat     0.7241   0.5243   0.1105   0.6709    1.49     -0.0001
#
# Two findings, both of which change what this training arm is for:
#  (a) coll - a^2 ~ 0 holds on 6/6 (backbone x dataset) to four decimals -> the central claim
#      ("all apparent inter-class similarity IS the shared component; the class residuals were
#      already mutually orthogonal") is now a six-point result, not a single measurement.
#  (b) The defect EXISTS on L/14 but is MILDER: the recoverable factor 1/rho - 1 drops by
#      20% (IN), 23% (PL), 45% (iNat). A larger, better-trained text encoder has a less
#      degenerate prototype space -- the defect shrinks with scale but does not vanish.
#
# So the diagnosis already generalizes with zero GPU hours. What this arm now tests is the
# QUANTITATIVE link "size of the defect predicts size of the gain", which is a separate and
# stronger claim than "it works on another backbone".
#
# PRE-REGISTERED PREDICTION (written before running; scale the B/16 gain by the ratio of
# recoverable factors (1/rho - 1)):
#   ImageNet-LT  B/16 measured Delta-Few +1.66  ->  L/14 predicted +1.34
#   Places-LT    B/16 measured Delta-Few +1.84  ->  L/14 predicted +1.42
#   L/14 also starts from a higher baseline, so headroom shrinks too and the true value should sit
#   at or BELOW these numbers. Prediction interval: Delta-Few in [+1.0, +1.4], sign positive.
#   FALSIFICATION: if L/14's gain equals or exceeds B/16's, "recoverable norm budget determines
#   the gain" is wrong and the mechanism section needs revising. If it lands in the interval, this
#   becomes the second successful pre-registered quantitative prediction alongside IR40/IR200.
#
# !! Also correct the draft: the "1.9x logit-gap loss" figure was measured on iNat class names.
#    Per dataset on B/16 it is 1.60x (ImageNet-LT), 1.95x (Places-LT), 1.88x (iNat). Do not quote
#    a single 1.9x next to an ImageNet-LT table -- say "1.6-2.0x depending on the dataset".
#
# CAVEATS -- read before launching
#  1. classifier_scale=25 was chosen on ViT-B/16. L/14 has a different embedding dimension (768 vs
#     512) and different feature-norm statistics, so 25 may not be the right logit scale. If the
#     L/14 baseline lands far below its published/expected accuracy, sweep SCALE before concluding
#     anything about centering:  SCALE=30 bash scripts/run_backbone_l14.sh
#  2. L/14 is ~3x the compute of B/16 (24 layers, width 1024). If it OOMs on a 24GB card, raise
#     ACCUM (gradient accumulation) rather than changing the batch size, so the effective batch --
#     and therefore comparability with the B/16 numbers -- is preserved:  ACCUM=2 or ACCUM=4.
#  3. Keep everything else identical to the B/16 runs (semantic init, aux off, mda+tte, 5 epochs).
#     The only intended differences vs Table 1 are the backbone and, if forced, ACCUM.
#
# Cost estimate (single GPU, rough, scaled from B/16's 15 min / 5 min per run):
#   ImageNet-LT ~45-60 min/run, Places-LT ~15-20 min/run
#   2 variants x 1 seed  x (IN + PL) ~ 1.5-2.5 h      <- SEEDS="0", the minimum credible arm
#   2 variants x 3 seeds x (IN + PL) ~ 4.5-7.5 h      <- SEEDS="0 1 2", what a main table wants
#
#   SEEDS="0" bash scripts/run_backbone_l14.sh          # first look
#   bash scripts/run_backbone_l14.sh                    # 3 seeds
#   python scripts/agg_runs.py output/backbone_l14 --sort path
#   python scripts/diag_rho_scg.py --baseline output/backbone_l14/imagenet_lt/baseline_seed0 \
#          --runs output/backbone_l14/imagenet_lt/baseline_seed0 output/backbone_l14/imagenet_lt/center_seed0 \
#          --label "ViT-L/14 ImageNet-LT"
#
# Reference to diff against (ViT-B/16, 5 seeds):
#   IN  baseline 78.32+-.08 / 81.19 / 77.37 / 73.51    center 78.51+-.03 / 81.06 / 77.42 / 75.17
#   PL  baseline 52.10+-.12 / 51.37 / 53.00 / 51.40    center 52.24+-.08 / 51.18 / 52.65 / 53.23
#   The claim to reproduce is the SIGN and rough SIZE of the Few gain (+1.7 / +1.8), not the
#   absolute accuracy -- L/14 is a stronger backbone and will sit higher everywhere.
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"baseline center"}
SEEDS=${SEEDS:-"0"}
BACKBONE=${BACKBONE:-clip_vit_l14}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}
ACCUM=${ACCUM:-1}
OUT_ROOT=${OUT_ROOT:-"backbone_l14"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }
[ -f "configs/backbone/${BACKBONE}.yaml" ] || { echo "ERROR: missing configs/backbone/${BACKBONE}.yaml"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}"
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True num_epochs "${EPOCHS}" PEFT_WARMUP False
  accum_step "${ACCUM}"
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

echo "backbone=${BACKBONE}  scale=${SCALE}  accum=${ACCUM}  epochs=${EPOCHS}"
for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    for s in ${SEEDS}; do
      out="${OUT_ROOT}/${data}/${v}_seed${s}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] ${BACKBONE} ${v} seed ${s} ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b "${BACKBONE}" -m lift+ \
        "${BASE_ARGS[@]}" ${va} seed "${s}" output_dir "${out}"
    done
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read the SIGN and SIZE of the Few gap, not absolute accuracy (L/14 sits higher overall)."
