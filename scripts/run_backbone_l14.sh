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
# DO THIS FIRST (no GPU hours, needs the L/14 weights downloaded):
#   python scripts/measure_anisotropy_backbone.py --backbones ViT-B/16 ViT-L/14
# If L/14 shows |mu| ~ 0.8, rho ~ 0.55 and coll - a^2 ~ 0 as B/16 does, the problem statement
# already generalizes and the training arm below is confirming the *remedy*, not the *diagnosis*.
# Reference (B/16, measured): |mu| 0.77-0.87, rho 0.49-0.63, coll - a^2 = -0.001..-0.013 across
# ImageNet-LT / Places-LT / iNat2018.
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
