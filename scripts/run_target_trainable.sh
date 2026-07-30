#!/bin/bash
#
# TRAINABLE target comparison -- does a data-aligned target beat centered-text once the
# classifier can also train? (frozen diagnostic freeze_target25 showed image-mean BEATS center
# on Places but loses on ImageNet, with a Head cost from raw-CLIP staleness. Trainable should
# let the classifier chase the drifting features -> the staleness Head cost may vanish.)
#
#   baseline : semantic (raw text)
#   center   : semantic + PROMPT_CENTER global (current method)
#   imgmean  : class_mean (per-class image-feature mean)
#   blend    : img_shrink (count-adaptive imagemean<->centered-text, IMG_SHRINK_KAPPA)
#
# ============================================================================================
# ALREADY RUN (seed 0, on vast.ai; the output/ dirs were never copied back to this repo, so the
# completed() check below will NOT skip and a bare re-run would repeat ~80 min of work for
# nothing). Results, transcribed:
#
#   ImageNet-LT            All      Head      Med       Few
#     baseline           78.28     81.03    77.43     73.49
#     center             78.51     81.01    77.46     75.12   (+1.63 Few)
#     imgmean            77.50     81.27    76.78     69.38   (-4.11 Few)
#     blend              78.25     80.85    77.03     75.22   (+1.73 Few)
#
#   Places-LT              All      Head      Med       Few
#     baseline           52.16     51.67    52.93     51.36
#     center             52.32     51.23    52.64     53.58   (+2.22 Few)
#     imgmean            51.15     50.79    52.43     48.90   (-2.46 Few)
#     blend              52.01     50.95    52.33     53.25   (+1.89 Few)
#
# CONCLUSIONS (judged against the 5-seed Few sigma = 0.32, so diff sigma = 0.45):
#  1. imgmean LOSES DECISIVELY to center on the tail: -5.74 Few on IN (-12.7 sigma) and -4.68 on
#     PL (-10.3 sigma), and its Overall is below even the raw baseline. This closes the "why not
#     just initialize from image class means?" objection with data. Mechanism is obvious in
#     hindsight: a Few class has <20 images, so its image mean is a noisy estimate exactly where
#     the paper's regime lives.
#  2. It also RESOLVES the frozen anomaly noted above -- frozen Places had imgmean (44.93) beating
#     center (42.51), which did NOT transfer to the trainable setting (48.90 vs 53.58). Say this
#     explicitly in the paper so a reviewer reading the frozen table does not raise it.
#  3. blend TIES center on Few (IN +0.10 = +0.2 sigma, PL -0.33 = -0.7 sigma) but is worse on
#     All/Head/Med on both datasets, while additionally requiring a full forward pass over the
#     train set and a kappa hyper-parameter that center does not need. center dominates once cost
#     is accounted for. The lambda axis is therefore NOT wrong -- blend behaves exactly as designed
#     (tail lambda ~0.09-0.20 -> inherits center's tail gain; head lambda ~0.98 -> inherits
#     imgmean's head behaviour); it simply buys nothing.
#  4. Suggestive: on ImageNet, blend's Head (80.85) is BELOW both pure options (center 81.01,
#     imgmean 81.27). Putting head classes in image space and tail classes in text space makes the
#     coordinate system inconsistent ACROSS the class set. Same shape as the genus finding (28%
#     local / 72% global was worse than uniform global). Independent support for the "uniform"
#     clause of the global/uniform/minimal design law. Holds clearly on IN, only partially on PL
#     (blend Head sits between the two there), so report it as an observation, not a claim.
#
# Only re-run this if the raw logs/checkpoints are needed (e.g. to compute rho/coll diagnostics
# on the imgmean and blend inits via scripts/diag_rho_scg.py, which needs ckpts/init).
# ============================================================================================
#
#   bash scripts/run_target_trainable.sh
#   python scripts/agg_runs.py output/target_trainable25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"baseline center imgmean blend"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
KAPPA=${KAPPA:-20}
OUT_ROOT=${OUT_ROOT:-"target_trainable25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_scale "${SCALE}"
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True PEFT_WARMUP False
)
variant_args(){ case "$1" in
  baseline) echo "classifier_init semantic PROMPT_CENTER False" ;;
  center)   echo "classifier_init semantic PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  imgmean)  echo "classifier_init class_mean PROMPT_CENTER False" ;;
  blend)    echo "classifier_init img_shrink PROMPT_CENTER False IMG_SHRINK_KAPPA ${KAPPA}" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "$data" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] trainable target=${v} (scale ${SCALE}, ${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
      -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" \
      output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
