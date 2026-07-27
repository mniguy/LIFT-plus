#!/bin/bash
#
# #5 rarity-function follow-up (2026-07-24 brainstorm). PROMPT_CENTER_MODE=kappa and =logcount
# were added to _center_prototypes (trainer.py) as alternatives to tail's rank-based rarity.
# (#4, a normalize-fix that was tried in the same session, was REVERTED after testing worse --
# _center_prototypes is back to its original un-normalized-input behavior. Not part of this run.)
#
#   tail     (existing): rarity_c = 1 - rank_c/(C-1)        -- linear in ORDINAL RANK, ignores count gaps
#   kappa    (new):       rarity_c = kappa/(n_c+kappa)       -- hyperbolic in COUNT, kappa = half-strength count
#   logcount (new):       rarity_c = (log n_max - log n_c)/(log n_max - log n_min), clipped [0,1]
#                          -- parameter-free, linear in LOG-count. Measured stronger than tail in the
#                          mid/upper range on both IN and iNat (their count range spans ~2-3 decades),
#                          so expect logcount to behave more aggressively than tail, not "in between".
#
# First-round results (single-seed, output/center_kappa25, already done):
#   IN  kappa5/20/50:  Head -0.24/-0.32/-0.22  Med +0.20/+0.26/+0.26  Few +0.73/+1.18/+1.04
#   PL  kappa5/20/50:  Head +0.14/+0.12/+0.03  Med -0.18/-0.34/-0.23  Few +0.84/+1.59/+1.68
#   -> Places: kappa50 nearly matches global's Few (+1.68 vs +1.61) while keeping Head ~flat
#      (+0.03 vs global's -0.26) -- promising, needs confirming. ImageNet: no kappa beat tail/global.
#
# Reference anchors (pre-existing, for diffing new results against):
#   baseline (center_seeds25, 5-seed): IN 78.33/81.20/77.37/73.58   PL 52.15/51.45/52.94/51.62
#   global   (center_seeds25, 5-seed): IN 78.51/81.01/77.46/75.17   PL 52.24/51.19/52.65/53.23
#   tail     (center_geom25, 1-seed):  IN 78.52/80.98/77.65/74.62   PL 52.43/51.50/52.88/53.08
#   fewonly  (center_control25,1-seed):IN 78.38/80.94/77.47/74.31   PL 52.35/51.54/52.58/53.32
#
# This round's defaults (per dataset, see default_variants()):
#   imagenet_lt: kappa100 (matches the Many/Med=100 boundary; global already has low Head cost here,
#                so this is mostly a completeness check), logcount
#   places_lt:   kappa35, kappa75 (bracket the 20/50 sweet spot found above), logcount
#   inat2018:    kappa20 (same threshold convention as IN/PL, first iNat data point), kappa10 (near
#                iNat's p10 count=12 -- iNat's median count is only 22, right at the Few cutoff, so
#                kappa20 centers ~half the dataset at >=0.5 strength; kappa10 concentrates more on
#                the true deep tail). iNat needs 15 epochs (INAT_EPOCHS), not the 5 used for IN/PL.
#
# Override any of DATASETS / VARIANTS / EPOCHS / INAT_EPOCHS / SEED / GPU_ID / OUT_ROOT as env vars,
# e.g.:  DATASETS="places_lt" VARIANTS="kappa35 kappa75 logcount" bash scripts/run_center_kappa.sh
#
#   bash scripts/run_center_kappa.sh
#   python scripts/agg_runs.py output/center_kappa25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_kappa25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale 25 PROMPT_CENTER True
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True PEFT_WARMUP False
)
variant_args(){ case "$1" in
  global|tail|logcount) echo "PROMPT_CENTER_MODE $1" ;;
  kappa[0-9]*) echo "PROMPT_CENTER_MODE kappa PROMPT_CENTER_KAPPA ${1#kappa}" ;;  # kappaNN -> any integer NN
  *) return 1 ;; esac; }

default_variants(){ case "$1" in
  imagenet_lt) echo "kappa100 logcount" ;;
  places_lt)   echo "kappa35 kappa75 logcount" ;;
  inat2018)    echo "kappa50" ;;
  *) echo "" ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  variants="${VARIANTS:-$(default_variants "${data}")}"
  [ -z "${variants}" ] && { echo "  [skip dataset] ${data}: no variants (set VARIANTS to override)"; continue; }
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for v in ${variants}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] center-kappa ${v} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    PL:  compare kappa35/75 against kappa20(Head+0.12/Few+1.59)/kappa50(Head+0.03/Few+1.68)"
echo "    IN:  kappa100 vs tail(Head-0.22/Med+0.28/Few+1.04) and global(Head-0.19/Few+1.59)"
echo "    iNat: first data point for this whole mode family -- no prior kappa/logcount anchor yet"
