#!/bin/bash
#
# MUST-ADD (C, part 2 of 2) -- seed coverage for the single-seed evidence families that
# run_pij_close.sh does NOT already cover.
#
# WHY. Half the paper's causal skeleton is n=1 while the paper's own measured seed noise on
# Few is +/-0.23 (ImageNet-LT) and +/-0.33 (Places-LT). Claims currently resting on a
# single seed:
#     freeze   (tab:freeze)   -- THE causal claim      -> ALREADY COVERED, see below
#     controls (tab:controls) -- the selectivity claim -> ALREADY COVERED, see below
#     pca      (fig:pca)      -- the minimality design law        <- this script
#     local    (tab:in_local) -- the globality design law         <- this script
#     l14      (tab:backbone) -- the backbone-scale robustness    <- this script
#
# DO NOT run freeze/controls here. scripts/run_pij_close.sh already schedules exactly those
# two families at seeds {0,1,2} / {1,2}, reuses the existing seed-0 runs instead of repeating
# them, and adds a cell this script would not have: a FROZEN randdir control, without which
# "freezing reveals the centered init" is still compatible with "freezing helps any perturbed
# init". It writes to pij_frozen25/ and pij_control25/ and is read by scripts/agg_pij.py.
# It has not been run yet. Run it first:
#     GPU_ID=0 DATASETS=imagenet_lt bash scripts/run_pij_close.sh &
#     GPU_ID=1 DATASETS=places_lt   bash scripts/run_pij_close.sh &
#     python scripts/agg_pij.py
# Consider adding the positive control while you are there, which currently exists only at
# seed 0 and is absent from tab:controls even though it recovers most of the gain
# (Places Few 53.32 vs 53.58 for full centering):
#     TRAINABLE_VARIANTS="randdir headonly perclass_rand fewonly" bash scripts/run_pij_close.sh
#
# The three families below add seeds 1 and 2 into the SAME output root as their seed-0 runs,
# using per-seed directory names so nothing is overwritten and agg_runs.py picks up all of them.
# Protocol is copied verbatim from the scripts that produced the seed-0 runs
# (run_pca_sweep.sh, run_center_local.sh, run_backbone_l14.sh): scale 25, semantic init,
# MDA on, TTE on, 5 ep.
#
#   bash scripts/run_seed_boost.sh                       # pca + local + l14, seeds 1 2
#   FAMILIES=pca SEEDS="1 2 3 4" bash scripts/run_seed_boost.sh
#   python scripts/agg_runs.py output/pca_sweep25 --sort path
#
# Cost (default): pca 24 + local 16 + l14 8 = 48 runs (l14 is ~50 min/run, the rest ~5-15 min).
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
FAMILIES=${FAMILIES:-"pca local l14"}   # freeze/controls belong to run_pij_close.sh
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
SEEDS=${SEEDS:-"1 2"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON=( classifier_init semantic classifier_scale "${SCALE}" mda True tte True )
completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

# family -> (output root, backbone, extra base args, variant list)
family_root(){ case "$1" in
  pca)      echo "pca_sweep25" ;;
  local)    echo "center_local25" ;;
  l14)      echo "backbone_l14" ;;
  freeze|controls)
    echo "ERROR: family '$1' is owned by scripts/run_pij_close.sh -- see header" >&2; return 1 ;;
  *) return 1 ;; esac; }

family_backbone(){ case "$1" in
  l14) echo "clip_vit_l14" ;;
  *)   echo "clip_vit_b16" ;; esac; }

family_extra(){ case "$1" in
  *) echo "" ;; esac; }

family_variants(){ case "$1" in
  pca)   echo "k0 k1 k2 k5 k10 k20" ;;
  local) echo "knn20 knn50 cluster16 cluster32" ;;
  l14)   echo "baseline center" ;;
  *) return 1 ;; esac; }

variant_args(){ case "$1" in
  baseline)      echo "PROMPT_CENTER False" ;;
  center)        echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  headonly)      echo "PROMPT_CENTER True PROMPT_CENTER_MODE headonly" ;;
  fewonly)       echo "PROMPT_CENTER True PROMPT_CENTER_MODE fewonly" ;;
  randdir)       echo "PROMPT_CENTER True PROMPT_CENTER_MODE randdir" ;;
  perclass_rand) echo "PROMPT_CENTER True PROMPT_CENTER_MODE perclass_rand" ;;
  k[0-9]*)       echo "PROMPT_CENTER True PROMPT_CENTER_MODE pca PROMPT_CENTER_PCA_K ${1#k}" ;;
  knn[0-9]*)     echo "PROMPT_CENTER True PROMPT_CENTER_MODE knn PROMPT_CENTER_KNN_K ${1#knn}" ;;
  cluster[0-9]*) echo "PROMPT_CENTER True PROMPT_CENTER_MODE cluster PROMPT_CENTER_CLUSTER_SIZE ${1#cluster}" ;;
  *) return 1 ;; esac; }

for fam in ${FAMILIES}; do
  root=$(family_root "${fam}")   || { echo "unknown family ${fam}"; exit 1; }
  bb=$(family_backbone "${fam}")
  extra=$(family_extra "${fam}")
  for data in ${DATASETS}; do
    if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
    for v in $(family_variants "${fam}"); do
      va=$(variant_args "${v}") || { echo "unknown variant ${v}"; exit 1; }
      for s in ${SEEDS}; do
        out="${root}/${data}/${v}_seed${s}"
        completed "${out}" && { echo "  [skip] ${out}"; continue; }
        echo "=== [${fam}/${data}] ${v} seed=${s} (${bb}, ${ep} ep) ==="
        CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
          -d "${data}" -b "${bb}" -m lift+ \
          "${COMMON[@]}" num_epochs "${ep}" ${extra} ${va} \
          seed "${s}" output_dir "${out}"
      done
    done
  done
done
echo
echo "=== tabulate each family, e.g.: ==="
for fam in ${FAMILIES}; do echo "  ${PYTHON} scripts/agg_runs.py output/$(family_root "${fam}") --sort path"; done
echo "NOTE: seed-0 runs live in the same roots under their original (seedless) names."
