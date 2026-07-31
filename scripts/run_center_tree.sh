#!/bin/bash
#
# 2026-07-31 follow-ups, all on iNat (the only dataset with a real taxonomy):
#
#   sem_off (Q1): does semantic-aware init still matter on iNat at all? classifier_init None
#      leaves the cosine classifier at its random init and trains from there; everything else
#      identical to the 15-ep baseline. On ImageNet/Places semantic init is worth several points,
#      but iNat's 8142 latin binomials are the case where CLIP's text prior is weakest (and where
#      within-genus prototype cosine is 0.945 raw -- congeneric species are near-duplicates), so
#      the ablation sets the ceiling for how much any prototype-side trick can possibly move.
#
#   cascade_full (Q2): PROMPT_CENTER_CASCADE_MEAN=full. A fallback level's mean is now taken over
#      the WHOLE group instead of only the members still unassigned at that level -- a 4-species
#      genus inside an 11-species family subtracts the family's full mean, not the mean of the 4
#      leftovers. Verified offline on the real 8142 prototypes: coverage genus=2279 family=4518
#      order=1100 global=245 (vs residual's 4427/1068/368) and it is slightly LESS aggressive
#      (overall off-diag cos 0.0051 vs 0.0013, within-family 0.0052 vs -0.0114).
#
#   family / order (Q3): single-level centering, i.e. the "whole tree from the top" version. A
#      cascade with one level is the same thing as nested level-by-level subtraction (the level
#      means telescope), so this IS the tree variant, not an approximation of it. Offline these
#      are the interesting rows: family-only reaches global's isotropy (overall 0.0019 vs global
#      0.0042) while KEEPING the within-genus structure that cascade destroys (within-genus cos
#      0.569 vs cascade's 0.054; nearest-prototype-same-family 43% vs cascade's 20%). Whether that
#      retained structure helps or hurts is exactly what cascade (80.84) vs global (80.52) cannot
#      currently tell us.
#
#   cluster50 / cluster500 (Q4): PROMPT_CENTER_MODE=cluster -- k-means over the prototypes, each
#      class centered by its own cluster mean. Taxonomy-free, so unlike genus/cascade it also runs
#      on ImageNet/Places. Offline, k=50 is the most isotropic variant of all (overall 0.0002,
#      |mu| 0.019) while preserving family structure as well as raw prototypes do (nn-family 45.9%
#      vs raw 45.9%), and its clusters do recover the taxonomy (NMI vs genus 0.64, vs family 0.46;
#      k=500 -> 0.82/0.64). k=500 is the finer, more cascade-like end of the same knob.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   genus    (center_local25/inat2018/genus)     80.46  75.22  80.05  82.34
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50   <- residual means
#   NOTE the spread here is ~0.3 All / ~0.4 Head on ONE seed; iNat seed noise measured on 5-ep runs
#   was ~0.06 All / ~0.19 Few. Treat any single-run winner below as a candidate, not a result, and
#   replicate it over seeds before it goes anywhere near a paper.
#
#   bash scripts/run_center_tree.sh
#   python scripts/agg_runs.py output/center_tree25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_tree25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON_ARGS=(
  classifier_scale 25
  mda True tte True
)
variant_args(){ case "$1" in
  sem_off)      echo "classifier_init None PROMPT_CENTER False" ;;
  sem_on)       echo "classifier_init semantic PROMPT_CENTER False" ;;   # == breadth25 baseline_15ep
  cascade_full) echo "classifier_init semantic PROMPT_CENTER True PROMPT_CENTER_MODE cascade PROMPT_CENTER_CASCADE genus,family,order PROMPT_CENTER_CASCADE_MEAN full" ;;
  family)       echo "classifier_init semantic PROMPT_CENTER True PROMPT_CENTER_MODE cascade PROMPT_CENTER_CASCADE family" ;;
  order)        echo "classifier_init semantic PROMPT_CENTER True PROMPT_CENTER_MODE cascade PROMPT_CENTER_CASCADE order" ;;
  cluster[0-9]*) echo "classifier_init semantic PROMPT_CENTER True PROMPT_CENTER_MODE cluster PROMPT_CENTER_CLUSTER_K ${1#cluster}" ;;
  *) return 1 ;; esac; }

default_variants(){ case "$1" in
  inat2018)    echo "sem_off cascade_full family order cluster50 cluster500" ;;
  *)           echo "sem_off cluster50 cluster500" ;;   # taxonomy-free variants only
esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  variants="${VARIANTS:-$(default_variants "${data}")}"
  [ -z "${variants}" ] && { echo "  [skip dataset] ${data}: no variants (set VARIANTS to override)"; continue; }
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for v in ${variants}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] center-tree ${v} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${COMMON_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    sem_off vs breadth25/inat2018/baseline_15ep (80.63): size of the semantic-init effect on iNat"
echo "    cascade_full vs center_local25/inat2018/cascade (80.84): does full-group pooling beat residual?"
echo "    family/order vs global (80.52): does coarse-only centering keep the within-genus structure AND the gain?"
echo "    cluster50/500 vs cascade: is a taxonomy even needed, or do k-means groups suffice?"
