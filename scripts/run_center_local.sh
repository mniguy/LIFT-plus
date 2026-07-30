#!/bin/bash
#
# Q2/Q3/Q6 follow-up (2026-07-24 brainstorm): three new things added to trainer.py, need GPU runs.
#
#   genus (PROMPT_CENTER_MODE=genus, iNat only): subtract the per-GENUS mean instead of one global
#      mu -- classnames are "Genus species" binomials, genus = first token. Genera with <
#      PROMPT_CENTER_GENUS_MIN (default 5) species fall back to global mu (68% of iNat genera are
#      singletons -- subtracting a genus's own single member would zero that class out). Verified
#      offline: min_size=5 -> 5863/8142 classes fall back, Quercus's 28 species get a real local mean
#      (cos to global mu = 0.22, genuinely different direction).
#
#   cascade (PROMPT_CENTER_MODE=cascade, iNat only): hierarchical fallback genus -> family -> order ->
#      global. Fixes genus mode's coverage hole: genus-only sent 5863/8142 (72%) straight to global mu,
#      which left those classes carrying their own genus blob. Verified offline on all 8142 classes:
#      cascade assigns genus=2279, family=4427, order=1068, and only 368 (4.5%) reach global. Geometry
#      (CPU, real CLIP prototypes, 1824-species sample) is strictly better than BOTH global and genus:
#        overall coll  global 0.0045 | genus-only 0.0143 | cascade 0.0010
#        within-genus  global 0.8425 | genus-only 0.0323 | cascade 0.0265
#      CAVEAT: genus mode already showed that a better iNat init does NOT convert into trainable
#      accuracy (drift 0.85-0.89 overwrites it; genus's Med/Few drift was even HIGHER than baseline).
#      Cascade is more aggressive still, so a neutral-or-slightly-worse trainable result is the
#      expectation here -- the geometry claim is what this run actually tests.
#
#   knn (PROMPT_CENTER_MODE=knn, any dataset): taxonomy-free generalization of genus -- subtract the
#      mean of each class's PROMPT_CENTER_KNN_K nearest OTHER classes (by prototype cosine similarity)
#      instead of the fixed global mu. Sweep k to see where it sits relative to genus (iNat) and to
#      global/tail (IN/PL, no taxonomy available there).
#
#   bare (PROMPT_MODE=bare): drops the "a photo of a {}." template, using just "{}." -- tests whether
#      centering's benefit survives when the template boilerplate is removed. Measured OFFLINE (CPU,
#      real CLIP text encoder, 300 iNat names) that |mu| barely shrinks without the template (0.8223 ->
#      0.8034, ~2%) so centering is expected to remain just as necessary; this is a robustness/sanity
#      check, not expected to unlock a better init. Compare bare_raw (no centering) vs bare_global
#      (centered) to see if the SAME ~+1.6 Few delta from the templated version reproduces.
#
# Reference anchors (default template, for diffing against):
#   baseline (center_seeds25, 5-seed): IN 78.33/81.20/77.37/73.58   PL 52.15/51.45/52.94/51.62
#   global   (center_seeds25, 5-seed): IN 78.51/81.01/77.46/75.17   PL 52.24/51.19/52.65/53.23
#   (iNat trainable centering is ~neutral per breadth25: baseline_15ep vs center_15ep ~ -0.11/-0.23 Few)
#
#   bash scripts/run_center_local.sh
#   python scripts/agg_runs.py output/center_local25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_local25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
)
variant_args(){ case "$1" in
  genus)        echo "PROMPT_CENTER True PROMPT_CENTER_MODE genus" ;;
  cascade)      echo "PROMPT_CENTER True PROMPT_CENTER_MODE cascade PROMPT_CENTER_CASCADE genus,family,order" ;;
  cascade_gf)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE cascade PROMPT_CENTER_CASCADE genus,family" ;;
  knn[0-9]*)    echo "PROMPT_CENTER True PROMPT_CENTER_MODE knn PROMPT_CENTER_KNN_K ${1#knn}" ;;
  bare_raw)     echo "PROMPT_MODE bare PROMPT_CENTER False" ;;
  bare_global)  echo "PROMPT_MODE bare PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

default_variants(){ case "$1" in
  imagenet_lt) echo "knn10 knn20 knn50 bare_raw bare_global" ;;
  places_lt)   echo "knn10 knn20 knn50 bare_raw bare_global" ;;
  inat2018)    echo "cascade" ;;
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
    echo "=== [${data}] center-local ${v} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${COMMON_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    genus/knn: compare Med/Head preservation against tail(center_geom25)/kappa(center_kappa25)"
echo "    bare_global vs bare_raw: does the templated version's ~+1.6 Few delta reproduce w/o template?"
