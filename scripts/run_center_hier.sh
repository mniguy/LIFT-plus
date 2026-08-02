#!/bin/bash
#
# baseline vs global centering vs cascade, on every dataset that HAS a hierarchy.
#
# Point of the run: cascade is currently a one-dataset result (iNat 80.84, the only thing that beats
# the iNat baseline). It needs a second hierarchical dataset or it stays an appendix. ImageNet-LT
# has one -- its classes are WordNet synsets -- it just was not wired up until now:
#   python scripts/make_imagenet_taxonomy.py --wordnet ~/nltk_data/corpora/wordnet
# writes datasets/ImageNet_LT/categories.json in the schema _load_taxonomy() already reads, with
# levels h1..h8 = hops up the hypernym chain from the class. No trainer change needed.
#
# Assignment actually produced (verified on all 1000 real prototypes, min_size=5):
#   cascade        h1,h2,h3,h4      -> h1=197 h2=389 h3=161 h4=78            global=175
#   cascade_coarse h3,h4,h5         -> h3=791 h4=43  h5=51                   global=115
#   cascade_wide   h2,h3,h4,h5,h6   -> h2=603 h3=156 h4=74  h5=59  h6=47     global=61
# (iNat cascade for comparison: genus=2279 family=4427 order=1068 global=368.)
#
# PRE-REGISTERED PREDICTION: cascade is expected to LOSE on ImageNet. The granularity sweep in
# center_cluster25 showed the optimum locality on IN/PL is ZERO -- accuracy rises monotonically as
# groups get coarser, all the way to a single group (= global): IN k=62 78.21 < k=31 78.27 <
# global 78.51; PL k=50 51.94 < k=23 51.82* < k=11 52.23 < global 52.32. iNat runs the other way
# (k=50 80.52 < k=254 80.65 < k=500 80.75 < cascade 80.84). Cascade's ImageNet init is MORE
# aggressive than global (mean cos to the 5 nearest other classes 0.335 vs global's 0.497), i.e.
# further along the axis that hurts here. If cascade nonetheless wins, the "locality only pays on
# fine-grained taxonomies" story is wrong and needs rewriting; if it loses, cascade stays an iNat
# extension and the two datasets together make that boundary a measured claim rather than a guess.
# cascade_coarse is the hedge: h3,h4,h5 gives 791 classes a level-3 mean and only 115 fall to
# global, i.e. locality at the coarsest level that still localizes.
#
# Reference anchors (seed 0, scale 25, mda+tte, same args as below -- baseline/center ALREADY EXIST,
# so `VARIANTS="cascade cascade_coarse"` skips 30 min of rerunning them):
#   imagenet_lt 5ep  baseline (eval_center25/imagenet_lt/base)     78.28  81.03  77.43  73.49
#                    center   (prompt_center25/imagenet_lt/center) 78.51  81.01  77.46  75.12
#   inat2018   15ep  baseline (breadth25/inat2018/baseline_15ep)   80.63  74.62  80.50  82.36
#                    center   (breadth25/inat2018/center_15ep)     80.52  74.86  80.41  82.13
#                    cascade  (center_local25/inat2018/cascade)    80.84  75.81  80.57  82.50
#   Noise (11-seed B/16): IN All 0.056 Head 0.083 Med 0.102 Few 0.154.
#
#   bash scripts/run_center_hier.sh                                  # IN: all 4 variants
#   VARIANTS="cascade cascade_coarse" bash scripts/run_center_hier.sh # only the new ones
#   DATASETS="imagenet_lt inat2018" bash scripts/run_center_hier.sh
#   python scripts/agg_runs.py output/center_hier25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt"}
VARIANTS=${VARIANTS:-"baseline center cascade cascade_coarse"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_hier25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
)

# cascade level names are per-dataset: WordNet hops on ImageNet, Linnaean ranks on iNat.
cascade_levels(){ case "$1" in
  imagenet_lt) echo "h1,h2,h3,h4" ;;
  inat2018)    echo "genus,family,order" ;;
  *) return 1 ;; esac; }
cascade_coarse_levels(){ case "$1" in
  imagenet_lt) echo "h3,h4,h5" ;;
  inat2018)    echo "family,order" ;;
  *) return 1 ;; esac; }

variant_args(){ case "$2" in
  baseline)       echo "PROMPT_CENTER False" ;;
  center)         echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  cascade)        echo "PROMPT_CENTER True PROMPT_CENTER_MODE cascade PROMPT_CENTER_CASCADE $(cascade_levels "$1")" ;;
  cascade_coarse) echo "PROMPT_CENTER True PROMPT_CENTER_MODE cascade PROMPT_CENTER_CASCADE $(cascade_coarse_levels "$1")" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

# cfg.dataset (the directory _load_taxonomy looks in), per data config name
data_dir(){ case "$1" in
  imagenet_lt) echo "ImageNet_LT" ;;
  inat2018)    echo "iNaturalist2018" ;;
  *) return 1 ;; esac; }

for data in ${DATASETS}; do
  dd=$(data_dir "${data}") || { echo "  [skip dataset] ${data}: no hierarchy wired up"; continue; }
  [ -f "datasets/${dd}/categories.json" ] || {
    echo "  [skip dataset] ${data}: datasets/${dd}/categories.json missing"
    echo "    build it: ${PYTHON} scripts/make_imagenet_taxonomy.py --wordnet ~/nltk_data/corpora/wordnet"
    continue; }
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for v in ${VARIANTS}; do
    va=$(variant_args "${data}" "$v") || { echo "unknown variant $v for ${data}"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] ${v} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" num_epochs "${ep}" ${va} seed "${SEED}" output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    verify the assignment line first: grep -h 'PROMPT_CENTER cascade' output/${OUT_ROOT}/*/*/log-*.txt"
echo "    read cascade against center (78.51 IN), not against baseline: the question is whether a"
echo "    hierarchy beats plain global centering, and on IN global is already strong (Few +1.63)."