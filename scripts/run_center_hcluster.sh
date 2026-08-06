#!/bin/bash
#
# C) PROMPT_CENTER_MODE=hcluster -- taxonomy-free HIERARCHICAL cascade (2026-08-06 brainstorm).
#
# Motivation: 'cascade' (genus->family->order) is iNat's best result (80.84) but needs a real
# taxonomy, so it cannot transfer to any dataset without one. 'cluster' (flat k-means, single level)
# is taxonomy-free but already gets close (cluster,k=500: 80.75 All / 76.01 Head / 82.60 Few --
# actually BEATS cascade on Head and Few individually, just loses narrowly on All/Med). hcluster asks
# whether stacking cluster into MULTIPLE nested levels (mimicking cascade's finest-first fallback,
# but with the levels coming from one dendrogram instead of genus/family/order) closes that last gap
# AND generalizes to datasets that have no taxonomy at all (ImageNet-LT, Places-LT).
#
# HOW IT WORKS (trainer.py, mode=hcluster): builds ONE agglomerative dendrogram (complete linkage,
# cosine distance) over the prototypes, then cuts it at k=round(C/size) for each size in
# PROMPT_CENTER_HCLUSTER_SIZES (finest/smallest size first). Reuses cascade's exact residual-mean +
# min_size fallback logic. Levels are cut from the SAME tree, so they are mathematically GUARANTEED
# to nest (a fine cluster never straddles two coarse clusters) -- the same nesting property
# genus/family/order has, unlike running independent KMeans at each granularity.
#
# LINKAGE CHOICE, revised after measurement: first version used "average" linkage and chained badly --
# one cluster absorbed 4549/8142 classes at k=509 (median cluster size only 3), a "rich get richer"
# blob that is barely different from plain global centering for most of what it nominally "covers".
# Switched to "complete" linkage (merge distance = worst-case pairwise distance between two clusters,
# not the average), which resists this: same k=509, max cluster size drops to 893, median rises to 6.
#
# OFFLINE SANITY CHECK (2026-08-06, GPU-free, real iNat raw prototypes from
# output/breadth25/inat2018/baseline_15ep/ckpts/init, sizes=16,64,256, complete linkage):
#   coverage: size16(k=509)=7721  size64(k=127)=268  size256(k=32)=119  global=34
#   -> 99.6% of classes get SOME local mean (vs cascade's 95.5%, vs genus-only's 28%)
#   nesting check (on the size16/size256 pair): 0 fine clusters straddle >1 coarse cluster
#   (guarantee verified, not assumed)
#
# HONEST CALIBRATION AGAINST 'cluster' (flat KMeans), measured 2026-08-06 at matched k=509: KMeans
# actually finds TIGHTER, more balanced, higher-coverage groups than this dendrogram-cut approach at
# the SAME single k (within-cluster raw cosine 0.8775 vs 0.8607; coverage 7789 vs 7486 classes in
# clusters >=5 before the linkage fix). hcluster's case is NOT "finds better single-level clusters
# than KMeans" -- on that axis KMeans currently wins. Its case is the NESTING guarantee across
# multiple levels, which independent KMeans calls at different k cannot provide (a class's k=509
# cluster is not guaranteed to sit inside any single one of its k=127 cluster). Whether that
# structural property translates into a real accuracy edge over 'cluster,k=500' (80.75/76.01/80.26/
# 82.60, already close to cascade) is untested -- this run is what actually answers that.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   genus    (center_local25/inat2018/genus)     80.46  75.22  80.05  82.34
#   cluster,k=500 (center_tree25/inat2018)       80.75  76.01  80.26  82.60
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50
#   ImageNet-LT/Places-LT global anchors: see run_center_cluster.sh header.
#
#   bash scripts/run_center_hcluster.sh                                   # iNat only, default sizes
#   SIZES_LIST="16,64,256 8,32,128" bash scripts/run_center_hcluster.sh   # sweep two size ladders
#   DATASETS="imagenet_lt places_lt inat2018" bash scripts/run_center_hcluster.sh
#   python scripts/agg_runs.py output/center_hcluster25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
SIZES_LIST=${SIZES_LIST:-"16,64,256"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_hcluster25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
  PROMPT_CENTER True PROMPT_CENTER_MODE hcluster
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for sizes in ${SIZES_LIST}; do
    tag=$(echo "${sizes}" | tr ',' '-')
    out="${OUT_ROOT}/${data}/sizes${tag}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] hcluster sizes=${sizes} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${COMMON_ARGS[@]}" num_epochs "${ep}" PROMPT_CENTER_HCLUSTER_SIZES "${sizes}" \
      seed "${SEED}" output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    check each log's '[PROMPT_CENTER hcluster] sizes=.. -> ...' coverage line first."
echo "    iNat: compare against cascade (80.84/75.81/80.57/82.50) and cluster,k=500 (80.75/76.01/80.26/82.60)."
echo "    IN/PL: expected to lose to global, same as cascade/cluster already do -- confirms the spike"
echo "    is dataset-intrinsic (Sec. discussion), not an artifact of needing a real taxonomy."
