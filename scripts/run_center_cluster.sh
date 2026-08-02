#!/bin/bash
#
# PROMPT_CENTER_MODE=cluster at a matched k across all three LT datasets (default k=100).
#
# cluster = k-means over the prompt prototypes (unit-normalized rows, sklearn KMeans, k-means++,
# n_init=10, random_state=seed), each class centered by its own cluster's mean; clusters smaller
# than PROMPT_CENTER_GENUS_MIN (5) fall back to global mu. It is cascade with an embedding
# partition in place of the taxonomy, so unlike genus/cascade it runs on ImageNet/Places too --
# which is the generality gap cascade cannot close.
#
# Why this run: on iNat, cluster500 (80.75 All / 76.01 Head / 82.60 Few) came within 0.09 of
# cascade WITHOUT a taxonomy, while cluster50 (80.52) sat at global. So k matters and its useful
# range is unknown; k=100 is the matched point across datasets.
#
# TWO WAYS TO SET THE GRANULARITY:
#   KS="100"     absolute cluster count. NOT comparable across datasets -- k=100 is 3 classes per
#                cluster on Places (C=365) and 57 on iNat (C=8142), i.e. two different experiments.
#   SIZES="16"   dataset-relative (PROMPT_CENTER_CLUSTER_SIZE): fixes the AVERAGE classes per
#                cluster and derives k=round(C/size), so one number means the same granularity
#                everywhere. Same idea as kappa parameterizing rarity by count instead of rank.
#
# MEASURED BEFORE RUNNING (real CLIP B/16 prototypes, CPU, same code path -- classes in clusters
# smaller than PROMPT_CENTER_GENUS_MIN fall back to plain global centering, so a high fallback rate
# means the run is largely a global rerun):
#   absolute k=100      k   clusters>=5  median  fallback%   cos(W, W_global)
#     imagenet_lt      100       80         8       5.7%         0.790
#     places_lt        100       28         3      43.8% <-- !   0.864
#     inat2018         100       99        57       0.0%         0.793
#   relative size=16    k   median  p10  p90  fallback%   cos(W, W_global)
#     imagenet_lt       62     14     6   28      1.1%        0.805
#     places_lt         23     13     6   30      0.3%        0.850
#     inat2018         509     10     2   35      3.9%        0.719
#   (other sizes: size=8 -> fallback 10.6/7.9/13.7%, already into the harmful small-group regime;
#    size=32 -> 0.3/0.0/0.8%; size=64 -> 0/0/0% but Places has only 6 clusters left, i.e. ~global.)
# -> Places at ABSOLUTE k=100 is half a global rerun (43.8% fallback), the same 28/72 mixture that
#    made genus fail on iNat; the relative setting removes that failure mode by construction.
# -> size=16 is anchored by an existing positive result: on iNat it gives k=509 ~= the cluster500
#    run (80.75 All / 76.01 Head / 82.60 Few, best Head AND Few of all 10 iNat runs). So for iNat,
#    SIZES=16 is nearly a rerun of what we have -- use DATASETS="imagenet_lt places_lt" to skip it.
#
# Reference anchors (seed 0, scale 25, mda+tte, identical args to below):
#   imagenet_lt 5ep   baseline (eval_center25/imagenet_lt/base)      78.28  81.03  77.43  73.49
#                     global   (prompt_center25/imagenet_lt/center)  78.51  81.01  77.46  75.12
#                     knn20/knn50 (center_local25)                   78.41 / 78.49  <- local failed here
#   places_lt   5ep   baseline (eval_center25/places_lt/base)        52.17  51.67  52.93  51.37
#                     global   (prompt_center25/places_lt/center)    52.32  51.23  52.64  53.58
#                     cluster50 (center_tree25)                      51.94  51.37  52.43  51.86
#   inat2018   15ep   baseline (breadth25/inat2018/baseline_15ep)    80.63  74.62  80.50  82.36
#                     global   (breadth25/inat2018/center_15ep)      80.52  74.86  80.41  82.13
#                     cluster50 / cluster500 (center_tree25)         80.52 / 80.75
#                     cascade  (center_local25/inat2018/cascade)     80.84  75.81  80.57  82.50
#   Noise (11-seed, B/16): IN All 0.056 Head 0.083 Few 0.154 | PL All 0.097 Head 0.19 Few 0.201.
#   iNat has NO 15-ep multi-seed baseline yet (only estimate: All 0.06, Head 0.74 at 5ep/scale30),
#   so iNat Head differences here are not interpretable until that baseline exists.
#
#   bash scripts/run_center_cluster.sh                                   # size=16 on all three
#   KS="100" SIZES="" bash scripts/run_center_cluster.sh                 # absolute k=100 instead
#   SIZES="8 16 32" DATASETS=places_lt bash scripts/run_center_cluster.sh
#   python scripts/agg_runs.py output/center_cluster25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt inat2018"}
SIZES=${SIZES:-"16 32"}
KS=${KS:-""}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_cluster25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
  PROMPT_CENTER True PROMPT_CENTER_MODE cluster
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for s in ${SIZES}; do              # dataset-relative: k = round(C / s), computed in the trainer
    out="${OUT_ROOT}/${data}/size${s}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] cluster target size=${s} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${COMMON_ARGS[@]}" num_epochs "${ep}" PROMPT_CENTER_CLUSTER_SIZE "${s}" \
      seed "${SEED}" output_dir "${out}"
  done
  for k in ${KS}; do                 # absolute cluster count (not comparable across datasets)
    out="${OUT_ROOT}/${data}/cluster${k}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] cluster k=${k} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${COMMON_ARGS[@]}" num_epochs "${ep}" PROMPT_CENTER_CLUSTER_K "${k}" \
      seed "${SEED}" output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    check each log's '[PROMPT_CENTER cluster] k=.. -> N/C classes fell back' line FIRST:"
echo "    a high N means the run is mostly plain global centering, not a clustering result."
echo "    IN/PL: does cluster beat global's Few (+1.63 IN / +2.21 PL over baseline), or repeat knn's failure?"
echo "    iNat:  k=100 vs k=50 (80.52) and k=500 (80.75) -- is there a real k trend or is it flat?"
