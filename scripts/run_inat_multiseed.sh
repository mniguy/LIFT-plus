#!/bin/bash
#
# E4 -- iNaturalist-2018 MULTI-SEED logit dump (closes C6: the headline iNat result is
# currently single-seed/trials=2 with NO model-seed std). RUN ON THE GPU BOX.
#
# Trains LIFT+ FRESH at 3 seeds into the same test_agnostic_ms/<ds>/seed<N> layout the
# ImageNet/Places multi-seed dumps use (homogeneous provenance -> clean std), each writing
# logits.npy via SAVE_LOGITS. Mirrors scripts/run_dump_logits.sh's flags EXACTLY so the
# seed0 dump reproduces output/test_agnostic/inat2018/lift+ (~76.1 All; sanity-check below).
#
# iNat is heavy (8142 cls); expect the longest single-dataset run in the paper. Re-running
# overwrites partial dumps. After it lands, ALL tables are post-hoc (no GPU).
#
#   bash scripts/run_inat_multiseed.sh
#   GPU_ID=1 SEEDS="1 2" bash scripts/run_inat_multiseed.sh   # add seeds to an existing seed0
set -euo pipefail

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"0 1 2"}
COMMON="classifier_init semantic TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0 PRIOR_REG_MODE fixed tte True SAVE_LOGITS True"

[ -f main.py ] || { echo "ERROR: run from repo root (main.py not found)"; exit 1; }

for SEED in ${SEEDS}; do
  echo "=== [inat2018 seed${SEED}] train LIFT+ + dump logits ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    ${COMMON} seed ${SEED} \
    output_dir test_agnostic_ms/inat2018/seed${SEED}
done
# SANITY: seed0's training log should print "* Overall accuracy: ~77.3%" (native uniform
# test), matching output/test_agnostic/inat2018/lift+ (77.31%). NOTE: the post-hoc no-adapt
# "All" in compare_baselines is ~76.1 -- a DIFFERENT number (mean over resampled
# forward/uniform/backward priors), not the native test accuracy; don't confuse the two.

cat <<'EOF'

Next (any machine, no GPU) -- iNat now has model-seed std; feed all 3 roots:
  python scripts/compare_baselines.py \
    ImageNet-LT:output/test_agnostic_ms/imagenet_lt/seed0,...seed1,...seed2 \
    Places-LT:output/test_agnostic_ms/places_lt/seed0,...seed1,...seed2 \
    iNat2018:output/test_agnostic_ms/inat2018/seed0,output/test_agnostic_ms/inat2018/seed1,output/test_agnostic_ms/inat2018/seed2 \
    --trials 10 --inat-trials 10 --out output/paper/tables_baselines.tex

  # same 3-root iNat spec works for: make_beta_tables.py, ablation_2x2.py, cost_table.py
EOF
