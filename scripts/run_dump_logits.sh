#!/bin/bash
#
# Dump test logits for the beta (test-agnostic prior-adaptation) paper. RUN ON THE GPU
# BOX (needs CUDA + the datasets). Every run writes logits.npy (+ y_true / cls_num) via
# SAVE_LOGITS; all analysis afterwards is post-hoc on logits (no GPU):
#   scripts/structured_prior_adapt.py | scripts/sweep_beta.py | scripts/make_beta_tables.py
#
# (1) multi-seed std on ImageNet-LT + Places-LT   (2) generality: iNat2018 + CIFAR-100-LT
#
# COMMON flags reproduce the LIFT+ baseline used for the seed0 logits in
# output/test_agnostic/* (semantic init, NO text-prior, TTE on).

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
# Dataset paths come from configs/data/*.yaml (imagenet_lt -> /workspace/data/ImageNet, etc.),
# same as every other script -- no 'root' override needed.
COMMON="classifier_init semantic TEXT_REG_LAMBDA 0 INFONCE_LAMBDA 0 PRIOR_REG_MODE fixed tte True SAVE_LOGITS True"

# ---------- (1) multi-seed on ImageNet-LT + Places-LT: train LIFT+ fresh at 3 seeds ----------
# Each run trains AND (via SAVE_LOGITS) dumps logits.npy into output_dir; the trained
# checkpoint.pth.tar is saved there too, so all 3 seeds share one identical pipeline
# (homogeneous provenance -> clean std). Re-running overwrites any earlier partial dumps.
for ds in imagenet_lt places_lt; do
  for SEED in 0 1 2; do
    echo "=== [${ds} seed${SEED}] train LIFT+ + dump logits ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d ${ds} -b clip_vit_b16 -m lift+ \
      ${COMMON} seed ${SEED} \
      output_dir test_agnostic_ms/${ds}/seed${SEED}
  done
done
# SANITY: dumped seed0 All should be ~78.4 (ImageNet) / ~52.0 (Places).

# ---------- (2) generality: iNat2018 (8142-cls; grouping should help most) ----------
for ds in inat2018; do
  echo "=== [${ds}] train LIFT+ + dump logits ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d ${ds} -b clip_vit_b16 -m lift+ \
    ${COMMON} seed 0 output_dir test_agnostic/${ds}/lift+
done

# ---------- analyze (no GPU) ----------
cat <<'EOF'

Next (any machine, no GPU):
  # final multi-seed + generality table:
  python scripts/make_beta_tables.py \
    ImageNet-LT:output/test_agnostic_ms/imagenet_lt/seed0,output/test_agnostic_ms/imagenet_lt/seed1,output/test_agnostic_ms/imagenet_lt/seed2 \
    Places-LT:output/test_agnostic_ms/places_lt/seed0,output/test_agnostic_ms/places_lt/seed1,output/test_agnostic_ms/places_lt/seed2 \
    iNat2018:output/test_agnostic/inat2018/lift+

  # per-dataset frontier figure:
  for d in imagenet_lt places_lt inat2018; do
    python scripts/plot_beta_frontier.py --root output/test_agnostic/$d/lift+ --title $d \
      --out output/paper/fig_frontier_$d.pdf
  done
EOF
