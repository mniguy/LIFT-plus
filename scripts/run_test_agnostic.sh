#!/bin/bash
#
# #4: test-agnostic / prior-shift evaluation.
# Train LIFT+ (semantic, no text-prior), dump raw test logits (SAVE_LOGITS), then
# evaluate accuracy under forward/uniform/backward test priors with
# no-adapt vs EM-estimated-prior adaptation vs oracle (scripts/test_agnostic.py).
#
# The "method" here is unsupervised test-time prior adaptation (Saerens EM). The
# win comes from a setting where LIFT+ has headroom: it is tuned for the uniform
# test prior and degrades under shift, which adaptation recovers.
#
# To evaluate an EXISTING checkpoint instead of training, swap the main.py call for:
#   ... test_only True model_dir output/<run>/lift+_seed0 SAVE_LOGITS True ...

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}

for ds in ${DATASETS}; do
  echo "=== [${ds}] train LIFT+ + dump logits ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
    -d "${ds}" -b clip_vit_b16 -m lift+ tte True PEFT_WARMUP False seed "${SEED}" \
    classifier_init semantic TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0 PRIOR_REG_MODE fixed \
    SAVE_LOGITS True output_dir "test_agnostic/${ds}/lift+"

  ${PYTHON} scripts/test_agnostic.py --root "output/test_agnostic/${ds}/lift+"
done

echo ""
echo "=== (optional) also dump + evaluate the per-class-temperature model: ==="
echo "    add 'classifier CosineClassifierPCT' to the main.py call, output_dir test_agnostic/<ds>/lift_pct,"
echo "    then: ${PYTHON} scripts/test_agnostic.py --root output/test_agnostic/<ds>/lift_pct"
