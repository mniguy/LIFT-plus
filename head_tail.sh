#!/bin/bash

GPU_ID=0

DATASET="imagenet_lt"
BACKBONE="clip_vit_b16"
METHOD="lift+"

LAYERS=12

HEAD_SIZES=(4 6 8)
TAIL_SIZES=(4 6 8)
LORA_SCALE=16.0
ADAPT_SCALE=1.0

BASE_OUTPUT_DIR="HEAD_TAIL"

for HEAD in "${HEAD_SIZES[@]}"; do
  for TAIL in "${TAIL_SIZES[@]}"; do
    if (( HEAD + TAIL > LAYERS )); then
      echo "skip head=${HEAD} tail=${TAIL} (exceeds ${LAYERS} layers)"
      continue
    fi
    
    OUTPUT_DIR="${OUTPUT_DIR}/${HEAD}/${TAIL}"
    
    echo "Running head_tail: head=${HEAD} tail=${TAIL}"

    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
      -d "${DATASET}" \
      -b "${BACKBONE}" \
      -m "${METHOD}" \
      tte True \
      v.hybrid_mix_mode head_tail \
      v.hybrid_head_layers "${HEAD}" \
      v.hybrid_tail_layers "${TAIL}" \
      v.lora_gate_scale "${LORA_SCALE}" \
      v.adaptformer_gate_scale "${ADAPT_SCALE}" \
      output_dir ${OUTPUT_DIR}
  done
done
