# Visualization scripts for paper

## 0) Train baseline + final model

```bash
# Baseline: original LIFT (no hybrid/loss1/loss2/warmup/TTE)
bash scripts/run_baseline.sh

# Final method (already in run_final.sh as ep2_lr_1e-4)
```

Each run now saves:
- `<output_dir>/cls_accs.npy`        — per-class accuracy
- `<output_dir>/cls_num_list.npy`    — class frequencies
- `<output_dir>/ckpts/init/checkpoint.pth.tar`         — pre-training
- `<output_dir>/ckpts/after_warmup/checkpoint.pth.tar` — only when warmup ran
- `<output_dir>/checkpoint.pth.tar`  — final

---

## 1) Per-class accuracy plot

```bash
python scripts/viz/plot_per_class_acc.py \
    --baseline output/baseline_lift/cls_accs.npy \
    --ours     output/final_tte/warmup/ep2_lr_1e-4/cls_accs.npy \
    --freq     output/final_tte/warmup/ep2_lr_1e-4/cls_num_list.npy \
    --out      figures/per_class_acc.pdf
```

---

## 2) t-SNE (tail classes)

First extract features for both models:
```bash
# Ours (final checkpoint)
python scripts/viz/extract_features.py \
    --ckpt output/final_tte/warmup/ep2_lr_1e-4/checkpoint.pth.tar \
    --output_dir output/viz/ours \
    -d imagenet_lt -b clip_vit_b16 -m lift+ \
    PEFT_WARMUP True PEFT_WARMUP_EPOCHS 2 PEFT_WARMUP_LR 1e-4 tte True

# Baseline
python scripts/viz/extract_features.py \
    --ckpt output/baseline_lift/checkpoint.pth.tar \
    --output_dir output/viz/baseline \
    -d imagenet_lt -b clip_vit_b16 -m lift+ \
    classifier_init semantic TEXT_REG_LAMBDA 0 INFONCE_LAMBDA 0
```

Then plot:
```bash
python scripts/viz/plot_tsne.py \
    --baseline_dir output/viz/baseline \
    --ours_dir     output/viz/ours \
    --num_classes 10 \
    --out figures/tsne.pdf
```

---

## 3) Image-text similarity matrix (epoch evolution)

Extract features at three timepoints of the final model:
```bash
for STAGE in init after_warmup; do
    python scripts/viz/extract_features.py \
        --ckpt output/final_tte/warmup/ep2_lr_1e-4/ckpts/${STAGE}/checkpoint.pth.tar \
        --output_dir output/viz/ours_${STAGE} \
        -d imagenet_lt -b clip_vit_b16 -m lift+ \
        PEFT_WARMUP True PEFT_WARMUP_EPOCHS 2 PEFT_WARMUP_LR 1e-4 tte True
done
# (output/viz/ours already has the final-epoch features)
```

Then plot:
```bash
python scripts/viz/plot_sim_matrix.py \
    --dirs   output/viz/ours_init output/viz/ours_after_warmup output/viz/ours \
    --titles "Init" "After warmup" "Final" \
    --out    figures/sim_matrix.pdf
```
