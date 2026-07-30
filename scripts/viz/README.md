# Visualization scripts for paper

## 0) Train baseline + centered model

```bash
# Baseline: LIFT+ with plain semantic init (no centering)
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
    classifier_init semantic tte True output_dir baseline_lift

# Centered: same, with prototype centering
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
    classifier_init semantic PROMPT_CENTER True tte True output_dir center_lift
```

Each run saves:
- `<output_dir>/cls_accs.npy`        — per-class accuracy
- `<output_dir>/cls_num_list.npy`    — class frequencies
- `<output_dir>/ckpts/init/checkpoint.pth.tar`         — pre-training
- `<output_dir>/checkpoint.pth.tar`  — final

---

## 1) Per-class accuracy plot

```bash
python scripts/viz/plot_per_class_acc.py \
    --baseline output/baseline_lift/cls_accs.npy \
    --ours     output/center_lift/cls_accs.npy \
    --freq     output/center_lift/cls_num_list.npy \
    --out      figures/per_class_acc.pdf
```

---

## 2) t-SNE (tail classes)

First extract features for both models:
```bash
# Ours (final checkpoint)
python scripts/viz/extract_features.py \
    --ckpt output/center_lift/checkpoint.pth.tar \
    --output_dir output/viz/ours \
    -d imagenet_lt -b clip_vit_b16 -m lift+ \
    classifier_init semantic PROMPT_CENTER True tte True

# Baseline
python scripts/viz/extract_features.py \
    --ckpt output/baseline_lift/checkpoint.pth.tar \
    --output_dir output/viz/baseline \
    -d imagenet_lt -b clip_vit_b16 -m lift+ \
    classifier_init semantic
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

Extract features at the init checkpoint of the centered model:
```bash
python scripts/viz/extract_features.py \
    --ckpt output/center_lift/ckpts/init/checkpoint.pth.tar \
    --output_dir output/viz/ours_init \
    -d imagenet_lt -b clip_vit_b16 -m lift+ \
    classifier_init semantic PROMPT_CENTER True tte True
# (output/viz/ours already has the final-epoch features)
```

Then plot:
```bash
python scripts/viz/plot_sim_matrix.py \
    --dirs   output/viz/ours_init output/viz/ours \
    --titles "Init" "Final" \
    --out    figures/sim_matrix.pdf
```
