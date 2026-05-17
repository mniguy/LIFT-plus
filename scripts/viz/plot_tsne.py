"""t-SNE of tail-class image features + text prototypes (baseline vs ours).

Usage:
    python scripts/viz/plot_tsne.py \
        --baseline_dir output/viz/baseline \
        --ours_dir     output/viz/ours \
        --num_classes 10 \
        --out figures/tsne.pdf
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def _tsne_panel(ax, feats_dir, title, tail_class_ids):
    feats = np.load(os.path.join(feats_dir, "image_features.npy"))
    labels = np.load(os.path.join(feats_dir, "labels.npy"))
    text   = np.load(os.path.join(feats_dir, "text_prototypes.npy"))

    # L2 normalize
    feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
    text  = text  / (np.linalg.norm(text,  axis=-1, keepdims=True) + 1e-8)

    # Keep only tail classes
    mask = np.isin(labels, tail_class_ids)
    feats_tail = feats[mask]
    labels_tail = labels[mask]
    text_tail = text[tail_class_ids]

    # Stack image features + text prototypes; remember the split
    n_img = feats_tail.shape[0]
    X = np.concatenate([feats_tail, text_tail], axis=0)

    Z = TSNE(n_components=2, perplexity=30, init="pca",
             learning_rate="auto", random_state=0).fit_transform(X)
    Z_img, Z_txt = Z[:n_img], Z[n_img:]

    cmap = plt.cm.get_cmap("tab20", len(tail_class_ids))
    for i, c in enumerate(tail_class_ids):
        m = labels_tail == c
        ax.scatter(Z_img[m, 0], Z_img[m, 1], s=10, color=cmap(i), alpha=0.5, edgecolor="none")
        ax.scatter(Z_txt[i, 0], Z_txt[i, 1], s=180, color=cmap(i),
                   marker="*", edgecolor="black", lw=0.8)

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--ours_dir",     required=True)
    parser.add_argument("--num_classes",  type=int, default=10)
    parser.add_argument("--out",          required=True)
    parser.add_argument("--seed",         type=int, default=0)
    args = parser.parse_args()

    # Pick same tail classes for both panels (use ours_dir's cls_num_list)
    freq = np.load(os.path.join(args.ours_dir, "cls_num_list.npy"))
    few_idx = np.where(freq < 20)[0]
    rng = np.random.default_rng(args.seed)
    tail = rng.choice(few_idx, size=min(args.num_classes, len(few_idx)), replace=False)
    tail = np.sort(tail)
    print(f"Tail classes used: {tail.tolist()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _tsne_panel(axes[0], args.baseline_dir, "Baseline (LIFT)", tail)
    _tsne_panel(axes[1], args.ours_dir,     "Ours",            tail)

    fig.suptitle("t-SNE: tail-class image features (dots) + text prototypes (stars)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
