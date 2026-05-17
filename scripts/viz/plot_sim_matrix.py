"""Image-text similarity heatmap evolution: init -> after warmup -> final.

Run extract_features.py 3 times (one per checkpoint) into separate dirs, then:
    python scripts/viz/plot_sim_matrix.py \
        --dirs output/viz/init output/viz/after_warmup output/viz/final \
        --titles "Init" "After warmup" "Final" \
        --out figures/sim_matrix.pdf
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def _sim_matrix(feats_dir):
    """Return [C, C] class-mean cosine similarity matrix sorted by frequency."""
    img = np.load(os.path.join(feats_dir, "image_features.npy"))
    lbl = np.load(os.path.join(feats_dir, "labels.npy"))
    txt = np.load(os.path.join(feats_dir, "text_prototypes.npy"))
    freq = np.load(os.path.join(feats_dir, "cls_num_list.npy"))

    img = img / (np.linalg.norm(img, axis=-1, keepdims=True) + 1e-8)
    txt = txt / (np.linalg.norm(txt, axis=-1, keepdims=True) + 1e-8)

    C = txt.shape[0]
    mean_img = np.zeros((C, img.shape[1]), dtype=np.float32)
    for c in range(C):
        m = lbl == c
        if m.any():
            mean_img[c] = img[m].mean(axis=0)
    mean_img = mean_img / (np.linalg.norm(mean_img, axis=-1, keepdims=True) + 1e-8)

    sim = mean_img @ txt.T  # [C, C]

    # Sort by frequency descending so head classes are top-left
    order = np.argsort(-freq)
    return sim[order][:, order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs",   nargs="+", required=True, help="Feature dirs (>=2)")
    parser.add_argument("--titles", nargs="+", required=True)
    parser.add_argument("--out",    required=True)
    args = parser.parse_args()
    assert len(args.dirs) == len(args.titles)

    n = len(args.dirs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    vmin, vmax = -0.2, 0.6  # consistent scale across panels
    for ax, d, t in zip(axes, args.dirs, args.titles):
        sim = _sim_matrix(d)
        im = ax.imshow(sim, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(t)
        ax.set_xlabel("Text prototype (sorted by freq)")
        ax.set_ylabel("Image class-mean (sorted by freq)")

    fig.colorbar(im, ax=axes, shrink=0.8, label="cosine similarity")
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
