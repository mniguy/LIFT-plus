"""Plot per-class accuracy: baseline vs ours, sorted by class frequency.

Usage:
    python scripts/viz/plot_per_class_acc.py \
        --baseline output/baseline_lift/cls_accs.npy \
        --ours     output/center_lift/cls_accs.npy \
        --freq     output/center_lift/cls_num_list.npy \
        --out      figures/per_class_acc.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def smooth(x, window=21):
    """Running-mean smoothing for visual clarity."""
    k = np.ones(window) / window
    return np.convolve(x, k, mode="same")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ours",     required=True)
    parser.add_argument("--freq",     required=True, help="cls_num_list.npy")
    parser.add_argument("--out",      required=True)
    parser.add_argument("--window",   type=int, default=21)
    args = parser.parse_args()

    base = np.load(args.baseline)
    ours = np.load(args.ours)
    freq = np.load(args.freq)

    # Sort by frequency descending (head -> tail)
    order = np.argsort(-freq)
    base_s = base[order]
    ours_s = ours[order]
    freq_s = freq[order]

    x = np.arange(len(freq_s))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, smooth(base_s, args.window), label="Baseline (LIFT)", color="#888", lw=1.5)
    ax.plot(x, smooth(ours_s, args.window), label="Ours",            color="#d62728", lw=1.8)

    # Many/Med/Few boundaries (>100, 20-100, <20)
    many_end = int(np.searchsorted(-freq_s, -100))
    few_start = int(np.searchsorted(-freq_s, -20))
    ax.axvline(many_end,  color="gray", ls=":", lw=0.8)
    ax.axvline(few_start, color="gray", ls=":", lw=0.8)
    ax.text(many_end / 2,                     5, "Many", ha="center", color="gray")
    ax.text((many_end + few_start) / 2,       5, "Med",  ha="center", color="gray")
    ax.text((few_start + len(freq_s)) / 2,    5, "Few",  ha="center", color="gray")

    ax.set_xlabel("Class index (sorted by # samples)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xlim(0, len(freq_s))
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
