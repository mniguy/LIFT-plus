#!/usr/bin/env python
"""Threshold sensitivity of hybrid caption selection, from a DUMP_CAPTIONS json.

Per class, gathers the candidate sentence sims (from 'selected' if present, else the
recorded 'top_missed'), then reports at each SIM_THRESHOLD how many classes would
select >=1 caption and the mean #selected.

To measure the fixed (aligned) corpus properly, run the dump ONCE with a permissive
threshold so all top-k sims are recorded:
    # after swapping in wiki_aligned (scripts/realign_inat_wiki.py):
    DUMP_CAPTIONS=output/inat_caps_aligned.json SIM_THRESHOLD 0.0 \
      CUDA_VISIBLE_DEVICES=0 python main.py -d inat2018 -b clip_vit_b16 -m lift+ \
      classifier_init hybrid output_dir tmp/inat_capdump
    python scripts/analyze_caption_threshold.py output/inat_caps_aligned.json
"""
import argparse
import json

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def class_sims(rec):
    """All candidate sims recorded for a class (empty if no corpus)."""
    if rec.get("selected"):
        sims = [s["sim"] for s in rec["selected"]]
    else:
        sims = [s["sim"] for s in rec.get("top_missed", [])]
    return sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="DUMP_CAPTIONS json path")
    args = ap.parse_args()
    d = json.load(open(args.dump))
    n = len(d)
    sims = [class_sims(r) for r in d]
    with_corpus = sum(1 for s in sims if s)
    best = [max(s) if s else -1.0 for s in sims]

    print(f"classes: {n}   with candidate sentences: {with_corpus}")
    print(f"best-sim over classes-with-corpus: "
          f"mean {sum(b for b in best if b>=0)/max(with_corpus,1):.3f}, "
          f"max {max(best):.3f}\n")
    print(f"{'threshold':>9} | {'classes w/ caption':>18} | {'rate':>6} | {'mean #selected':>14}")
    print("-" * 58)
    for t in THRESHOLDS:
        picked = [sum(1 for x in s if x > t) for s in sims]
        n_cls = sum(1 for p in picked if p >= 1)
        mean_sel = (sum(p for p in picked if p >= 1) / n_cls) if n_cls else 0.0
        print(f"{t:>9.2f} | {n_cls:>18d} | {100*n_cls/n:>5.1f}% | {mean_sel:>14.2f}")


if __name__ == "__main__":
    main()
