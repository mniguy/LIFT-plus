#!/usr/bin/env python
"""Fix the iNat2018 wiki caption corpus misalignment.

BUG: datasets/iNaturalist2018/wiki/desc_{i}.txt is numbered by categories.json RAW
order (desc_0 = categories[0]['name']), but build_wiki_corpus / _compute_caption_features
index it by the dataset's SORTED class label (classnames = sorted(set(names))). So for
99.95% of classes the wrong species' article is loaded. It's a deterministic permutation
(no duplicate names), so it's perfectly recoverable:

    aligned desc for class label L  =  desc_{ name2raw[ classnames[L] ] }.txt

Writes a corrected corpus to --out (default datasets/iNaturalist2018/wiki_aligned/),
non-destructively. Swap it in to fix training:
    mv datasets/iNaturalist2018/wiki datasets/iNaturalist2018/wiki_broken
    mv datasets/iNaturalist2018/wiki_aligned datasets/iNaturalist2018/wiki

    python scripts/realign_inat_wiki.py            # generate wiki_aligned/
    python scripts/realign_inat_wiki.py --dry-run  # report mapping only, no files
"""
import argparse
import json
import os
import re
import shutil

ROOT = "datasets/iNaturalist2018"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default=f"{ROOT}/categories.json")
    ap.add_argument("--wiki", default=f"{ROOT}/wiki")
    ap.add_argument("--out", default=f"{ROOT}/wiki_aligned")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cats = json.load(open(args.categories))
    names = [c["name"] for c in cats]           # raw-id order == desc_{i} numbering
    dups = len(names) - len(set(names))
    name2raw = {}
    for i, nm in enumerate(names):
        name2raw.setdefault(nm, i)
    classnames = sorted(set(names))             # == dataset self.classnames (sorted)
    print(f"categories: {len(names)}  unique names: {len(classnames)}  duplicate names: {dups}")

    # build label -> raw_id mapping
    mapping = {}         # label -> raw_id
    missing_src = 0
    for lab, nm in enumerate(classnames):
        raw = name2raw[nm]
        src = os.path.join(args.wiki, f"desc_{raw}.txt")
        if not os.path.exists(src):
            missing_src += 1
            continue
        mapping[lab] = raw
    print(f"mapped {len(mapping)}/{len(classnames)} classes  (missing source desc: {missing_src})")

    if args.dry_run:
        # spot-check a few
        print("\nspot-check (class label -> raw_id -> desc first line):")
        for lab in [0, 100, 6502, len(classnames) - 1]:
            raw = mapping.get(lab)
            if raw is None:
                continue
            fl = open(os.path.join(args.wiki, f"desc_{raw}.txt"), errors="ignore").readline().strip()[:70]
            print(f"  class {lab} '{classnames[lab]}' -> desc_{raw}: {fl}")
        return

    # write aligned corpus
    os.makedirs(args.out, exist_ok=True)
    for lab, raw in mapping.items():
        shutil.copyfile(os.path.join(args.wiki, f"desc_{raw}.txt"),
                        os.path.join(args.out, f"desc_{lab}.txt"))
    json.dump({str(k): {"raw_id": v, "name": classnames[k]} for k, v in mapping.items()},
              open(os.path.join(args.out, "_realign_map.json"), "w"), ensure_ascii=False, indent=0)

    # verify: classname[L] appears in out/desc_{L}.txt
    ok = checked = 0
    for lab in mapping:
        checked += 1
        txt = open(os.path.join(args.out, f"desc_{lab}.txt"), errors="ignore").read().lower()
        if classnames[lab].lower() in txt:
            ok += 1
    print(f"wrote {len(mapping)} files to {args.out}")
    print(f"verify: classname present in its aligned desc: {ok}/{checked} ({100*ok/checked:.1f}%)")
    print(f"\nswap in with:\n  mv {args.wiki} {ROOT}/wiki_broken && mv {args.out} {args.wiki}")


if __name__ == "__main__":
    main()
