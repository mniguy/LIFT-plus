"""Build datasets/ImageNet_LT/categories.json (the WordNet hierarchy) so PROMPT_CENTER_MODE=cascade
works on ImageNet-LT, in the same schema _load_taxonomy() already reads for iNat.

Levels are HOPS FROM THE LEAF (h1 = the class's own hypernym, h2 = its hypernym, ...), not depth
from the root: ImageNet's WordNet depth ranges from 4 to 18 hops, so an absolute depth would mean a
wildly different granularity per class. Cascade's deepest-first fallback absorbs the unevenness.

Chose raw WordNet over the BREEDS calibrated hierarchy after measuring both on all 1000 classes:
BREEDS leaves 110/1000 classes out of its tree entirely (tusker, altar, barrel, bathtub, ...) and,
being only ~6 levels deep, collapses 23.5% of classes into >200-member groups by hop 4 and 78.9% by
hop 5 -- groups so large their mean is effectively the global mean. WordNet keeps 0% in >200-member
groups through hop 6 while reaching 95.3% coverage, and the resulting cascade init is geometrically
identical anyway (overall off-diag cos 0.0006 vs -0.0004, top5-confusion 0.335 vs 0.337).

Needs WordNet 3.0's data.noun (ImageNet wnids ARE WordNet 3.0 noun offsets). Either:
    python -c "import nltk; nltk.download('wordnet')"    # -> ~/nltk_data/corpora/wordnet/data.noun
    # or unzip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip

Usage:
    python scripts/make_imagenet_taxonomy.py --wordnet ~/nltk_data/corpora/wordnet
"""
import argparse, json, os
from collections import defaultdict

LEVELS = 8          # h1..h8; h1-h6 are the useful range, h7+ is near the WordNet root
MIN_SIZE = 5        # only for the coverage report -- matches PROMPT_CENTER_GENUS_MIN


def parse_wordnet(wn_dir):
    """data.noun -> (offset -> first hypernym offset, offset -> lemma)."""
    hypernym, lemma = {}, {}
    with open(os.path.join(wn_dir, "data.noun"), encoding="latin-1") as f:
        for line in f:
            if line.startswith("  "):          # licence header
                continue
            tok = line.partition("|")[0].split()
            off = tok[0]
            w_cnt = int(tok[3], 16)
            lemma[off] = tok[4]
            i = 4 + 2 * w_cnt
            p_cnt = int(tok[i]); i += 1
            for _ in range(p_cnt):
                sym, target, pos = tok[i], tok[i + 1], tok[i + 2]
                i += 4
                if sym in ("@", "@i") and pos == "n" and off not in hypernym:
                    hypernym[off] = target      # first hypernym: WordNet allows several
    return hypernym, lemma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wordnet", required=True, help="dir containing WordNet 3.0 data.noun")
    ap.add_argument("--out", default="datasets/ImageNet_LT/categories.json")
    args = ap.parse_args()

    # label -> wnid comes from the split file's image paths (train/n01440764/...). NOT from
    # id_to_name.json, whose key order is the original ILSVRC order, not the label order.
    lab2wnid = {}
    with open("datasets/ImageNet_LT/train.txt") as f:
        for line in f:
            path, lab = line.split()
            lab2wnid.setdefault(int(lab), path.split("/")[1])
    classnames = [l.strip() for l in open("datasets/ImageNet_LT/classnames.txt")]
    assert len(lab2wnid) == len(classnames), (len(lab2wnid), len(classnames))

    hypernym, lemma = parse_wordnet(args.wordnet)

    records, chains = [], []
    for i, name in enumerate(classnames):
        wnid = lab2wnid[i]
        cur, chain, seen = wnid[1:], [], set()
        while cur in hypernym and cur not in seen:
            seen.add(cur)
            cur = hypernym[cur]
            chain.append(f"{lemma.get(cur, '?')}.n.{cur}")   # readable AND unique (offset)
        chains.append(chain)
        rec = {"name": name, "wnid": wnid}
        for k in range(1, LEVELS + 1):
            if k <= len(chain):
                rec[f"h{k}"] = chain[k - 1]
        records.append(rec)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=1)

    C = len(classnames)
    print(f"wrote {args.out}: {C} classes, chain length min {min(len(c) for c in chains)} "
          f"max {max(len(c) for c in chains)}")
    # _load_taxonomy keys records by classname, so duplicated names collapse to one record.
    # ImageNet-LT has two such pairs (missile n03773504/n04008634, sunglasses n04355933/n04356056);
    # both members of a pair also get the SAME prompt, hence the same prototype, so the shared
    # taxonomy record changes nothing that was not already tied. Reported, not worked around.
    dupes = {n for n in classnames if classnames.count(n) > 1}
    if dupes:
        print(f"note: {len(dupes)} duplicated classname(s) share one record: {sorted(dupes)}")
    print(f"{'level':>6s} {'groups':>7s} {'median':>7s} {'max':>5s} {'cover>=5':>9s} {'in group>200':>13s}")
    for k in range(1, LEVELS + 1):
        groups = defaultdict(int)
        for c in chains:
            if k <= len(c):
                groups[c[k - 1]] += 1
        if not groups:
            continue
        sizes = sorted(groups.values())
        cover = sum(v for v in groups.values() if v >= MIN_SIZE)
        big = sum(v for v in groups.values() if v > 200)
        print(f"{'h'+str(k):>6s} {len(groups):7d} {sizes[len(sizes)//2]:7d} {sizes[-1]:5d} "
              f"{100*cover/C:8.1f}% {100*big/C:12.1f}%")
    print("\nexample:", json.dumps(records[0]))


if __name__ == "__main__":
    main()
