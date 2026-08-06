#!/bin/bash
#
# D) PROMPT_CENTER_MODE=genus_lex -- surgical lexical fix for the genus spike (2026-08-06 brainstorm).
#
# Motivation: we proved the iNat genus residual (raw within-genus cosine 0.835) is almost entirely a
# TEXT artifact, not real biological similarity -- re-encoding with the genus word stripped (species
# epithet alone, e.g. "alba" instead of "Quercus alba") collapses it to 0.026 (tables_cascade.tex,
# tab:epithet). Plain 'genus' mode still subtracts the group's FULL-embedding mean, which conflates
# (a) the repeated genus word with (b) whatever genuine within-genus content those species happen to
# share beyond the word. genus_lex isolates (a) specifically: encode each class TWICE (full binomial
# name, and species-epithet-only), take the per-class difference diff_i = embed(full_i) - embed(epithet_i)
# as that class's estimate of "what the genus word alone contributes" to its embedding, then subtract
# the GENUS-AVERAGE of that diff vector instead of the genus-average of the full embedding. Same
# PROMPT_CENTER_GENUS_MIN(5) fallback-to-global-mu guard as plain 'genus'.
#
# OFFLINE PLUMBING CHECK (2026-08-06, GPU-free, string/shape logic only -- the actual diff vectors
# need the real CLIP text encoder, only exercised on the GPU box): 0/8142 iNat classnames are
# single-token (every one splits cleanly into genus+epithet, so the "no epithet" fallback path in the
# code is never actually hit on this dataset); genus grouping/fallback count (5863/8142 fall back at
# min_size=5) matches plain 'genus' mode exactly, confirming the grouping logic itself is unchanged --
# only WHAT gets averaged and subtracted differs.
#
# HONEST UNCERTAINTY (stated before running, not after): plain 'genus' already showed the spike is
# ~97% explained by the repeated word (0.835 -> 0.026 on removal), so genus_lex's diff-vector mean and
# genus's full-embedding mean may end up numerically close for most genera, in which case this run
# should land near genus's own numbers (80.46/75.22/80.05/82.34) rather than clearly beating them. A
# genuinely different result (either direction) would mean the two are NOT interchangeable in
# practice, which is itself the informative outcome either way.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   genus    (center_local25/inat2018/genus)     80.46  75.22  80.05  82.34
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50
#
#   bash scripts/run_center_genuslex.sh
#   GENUS_MIN="5 10 20" bash scripts/run_center_genuslex.sh   # sweep the min-group-size gate too
#   python scripts/agg_runs.py output/center_genuslex25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
GENUS_MIN=${GENUS_MIN:-"5"}
INAT_EPOCHS=${INAT_EPOCHS:-15}
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"center_genuslex25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
  PROMPT_CENTER True PROMPT_CENTER_MODE genus_lex
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "${data}" != "inat2018" ]; then
    echo "  [skip dataset] ${data}: genus_lex needs binomial 'Genus species' classnames (iNat only)"
    continue
  fi
  ep=${INAT_EPOCHS}
  for gm in ${GENUS_MIN}; do
    out="${OUT_ROOT}/${data}/min${gm}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] genus_lex min_size=${gm} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${COMMON_ARGS[@]}" num_epochs "${ep}" PROMPT_CENTER_GENUS_MIN "${gm}" \
      seed "${SEED}" output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    check each log's '[PROMPT_CENTER genus_lex] ... mean|diff|=...' line: a mean|diff| near 0"
echo "    would mean the epithet-stripped encoding barely differs from the full one for this backbone,"
echo "    i.e. the isolation step is a no-op and results SHOULD match plain genus closely."
echo "    compare against genus (80.46/75.22/80.05/82.34) -- the whole point of this run."
