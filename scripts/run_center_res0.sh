#!/bin/bash
#
# PROMPT_CENTER_MODE=level -- single-taxonomy-level centering with NO fallback and NO min_size gate.
#
#   out = O - mean(O over the class's group at LEVEL)          [renormalized, as always]
#
# Every taxonomy mode so far (genus / cascade / nested) guards small groups: a group below
# PROMPT_CENTER_GENUS_MIN falls back to the global centroid (genus/cascade) or is skipped (nested).
# The guard exists because a SINGLETON group's mean is the class itself, so O - mu is exactly 0.
# This script removes the guard on purpose, to measure what the guard was buying -- i.e. is landing
# on the zero vector actually harmful, or was the fallback machinery solving a non-problem?
#
# 6 arms: genus, family, order, class, phylum, kingdom.
# LEVEL=global is NOT run here: global never had a fallback, so mode=level with LEVEL=global is
# bit-identical to PROMPT_CENTER_MODE=global (verified), and that number already exists (80.52).
#
# ============================ WHAT THE ZERO ROWS COST (categories.json census) ============================
#   level     groups   classes in a SINGLETON group -> classifier rows that init to exactly 0
#   genus      4401    3000 / 8142   (36.8%)   <- the arm where this is a real intervention
#   family     1118     463 / 8142   ( 5.7%)
#   order       272      64 / 8142   ( 0.8%)
#   class        57       9 / 8142
#   phylum       25       5 / 8142
#   kingdom       6       1 / 8142
# F.normalize leaves a zero row at zero, so with CosineClassifier those classes start with a logit
# that is identically 0 for every image -- a dead init that training has to recover from scratch.
# The trainer prints the exact zero-row count per run ("[PROMPT_CENTER level] ... rows are ZERO");
# read that line before reading the accuracies.
#
# PRE-REGISTERED PREDICTION: genus should be the only arm that visibly moves, and downward -- 37% of
# classes losing their init is far more damage than any centering variant has done on this dataset.
# family and coarser have <6% zero rows and should land near their guarded counterparts. If genus
# does NOT lose, the fallback machinery in genus/cascade/nested was never load-bearing, which is the
# more interesting outcome of the two.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   genus, guarded min_size=5 (center_local25)   80.46  75.22  80.05  82.34   <- the guarded twin of arm 'genus'
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50
#   iNat seed noise (5-ep/scale-30 proxy): All ~0.06, Head ~0.74, Med ~0.16, Few ~0.23.
#
#   bash scripts/run_center_res0.sh                       # all 6 arms (~6 x 15 ep)
#   ARMS="genus family" bash scripts/run_center_res0.sh   # the two that can move
#   python scripts/agg_runs.py output/center_res0 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
ARMS=${ARMS:-"genus family order class phylum kingdom"}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_res0"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for lv in ${ARMS}; do
  out="${OUT_ROOT}/inat2018/${lv}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  echo "=== [inat2018] level (O - mu) at ${lv}, no fallback (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE level PROMPT_CENTER_LEVEL "${lv}" \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER level] ... rows are ZERO' line FIRST -- that count is the"
echo "    size of the intervention (genus 3000, family 463, order 64, class 9, phylum 5, kingdom 1)."
echo "    genus (unguarded) vs the guarded genus arm 80.46/75.22/80.05/82.34 is the headline pair."
echo "    Companion: scripts/run_center_res1.sh, same levels with out = 2*O - mu (no zero rows)."
