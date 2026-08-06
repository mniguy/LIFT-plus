#!/bin/bash
#
# D+cascade hybrid) PROMPT_CENTER_MODE=cascade_lex (2026-08-06 brainstorm, follow-up to genus_lex).
#
# Motivation: standalone genus_lex compares against plain 'genus' (80.46/75.22/80.05/82.34), but
# 'genus' alone is not the paper's best iNat result -- 'cascade' (genus->family->order fallback,
# 80.84/75.81/80.57/82.50) is. genus_lex's surgical diff-vector subtraction (isolating the repeated
# genus WORD's own contribution instead of the raw group mean) only makes sense at the genus level --
# family/order names ("Fagaceae") never appear inside a classname, so there is no analogous
# epithet-style split for them. cascade_lex plugs the diff-vector subtraction into ONLY the genus
# level of cascade's existing fallback chain, leaving family/order exactly as in plain cascade, so
# this is the fair, apples-to-apples comparison against the paper's actual best number.
#
# OFFLINE PLUMBING CHECK (2026-08-06, GPU-free, string/shape logic only -- the real diff vectors need
# the CLIP text encoder, only exercised on the GPU box): coverage genus(lex)=2279 family=4427
# order=1068 global=368 -- matches plain cascade's coverage EXACTLY (same source/tables_cascade.tex
# assignment numbers), confirming the assignment logic is unchanged; only the genus level's SOURCE
# tensor (diff vs X) differs.
#
# HONEST UNCERTAINTY (stated before running): same caveat as genus_lex -- if the diff-vector mean and
# the full-embedding mean are numerically close for most genera (plausible, since the genus word was
# already shown to explain ~97% of the raw spike), cascade_lex should land close to plain cascade's
# numbers. The informative outcome either way: convergence would mean cascade's genus pass was
# already "mostly lexical" in effect even without this surgery; a real divergence would mean the
# genus-mates' non-lexical shared content (biology beyond the shared word) was doing more work in
# plain cascade than expected.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   genus    (center_local25/inat2018/genus)     80.46  75.22  80.05  82.34
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50   <- the number to beat
#
#   bash scripts/run_center_cascadelex.sh
#   python scripts/agg_runs.py output/center_cascadelex25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_cascadelex25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

out="${OUT_ROOT}/inat2018/cascade_lex"
if completed "${out}"; then
  echo "  [skip] ${out}"
else
  echo "=== [inat2018] cascade_lex (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE cascade_lex \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
fi
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    check the log's '[PROMPT_CENTER cascade_lex] ... genus(lex)=...' coverage line matches"
echo "    plain cascade's genus=2279 family=4427 order=1068 global=368 (it should, exactly)."
echo "    compare against cascade (80.84/75.81/80.57/82.50) -- the whole point of this run."
