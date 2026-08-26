#!/bin/bash
#
# The three best SINGLE levels, mixed (2026-08-26).   mode=shrink, no renorm anywhere.
#
#     out = O - s * sum_k w_k mu_k          w_k uniform, s = 0.963
#
# Motivation: on the single-level shrink sweep (s=0.963, iNat Few) the ranking was
#     genus 82.90 > phylum 82.64 > order 82.51 > kingdom 82.49 > class 82.30 > global 82.29
#     > family 82.18,  against cascade 82.50 and plain global 82.13.
# genus, phylum and order are the top three. This script mixes exactly those, nothing else.
#
# ============================ READ THE SCREEN BEFORE SPENDING GPU TIME ============================
# Measured on the real prototypes, per-class cosine of each mix to arms ALREADY RUN:
#
#   mix                    genus   order  phylum    sumA  cascade  global  g5glob5     MAX
#   genus,order           0.6791  0.9765  0.9269  0.9650   0.8318  0.8960   0.9009  0.9765
#   genus,phylum          0.6703  0.9364  0.9777  0.9833   0.8125  0.9426   0.9220  0.9833
#   genus,order,phylum    0.6185  0.9768  0.9810  0.9952   0.8128  0.9454   0.8936  0.9952
#
# Two things follow and neither is comfortable:
#   * genus,order,phylum is cos 0.9952 to sumA, which ALREADY RAN and scored Few 82.27 / All 80.59.
#     Expect that arm to reproduce it. It is kept because a 0.995 twin landing somewhere else would
#     itself be informative, but it is the first one to cut if GPU time is short.
#   * NONE of the three is close to the genus arm they were meant to build on (0.62 - 0.68). Uniform
#     weights do NOT combine the three levels' effects. The level MEANS have wildly different
#     magnitudes -- order and phylum means are essentially the global centroid (|mu| 7.26) while a
#     genus mean is small -- so an equal-weight sum is swallowed by the coarse terms and the genus
#     contribution is diluted to 1/k. That is why every mix leans coarse.
# The three mixes are also 0.96 - 0.99 to EACH OTHER, i.e. closer to one arm run three times than to
# three independent arms.
#
# ============================ IF YOU WANT THE GENUS-ANCHORED VERSION ============================
# Weighting genus up pulls away from sumA but straight into g5glob5 (Few 82.31), so there is no
# escape from this neighbourhood; the least-overlapping point measured anywhere in the family is
#     WEIGHTS="genus:0.7,order:0.3"      max cos 0.9351 (to g5glob5)
#     WEIGHTS="genus:0.8,order:0.2"      max cos 0.9473
# Run those instead with:  WEIGHTS="genus:0.7,order:0.3 genus:0.8,order:0.2" bash scripts/run_center_top3levels.sh
#
# ============================ CONTEXT ============================
# Every attempt to COMBINE levels has so far lost to using genus alone:
#     single genus 82.90 | cascade 82.50 | sumB 82.50 | g7_glob3_plain 82.37 | g7_glob3_norm 82.34
#     | g5_glob5_norm 82.31 | global 82.29 | sumA 82.27 | g5_fam5_norm 82.14
# Four different combination schemes (hierarchical fallback, mean mixture, output mixture with
# per-level renormalization, weighted mixture) all land in 82.1 - 82.5. This script asks whether
# restricting the mixture to the three levels that individually worked changes that.
#
# NO PREDICTION, per run_center_ms2.sh: cos-to-global correlates with All at r = -0.37 across the 15
# arms where both were measured. Init geometry does not predict accuracy on iNat.
# BASE RATE: 71 iNat centering arms span All 80.46 - 81.02.   THE NUMBER TO BEAT IS Few 82.90.
#
#   bash scripts/run_center_top3levels.sh
#   python scripts/agg_runs.py output/center_top3lv25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
S=${S:-0.963}                        # matches the single-level sweep, so those are the controls
WEIGHTS=${WEIGHTS:-"genus,order genus,order,phylum genus,phylum"}
OUT_ROOT=${OUT_ROOT:-"center_top3lv25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
  PROMPT_CENTER True PROMPT_CENTER_MODE shrink
  PROMPT_CENTER_MIX_NORM False        # no renorm anywhere: plain mean mixture, one subtraction
  PROMPT_CENTER_G 0.0
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for spec in ${WEIGHTS}; do
    name=$(echo "${spec}" | tr ',:' '_-')
    out="${OUT_ROOT}/${data}/${name}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] ${name}  levels=${spec} s=${S} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${COMMON_ARGS[@]}" num_epochs "${ep}" \
      PROMPT_CENTER_S "${S}" PROMPT_CENTER_LEVEL "${spec}" \
      seed "${SEED}" output_dir "${out}"
  done
done

echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER shrink] ... rows are UNCENTERED' line FIRST: a level with"
echo "    singleton groups leaves those classes at the raw prototype (family did this to 461 rows)."
echo "    Q1 does any mix beat single-level genus (Few 82.90)? five combination schemes have not."
echo "    Q2 genus_order_phylum vs sumA (82.27 / All 80.59): the inits are cos 0.9952, so a gap"
echo "       there is run-to-run variation, not a finding. Use it to size the noise floor."
