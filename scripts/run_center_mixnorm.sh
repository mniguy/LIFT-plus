#!/bin/bash
#
# PROMPT_CENTER_MIX_NORM -- combine the LEVEL ARMS' OUTPUTS, not their means (2026-08-25).
#
#     out = sum_k  w_k * normalize( O - s * mu_k )
#
# versus the default, which mixes the means first and subtracts once:
#
#     out = O - s * sum_k w_k mu_k
#
# The flag has existed since the shrink family was written and has NEVER BEEN RUN -- no log in
# output/ carries PROMPT_CENTER_MIX_NORM: True. This script is that gap.
#
# ============================ WHY THE PER-LEVEL NORMALIZE IS THE WHOLE IDEA ============================
# Without it the sum is degenerate: because the weights sum to 1, summing the raw per-level
# differences sum_k w_k (O - s mu_k) equals O - s sum_k w_k mu_k exactly, i.e. the default. Row
# normalization is per-row and nonlinear, so it makes the EFFECTIVE weight on a level depend on how
# much that level shrank THAT class -- a level that nearly annihilated a class contributes a unit
# vector just like a level that barely moved it. The mixture weights become class-dependent instead
# of fixed. (Same reason NESTED_RENORM breaks the telescoping identity in run_center_nested_shrink.sh.)
#
# It is also structurally safe against over-centering: every term is a unit vector, so the sum cannot
# be pushed past the origin. Measured across every weighting below, FLIPPED rows (init negatively
# correlated with its own raw prototype) = 0/8142. Compare PROMPT_CENTER_G=1 at s=0.963, where 6390
# of 8142 rows flip.
#
# ============================ WHY NOT ALL SEVEN LEVELS, EQUALLY WEIGHTED ============================
# That is the obvious configuration and it is a dead end. With uniform weights the per-level outputs
# are too similar for the renormalization to re-weight anything, so True and False collapse onto each
# other -- measured per-class cos 0.9925, and BOTH sit at cos 0.99 to sumA, which already ran and
# scored Few 82.27 against single-level genus's 82.90. Screening (s=0.963, MIX_NORM=True), max cosine
# to any arm already run:
#     genus:0.9,global:0.1   0.9962 (that IS the genus arm)   genus:0.8,phylum:0.2   0.9832
#     genus:0.7,family:0.3   0.9650                           genus:0.7,order:0.3    0.9597
#     genus:0.7,global:0.3   0.9595                           genus:0.5,family:0.5   0.8919
#     genus:0.6,family:0.2,global:0.2  0.9368                 genus:0.5,global:0.5   0.8780
# Genus-dominant mixtures are restatements of the 82.90 arm. The novelty is at 50/50.
#
# ============================ THE FOUR ARMS ============================
#   g5_glob5_norm    genus:0.5,global:0.5  True   most independent init in the family (max cos .8780
#                    to anything run). genus is the best single level (82.90); global is the only
#                    level that reaches all 8142 classes, including the 36.8% with no genus relative.
#   g5_fam5_norm     genus:0.5,family:0.5  True   the same slot filled by a TAXONOMY partner instead
#                    of global (max cos .8919). Paired with the above it answers one question:
#                    is genus better partnered with a coarser taxon or with the global centroid?
#                    family is the WORST single level (82.18), so this is not a foregone conclusion.
#   g7_glob3_norm    genus:0.7,global:0.3  True   where the normalization changes the result MOST
#                    (True-vs-False cos 0.9142, the minimum over the screen). Genus-anchored, so it
#                    also asks whether the best arm can be nudged upward rather than replaced.
#   g7_glob3_plain   genus:0.7,global:0.3  False  THE CONTROL for the arm above: identical weights,
#                    normalization off. Without this pair, a win by g7_glob3_norm cannot be
#                    attributed to the normalize rather than to the weighting.
#
# ============================ PREDICTION ============================
# NONE, following run_center_ms2.sh: cos-to-global correlates with All at r = -0.37 across the 15
# arms where both were measured, and the sumA control (78% of rows pointing AWAY from their class)
# scored 80.59 vs an 80.63 baseline. Init geometry does not predict accuracy on iNat.
# BASE RATE: 71 iNat centering arms span All 80.46 - 81.02.  THE NUMBER TO BEAT IS Few 82.90.
#
#   bash scripts/run_center_mixnorm.sh
#   python scripts/agg_runs.py output/center_mixnorm25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
S=${S:-0.963}                        # matches the single-level shrink sweep, so those are the controls
OUT_ROOT=${OUT_ROOT:-"center_mixnorm25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
  PROMPT_CENTER True PROMPT_CENTER_MODE shrink
  PROMPT_CENTER_G 0.0                # MIX_NORM=True requires g=0; kept explicit for both arms
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

run(){   # run <data> <name> <level-spec> <mix_norm>
  local data="$1" name="$2" lvl="$3" mn="$4"
  local out="${OUT_ROOT}/${data}/${name}"
  completed "${out}" && { echo "  [skip] ${out}"; return 0; }
  echo "=== [${data}] ${name}  levels=${lvl} mix_norm=${mn} (${ep} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
    "${COMMON_ARGS[@]}" num_epochs "${ep}" \
    PROMPT_CENTER_S "${S}" PROMPT_CENTER_LEVEL "${lvl}" PROMPT_CENTER_MIX_NORM "${mn}" \
    seed "${SEED}" output_dir "${out}"
}

for data in ${DATASETS}; do
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  run "${data}" g5_glob5_norm  "genus:0.5,global:0.5" True
  run "${data}" g5_fam5_norm   "genus:0.5,family:0.5" True
  run "${data}" g7_glob3_norm  "genus:0.7,global:0.3" True
  run "${data}" g7_glob3_plain "genus:0.7,global:0.3" False
done

echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER shrink] ... mix_norm=.. rows are FLIPPED' line FIRST:"
echo "    FLIPPED must be 0 on every mix_norm=True arm. Nonzero means the unit-vector argument broke."
echo "    Q1 does any arm beat single-level genus shrink (Few 82.90)? cascade 82.50, global 82.13."
echo "    Q2 g7_glob3_norm vs g7_glob3_plain: the normalize alone (inits differ at cos 0.9142)."
echo "    Q3 g5_glob5_norm vs g5_fam5_norm: is genus better partnered with global or with family?"
