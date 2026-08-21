#!/bin/bash
#
# Centering derived from the HIERARCHICAL (ANOVA) DECOMPOSITION of a prompt prototype (2026-08-21).
#
#   O_i = mu_global + sum_{k=1..6} e_k + e_species,     e_k = mu_k - mu_{k-1}
#         (k = kingdom, phylum, class, order, family, genus; mu_0 = global mean)
#
# Verified exactly on the real 8142 iNat prototypes: reconstruction error 8.9e-16, and the components
# are near-orthogonal (max pairwise |cos| = 0.124), i.e. this is a genuine ANOVA decomposition.
# Component sizes as a fraction of ||O|| (mean over classes):
#   mu_global 71.0% | kingdom 16.4% | phylum 7.0% | class 6.9% | order 16.9% | family 24.9%
#   | genus 46.4% | species-specific 15.9%
# phylum and class carry almost nothing, matching the earlier "the level ladder saturates past 3
# levels" observation in run_center_nested1.sh.
#
# KEY READING: r_k = O - mu_k is a PARTIAL SUM of this decomposition -- it keeps every component
# FINER than level k and discards the coarser ones. So the whole "O - mu_LEVEL" family (scripts/
# run_center_res0.sh) was really asking "how many levels of shared structure do I strip?", and
# e_k = r_{k-1} - r_k recovers the individual level effects. A weighted sum of the r_k is therefore
# identical to a weighted sum of the level effects; both are the single linear family
#     out = O - sum_k c_k mu_k .
#
# WHY level=genus (c_genus = 1) BLEW UP, restated: O - mu_genus IS e_species, the pure
# species-specific component. For a genus with one species mu_genus = O, so e_species = 0 -- not a
# numerical accident but the decomposition correctly reporting that the species-specific component is
# UNIDENTIFIABLE there (zero degrees of freedom; nothing to compare the species against). Measured:
# e_species is exactly zero for precisely the 3000 singleton-genus classes, and the genus effect is
# inflated to 46.4% because for those classes it absorbs the species content it cannot separate.
#
# ============================ THE TWO ARMS ============================
#
# (1) sum_all -- ADD UP EVERY RESIDUAL. ZERO free parameters.
#       out = sum_{k=0..6} r_k = 7 * (O - mean_k mu_k)
#     Row normalization removes the scale, so this is FULL-strength centering (alpha = 1), not a
#     weakened one. In decomposition terms (verified to 1.4e-14):
#       sum_k r_k = 7 * ( sum_j (j/7) e_j + e_species )
#     i.e. each level effect is kept on a LINEAR ramp -- kingdom 1/7, phylum 2/7, class 3/7,
#     order 4/7, family 5/7, genus 6/7, species 7/7. Coarse structure removed most, fine least.
#     MEASURED: zero rows 0, min row norm 19.65, cos-to-global 0.9687, top5conf 0.5792.
#     PREDICTION: 80.5 - 80.6, a TIE with mode=global (80.52). The ramp keeps 6/7 of the genus effect,
#     which alone is 46.4% of the norm, so almost nothing is actually removed beyond the global mean.
#     It is run because it is the only arm in this family with no constant to justify.
#
# (2) blend -- ONE knob, shrink every level effect by the same factor s. The telescoping sum folds up:
#       out = O - mu_global - s * sum_k e_k  =  O - (1-s) * mu_global - s * mu_LEVEL
#     "subtract a 1-s : s mixture of the global mean and the genus mean." s=0 IS mode=global
#     (verified bit-identical), s->1 approaches mode=level.
#     SINGLETON-SAFE WITHOUT A BRANCH: if the class is alone in its group then mu_LEVEL = O and the
#     expression collapses to (1-s)(O - mu_global) -- that class simply receives GLOBAL centering.
#     One formula, all classes, no guard, no fallback chain, no zero row.
#     MEASURED (LEVEL=genus), cos-to-global / top5conf / min row norm:
#       s=0.00 1.0000/0.6399/10.69 (=global)   s=0.50 0.9763/0.5646/5.42
#       s=0.75 0.8972/0.4856/2.71              s=0.90 0.7614/0.4502/1.08
#       s=0.92 0.7338/0.4495/0.87  <- in the 0.72-0.75 winning band
#       s=0.95 0.6873/0.4495/0.54  s=1.00 would be mode=level (3000 zero rows) and is REJECTED by the
#                                  config check rather than silently allowed.
#     PREDICTION: s0.92 sits where cascade (0.743 -> 80.84) sits, with a much better top5conf
#     (0.4495 vs ~0.60). PREDICT 80.7 - 81.0. Beating g_bottomup_fo (81.02) would be a new result.
#
# ============================ RELATION TO THE OTHER NEW MODE ============================
# blend s=0.92 and taxo_kernel gamma=0.03 (scripts/run_center_taxokernel.sh) both land in the band but
# are NOT the same arm -- measured per-class cos 0.9054. blend mixes only TWO means (global + genus),
# taxo_kernel mixes ALL levels by taxonomic distance. That difference is the question these two
# scripts jointly answer: do the intermediate levels (family, order) carry anything, or is
# global-vs-genus the whole story? Cross-cosines measured on the real prototypes:
#     taxo_kernel(0.03) x blend(0.92) = 0.9054   x cascade = 0.8900
#     blend(0.92)       x cascade     = 0.8165
# A geometric alternative that was MEASURED AND DROPPED: "O - mu_global - sum_k beta^k e_k" at
# beta=0.98 gives cos-to-global 0.7440, but it is per-class cos 0.9865 to blend s=0.92 -- the same arm
# with a far worse parameterization (all of its action is squeezed into beta in [0.95, 1.0]). Not queued.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50
#   g_bottomup_fo (center_nested25)              81.02  75.73  80.79  82.69   <- the number to beat
#   iNat seed noise (5-ep/scale-30 proxy): All ~0.06, Head ~0.74, Med ~0.16, Few ~0.23.
#   BASE RATE: 59 iNat centering arms measured so far span 80.46-81.02 (0.56). Expect a tie.
#
# NOTE: yacs is type-strict, so s MUST be written with a decimal point ("0.92", not "1").
#
# ============================ ADDED 2026-08-21: shrink, and multi-level blend ============================
# ARM SYNTAX: "sum_all" | "s<S>" = blend | "sh<S>" = shrink. LEVEL may be a comma-separated list for
# blend/shrink (the weight S is split evenly over the listed levels).
#
# (3) shrink -- out = O - s * mu_LEVEL, i.e. blend WITHOUT the global term. s=0.5 reproduces
#     mode=level_keep exactly (verified per-class cos 1.00000000, already run: 80.59), s=1 is
#     mode=level. Zero-row-safe for s<1, but a class alone in its group collapses to (1-s)*O, i.e. the
#     RAW uncentered prototype -- it receives NO centering at any s. Measured cos to the raw direction,
#     singleton vs non-singleton, LEVEL=genus:
#         s=0.50  1.0000/0.9747   s=0.80  1.0000/0.8105   s=0.90  1.0000/0.6196   s=0.98  1.0000/0.3464
#     versus blend s=0.92 (0.7198/0.4149), global (0.7198/0.7140), sum_all (0.6888/0.6904).
#     So raising s does not strengthen the method uniformly; it SPLITS the init into an untouched 37%
#     and a heavily centered 63%. The trainer logs "N rows are UNCENTERED" so this is visible per run.
#
# THE CONTROLLED PAIR: shrink s=0.963 matches blend s=0.92 on the non-singleton classes (cos-to-raw
# 0.4141 vs 0.4149) while leaving the 3000 singletons at exactly 1.0000. The two inits are per-class
# cos 0.8851. Running both isolates ONE question that has never been tested separately:
#     should a class with no relatives receive GLOBAL centering, or nothing at all?
# blend s=0.92 already ran (80.64), so only the shrink arm is outstanding.
#
# MULTI-LEVEL BLEND: listing several levels turns the flat keep-profile into a staircase. Measured
# keep-profiles at s=0.92 (kingdom..genus, species always 1.0):
#     genus              0.080 0.080 0.080 0.080 0.080 0.080
#     family,genus       0.080 0.080 0.080 0.080 0.080 0.540
#     order,family,genus 0.080 0.080 0.080 0.080 0.387 0.693
#     all six            0.080 0.233 0.387 0.540 0.693 0.847
#     (sum_all           0.143 0.286 0.429 0.571 0.714 0.857)
# The profile is always monotone (C_j is a tail sum of non-negative weights), so fine effects are
# always kept at least as much as coarse ones, and e_species is kept in full for every choice.
# REDUNDANCY, measured: "all six" is per-class cos 0.9997 to sum_all -- the SAME arm, already run
# (80.68). "order,family,genus" is 0.9722 to sum_all. Only "family,genus" is genuinely off every
# curve already run (<= 0.94 to all of them). kingdom/phylum/class as single levels are mutually
# cos 0.98-0.99, i.e. one arm in triplicate.
#   => of this whole family, exactly two arms are worth GPU: blend LEVEL="family,genus", and
#      shrink s=0.963.
#
#   ARMS="sh0.963" bash scripts/run_center_decomp.sh                       # the singleton-policy control
#   ARMS="s0.92" LEVEL="family,genus" bash scripts/run_center_decomp.sh    # the one new keep-profile
#
#   bash scripts/run_center_decomp.sh                          # sum_all + blend s=0.92
#   ARMS="s0.5 s0.75 s0.92 s0.95" bash scripts/run_center_decomp.sh   # the s axis
#   LEVEL=family bash scripts/run_center_decomp.sh             # blend against the family mean instead
#   python scripts/agg_runs.py output/center_decomp --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
ARMS=${ARMS:-"sum_all s0.92"}
LEVEL=${LEVEL:-genus}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_decomp"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for arm in ${ARMS}; do
  case "${arm}" in
    sum_all) extra=(PROMPT_CENTER_MODE sum_all); tag="sum_all" ;;
    sh*)     sv="${arm#sh}"
             case "${sv}" in *.*) ;; *) echo "ERROR: s must have a decimal point (yacs is type-strict): '${sv}'"; exit 1;; esac
             extra=(PROMPT_CENTER_MODE shrink PROMPT_CENTER_S "${sv}" PROMPT_CENTER_LEVEL "${LEVEL}")
             tag="shrink_$(echo "${LEVEL}" | tr ',' '-')_s$(echo "${sv}" | tr -d '.')" ;;
    s*)      sv="${arm#s}"
             case "${sv}" in *.*) ;; *) echo "ERROR: s must have a decimal point (yacs is type-strict): '${sv}'"; exit 1;; esac
             extra=(PROMPT_CENTER_MODE blend PROMPT_CENTER_S "${sv}" PROMPT_CENTER_LEVEL "${LEVEL}")
             tag="blend_$(echo "${LEVEL}" | tr ',' '-')_s$(echo "${sv}" | tr -d '.')" ;;
    *) echo "unknown arm ${arm}"; exit 1 ;;
  esac
  out="${OUT_ROOT}/inat2018/${tag}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  echo "=== [inat2018] ${arm} (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True "${extra[@]}" \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    every arm must log '0/8142 rows are ZERO'; anything else means the construction is broken."
echo "    sum_all is the zero-constant arm -- predicted to TIE global 80.52; that is the point of it."
echo "    blend s0.92 vs cascade 80.84/75.81/80.57/82.50 is the headline comparison, and vs"
echo "    taxo_kernel g003 it answers whether family/order carry anything beyond global-vs-genus."
