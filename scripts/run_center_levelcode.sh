#!/bin/bash
#
# Taxonomy-level chains addressed by DIGIT CODE (2026-08-27; no-global grid 2026-08-31).
#
#     0 global   1 kingdom   2 phylum   3 class   4 order   5 family   6 genus
#
# A code is the chain, read left to right, coarse to fine:
#     6        subtract the genus mean only
#     06       global centering, then genus centering on the residual
#     0123456  every level, top-down
# Each step is  X <- X - mu_level(X),  the mean recomputed on the running residual
# (PROMPT_CENTER_MODE=nested, NESTED_MEAN=recompute). Plain centering: the full mean.
#
# ============================ SINGLE DIGITS ARE IN THE GRID ============================
# 1-6 are run here under mode=nested, the same code path as every multi-level code, so the whole
# table is one mode. Code 0 (global alone) is left out: it is already known.
#
# mode=shrink ran the same FORMULA at these levels (s=0.963) and those numbers are the reference,
# but they came from a different mode, so they are not this grid's baseline -- these runs are:
#     code  level     All     Few          (output/center_shrink/inat2018, mode=shrink)
#      0    global   80.58   82.29
#      1    kingdom  80.71   82.49
#      2    phylum   80.75   82.64
#      3    class    80.65   82.30
#      4    order    80.80   82.51
#      5    family   80.58   82.18
#      6    genus    80.85   82.90
#
# NOTE what a single level does to a class ALONE in its group: there is no size gate anywhere in this
# script, and none is needed -- the expression degenerates on its own. That class's group mean IS its
# own prototype, so O - s*O = (1-s)*O, which the trailing normalize turns back into the RAW direction.
# It receives NO centering at all (it does not fall back to a coarser level -- that is what cascade's
# GENUS_MIN did, and this family has no fallback). Verified, rows identical to raw vs singleton count:
#     code 1 kingdom 1/1   2 phylum 5/5   3 class 9/9
#     code 4 order 64/64   5 family 463/463   6 genus 3000/3000
# So code 6 alone leaves 36.8% of iNat uncentered. Prefixing 0 fixes it: global is one group over all
# 8142 classes, so it has no singletons and reaches every class.
#
# ============================ renorm IS OFF BY DEFAULT ============================
# RENORM=False here. What renorm does, measured on this exact setting (s=0.963, gate 1):
#     chain                renorm off   renorm on    delta
#     global,genus         80.62        80.54        -0.08   (2 levels: nothing after genus to protect)
#     global,genus,family  80.67        81.02        +0.35   (3 levels: family's mean gets fixed)
# It only has a job when a LATER level still has to compute a mean: it stops the post-centering norm
# spread (2.2x -> 13.2x after the genus step) from letting barely-centered classes dominate the next
# level's mean. So the +0.35 above is real and this grid gives it up on purpose, to keep the codes
# comparable to the single-level runs and to the rest of the nested family. Flip it per run with
# RENORM=True if the shape results argue for it.
#
# COST OF LEAVING IT OFF, measured (final residual norm before the trailing normalize):
#     code      min |row|   rows <1e-4   rows <1e-6
#     012        7.5e-03        0            0
#     034        5.3e-03        0            0
#     056        4.9e-03        0            0
#     0123       2.8e-04        0            0
#     0135       2.8e-04        0            0
#     0246       2.0e-04        0            0
#     0123456    1.4e-08        9            5      <- the only code at risk
# A class alone in EVERY group of the chain is multiplied by (1-s) at each level, so seven levels
# give (1-0.963)^6 = 2.6e-9. iNat has such classes (Bacteria holds exactly one species, alone from
# kingdom all the way down). The trailing F.normalize does NOT produce zero rows or NaN -- it rescales
# them back to unit length -- but their DIRECTION is float noise, so those 5-9 classes out of 8142
# start from a random classifier row. Small, but it is a defect only code 0123456 has, and it is the
# one arm where RENORM=True is worth using anyway.
#
# ============================ THE GRID (2026-08-31): NO-GLOBAL CHAINS ============================
#   12  34  135  123      stop before genus
#   56  246  456  123456  reach genus
# Eight codes, none with a leading 0. That is NOT simply "centering without global" -- read the
# telescoping note below before interpreting a single number here.
#
# THE LEADING 0 CANCELS. With NESTED_MEAN=recompute and renorm off, each level's mean is taken on
# the RUNNING RESIDUAL, so for any class COVERED at the first non-global level L:
#     X1 = X - mean(X);   shift = mu_L(X1) = mu_L(X) - mean(X);   X1 - shift = X - mu_L(X)
# the global term drops out exactly. (trainer.py: src = X_out when mean_mode == "recompute".)
# So c0L... and cL... build the SAME prototypes EXCEPT for classes the GENUS_MIN=2 gate SKIPS at L:
# a skipped class gets shift=0 and therefore KEEPS the global centering it would otherwise lose.
# Measured on the 2026-08-31 launch, final residual norms:
#     c12  15.321  vs c012  15.321   (identical -- kingdom skips 1 class)
#     c34  14.426  vs c034  14.418
#     c246  8.979  vs c0246  8.974
#     c56   9.286  vs c056   8.824   <- the big one: family skips 463 classes
# Skips at each level, of 8142: kingdom 1, phylum 5, class 9, order 64, family 463, genus 3000.
#
# SO THIS SET BUYS TWO THINGS AT ONCE:
#   1. what the global step is worth to exactly the classes a level SKIPS (c56/c056 is the probe;
#      c12/c012 differs on ~1 class and should be indistinguishable from noise)
#   2. a NOISE FLOOR from seven near-replicate pairs -- far better than the single cosine-1.0
#      pair the estimate has rested on until now
#
# ============================ HOW TO READ THE RESULT ============================
# BASE RATE: 71 iNat centering arms span All 80.46 - 81.02.  Best ever is g_bottomup_ms2 at 81.24;
# best in this grid so far is c0123456_rnT at 81.11.
# NOISE FLOOR, measured on this grid: two arms whose inits are cosine 1.0000000000 apart (global,genus
# vs genus,global, identical to float32 rounding) scored All 80.62 vs 80.62, but Head 75.26 vs 74.74;
# a 13-run regression left residual SD 0.082. So differences under ~0.1 in All mean nothing, and
# HEAD carries +-0.5 of pure noise and must not be interpreted. Rank on All.
#
# WHAT THE FIRST 17 RUNS ACTUALLY SHOWED (2026-08-29) -- read before adding arms:
#   - The one variable that predicts All is BINARY: does the chain REACH GENUS.
#         reaches genus  n=8  All 80.968 (sd 0.106)
#         does not       n=9  All 80.707 (sd 0.082)      diff +0.261, t=5.69 (df=15)
#     R2 0.684, beating residual norm 0.630 and chain length 0.365.
#   - RENORM=True IS A DEAD KNOB. Paired dAll: +0.18 (0123456), +0.06 (056), -0.07 (0246);
#     mean +0.057 against a 0.082 noise floor, and the rnF/rnT rankings of the same four arms
#     ANTI-correlate (r=-0.28). It moves the pre-norm residual norm 9x while the init the model
#     sees barely moves, because the classifier is F.normalize'd regardless. The "+0.35 on three
#     levels" recorded above did NOT replicate -- treat that older number as superseded.
#   - RESIDUAL NORM IS NOT CAUSAL. Within genus-reaching arms r=-0.416 (ns); within non-genus
#     r=-0.236 (ns). Its -0.83 across all arms was a proxy for "reaches genus", nothing more.
#   - THE ROBUST MECHANISM is a training-curve crossover at epoch 4->5: corr(train acc, test All)
#     is -0.71 at ep1 and +0.85 from ep5 onward (n=17). Chains that centre harder start much worse
#     and finish better -- centering redirects the optimisation, it is not an init trick that
#     washes out. This is why iNat's 15 epochs can afford aggressive chains and a short one cannot.
# NO GEOMETRIC PREDICTION is offered: run_center_ms2.sh measured cos-to-global correlating with All
# at r = -0.37 across 15 arms, and a control with 78% of rows pointing away from their own class
# still scored 80.59 against an 80.63 baseline.
#
#   bash scripts/run_center_levelcode.sh
#   ARMS="56 246" bash scripts/run_center_levelcode.sh
#   RENORM=True SUFFIX=_rnT ARMS="56 246" bash scripts/run_center_levelcode.sh
#   python scripts/agg_runs.py output/center_levelcode25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
RENORM=${RENORM:-False}   # see "renorm IS OFF BY DEFAULT" above
ARMS=${ARMS:-"12 34 56 135 246 123 456 123456"}
SUFFIX=${SUFFIX:-""}       # appended to the output dir, e.g. SUFFIX=_rnT for a RENORM=True set.
                          # Without it a RENORM=True rerun lands on top of the RENORM=False
                          # result for the same code, because the dir is bare c<code>.
OUT_ROOT=${OUT_ROOT:-"center_levelcode25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

LEVELS=(global kingdom phylum class order family genus)

expand(){   # digit code -> comma-separated level chain, in the order written
  local code="$1" out="" i c
  for (( i=0; i<${#code}; i++ )); do
    c="${code:$i:1}"
    case "$c" in
      [0-6]) out="${out:+${out},}${LEVELS[$c]}" ;;
      *) echo "ERROR: bad digit '$c' in code '$code' (valid: 0-6)" >&2; return 1 ;;
    esac
  done
  echo "$out"
}

COMMON_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
  PROMPT_CENTER True PROMPT_CENTER_MODE nested
  PROMPT_CENTER_NESTED_MEAN recompute
  PROMPT_CENTER_GENUS_MIN 2          # a group of 1 is SKIPPED at that level (ms2), not zeroed
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for code in ${ARMS}; do
    chain=$(expand "${code}") || exit 1
    out="${OUT_ROOT}/${data}/c${code}${SUFFIX}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] code ${code} = ${chain} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${COMMON_ARGS[@]}" num_epochs "${ep}" \
      PROMPT_CENTER_NESTED_LEVELS "${chain}" \
      PROMPT_CENTER_NESTED_RENORM "${RENORM}" \
      seed "${SEED}" output_dir "${out}"
  done
done

echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER nested] ... kingdom(|mu|=..) phylum(|mu|=..) ...' line FIRST:"
echo "    a level with |mu| ~ 0 did no work, so that code is really a shorter code. phylum and class"
echo "    are the ones to watch -- 14 of 25 phyla contain a single class, so they branch ~1:1."
echo "    Q1 does the leading 0 matter? pair each cX against c0X --"
echo "       12/012 34/034 56/056 135/0135 246/0246 123/0123 456/0456 123456/0123456."
echo "       Only classes SKIPPED at the first level can differ (see the telescoping note above),"
echo "       so 56 vs 056 (463 skipped) should move and 12 vs 012 (1 skipped) should not."
echo "    Q2 sd of those paired deltas IS a noise floor -- 7 near-replicates, the best estimate yet."
echo "    Q3 among genus-reaching chains (56 246 456 123456), does anything beyond reaching"
echo "       genus matter? The 17-run answer so far is no -- this set is the check."
echo "    Rank on All. Reference: reaches-genus arms average 80.97, others 80.71 (t=5.69, n=17)."
