#!/bin/bash
#
# CHAINED PARTIAL centering, no renorm anywhere (2026-08-26).
#
#     X <- X - s * mu_level        for each level in the chain, in order, mean recomputed on the residual
#
# i.e. mode=shrink's partial-subtraction axis (s) applied to every link of mode=nested's chain.
# This is g_bottomup_ms2 -- the current project best -- with the subtraction made partial.
#
# ============================ THE ARM THIS BUILDS ON ============================
# Ranked by All (NOT by Few: the two disagree here and All is the headline the ms2 header uses):
#     g_bottomup_ms2  81.24  global,genus,family,order  ms=2   <- project best
#     g_bottomup_fo   81.02  |  bottomup3_gL 81.02  |  bottomup3_rn 80.97  |  g_bottomup_gf 80.95
#     g_bottomup      80.93  (same chain at ms=5)     |  cascade_ms2 80.90
#     genus (shrink)  80.85  Few 82.90                |  cascade 80.84  |  global 80.52
# Chains BEAT single-level genus on All (+0.39) while losing to it on Few (-0.22). Every MIXTURE
# scheme loses on both (mixnorm 80.33-80.62, sumA 80.59, sumB) -- the split is chain vs mixture,
# not "combining always loses".
# Verified: this script at s=1.0 with GENUS_MIN=2 reproduces g_bottomup_ms2 EXACTLY (per-class
# cosine 1.0000), so the s axis below is anchored on a known point.
#
# ============================ WHY PARTIAL, AND WHY THE GATE DISAPPEARS ============================
# mode=nested needs GENUS_MIN >= 2 at s=1 because a singleton's group mean is its own row, so the
# subtraction zeroes it. At s < 1 that is impossible at ANY group size: a singleton gets (1-s)*X, a
# positive rescale, and the trailing normalize makes it a no-op for that class. GENUS_MIN is set to 1
# here and is inert. That matters because the gate is what ms2 was sweeping: 5 -> 2 took g_bottomup
# from 80.93 to 81.24, and s < 1 is the limit of that direction -- no gate and no fallback at all.
#
# ============================ NO RENORM, DELIBERATELY ============================
# Measured on this chain: g_bottomup 80.93 with renorm OFF vs g_bottomup_rn 80.68 with it ON (-0.25).
# And at s < 1 renorm has little left to do anyway: the partial subtraction ALREADY breaks the
# telescoping identity that renorm exists to break, and turning it on makes the chain lengths MORE
# alike, not less (lv4-vs-lv2 at s=0.963: 0.9407 renorm off, 0.9608 renorm on). s and renorm are
# substitutes for the same degeneracy, so this grid uses s and leaves renorm off everywhere.
#
# ============================ WHY s = 0.963 AND 0.7 ============================
# 0.963 matches the single-level shrink sweep exactly, so those runs are its controls. The second
# value was chosen by measuring where the CHAIN-LENGTH axis is most alive -- this grid's whole point
# is lv2 vs lv3 vs lv4, and that contrast dies at both ends of s:
#     s      lv2-lv4   vs g_bottomup_ms2   flipped rows
#     0.963   0.9407        0.9177             48
#     0.900   0.9311        0.9099             18
#     0.800   0.9069        0.8771              3
#     0.700   0.8854        0.8318              0     <- chains most separated, first s with 0 flips
#     0.500   0.8864        0.7097              0
#     0.300   0.9495        0.5288              0     <- too little removed, chains collapse together
# CAVEAT: global,genus,family and global,genus,family,order sit at cos 0.988-0.994 at every s --
# adding `order` on top of `family` barely moves the init (that step's |mu| is 0.052 against genus's
# 0.916). Expect those two to land together; if they do not, that is the noise floor talking. They
# are DEFERRED out of the default grid for that reason; CHAINS_DEFERRED below re-adds them.
#
# ============================ THE TWO SHORT CHAINS, AND WHY ============================
# genus,global and genus,order were added because every chain arm run so far uses only
# global/genus/family/order with global FIRST. Screened against the eight chain arms already run
# (max per-class cosine to ANY of them, renorm off, gate 1):
#     chain            s=0.963   s=0.7
#     genus,global      0.9081   0.8398   <- the most independent point measured in this family
#     genus,order       0.9081   0.8532
# genus,global is the ORDER question. Where "global" sits mattered a lot while the size gate existed:
#     global first  g_bottomup   80.93 (ms=5) -> 81.24 (ms=2)
#     global last   bottomup3_gL 81.02 (ms=5) -> 79.97 (ms=2)   <- collapsed once small groups entered
# But that was a GATE effect, and at s < 1 there is no gate: singletons pass through every level
# untouched on their own. The ordering verdict therefore has to be re-taken here.
# genus,order asks whether the level after genus should be its immediate parent (family, whose rho
# is 0.020 once genus has run) or a coarser one. Note g_bottomup_fo = global,family,order scored
# All 81.02 with NO genus level at all, so "genus then something" is not obviously the right shape.
#
# ============================ PREDICTION ============================
# NONE, per run_center_ms2.sh: across the 15 arms where both were measured cos-to-global correlates
# with All at r = -0.37, and the sumA control (78% of rows initialized pointing AWAY from their own
# class) scored 80.59 against an 80.63 baseline. Init geometry does not predict accuracy on iNat.
# BASE RATE: 71 iNat centering arms span All 80.46 - 81.02.  THE NUMBER TO BEAT IS All 81.24.
#
#   bash scripts/run_center_nested_shrink.sh
#   python scripts/agg_runs.py output/center_nestshrink25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
# The two family/order chains are DEFERRED, not dropped -- they sit at cos 0.988-0.994 to each
# other (see the CAVEAT above), so they are the least informative pair to spend GPU time on.
#   add them back:  CHAINS="$CHAINS_DEFERRED" bash scripts/run_center_nested_shrink.sh
#   or run both:    CHAINS="global,genus genus,global genus,order $CHAINS_DEFERRED" bash ...
CHAINS_DEFERRED="global,genus,family global,genus,family,order"
CHAINS=${CHAINS:-"global,genus genus,global genus,order"}
S_VALUES=${S_VALUES:-"0.963 0.7"}
OUT_ROOT=${OUT_ROOT:-"center_nestshrink25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
  PROMPT_CENTER True PROMPT_CENTER_MODE nested
  PROMPT_CENTER_NESTED_MEAN recompute
  PROMPT_CENTER_NESTED_RENORM False   # see "NO RENORM, DELIBERATELY" above
  PROMPT_CENTER_GENUS_MIN 1           # inert at s < 1; explicit so the log shows the gate is gone
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for chain in ${CHAINS}; do
    # name by COMPOSITION, not by length: genus,global and global,genus are both 2 levels but are
    # different arms (the chain order decides which level absorbs the global component first).
    tag=$(echo "${chain}" | tr ',' '_')
    for s in ${S_VALUES}; do
      out="${OUT_ROOT}/${data}/${tag}_s${s}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] ${tag}_s${s} (${ep} ep) ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
        "${COMMON_ARGS[@]}" num_epochs "${ep}" \
        PROMPT_CENTER_NESTED_LEVELS "${chain}" PROMPT_CENTER_NESTED_S "${s}" \
        seed "${SEED}" output_dir "${out}"
    done
  done
done

echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER nested] ... global(|mu|=..) genus(|mu|=..) family(..)' line:"
echo "    a level whose |mu| is ~0 did no work, so that chain is really the shorter chain."
echo "    Q1 does any arm beat g_bottomup_ms2 (All 81.24)? that is s=1 on global,genus,family,order."
echo "    Q2 chain SHAPE at fixed s: global_genus (baseline) vs genus_global (order of the"
echo "       global level) vs genus_order (which partner follows genus)."
echo "       the chain-LENGTH question is NOT in this grid -- add CHAINS_DEFERRED for it."
echo "    Q3 s=0.963 vs s=0.7 at fixed chain: does leaving residue for later levels help?"
echo "    compare on All AND Few -- the chain and single-genus families disagree on which wins."
