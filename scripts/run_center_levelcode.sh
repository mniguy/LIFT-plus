#!/bin/bash
#
# Taxonomy-level chains addressed by DIGIT CODE (2026-08-27).
#
#     0 global   1 kingdom   2 phylum   3 class   4 order   5 family   6 genus
#
# A code is the chain, read left to right, coarse to fine:
#     6        subtract the genus mean only
#     06       global centering, then genus centering on the residual
#     0123456  every level, top-down
# Each step is  X <- normalize( X - s * mu_level(X) ),  the mean recomputed on the running residual
# (PROMPT_CENTER_MODE=nested, NESTED_MEAN=recompute), s < 1 so no size gate is needed.
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
# ============================ THE GRID ============================
#   012   0123           global + the coarse end, 3 and 4 levels deep
#   034                  global + the middle (class, order)
#   056                  global + the fine end (family, genus)
#   0135  0246           alternating levels: odd (kingdom,class,family) vs even (phylum,order,genus)
#   0123456              everything
# The pairs are the point: 012 vs 034 vs 056 asks WHERE in the hierarchy the useful structure is,
# holding chain length fixed at 3. 0135 vs 0246 asks the same at length 4 while interleaving.
# NOTE 0123456 is the same arm as the one queued in run_center_nested_shrink.sh
# (global,kingdom,phylum,class,order,family,genus | True). Run it in one place only.
#
# ============================ HOW TO READ THE RESULT ============================
# BASE RATE: 71 iNat centering arms span All 80.46 - 81.02.  Best ever is g_bottomup_ms2 at 81.24.
# NOISE FLOOR, measured on this grid: two arms whose inits are cosine 1.0000000000 apart (global,genus
# vs genus,global, identical to float32 rounding) scored All 80.62 vs 80.62, but Head 75.26 vs 74.74.
# So All is stable to ~0.0 and differences under ~0.1 mean nothing; HEAD carries +-0.5 of pure noise
# and must not be interpreted. Rank on All.
# NO GEOMETRIC PREDICTION is offered: run_center_ms2.sh measured cos-to-global correlating with All
# at r = -0.37 across 15 arms, and a control with 78% of rows pointing away from their own class
# still scored 80.59 against an 80.63 baseline.
#
#   bash scripts/run_center_levelcode.sh
#   ARMS="056 0246" bash scripts/run_center_levelcode.sh
#   python scripts/agg_runs.py output/center_levelcode25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
S=${S:-0.963}
RENORM=${RENORM:-False}   # see "renorm IS OFF BY DEFAULT" above
ARMS=${ARMS:-"1 2 3 4 5 6 012 034 056 0135 0246 0123 0123456"}
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
  PROMPT_CENTER_GENUS_MIN 1          # inert at s < 1; explicit so the log shows the gate is gone
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for code in ${ARMS}; do
    chain=$(expand "${code}") || exit 1
    out="${OUT_ROOT}/${data}/c${code}_s${S}_rn${RENORM}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] code ${code} = ${chain} (${ep} ep) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${COMMON_ARGS[@]}" num_epochs "${ep}" \
      PROMPT_CENTER_NESTED_LEVELS "${chain}" PROMPT_CENTER_NESTED_S "${S}" \
      PROMPT_CENTER_NESTED_RENORM "${RENORM}" \
      seed "${SEED}" output_dir "${out}"
  done
done

echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER nested] ... kingdom(|mu|=..) phylum(|mu|=..) ...' line FIRST:"
echo "    a level with |mu| ~ 0 did no work, so that code is really a shorter code. phylum and class"
echo "    are the ones to watch -- 14 of 25 phyla contain a single class, so they branch ~1:1."
echo "    Q1 012 vs 034 vs 056: at fixed length 3, WHERE in the hierarchy is the useful structure?"
echo "    Q2 0135 vs 0246: does it matter which levels you interleave?"
echo "    Q3 0123 vs 0123456: does adding the fine end to the coarse chain help?"
echo "    Rank on All. Single levels for reference: 6=80.85 4=80.80 2=80.75 1=80.71 3=80.65 0=5=80.58."
