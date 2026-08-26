#!/bin/bash
#
# CHAINED PARTIAL centering with renorm between levels (2026-08-25).
#
#     X <- normalize( X - s * mu_level )     for each level in the chain, in order
#
# i.e. mode=shrink's partial-subtraction axis (s) applied to every link of mode=nested's chain,
# with PROMPT_CENTER_NESTED_RENORM breaking the telescoping identity in between.
#
# ============================ WHY THIS IS NOT AN ARM ALREADY RUN ============================
# Every multi-level arm measured so far MIXES the level means and subtracts the mixture ONCE:
#   sumA / sumB / blend / taxo_kernel / sum_all   out = O - sum_k w_k mu_k
# This one SUBTRACTS SEQUENTIALLY and RENORMALIZES BETWEEN LEVELS, which matters because with a
# single trailing normalize the chain telescopes exactly onto "subtract the finest level's mean"
# (verified in the repo: per-class cos(topdown3, cascade) = 1.0000). Row rescaling is nonlinear and
# per-row, so the levels stop collapsing into one another.
#
# The single-level shrink sweep is the reason to try stacking at all -- and also the reason to be
# skeptical of the mixtures (s=0.963, iNat Few):
#     genus 82.90   phylum 82.64   order 82.51   kingdom 82.49   class 82.30   global 82.29
#     family 82.18  |  MIXED six levels: sumB 82.50, sumA 82.27, sumA_mild(s=.3) 82.23
# Single-level genus is the best arm on this axis and beats cascade (82.50), yet every attempt to
# combine levels so far LOST to it. If mixing is what breaks and chaining is not, this grid finds
# it; if the chains also land at ~82.3, the "one level is enough" reading is confirmed properly.
# THE NUMBER TO BEAT IS 82.90.
#
# ============================ WHAT s < 1 BUYS ============================
# mode=nested previously required GENUS_MIN >= 2 because a singleton's group mean is its own row, so
# a full subtraction zeroes it. With s < 1 that cannot happen at any group size: a singleton gets
# (1 - s) * X, a positive rescale that renorm undoes EXACTLY, so the class passes through the level
# untouched. GENUS_MIN is therefore set to 1 here and is doing nothing.
# It also repairs mode=shrink's documented split. There, singletons sit at cos-to-raw 1.0000 for
# every s -- 36.8% of iNat receives no centering whatsoever. Starting the chain with the "global"
# pseudo-level (no group, no gate) reaches every class first. Measured on the real prototypes at
# s=0.92, chain global,genus:  singleton cos-to-raw 0.7155 vs mode=blend's documented 0.7198, i.e.
# singletons now get exactly what plain global centering gives them, and non-singletons 0.3400.
#
# ============================ s CONTROLS HOW MANY LEVELS GET WORK ============================
# Measured |mu| per level on the real prototypes (chain global,genus,family,order, renorm on):
#     s=0.50   global 7.259  genus 0.971  family 0.717  order 0.442
#     s=0.92   global 7.259  genus 0.952  family 0.321  order 0.066
# A low s deliberately leaves residue for the next level, so the deeper levels still have something
# to remove; a high s lets genus take nearly everything and the tail of the chain idles. That is the
# axis this grid sweeps, and it is exactly what a one-shot mixture cannot express.
# Geometry (cos to plain global centering): 0.63 - 0.88 across the grid, all distinct arms.
# Against the arm to beat, single-level genus shrink (82.90), at s=0.963 the chains sit at
# lv2 0.8641, lv3 0.8131, lv4 0.7987 -- all genuinely different inits, not restatements of it.
#
# ============================ PREDICTION ============================
# NONE OFFERED, following run_center_ms2.sh: across the 15 arms where both were measured,
# cos-to-global correlates with All at r = -0.37, and the sumA control (78% of rows initialized
# pointing AWAY from their own class) scored 80.59 vs an 80.63 baseline. Init geometry does not
# predict accuracy on iNat. BASE RATE: 71 iNat centering arms span All 80.46 - 81.02.
#
#   bash scripts/run_center_nested_shrink.sh
#   python scripts/agg_runs.py output/center_nestshrink25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"inat2018"}
EPOCHS=${EPOCHS:-5}
INAT_EPOCHS=${INAT_EPOCHS:-15}
# chain length axis. "global" first is load-bearing: it is the only level that reaches every class.
CHAINS=${CHAINS:-"global,genus,family global,genus,family,order"}
# 0.963 matches the single-level shrink sweep exactly, so those runs are the controls for it.
# 0.5 is the "leave residue for the deeper levels" end of the axis.
S_VALUES=${S_VALUES:-"0.963"}          # 0.963 matches the single-level shrink sweep exactly
RENORM_CONTROL=${RENORM_CONTROL:-1}     # also run the LONGEST chain with renorm OFF, to isolate renorm
OUT_ROOT=${OUT_ROOT:-"center_nestshrink25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

COMMON_ARGS=(
  classifier_init semantic classifier_scale 25
  mda True tte True
  PROMPT_CENTER True PROMPT_CENTER_MODE nested
  PROMPT_CENTER_NESTED_MEAN recompute
  PROMPT_CENTER_GENUS_MIN 1          # inert at s < 1; kept explicit so the log shows it
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

run(){   # run <name> <extra cfg opts...>
  local data="$1"; local name="$2"; shift 2
  local out="${OUT_ROOT}/${data}/${name}"
  completed "${out}" && { echo "  [skip] ${out}"; return 0; }
  echo "=== [${data}] ${name} (${ep} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
    "${COMMON_ARGS[@]}" num_epochs "${ep}" "$@" \
    seed "${SEED}" output_dir "${out}"
}

for data in ${DATASETS}; do
  if [ "${data}" = "inat2018" ]; then ep=${INAT_EPOCHS}; else ep=${EPOCHS}; fi
  for chain in ${CHAINS}; do
    n_lv=$(( $(echo "${chain}" | tr -cd ',' | wc -c) + 1 ))
    for s in ${S_VALUES}; do
      run "${data}" "lv${n_lv}_s${s}" \
        PROMPT_CENTER_NESTED_LEVELS "${chain}" \
        PROMPT_CENTER_NESTED_S "${s}" \
        PROMPT_CENTER_NESTED_RENORM True
    done
  done
  if [ "${RENORM_CONTROL}" != "0" ]; then
    # renorm OFF on the LONGEST chain, because that is where renorm has the most to do: measured
    # cos(renorm on, off) at s=0.963 is lv2 0.9967, lv3 0.9715, lv4 0.9651. With only one taxonomy
    # level after "global" there is nothing for the telescoping to collapse, so an lv2 control would
    # be comparing an arm against itself.
    last_chain=""; for c in ${CHAINS}; do last_chain="${c}"; done
    n_lv=$(( $(echo "${last_chain}" | tr -cd ',' | wc -c) + 1 ))
    for s in ${S_VALUES}; do
      run "${data}" "lv${n_lv}_s${s}_norenorm" \
        PROMPT_CENTER_NESTED_LEVELS "${last_chain}" \
        PROMPT_CENTER_NESTED_S "${s}" \
        PROMPT_CENTER_NESTED_RENORM False
    done
  fi
done

echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER nested] ... global(...) genus(|mu|=..) family(|mu|=..)' line:"
echo "    a level whose |mu| is ~0 did no work, so that chain is really the shorter chain."
echo "    Q1 does any chain beat single-level genus shrink (Few 82.90)?"
echo "    Q2 lv4_s0.963 vs lv4_s0.963_norenorm: renorm alone. inits differ at cos 0.9651,"
echo "       so a null result here means renorm is cosmetic on iNat, not that it was untested."
echo "    Q3 does Few rise or fall with chain length at fixed s? that is 'is stacking worth it'."
