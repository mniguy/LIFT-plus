#!/bin/bash
#
# GRANULARITY vs MEANING: is genus useful because of how FINE it is, or because it is biology?
# (2026-08-31)
#
# ============================ WHY THIS EXISTS ============================
# 21 runs of the digit-code grid settled on ONE variable: whether the chain reaches genus.
#     reaches genus  n=8  All 80.968 (sd 0.106)
#     does not       n=9  All 80.707 (sd 0.082)     diff +0.261, t=5.69 (df=15)
# Everything else -- which coarse levels, chain length, RENORM, residual norm -- washed out.
# But "reaches genus" confounds two things that the taxonomy can never separate, because on iNat
# genus IS the finest level:
#     (A) GRANULARITY -- genus groups average 1.85 classes, so the subtracted mean is the shared
#         content of a handful of near-identical species
#     (B) MEANING -- genus is a biological grouping, not just a small one
# k-means over the prototypes gives (A) as a CONTINUOUS knob with no (B) at all. If a cluster
# partition at genus granularity matches c06, the answer is (A) and the taxonomy is dispensable
# -- which also makes the method work on ImageNet/Places, where there is no taxonomy at all.
# If c06 stays ahead, it is (B) and the taxonomy is carrying real information.
#
# The same sweep answers the second open question: the single-level arms 1-6 looked U-shaped
# (kingdom 80.83, order 80.63, genus 80.80) but that U does NOT survive scrutiny -- the
# independent mode=shrink measurement of the same six levels correlates with it at r=+0.166 and
# puts its minimum at a different level, and a quadratic fits the six points to sd 0.019, four
# times BELOW the 0.075 noise floor, i.e. it is fitting noise. Six discrete points cannot settle
# it. This sweep turns granularity into ~8 points on one axis with coverage held constant.
#
# ============================ THE TWO ARM KINDS ============================
#   s<N>    mode=cluster, PROMPT_CENTER_CLUSTER_SIZE=N  -- k-means, k = round(8142/N), each class
#           centered by its own cluster mean; a cluster smaller than GENUS_MIN falls back to the
#           GLOBAL mean (trainer.py, mode=="cluster"). No taxonomy is read.
#   t<code> mode=nested with the levelcode digits (0 global 1 kingdom .. 6 genus) -- the taxonomy
#           anchor, run with settings IDENTICAL to the digit-code grid so its numbers drop
#           straight into that table.
#
# THESE TWO ARE THE SAME CONSTRUCTION, which is the whole reason the comparison is fair. For a
# nested chain "0,L", telescoping (see run_center_levelcode.sh) gives
#       covered class  -> X - mu_L(X)          skipped class -> X - mu_global(X)
# and mode=cluster gives exactly that, with a k-means part in place of L and its own fallback.
# So s<N> vs t0<L> differs in the PARTITION and nothing else. Matched granularities, measured
# from datasets/iNaturalist2018/categories.json (8142 classes):
#       level    groups   avg size   singletons
#       genus      4401      1.85       3000        <- s2
#       family     1118      7.28        463        <- s7
#       order       272     29.93         64        <- s30
#       class        57    142.84          9        <- s143
#       phylum       25    325.68          5        <- s326
#       kingdom       6   1357.00          1
#
# ============================ GENUS_MIN 2 IS LOAD-BEARING ============================
# The trainer's default is 5. At s2 the average cluster holds 1.85 classes, so with the default
# nearly every cluster would fall UNDER the gate and be replaced by the global mean -- the fine
# end of the sweep would silently become a global-centering rerun, which is exactly the failure
# run_center_cluster.sh documents for Places at absolute k=100 (43.8% fallback). GENUS_MIN=2 here
# also matches the digit-code grid, so the anchors stay comparable to the 21 existing runs.
# READ THE FALLBACK LINE IN EVERY LOG BEFORE TRUSTING AN ARM:
#     [PROMPT_CENTER cluster] k=.. (target size=..) min_size=2 -> N/8142 classes fell back
# A large N means that arm is partly a global rerun and its granularity label is a lie.
#
# ============================ HOW TO READ THE RESULT ============================
# Baseline 80.63 / global-only 80.52. Noise floor ~0.075 in All (sd of 4 twin deltas 0.072;
# 13-run regression residual 0.082). HEAD carries +-0.5 of noise -- do not interpret it.
#   Q1 s2 vs t06: granularity (A) or biology (B)? Within ~0.1 => (A), and the taxonomy is not needed.
#   Q2 the s-curve over 2..326: is there really a U, or is accuracy flat in granularity?
#      CAVEAT, and it is not small: every class gets *some* centering (a small cluster falls back
#      to the global mean, never to raw), but the fraction getting a LOCAL mean varies down the
#      sweep -- measured s2 33.6% fallback, s4 8.9%, s7 1.6%, s30+ 0.0%. So the fine end is
#      diluted with global centering, which is the worst fallback measured. The sweep can rule a
#      U OUT; it cannot show granularity is irrelevant. Only the s2/t06 pair is clean, because
#      there the fallback rates match (33.6% vs genus skipping 36.8%).
#   Q3 s7 vs t05, s30 vs t04: does the cluster/taxonomy gap grow or shrink as groups get coarser?
#
#   bash scripts/run_center_granularity.sh                  # every arm, serially, on GPU_ID
#   ARMS="s2 t06" bash scripts/run_center_granularity.sh    # just the headline pair
#   bash scripts/run_center_granularity_4gpu.sh             # the real way: fan out over 4 cards
#   python scripts/agg_runs.py output/center_gran25 --sort all
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATA=${DATA:-"inat2018"}
EPOCHS=${EPOCHS:-15}
GENUS_MIN=${GENUS_MIN:-2}          # see "GENUS_MIN 2 IS LOAD-BEARING" above -- do not raise for s2/s4
ARMS=${ARMS:-"s2 s4 s7 s15 s30 s64 s143 s326 t06 t05"}
OUT_ROOT=${OUT_ROOT:-"center_gran25"}
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
  PROMPT_CENTER True
  PROMPT_CENTER_GENUS_MIN "${GENUS_MIN}"
)

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for arm in ${ARMS}; do
  out="${OUT_ROOT}/${DATA}/${arm}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  case "${arm}" in
    s*)                                        # k-means granularity, no taxonomy
      size="${arm#s}"
      case "${size}" in ''|*[!0-9]*) echo "ERROR: bad cluster size in arm '${arm}'" >&2; exit 1;; esac
      echo "=== [${DATA}] cluster target size=${size}  (k=round(8142/${size}), ${EPOCHS} ep) ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${DATA}" -b clip_vit_b16 -m lift+ \
        "${COMMON_ARGS[@]}" PROMPT_CENTER_MODE cluster \
        PROMPT_CENTER_CLUSTER_SIZE "${size}" num_epochs "${EPOCHS}" \
        seed "${SEED}" output_dir "${out}"
      ;;
    t*)                                        # taxonomy anchor, identical to the digit-code grid
      code="${arm#t}"
      chain=$(expand "${code}") || exit 1
      echo "=== [${DATA}] taxonomy anchor ${code} = ${chain} (${EPOCHS} ep) ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${DATA}" -b clip_vit_b16 -m lift+ \
        "${COMMON_ARGS[@]}" PROMPT_CENTER_MODE nested \
        PROMPT_CENTER_NESTED_MEAN recompute PROMPT_CENTER_NESTED_LEVELS "${chain}" \
        PROMPT_CENTER_NESTED_RENORM False num_epochs "${EPOCHS}" \
        seed "${SEED}" output_dir "${out}"
      ;;
    *) echo "ERROR: arm '${arm}' must start with 's' (cluster size) or 't' (taxonomy code)" >&2; exit 1 ;;
  esac
done

echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort all ==="
echo "    FIRST grep the fallback line of every s-arm:"
echo "      grep -h 'PROMPT_CENTER cluster' output/${OUT_ROOT}/${DATA}/s*/log-*.txt"
echo "    an arm with a large 'fell back' count is partly a global rerun, not a granularity point."
echo "    Q1 s2 vs t06  -- within ~0.1 means granularity, not biology, and the taxonomy is dispensable."
echo "    Q2 s2..s326   -- the honest granularity curve; the 1-6 'U' did not survive (r=+0.166 vs shrink)."
echo "    Q3 s7 vs t05, s30 vs t04 -- does the cluster/taxonomy gap open up at coarser groups?"
