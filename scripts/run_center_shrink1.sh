#!/bin/bash
#
# PROMPT_CENTER_MODE=shrink -- out = O - g*mu_global - s*mean(mu_LEVELs)
#
# Two knobs, and they select three regimes:
#   g=0, one level    : O - s*mu_LEVEL          partial centering, no global term
#   g=0, N levels     : O - (s/N)*sum_k mu_k    == "sum_k (O - s*mu_k)" in closed form
#   g=1, N levels     : coefficients sum to 1+s == "sum_k (O - mu_global - s*mu_k)"  -> OVER-centering
# s=0.5 with one level reproduces mode=level_keep EXACTLY (2O - mu is a positive multiple of
# O - 0.5 mu; verified per-class cos 1.00000000; that arm ran: 80.59). s=1 would be mode=level, whose
# zero rows destroyed the run (All 0.01), so s >= 1 is rejected.
#
# ============================ WHY g=0 SPLITS THE INIT, AND WHY ADDING LEVELS FIXES IT ============================
# A class alone in its group has mu_LEVEL = O, so with one level the expression collapses to (1-s)*O:
# after row normalization, the RAW uncentered prototype. That class gets no centering at ANY s.
# Adding coarser levels repairs this WITHOUT a global term, because the mean over several levels only
# equals O if the class is a singleton at EVERY one of them. Measured at s=0.963 on the real 8142 iNat
# prototypes ("UNCENTERED" = rows whose direction is identical to raw O):
#
#   levels subtracted            UNCENTERED   cos-to-global   top5conf   nearest already-run arm
#   genus                           3000        0.5583        0.5627     blend s=0.92   cos 0.8851
#   genus,family                     461        0.8108        0.4841     sum_all        cos 0.8820
#   genus,family,order                63        0.8936        0.5165     sum_all        cos 0.9603
#   all six                            1        0.9573        0.5752     sum_all        cos 0.9979  <- REDUNDANT
#
# So the full six-level sum the idea started from is mode=sum_all, which already ran (80.68). The
# informative arms are the PARTIAL sums, and they form a clean dose axis for the inhomogeneity:
# 3000 -> 461 -> 63 untouched classes, with the genus term subtracted throughout, so only the
# singleton-filling changes between arms. (A single-level sweep genus/family/order gives the same
# doses 3000/463/64 but also swaps out WHAT is subtracted, so it is the less controlled version;
# LEVELS="genus family order" runs it if wanted.)
#
# ============================ g=1 IS THE OVER-CENTERING CONTROL ============================
# With g=1 the subtracted coefficients sum to 1+s > 1, so rows are pushed PAST the origin instead of
# towards it. Measured with all six levels, classes whose init ends up NEGATIVELY correlated with
# their own raw prototype ("FLIPPED", also logged per run):
#     s=0.3  ->     0/8142   cos-to-global 0.9390   mean cos to raw +0.444
#     s=0.5  ->   130/8142   cos-to-global 0.8331   mean cos to raw +0.230
#     s=0.963-> 6390/8142    cos-to-global 0.5439   mean cos to raw -0.167
# At s=0.963 the classifier row for a species points AWAY from that species for 78% of classes. This
# project pre-registered such an arm once (nested's PROMPT_CENTER_NESTED_MEAN=static, "expected to
# lose") and never ran it, so over-centering is still untested on iNat.
#
# ============================ PRE-REGISTERED PREDICTIONS ============================
# BASE RATE: 65 iNat centering arms measured so far span 80.46-81.02.
#
# (B) dose axis, g=0, s=0.963. The one previous arm that split iNat this way was plain 'genus'
#     centering under the min_size=5 guard (72% global fallback / 28% genus-centered), which scored
#     80.46 -- the LOWEST of every centering arm here. If that "mixture pathology" reading is right,
#     accuracy should RISE as the untouched count falls:
#       genus (3000)              predict 80.40 - 80.65   (its matched partner blend s=0.92 got 80.64)
#       genus,family (461)        predict 80.50 - 80.75
#       genus,family,order (63)   predict 80.55 - 80.80   (near sum_all 80.68)
#     FALSIFIER: if the three tie, inhomogeneity is NOT what hurt 'genus' mode, and the fallback-design
#     argument behind cascade / blend / taxo_kernel loses its main justification.
#
# (C) over-centering, g=1, all six levels. s=0.963 flips 78% of the rows and is the most broken
#     initialization this project can construct on purpose. PREDICT CLEARLY BELOW BASELINE -- if it
#     lands anywhere near 80.5, then the iNat init geometry does not matter AT ALL, which after 65
#     tied arms would be the cleanest possible answer to the whole question. That is why it is worth
#     one run despite the confident prediction. s=0.3 (0 flipped, cos-to-global 0.939) is the mild
#     companion and should tie global (80.52).
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline 80.63 74.62 80.50 82.36 | global 80.52 74.86 80.41 82.13 | guarded genus 80.46 75.22 80.05 82.34
#   blend s=0.92 80.64 75.34 80.50 82.21 | sum_all 80.68 75.22 80.38 82.48
#   cascade 80.84 75.81 80.57 82.50 | g_bottomup_fo 81.02 75.73 80.79 82.69
#
# NOTE: yacs is type-strict, so S and G MUST be written with a decimal point ("0.963", not "1").
#
# ============================ ARMS ============================
# Every arm is named explicitly; the default runs all five below. All seven are independent of every
# already-run arm (max per-class cos 0.9686), and mutually distinct (max 0.9773).
#
#   arm        levels subtracted          g     UNCENT / FLIP     cos-to-global   nearest run arm
#   genus      genus                     0.0    3000 UNCENT          0.5583       blend s=.92  0.8851
#   family     family                    0.0     463 UNCENT          0.8304       sum_all      0.8967
#   order      order                     0.0      64 UNCENT          0.9158       sum_all      0.9686
#   sumB       all six                   0.0       1 UNCENT          0.9573       sum_all      0.9979 (!)
#   sumA       all six                   1.0    6390 FLIP            0.5439       sum_all      0.5759
#   -- optional --
#   sumB_gf    genus,family              0.0     461 UNCENT          0.8108       sum_all      0.8820
#   sumB_gfo   genus,family,order        0.0      63 UNCENT          0.8936       sum_all      0.9603
#   sumA_mild  all six                   1.0       0 FLIP            0.9390       global       0.9390
#
# HONEST NOTE ON sumB: the literal "sum over ALL six levels" is per-class cos 0.9979 to mode=sum_all,
# which already ran (80.68). Keeping it is a replication through a different code path -- useful only
# as a check that the two agree. If GPU is tight, run sumB_gf / sumB_gfo instead: those are the
# PARTIAL sums, they are genuinely new (0.88 / 0.96 to sum_all), and together with the single-level
# arms they give two independent readings of the same inhomogeneity dose axis.
#
#   bash scripts/run_center_shrink.sh                          # genus family order sumB sumA
#   ARMS="sumB_gf sumB_gfo" bash scripts/run_center_shrink.sh  # the partial sums instead of sumB
#   ARMS="sumA sumA_mild" bash scripts/run_center_shrink.sh    # the over-centering pair only
#   python scripts/agg_runs.py output/center_shrink --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
ARMS=${ARMS:-"sumB sumA sumA_mild"}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_shrink"}
ALL6="genus,family,order,class,phylum,kingdom"
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# arm -> "<levels> <s> <g>".  yacs is type-strict, so s and g always carry a decimal point.
arm_spec(){ case "$1" in
  genus)      echo "genus 0.963 0.0" ;;
  family)     echo "family 0.963 0.0" ;;
  order)      echo "order 0.963 0.0" ;;
  sumB)       echo "${ALL6} 0.963 0.0" ;;
  sumB_gf)    echo "genus,family 0.963 0.0" ;;
  sumB_gfo)   echo "genus,family,order 0.963 0.0" ;;
  sumA)       echo "${ALL6} 0.963 1.0" ;;
  sumA_mild)  echo "${ALL6} 0.3 1.0" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for arm in ${ARMS}; do
  spec=$(arm_spec "${arm}") || { echo "unknown arm ${arm}"; exit 1; }
  set -- ${spec}; lv="$1"; sv="$2"; gv="$3"
  out="${OUT_ROOT}/inat2018/${arm}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  echo "=== [inat2018] shrink ${arm}: levels=${lv} s=${sv} g=${gv} (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE shrink \
    PROMPT_CENTER_S "${sv}" PROMPT_CENTER_G "${gv}" PROMPT_CENTER_LEVEL "${lv}" \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER shrink] ...' line FIRST. Expected:"
echo "      genus 3000 UNCENTERED | family 463 | order 64 | sumB 1 | sumA 6390 FLIPPED"
echo "      ZERO must be 0 everywhere; FLIPPED must be 0 for every g=0 arm."
echo "    genus/family/order and sumB_gf/sumB_gfo are two independent readings of the same dose axis;"
echo "    read each set as a trend. sumA is the over-centering control -- predicted clearly worst."
