#!/bin/bash
#
# min_size 2 versions of the three best taxonomy arms (2026-08-21).
#
# ============================ WHAT min_size ACTUALLY CONTROLS ============================
# It is NOT "centered vs not centered". Every one of these three arms centers every class:
#   * cascade  initializes local_mu to the GLOBAL mean, so a class that qualifies at no level still
#              gets global centering (the census line ends with "global=N", those are the fallbacks).
#   * g_bottomup and bottomup3_gL both contain the "global" pseudo-level, which has NO size gate at
#              all, so every class receives at least global centering there too.
# What min_size controls is HOW LOCAL a mean each class gets. Lowering it 5 -> 2 moves classes from
# coarse means to their own genus mean. Census measured on the real prototypes:
#
#   cascade      ms=5:  genus 2279  family 4427  order 1068  global 368
#                ms=2:  genus 5142  family 2446  order  438  global 116
#   (genus coverage 28% -> 63%; the global fallback shrinks from 368 classes to 116)
#
# For the nested arms the same shift shows up as the later levels losing their work, because those
# levels only had something to remove while the gate-failing classes were still raw:
#
#   g_bottomup    ms=5:  |mu| global 16.29  genus 16.25  family 4.59  order 0.95
#                 ms=2:  |mu| global 16.29  genus 15.27  family 2.49  order 0.43
#   bottomup3_gL  ms=5:  |mu| genus 21.95  family 12.20  order 2.59  global 0.52
#                 ms=2:  |mu| genus 22.04  family  6.32  order 0.98  global 0.14
#
# So at ms=2 all three chains collapse towards "genus centering plus cleanup". That is the mechanism
# to watch, and it is why these are not just three more points on a smooth axis.
#
# ============================ GEOMETRY ============================
#   arm                cos-to-global   top5conf     cos(ms5, ms2)
#   cascade      ms5      0.7462        0.4574
#   cascade      ms2      0.5453        0.4139         0.7679
#   g_bottomup   ms5      0.7030        0.4783
#   g_bottomup   ms2      0.5131        0.4275         0.7430
#   bottomup3_gL ms5      0.6540        0.6218
#   bottomup3_gL ms2      0.4389        0.6343         0.6329
# All three ms=2 arms are independent of every already-run arm (closest is taxo_kernel gamma=0 at
# cos 0.9680 / 0.9229 / 0.7514) and mutually distinct (0.75 - 0.93).
#
# ============================ PREDICTION ============================
# NO GEOMETRIC PREDICTION IS OFFERED, deliberately. Across the 15 arms where both were measured,
# cos-to-global correlates with All at r = -0.37 and top5conf at r = -0.36, and the sumA control
# (78% of rows initialized pointing AWAY from their own class) scored 80.59 against a baseline of
# 80.63. Init geometry does not predict accuracy on this dataset, so the 0.72-0.75 "winning band"
# reasoning used in the older headers is retired and is not applied here.
# What can be said: the ms=5 versions scored cascade 80.84, g_bottomup 80.93, bottomup3_gL 81.02 --
# the top three arms of the whole project. The question is only whether the extra genus coverage
# helps, hurts, or does nothing. BASE RATE: 71 iNat centering arms now span 80.46 - 81.02.
# The repo's earlier min_size sweep (run_center_nested1.sh, RETIRED block) picked ms=5 on cos-to-global
# grounds -- that justification no longer holds, which is precisely why ms=2 is worth measuring
# directly rather than being ruled out by geometry.
#
# bottomup3_rn (renorm=True) geometry, measured on the real prototypes:
#   arm                cos-to-global  top5conf  within-genus   family |mu| after genus step
#   bottomup3    ms5      0.6527       0.6201      0.6089           12.195
#   bottomup3_rn ms5      0.6993       0.5323      0.1865            0.540
#   bottomup3    ms2      0.4390       0.6304      0.2110            6.317
#   bottomup3_rn ms2      0.5042       0.5016     -0.1177            0.289
# within-genus cosine going NEGATIVE at ms=2 with renorm means genus-mates end up pointing away from
# each other -- the most aggressive separation of near-relatives measured in this project.
# cos(bottomup3_rn ms5, ms2) = 0.7071; cos(bottomup3_rn ms2, bottomup3 ms2) = 0.9256; closest already-run
# arm is taxo_kernel gamma=0 at 0.9037. Independent on both counts.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline      80.63 74.62 80.50 82.36    global        80.52 74.86 80.41 82.13
#   cascade  ms5  80.84 75.81 80.57 82.50    g_bottomup ms5 80.93 75.18 80.78 82.61
#   bottomup3_gL ms5 81.02 75.93 80.71 82.75  <- the best arm measured in this project
#   bottomup3_rn ms5 80.97 75.42 80.84 82.58
#   iNat seed noise (5-ep/scale-30 proxy): All ~0.06, Head ~0.74, Med ~0.16, Few ~0.23.
#
#   bash scripts/run_center_ms2.sh
#   ARMS="cascade_ms2" bash scripts/run_center_ms2.sh
#   MS=3 bash scripts/run_center_ms2.sh          # a different gate value on the same three arms
#   python scripts/agg_runs.py output/center_ms2 --sort path
# ============================ NOTE ON "size < 2 -> do not center" ============================
# min_size=2 already IS that rule, and it means different things in the two modes -- both correct:
#   nested  : every class passes through EVERY level. A level whose group has < min_size members is
#             skipped FOR THAT CLASS ONLY; the class still goes on to the next level. Subtractions
#             therefore ACCUMULATE. Measured at ms=2 on g_bottomup: global 8142, genus 5142,
#             family 7679, order 8078 -- 29041 shifts over 8142 classes, and 5140 of the 5142
#             genus-centered classes are centered again at family and again at order.
#   cascade : a class is centered at exactly ONE level. Measured at ms=2: genus 5142 + family 2446
#             + order 438 + unassigned 116 = 8142 exactly, a clean partition.
# The two nested arms keep the "global" pseudo-level, which has no size gate, so every class receives
# global centering there (|mu| 16.29 when global runs first, 0.14 when it runs last).
#
# PROMPT_CENTER_CASCADE_NOFALL makes cascade's final fallback subtract NOTHING instead of the global
# mean, so those classes keep their raw prototype. It is ON for cascade_ms2 below. On the full genus,family,order chain only 116 classes ever reach the fallback, so
# nofall=True vs False is per-class cos 0.9960 -- effectively the same arm. It only bites on a short
# chain: genus,family leaves 554 uncentered, genus alone leaves 3000 (and that one is cos 0.9907 to
# shrink genus s=0.963, already run at 80.85).
#
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
ARMS=${ARMS:-"cascade_ms2 g_bottomup_ms2 bottomup3_gL_ms2 bottomup3_rn_ms2"}
MS=${MS:-2}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_ms2"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# arm -> "<mode> <levels> <flag>".  The gate value comes from MS so all arms move together.
# <flag> means different things per mode, because the two modes need different options:
#   cascade -> PROMPT_CENTER_CASCADE_NOFALL : what a class that qualifies at NO level receives.
#              True = subtract nothing (raw O), False = subtract the global mean.
#   nested  -> PROMPT_CENTER_NESTED_RENORM  : row-normalize after every level, or only at the end.
# nested has no "nofall" option and does not need one: a class whose group fails the gate simply gets
# shift = 0 at that level, i.e. nothing is subtracted there already.
arm_spec(){ case "$1" in
  cascade_ms2)       echo "cascade genus,family,order          True" ;;
  g_bottomup_ms2)    echo "nested  global,genus,family,order   False" ;;
  bottomup3_gL_ms2)  echo "nested  genus,family,order,global   False" ;;
  bottomup3_rn_ms2)  echo "nested  genus,family,order          True" ;;
  # optional: shorter cascade chains, where the "leave it alone" fallback reaches far more classes.
  # On the full genus,family,order chain only 116 classes ever get there, so the policy moves 1.4% of
  # the rows; dropping levels is what makes it bite.
  cascade_gf_nofall) echo "cascade genus,family                True" ;;   # 554 uncentered
  cascade_g_nofall)  echo "cascade genus                       True" ;;   # 3000; cos 0.9907 to shrink genus
  # the old behaviour, kept as the A/B partner for cascade_ms2 (cos 0.9960 between them)
  cascade_globalfall) echo "cascade genus,family,order         False" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for arm in ${ARMS}; do
  spec=$(arm_spec "${arm}") || { echo "unknown arm ${arm}"; exit 1; }
  # 3rd field: NOFALL when mode=cascade, RENORM when mode=nested (see arm_spec)
  set -- ${spec}; mode="$1"; lv="$2"; flag="$3"
  out="${OUT_ROOT}/inat2018/${arm}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  if [ "${mode}" = "cascade" ]; then
    extra=(PROMPT_CENTER_MODE cascade PROMPT_CENTER_CASCADE "${lv}" \
           PROMPT_CENTER_CASCADE_MEAN residual PROMPT_CENTER_CASCADE_NOFALL "${flag}")
  else
    extra=(PROMPT_CENTER_MODE nested PROMPT_CENTER_NESTED_LEVELS "${lv}" \
           PROMPT_CENTER_NESTED_MEAN recompute PROMPT_CENTER_NESTED_RENORM "${flag}")
  fi
    if [ "${mode}" = "cascade" ]; then flagname="nofall"; else flagname="renorm"; fi
  echo "=== [inat2018] ${arm}: mode=${mode} levels=${lv} min_size=${MS} ${flagname}=${flag} (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True "${extra[@]}" PROMPT_CENTER_GENUS_MIN "${MS}" \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    check the census line of each log FIRST. Expected at min_size=2:"
echo "      cascade_gfo -> genus=5142 family=2446 order=438 UNCENTERED=116"
echo "      cascade_gf  -> genus=5142 family=2446 UNCENTERED=554"
echo "      cascade_g   -> genus=5142 UNCENTERED=3000"
echo "      bottomup3_rn_ms2 -> |mu| genus 22.04 / family 0.29 / order 0.09, all row norms exactly 1.000"
echo "    The word must read UNCENTERED, not global -- 'global=N' means nofall did not take effect."
echo "    Anchors: cascade ms5 80.84, g_bottomup ms5 80.93, bottomup3_gL ms5 81.02, shrink genus 80.85."
