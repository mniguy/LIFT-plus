#!/bin/bash
#
# PROMPT_CENTER_MODE=nested -- REPEATED centering down (or up) the taxonomy (2026-08-06 brainstorm).
#
# Motivation: 'cascade' centers each class at exactly ONE level (deepest available, then stop).
# 'nested' instead centers every class at EVERY level it has a big-enough group for, so the
# subtractions stack: center by order, then by family, then by genus. The order of
# PROMPT_CENTER_NESTED_LEVELS IS the direction -- "order,family,genus" is top-down (coarse->fine),
# "genus,family,order" is bottom-up (fine->coarse). With the default
# PROMPT_CENTER_NESTED_MEAN=recompute, each level's mean is taken on the CURRENT residual, so a level
# removes only what the levels before it did not already explain (an ANOVA-style decomposition), and
# the two directions are genuinely different operations rather than a reordering of the same sum.
#
# ============================ OFFLINE MEASUREMENTS (2026-08-06, GPU-free) ============================
# All on the real 8142 iNat raw prototypes (output/breadth25/inat2018/baseline_15ep/ckpts/init).
#
# (1) HOW MANY LEVELS ARE WORTH USING -- top-down ladder, incremental |mu| removed at each level:
#       1 lvl  genus                                  cos-to-global 0.5153
#       2 lvl  family,genus                           0.6734    fam=0.882 gen=0.321
#       3 lvl  order,family,genus                     0.7164    ord=0.859 fam=0.172 gen=0.321
#       4 lvl  class,order,family,genus               0.7245    cla=0.845 ord=0.130 fam=0.172 gen=0.321
#       5 lvl  phylum,class,order,family,genus        0.7260    phy=0.842 cla=0.054 ...
#       6 lvl  kingdom,phylum,class,order,family,genus 0.7265   kin=0.838 phy=0.056 cla=0.054 ...
#     -> Two readings. (a) WHICHEVER level goes FIRST absorbs the global component (|mu| 0.84-0.98),
#     because nothing has removed it yet -- so top-down is really "global centering, then progressive
#     local corrections". (b) Past 3 levels it saturates: class/phylum/kingdom each contribute an
#     incremental |mu| of only ~0.05 and move cos-to-global by <0.01. Hence 3 levels (order,family,
#     genus) is the default here and topdown4 is included only as the null anchor that should tie it.
#
# (2) BOTTOM-UP IS NOT A CLEAN REVERSE -- it is alive only through the min_size coverage holes.
#     If every class got genus-centered, each genus mean becomes 0, so the family mean of
#     genus-centered data would be exactly 0 and every later level would be a no-op. Measured:
#       min_size=2  (63% genus coverage)  genus|mu|=0.977  family|mu|=0.298  order|mu|=0.045  <- collapsing
#       min_size=5  (28% genus coverage)  genus|mu|=0.975  family|mu|=0.586  order|mu|=0.124
#       min_size=10 (13% genus coverage)  genus|mu|=0.975  family|mu|=0.715  order|mu|=0.209
#     -> family/order only have something to remove because the 72% of classes that failed the genus
#     gate are still raw and dominate those means. This is the SAME 28/72 mixture pathology that made
#     plain 'genus' mode fail (see run_center_local.sh). Predict bottom-up therefore behaves like
#     "genus mode with cleanup", not like a real reverse of top-down.
#
# (3) STATIC SUM (=PROMPT_CENTER_NESTED_MEAN=static) OVER-SUBTRACTS, as the "중복해서 빼기" worry
#     expected. Every level's mean comes from the raw prototypes, so the shared component is removed
#     once per level. Pre-normalization row norms: mean 1.128 / max 2.016 / min 0.208 (raw rows are
#     1.0) -- rows longer than the original mean the vector was pushed PAST the origin into negation,
#     and the 10x spread between min and max means classes are treated wildly unequally.
#     cos-to-global 0.2592, far outside anything measured on this dataset. Included as a control that
#     is EXPECTED TO LOSE; if it does not, the "minimal removal" design law needs revisiting.
#
# (4) cos-to-global and norm summary for the arms below (norm = pre-normalization, raw = 1.0):
#       arm         chain                      cos-to-global   norm mean/min
#       topdown2    family,genus                   0.6734       0.490/0.0712
#       topdown3    order,family,genus             0.7164       0.420/0.0712
#       topdown4    class,order,family,genus       0.7245       0.408/0.0712
#       bottomup3   genus,family,order             0.6047       0.561/0.0966
#       static3     order,family,genus (static)    0.2592       1.128/0.2078
#     RISK to watch: min row norm 0.0712 means some class lost 93% of its norm before renormalization,
#     which then amplifies whatever little is left ~14x. If nested underperforms, check whether the
#     worst-shrunk classes are the ones that regressed.
#
# ============================ PRE-REGISTERED PREDICTIONS (written before running) ============================
# iNat's measured localization trend (tables_cascade.tex, tab:localization_axis) is r(cos-to-global,
# Few) = -0.89: on THIS dataset, further from global = better, over the measured range
# cos-to-global 0.719-1.000 (Few 82.13 -> 82.60).
#   * topdown3 (0.7164) sits just past the most-local measured point (cluster500, 0.719 -> Few 82.60),
#     so the trend line predicts Few ~82.6 and Overall competitive with cascade's 80.84. This is a
#     genuine interpolation, not an extrapolation -- the strongest prediction here.
#   * topdown4 (0.7245) should be INDISTINGUISHABLE from topdown3 (offline they differ by 0.008
#     cos-to-global and only ~0.05 incremental |mu|); if it differs materially, the saturation reading
#     in (1) is wrong.
#   * topdown2 (0.6734) and bottomup3 (0.6047) are both BEYOND the measured range, i.e. genuine
#     extrapolation. The trend says better, the "global/uniform/minimal" design law says worse. This
#     is the informative disagreement in this run -- predict the trend BREAKS somewhere in here
#     (nonmonotone), because unlike cascade these arms subtract multiple times and shrink norms hard.
#   * static3 (0.2592): predict clearly worse than baseline (80.63), for the reason in (3).
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   genus    (center_local25/inat2018/genus)     80.46  75.22  80.05  82.34
#   cluster,k=500 (center_tree25/inat2018)       80.75  76.01  80.26  82.60
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50   <- the number to beat
#   iNat seed noise (5-ep/scale-30 proxy, the only multi-seed estimate available): All ~0.06,
#   Head ~0.74, Med ~0.16, Few ~0.23. Head differences below ~1.5 pts here are NOT interpretable.
#
# ============================ ARM LADDER (measured offline, 2026-08-06) ============================
#   arm                 chain                            min_sz  cos-g   norm mean/min  per-level |mu|
#   T1 topdown3         order,family,genus                  5   0.7164  0.420/0.0712  .859 .172 .321
#   T1 bottomup3        genus,family,order                  5   0.6047  0.561/0.0966  .975 .586 .124
#   T1 g_topdown        global,order,family,genus           5   0.7291  0.405/0.0712  .828 .219 .172 .321
#   T1 g_bottomup       global,genus,family,order           5   0.7015  0.408/0.0902  .828 .576 .159 .033
#   T2 bottomup3_ms2    genus,family,order                  2   0.3859  0.453/0.0634  .977 .298 .045
#   T2 bottomup3_ms10   genus,family,order                 10   0.6953  0.597/0.1590  .975 .715 .209
#   T2 g_bottomup_ms2   global,genus,family,order           2   0.5122  0.301/0.0614  .828 .533 .085 .015
#   T3 topdown2         family,genus                        5   0.6734  0.490/0.0712  .882 .321
#   T3 bottomup2        genus,family                        5   0.6080  0.581/0.0867  .975 .586
#   T3 topdown4         class,order,family,genus            5   0.7245  0.408/0.0712  .845 .130 .172 .321
#   T3 topdown_skip     order,genus                         5   0.7474  0.430/0.0712  .859 .458
#   T3 bottomup_skip    genus,order                         5   0.6620  0.559/0.1105  .975 .618
#   T3 static3          order,family,genus (static)         5   0.2592  1.128/0.2078  (over-subtracts)
#   T4 bottomup_fo      family,order                        5   0.7729  0.524/0.0966  .882 .124
#   T4 topdown_of       order,family                        5   0.8396  0.475/0.0867  .859 .172
#   T4 g_bottomup_fo    global,family,order                 5   0.8490  0.461/0.0902  .828 .295 .033
#   T4 g_topdown_of     global,order,family                 5   0.8523  0.460/0.0867  .828 .219 .172
#
# T4 = DROP GENUS ENTIRELY. Motivated by the observation that genus is bottom-up's pathological level
# (28% coverage at min_size=5, and its group mean is estimated from as few as 5 species). Starting the
# chain at family raises first-step coverage to 83% and uses a far more stable mean.
#
#   PROVEN IDENTITY found while checking this (holds for any nested taxonomy, not just iNat): in
#   bottom-up, prepending genus changes NOTHING about what the later levels see. Genus groups are
#   nested inside family groups, so a class failing the family size gate necessarily fails the genus
#   gate too; and family centering forces every covered family's mean to exactly 0 regardless of what
#   genus did to the individual vectors. So the group means entering the order level are identical
#   with and without genus. Verified numerically -- order-level |mu| is bit-identical at every
#   min_size tested (ms=2: .045/.045, ms=5: .124/.124, ms=10: .209/.209). The per-class vectors still
#   differ (cos-g 0.6047 vs 0.7729); only the later levels' group statistics coincide.
#
#   HONEST FRAMING: dropping genus does fix bottom-up's coverage pathology, but it also dissolves the
#   very contrast being tested. Without genus, and with global controlled, top-down and bottom-up
#   become nearly the same operation (cos 0.9890; 0.9232 even uncontrolled), and the order level is
#   still near-dead (|mu| .124 uncontrolled, .033 controlled). So T4 is effectively "family centering
#   plus a whisper". PREDICTION: bottomup_fo lands near the already-measured single-level family arm
#   (80.60/75.10/80.51/82.15), i.e. BELOW cascade's 80.84 -- and bottomup_fo vs topdown_of should be
#   a near-tie. Run T4 to establish "bottom-up minus its pathological level stops being bottom-up",
#   not in the expectation of a new best number.
#
# WHY TIERED (GPU budget: each iNat arm is 15 ep; do not run all 13 blindly):
#   T1 (4 arms) answers the direction question BOTH ways -- uncontrolled and global-controlled. Run
#      these first; they are the only ones needed to decide top-down vs bottom-up.
#   T2 (3 arms) is the bottom-up MECHANISM sweep. Predicted: as min_size falls, genus coverage rises,
#      the later levels lose their remaining work (see the ms=2 row: fam .298 ord .045), and the init
#      drifts far outside the measured range (cos-g 0.386). Only worth running if T1's bottom-up arm
#      is interesting rather than clearly dead.
#   T3 (6 arms) is ladder depth + skip-a-level + the over-subtraction control. Cheapest to defer.
#
# ADDITIONAL PRE-REGISTERED PREDICTIONS for the new arms:
#   * g_topdown vs g_bottomup: with global equalized, top-down's three levels all still contribute
#     (.219/.172/.321) while bottom-up's last level is dead (.033). Predict g_topdown >= g_bottomup,
#     and predict the GAP between them is SMALLER than the topdown3-vs-bottomup3 gap (their inits are
#     cos 0.9246 apart vs 0.7382) -- i.e. much of any raw direction effect is really a global-estimate
#     effect. If instead g_bottomup wins, the "fine-first destroys the coarser levels' work" reading
#     is wrong and the whole nested story needs rethinking.
#   * topdown_skip (order,genus; cos-g 0.7474) vs topdown3 (0.7164): family contributes the least of
#     the three top-down levels (|mu| .172). If skipping it ties, the chain can be shortened to two
#     levels and "how many levels" has a cheaper answer than 3.
#   * bottomup3_ms2 (cos-g 0.3859) is the furthest-from-global init in this entire project, well past
#     anything on the measured iNat trend (min 0.719). Predict it BREAKS the r=-0.89 trend and loses
#     -- it is the cleanest test of whether "further from global is better" has a floor on iNat.
#
#   bash scripts/run_center_nested.sh                            # T1 only (default)
#   ARMS="$T2" bash scripts/run_center_nested.sh                 # bottom-up mechanism sweep
#   ARMS="$T4" bash scripts/run_center_nested.sh                 # drop-genus (no-genus) tier
#   ARMS="topdown3 g_topdown" bash scripts/run_center_nested.sh  # ad-hoc subset
#   python scripts/agg_runs.py output/center_nested25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
T1="topdown3 bottomup3 g_topdown g_bottomup"
T2="bottomup3_ms2 bottomup3_ms10 g_bottomup_ms2"
T3="topdown2 bottomup2 topdown4 topdown_skip bottomup_skip static3"
T4="bottomup_fo topdown_of g_bottomup_fo g_topdown_of"
ARMS=${ARMS:-"${T1}"}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_nested25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# each arm -> "<levels> <min_size> <mean_mode>"
arm_spec(){ case "$1" in
  topdown2)         echo "family,genus 5 recompute" ;;
  topdown3)         echo "order,family,genus 5 recompute" ;;
  topdown4)         echo "class,order,family,genus 5 recompute" ;;
  topdown_skip)     echo "order,genus 5 recompute" ;;
  bottomup2)        echo "genus,family 5 recompute" ;;
  bottomup3)        echo "genus,family,order 5 recompute" ;;
  bottomup_skip)    echo "genus,order 5 recompute" ;;
  bottomup3_ms2)    echo "genus,family,order 2 recompute" ;;
  bottomup3_ms10)   echo "genus,family,order 10 recompute" ;;
  g_topdown)        echo "global,order,family,genus 5 recompute" ;;
  g_bottomup)       echo "global,genus,family,order 5 recompute" ;;
  g_bottomup_ms2)   echo "global,genus,family,order 2 recompute" ;;
  bottomup_fo)      echo "family,order 5 recompute" ;;
  topdown_of)       echo "order,family 5 recompute" ;;
  g_bottomup_fo)    echo "global,family,order 5 recompute" ;;
  g_topdown_of)     echo "global,order,family 5 recompute" ;;
  static3)          echo "order,family,genus 5 static" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for arm in ${ARMS}; do
  spec=$(arm_spec "${arm}") || { echo "unknown arm ${arm}"; exit 1; }
  set -- ${spec}; lv="$1"; ms="$2"; mm="$3"
  out="${OUT_ROOT}/inat2018/${arm}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  echo "=== [inat2018] nested ${arm}: levels=${lv} min_size=${ms} mean=${mm} (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE nested \
    PROMPT_CENTER_NESTED_LEVELS "${lv}" PROMPT_CENTER_NESTED_MEAN "${mm}" \
    PROMPT_CENTER_GENUS_MIN "${ms}" \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER nested] ... | pre-norm row norm ...' line FIRST:"
echo "    a min norm near 0 means some class was nearly annihilated and then amplified by renorm."
echo "    topdown3 vs cascade (80.84/75.81/80.57/82.50) is the headline comparison."
echo "    topdown4 vs topdown3 tests the 'saturates past 3 levels' reading -- expect a tie."
echo "    bottomup3 vs genus (80.46/75.22/80.05/82.34) tests 'bottom-up == genus + cleanup'."
echo "    g_topdown vs g_bottomup is the DECONFOUNDED direction test; their gap should be smaller"
echo "    than topdown3-vs-bottomup3's, since prepending global equalizes who estimates it."
