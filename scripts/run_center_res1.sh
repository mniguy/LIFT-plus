#!/bin/bash
#
# PROMPT_CENTER_MODE=level_keep -- single-level centering that ADDS THE PROTOTYPE BACK.
#
#   out = 2*O - mean(O over the class's group at LEVEL)        [renormalized, as always]
#
# Same no-fallback, no-min_size setup as scripts/run_center_res0.sh, with one change that removes
# that script's failure mode: a SINGLETON group gives 2*O - O = O, i.e. the class degrades to its
# RAW uncentered prototype instead of to the zero vector. The extra +O is applied to EVERY class,
# not just singletons, so the operation stays uniform across the class list (no per-class branch).
#
# WHAT THIS IS, GEOMETRICALLY: rows are renormalized at the end, so 2*O - mu points the same
# direction as O - 0.5*mu. level_keep is therefore exactly the HALF-STRENGTH point of the shrinkage
# family O - alpha*mu; run_center_res0.sh's 'level' is alpha = 1. Read the pair as an alpha sweep at
# two points, not as two unrelated formulas -- if level_keep wins across the board it is evidence
# about centering STRENGTH, and the singleton-repair story is a secondary effect that only touches
# the few classes listed below.
#
# 7 arms: global, genus, family, order, class, phylum, kingdom. global IS run here (unlike res0):
# 2*O - mu_global is a genuinely new arm, not a rerun of PROMPT_CENTER_MODE=global.
#
# HOW MUCH OF THIS IS THE SINGLETON REPAIR (categories.json census, classes in a singleton group):
#   genus 3000 / family 463 / order 64 / class 9 / phylum 5 / kingdom 1 / global 0
# Only at genus is the repair a large-scale effect; at order and coarser, level_keep vs level is
# essentially a pure alpha=0.5-vs-1.0 contrast on ~99% of the classes.
#
# PRE-REGISTERED PREDICTIONS:
#   * global: half-strength global centering. Full-strength global is 80.52 vs baseline 80.63, i.e.
#     roughly a wash on iNat, so predict this lands between them -- an interpolation, and a cheap
#     check that the alpha axis behaves monotonically before reading anything into the other arms.
#   * genus: predict this beats res0's genus clearly, since res0's genus kills 37% of the init.
#     The real question is whether it also beats the GUARDED genus (80.46) -- if it does, "add O
#     back" is a better small-group fix than "fall back to global mu", and cascade/nested inherit a
#     simpler repair.
#   * order/class/phylum/kingdom: mu is close to the global centroid at these levels, so these are
#     near-duplicates of the 'global' arm at half strength. Expect a tight cluster; a spread wider
#     than the ~0.06 All / ~0.23 Few seed noise would be surprising and worth chasing.
#
#
# ==================== OFFLINE GEOMETRY (2026-08-21, CPU, real 8142 iNat prototypes) ====================
# Same measurement pipeline as scripts/run_center_res0.sh (see that header for validation).
#   arm              zero  cos-to-global  top5conf  within-genus
#   raw (baseline)      0     0.5668       0.9050      0.9424
#   global (=ref)       0     1.0000       0.6536      0.8256
#   keep/global         0     0.8084       0.8004      0.8846
#   keep/genus          0     0.5930       0.8604      0.8060
#   keep/family         0     0.7292       0.8020      0.8482
#   keep/order          0     0.7718       0.7906      0.8679
#   keep/class          0     0.7889       0.7916      0.8773
#   keep/phylum         0     0.7927       0.7928      0.8790
#   keep/kingdom        0     0.7964       0.7939      0.8806
# Zero rows are 0 everywhere, as designed -- the +O term removes run_center_res0.sh's failure mode
# entirely (that script's header explains why a zero row detonates the cosine classifier).
#
# ==================== PRE-REGISTERED PREDICTIONS (written 2026-08-21, before running) ====================
# BASE RATE, stated first because it dominates: all 55 iNat centering arms measured in this project so
# far land in 80.46 - 81.02, a total spread of 0.56 pts. Even diff_init, pre-registered as risky at
# cos-to-global 0.56, came back 80.57 -- a tie with baseline 80.63. iNat is nearly insensitive to init
# geometry, and these arms have no zero rows and only half-strength centering. PREDICT A WASH.
#   arm            cos-g    All predicted   reasoning
#   keep/family    0.7292     80.6 - 80.9   dead centre of the 0.72-0.75 band where every arm that has
#                                           won on this dataset sits (cascade 0.743 -> 80.84). Best bet.
#   keep/order     0.7718     80.5 - 80.8   band edge
#   keep/class     0.7889     80.5 - 80.8   |
#   keep/phylum    0.7927     80.5 - 80.8   |- mu is nearly the global mean at these levels; expect
#   keep/kingdom   0.7964     80.5 - 80.8   |  these three to cluster within 0.1 of keep/global
#   keep/global    0.8084     80.5 - 80.8   alpha=0.5 global; should land between full-strength global
#                                           (80.52) and baseline (80.63)
#   keep/genus     0.5930     80.4 - 80.7   outside the band (diff_init territory) and top5conf 0.8604
#                                           is poor; expect it near the guarded genus arm (80.46)
# PREDICT NONE OF THESE BEAT cascade (80.84) OR g_bottomup_fo (81.02). Mechanism: 2*O - mu only pulls
# top5conf down to 0.79-0.86 where global centering alone reaches 0.6536, and top5conf (confusion with
# the 5 nearest classes) is iNat's actual bottleneck. These arms are "weak global centering", and
# global centering on iNat is already a measured tie (80.52 vs baseline 80.63).
#   FALSIFIER / NEXT STEP: if keep/family clears 81.0, alpha is a real lever -- open it as a config
#   knob next to PROMPT_CENTER_LEVEL and sweep alpha in {0.25, 0.5, 0.75, 1.0}. If all 7 land within
#   0.3 of each other, the alpha axis is dead and should not be pursued.
#
# ==================== RESULT (2026-08-21, 4 of 7 arms run) ====================
#   arm       All    Head    Med    Few
#   global   80.59  74.70  80.28  82.53
#   genus    80.59  74.70  80.32  82.45
#   order    80.55  75.10  80.32  82.27
#   family   80.52  74.66  80.08  82.60
#   TOTAL SPREAD ON All = 0.07, vs seed noise sigma_All ~ 0.06. The four arms are indistinguishable.
#
# PRE-REGISTERED PREDICTION SCORING (see the block above; nothing edited after the fact):
#   "PREDICT A WASH"                                          -> CORRECT (0.07 spread)
#   keep/global lands between global 80.52 and baseline 80.63 -> CORRECT (80.59)
#   keep/genus 80.4-80.7 / keep/order 80.5-80.8               -> CORRECT
#   keep/family 80.6-80.9, "best bet"                         -> WRONG (80.52, and it came LAST;
#                                                                the ranking claim was never
#                                                                resolvable at this noise level)
#   "none of these beat cascade 80.84 / g_bottomup_fo 81.02"   -> CORRECT (max 80.59)
#   The stated falsifier fires: "if all 7 land within 0.3 of each other, the alpha axis is dead and
#   should not be pursued." Spread is 0.07. THE ALPHA AXIS IS DEAD. Do not open alpha as a config knob.
#
# WHY, mechanically: normalize is scale-invariant, so 2*O - mu == O - 0.5*mu after row-normalization,
# i.e. these arms are the alpha = 0.5 midpoint of the shrinkage family O - alpha*mu (alpha = 0 is raw
# baseline 80.63, alpha = 1 is full centering / mode=global 80.52). Those two ENDPOINTS differ by 0.11
# (~2 sigma), so nothing strictly between them can differ by more. Measured per-class cosine between
# the four arms' inits: 0.9485 - 0.9847, i.e. they are nearly the same initialization.
#
# STRONGEST EVIDENCE THAT THE OPERATION ITSELF DOES NOTHING ON iNat: in keep/genus, 3000/8142 classes
# (36.8%) revert to 2*O - O = O and receive NO centering at all, while keep/global centers every class
# uniformly. Those two arms scored IDENTICALLY (80.59 both). The composition of the arm does not matter
# because the centering is not doing measurable work here.
#
# => class / phylum / kingdom (cos-to-global 0.789-0.796, i.e. near-duplicates of keep/global) are
#    guaranteed ties. Not worth the GPU.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   genus, guarded min_size=5 (center_local25)   80.46  75.22  80.05  82.34
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50
#   iNat seed noise (5-ep/scale-30 proxy): All ~0.06, Head ~0.74, Med ~0.16, Few ~0.23.
#
#   bash scripts/run_center_res1.sh                        # all 7 arms (~7 x 15 ep)
#   ARMS="global genus" bash scripts/run_center_res1.sh    # the two informative ones first
#   python scripts/agg_runs.py output/center_res1 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
ARMS=${ARMS:-"class phylum kingdom"}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_res1"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for lv in ${ARMS}; do
  out="${OUT_ROOT}/inat2018/${lv}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  echo "=== [inat2018] level_keep (2*O - mu) at ${lv}, no fallback (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE level_keep PROMPT_CENTER_LEVEL "${lv}" \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    every arm here must report '0/8142 rows are ZERO' -- if not, the +O term did not apply."
echo "    Pair each arm with the same level in output/center_res0 (alpha=1.0 vs alpha=0.5)."
echo "    genus is the only level where the singleton repair touches many classes (3000/8142)."
