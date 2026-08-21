#!/bin/bash
#
# PROMPT_CENTER_MODE=level -- single-taxonomy-level centering with NO fallback and NO min_size gate.
#
#   out = O - mean(O over the class's group at LEVEL)          [renormalized, as always]
#
# Every taxonomy mode so far (genus / cascade / nested) guards small groups: a group below
# PROMPT_CENTER_GENUS_MIN falls back to the global centroid (genus/cascade) or is skipped (nested).
# The guard exists because a SINGLETON group's mean is the class itself, so O - mu is exactly 0.
# This script removes the guard on purpose, to measure what the guard was buying -- i.e. is landing
# on the zero vector actually harmful, or was the fallback machinery solving a non-problem?
#
# 6 arms: genus, family, order, class, phylum, kingdom.
# LEVEL=global is NOT run here: global never had a fallback, so mode=level with LEVEL=global is
# bit-identical to PROMPT_CENTER_MODE=global (verified), and that number already exists (80.52).
#
# ============================ WHAT THE ZERO ROWS COST (categories.json census) ============================
#   level     groups   classes in a SINGLETON group -> classifier rows that init to exactly 0
#   genus      4401    3000 / 8142   (36.8%)   <- the arm where this is a real intervention
#   family     1118     463 / 8142   ( 5.7%)
#   order       272      64 / 8142   ( 0.8%)
#   class        57       9 / 8142
#   phylum       25       5 / 8142
#   kingdom       6       1 / 8142
# F.normalize leaves a zero row at zero, so with CosineClassifier those classes start with a logit
# that is identically 0 for every image -- a dead init that training has to recover from scratch.
# The trainer prints the exact zero-row count per run ("[PROMPT_CENTER level] ... rows are ZERO");
# read that line before reading the accuracies.
#
# PRE-REGISTERED PREDICTION: genus should be the only arm that visibly moves, and downward -- 37% of
# classes losing their init is far more damage than any centering variant has done on this dataset.
# family and coarser have <6% zero rows and should land near their guarded counterparts. If genus
# does NOT lose, the fallback machinery in genus/cascade/nested was never load-bearing, which is the
# more interesting outcome of the two.
#
#
# ==================== OFFLINE GEOMETRY (2026-08-21, CPU, real 8142 iNat prototypes) ====================
# Encoded "a photo of a {}." with CLIP ViT-B/16 (pre-projection, as PEFT_Text returns), then applied
# init_classifier_weight's text_proj/image_proj chain. Validated against this repo's earlier offline
# numbers: raw within-genus 0.944 -> 0.9424, global-centered 0.834 -> 0.8256.
#   arm              zero  cos-to-global  top5conf  within-genus
#   raw (baseline)      0     0.5668       0.9050      0.9424
#   global (=ref)       0     1.0000       0.6536      0.8256
#   level/genus      3000     0.3635       0.2732     -0.1729
#   level/family      463     0.8152       0.5071      0.5529
#   level/order        64     0.9076       0.5837      0.7423
#   level/class         9     0.9490       0.6138      0.7970
#   level/phylum        5     0.9592       0.6208      0.8018
#   level/kingdom       1     0.9689       0.6279      0.8101
# (cos-to-global is computed over the SURVIVING rows only; the zero rows have no direction.)
# level/kingdom at 0.9689 is effectively a re-run of PROMPT_CENTER_MODE=global.
#
# ==================== THE ZERO ROW IS AN OPTIMIZER BUG, NOT A WEAK INIT ====================
# CosineClassifier.forward does F.normalize(self.weight) EVERY step, and F.normalize is
# w / clamp_min(||w||, 1e-12). At w = 0 the clamp is active, so the denominator is a CONSTANT 1e-12
# and the backward pass multiplies the incoming gradient by 1e12. Measured on a 3-row toy:
#     grad norm per row: [9.4e12, 1.99, 2.76]      <- zero row vs two normal rows
# Simulated with this repo's actual optimizer (SGD lr=0.02 momentum=0.9 wd=5e-4), 40% dead rows:
#     step 0: |grad| dead=1.08e+11  alive=1.66e-01   |w| dead=0
#     step 1: |grad| dead=4.93e-09  alive=1.42e-01   |w| dead=2.16e+09
#     after 3000 steps: |w| dead=1.6e10 / alive=1.37; cos(dead,target)=0.129 vs cos(alive,target)=0.898
# So a zero row EXPLODES to ||w|| ~1e9-1e10 in ONE step and is then FROZEN, because its gradient is
# suppressed by 1/||w|| forever after. Weight decay only shrinks it ~40% over 15 epochs (51k steps).
# AMP does not save it: 1e12 is finite, so GradScaler does not skip the step.
#   => the PROMPT_CENTER_GENUS_MIN guard in genus/cascade/nested was never protecting against "a zero
#      vector is a weak starting point". It was protecting against detonating the classifier. THAT is
#      the finding of this script; the accuracies below are just how much damage it does.
#   => iNat val is exactly 3 images/class (24426 = 8142 x 3), so All == mean per-class accuracy and
#      the damage is a clean fraction of the class list.
#   => singleton-genus classes are spread across all three splits: 24.3% of Many, 39.7% of Med,
#      36.5% of Few. This is NOT a tail-only intervention.
#
# ==================== PRE-REGISTERED PREDICTIONS (written 2026-08-21, before running) ====================
#   arm      dead classes   All predicted   reasoning
#   genus     3000 (36.8%)    50 - 62       dead rows freeze at a near-common direction; 3000 classes
#                                           cannot be rescued by the encoder (they are mutually
#                                           indistinguishable), so they score ~5-25% instead of ~80%
#   family     463 ( 5.7%)    76.5 - 78.0   5.7% x (80.6 - ~15)
#   order       64 ( 0.8%)    80.0 - 80.4   just above seed noise (sigma_All ~ 0.06)
#   class        9            80.4 - 80.8   -0.07 expected; buried in noise
#   phylum       5            80.4 - 80.8   -0.04; indistinguishable
#   kingdom      1            80.4 - 80.8   -0.008; a duplicate of mode=global (80.52)
#   FALSIFIERS: genus >= 78 means the gradient analysis above is wrong -- check the classifier weight
#   norms in that run before believing the accuracy. family >= 80 means the encoder CAN rescue dead
#   classes, which would be the more interesting outcome and would reopen the guard question.
#
# ==================== WHAT ACTUALLY HAPPENED (2026-08-21, run) ====================
# ALL THREE ARMS RUN (genus 3000 dead, family 463, order 64) CAME BACK ~0 ON ALL / HEAD / MED / FEW.
# The predictions above (proportional damage: family ~77, order ~80) were WRONG. The damage is NOT
# proportional to the dead-row count -- ONE dead row is enough to zero the entire evaluation. Stage 1
# below is the gradient explosion already documented; stage 2 is what makes it global.
#
#   STAGE 1 (training, fp32 -- localized, and SILENT):  F.normalize(0) backward multiplies by 1e12,
#     the row jumps to ||w|| ~1e6-1e9 in ONE step and then freezes. Verified in a faithful fp32
#     reproduction with this repo's optimizer (SGD 0.02/0.9/5e-4) and a trainable encoder:
#         dead rows   0     8      60     400   (of 1000 classes)
#         All acc   1.000  0.992  0.943  0.641
#         alive-class acc 1.000 in every case; encoder never corrupted; loss never NaN.
#     So in fp32 this stage alone would have produced roughly the predicted proportional numbers, and
#     THE TRAINING LOG LOOKS COMPLETELY NORMAL -- loss curves give no warning at all.
#
#   STAGE 2 (evaluation, fp16 -- GLOBAL):  utils/config.py sets prec_test = "fp16" and trainer.py
#     calls self.model.half() before testing. An exploded row (elements ~1e7) OVERFLOWS fp16
#     (max 65504) -> inf; then CosineClassifier's F.normalize(inf_vector) = inf/inf = NaN. Measured:
#         3 exploded rows of 1000 -> 2299/2304 entries inf after .half() -> 3 NaN rows
#         -> 3 NaN logit COLUMNS -> every one of 64 test samples has a NaN in its logit row
#         -> argmax returns the first NaN index for EVERY image -> all predictions collapse to one
#            dead class -> accuracy 0 on every split.
#     The count of dead rows is irrelevant: 1 is as fatal as 3000. That is why genus, family and
#     order returned the same number.
#
#   => class / phylum / kingdom (9 / 5 / 1 dead rows) WILL ALSO RETURN 0. Do not spend GPU on them.
#   => This script therefore CANNOT answer its own scientific question ("is landing on 0 actually
#      bad for those classes?"). It dies of a numerical bug before the question is reached. To make
#      it measurable, add `prec_test fp32` to the command below -- that removes stage 2 and leaves
#      the proportional stage-1 damage, which is what the predictions above describe.
#   => scripts/run_center_res1.sh (level_keep, 2*O - mu) has NO zero rows and is unaffected.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   genus, guarded min_size=5 (center_local25)   80.46  75.22  80.05  82.34   <- the guarded twin of arm 'genus'
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50
#   iNat seed noise (5-ep/scale-30 proxy): All ~0.06, Head ~0.74, Med ~0.16, Few ~0.23.
#
#   bash scripts/run_center_res0.sh                       # all 6 arms (~6 x 15 ep)
#   ARMS="genus family" bash scripts/run_center_res0.sh   # the two that can move
#   python scripts/agg_runs.py output/center_res0 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
ARMS=${ARMS:-"genus family order class phylum kingdom"}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_res0"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for lv in ${ARMS}; do
  out="${OUT_ROOT}/inat2018/${lv}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  echo "=== [inat2018] level (O - mu) at ${lv}, no fallback (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE level PROMPT_CENTER_LEVEL "${lv}" \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER level] ... rows are ZERO' line FIRST -- that count is the"
echo "    size of the intervention (genus 3000, family 463, order 64, class 9, phylum 5, kingdom 1)."
echo "    genus (unguarded) vs the guarded genus arm 80.46/75.22/80.05/82.34 is the headline pair."
echo "    Companion: scripts/run_center_res1.sh, same levels with out = 2*O - mu (no zero rows)."
