#!/bin/bash
#
# PROMPT_CENTER_MODE=proj -- LEAST-SQUARES SUBSPACE REMOVAL (2026-08-21).
#
#   c_hat = argmin_c || O - sum_k c_k mu_k ||          out = O - sum_k c_hat_k mu_k
#
# i.e. project the prototype onto the orthogonal complement of the subspace its level means span.
#
# ============================ WHY THIS IS A DIFFERENT KIND OF COMBINATION ============================
# Every other way of combining the level means fixes the weights and subtracts a weighted average:
#     sum_all / sumB   equal weights (a linear ramp on the level EFFECTS)
#     blend            one knob s, uniform shrinkage of every level effect
#     taxo_kernel      geometric decay gamma^d in taxonomic distance
#     nested           sequential subtraction on the running residual
# This one does not fix them at all -- it SOLVES for the coefficients per class, so the weights adapt
# to each prototype. Zero free parameters beyond which levels are allowed into the span.
#
# TWO COMBINATION SCHEMES THAT WERE MEASURED AND DROPPED (both collapse onto arms already run):
#   * adding the global term to sumB, i.e. summing all 7 residuals: that is O - (s/7)*sum mu_k versus
#     sum_all's O - (1/7)*sum mu_k -- the same arm at alpha 0.963 vs 1. Measured cos 0.9987 to
#     sum_all (80.68). Not queued.
#   * reliability weighting w_k = (n_k - 1)/(n_k - 1 + tau), which weights a level by how many
#     members that class's group has and gives a singleton group weight EXACTLY 0 with no branch.
#     Elegant, but iNat's taxonomy groups are mostly large enough that w saturates: at tau = 4 the
#     mean weights are global .171 kingdom .171 phylum .169 class .167 order .155 family .122
#     genus .046, i.e. nearly uniform, and the result is cos 0.996-0.999 to sum_all for tau in
#     [1, 16]. Not queued.
#
# ============================ THE GATE IS LOAD-BEARING ============================
# If a class is alone in its group then mu_LEVEL = O, so O lies exactly IN the span and the residual
# is exactly ZERO. Measured with no gate: 2579 zero rows -- the failure that gave mode=level an
# accuracy of 0.01 (see scripts/run_center_res0.sh for the mechanism). A level therefore enters the
# span only when that class's group has >= PROMPT_CENTER_GENUS_MIN members. At its natural value 2
# this is not an arbitrary cutoff: a group of one carries no information about the class beyond the
# class itself, so admitting it is exactly the degenerate case.
#
# ============================ ARMS: a 2x2 FACTORIAL ============================
# Two factors, crossed, so the gate value and the presence of the genus level can be read separately:
#
#   arm            levels                gate   zero  span dim  cos-to-global  top5conf
#   proj             all 7                 2      0     6.56/7     0.5271       0.3963
#   proj_ms5         all 7                 5      0     6.08/7     0.6966       0.4313
#   proj_nog         genus excluded (6)    2      0     5.93/6     0.7323       0.4749
#   proj_nog_ms5     genus excluded (6)    5      0     5.80/6     0.7672       0.4859
#
# All four are independent of every already-run arm (closest: blend s=0.92 at 0.8918) and mutually
# distinct -- the tightest pair is proj_nog vs proj_nog_ms5 at per-class cos 0.9592, the widest is
# proj vs proj_nog_ms5 at 0.6824. Full matrix:
#            proj   proj_ms5  proj_nog  proj_nog_ms5
#   proj    1.0000    0.7914    0.7200      0.6824
#   ms5     0.7914    1.0000    0.8502      0.8910
#   nog     0.7200    0.8502    1.0000      0.9592
#   nog_ms5 0.6824    0.8910    0.9592      1.0000
#
# WHAT THE TWO FACTORS MEAN:
#   gate 2 vs 5  -- 2 is forced (a group of one carries no information beyond the class itself), 5 is
#                   the constant cascade and nested both fix, whose original justification (a
#                   cos-to-global sweep) no longer holds. Contrast isolates whether it matters.
#   genus in/out -- genus is the level with the coverage problem (3000 of 8142 classes are alone in
#                   theirs) and also the level carrying the most shared structure (46.4% of the
#                   prototype norm). Dropping it removes both at once.
# proj at gate 2 has top5conf 0.3963, the LOWEST measured anywhere in this project (previous best
# taxo_kernel gamma=0 at 0.4167; global centering is 0.6536).
#
# DROPPED after measuring: the span WITHOUT the global term (kingdom..genus, gate 2). mu_kingdom is
# nearly mu_global -- kingdom has only 6 groups and Animalia dominates -- so the span hardly changes:
# per-class cos 0.9848 to proj. Available as arm proj_noglob if wanted.
#
# ============================ PREDICTION ============================
# NO GEOMETRIC PREDICTION. On this dataset cos-to-global correlates with All at r = -0.37 across the
# 15 arms where both were measured, top5conf at r = -0.36, and the sumA control (78% of rows starting
# out pointing AWAY from their own class) still scored 80.59 against a baseline of 80.63. The
# 0.72-0.75 "winning band" reasoning from the older headers is retired.
# What this pair does test, cleanly: proj vs proj_ms5 differ in ONE variable, the gate. cascade and
# nested both use 5, and the justification for that value (a cos-to-global sweep) no longer holds.
# If the gate factor is flat, proj at gate 2 stands as a zero-constant method. BASE RATE: 71 iNat centering arms span 80.46 - 81.02, and no arm has yet
# beaten bottomup3_gL at 81.02.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline 80.63 74.62 80.50 82.36   global 80.52 74.86 80.41 82.13   sum_all 80.68 75.22 80.38 82.48
#   shrink genus 80.85 75.18 80.41 82.90   cascade 80.84 75.81 80.57 82.50
#   bottomup3_gL 81.02 75.93 80.71 82.75  <- the best arm measured in this project
#   iNat seed noise (5-ep/scale-30 proxy): All ~0.06, Head ~0.74, Med ~0.16, Few ~0.23.
#
#   bash scripts/run_center_proj.sh                       # the 2x2: proj proj_ms5 proj_nog proj_nog_ms5
#   ARMS="proj_noglob" bash scripts/run_center_proj0.sh   # the near-redundant no-global span
#   python scripts/agg_runs.py output/center_proj --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
ARMS=${ARMS:-"ridge01 ridge05 pick"}   # the 2x2 (proj proj_ms5 proj_nog proj_nog_ms5) is already run
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_proj"}
ALL7="global,kingdom,phylum,class,order,family,genus"
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# arm -> "<mode> <levels> <gate> <ridge>"
arm_spec(){ case "$1" in
  # the 2x2 factorial (already run): proj 80.75, proj_ms5 80.58, proj_nog 80.51, proj_nog_ms5 80.74
  proj)         echo "proj ${ALL7} 2 0.00000001" ;;
  proj_ms5)     echo "proj ${ALL7} 5 0.00000001" ;;
  proj_nog)     echo "proj global,kingdom,phylum,class,order,family 2 0.00000001" ;;
  proj_nog_ms5) echo "proj global,kingdom,phylum,class,order,family 5 0.00000001" ;;
  proj_noglob)  echo "proj kingdom,phylum,class,order,family,genus 2 0.00000001" ;;
  # --- RIDGE: the size gate replaced by smooth regularization (gate switched off at 1) ---
  # lambda -> 0 is the plain projection, lambda -> inf leaves the prototype untouched. A singleton
  # level puts O inside the span and blows the coefficients up; the ridge caps that with no cutoff.
  # Measured with the gate off, zero rows / min pre-norm row norm / cos-to-global / top5conf:
  #   1e-2  0 / 0.042 / 0.5669 / 0.4039   <- smallest rows are numerically at the degenerate case
  #   1e-1  0 / 0.392 / 0.7254 / 0.4109
  #   5e-1  0 / 1.692 / 0.9009 / 0.4654
  # 1e-2 is NOT queued: a min row norm of 0.042 against a raw row of ~23 is the same near-annihilation
  # the gate exists to prevent. Distinctness: ridge01 is cos 0.9287 to proj, ridge05 is cos 0.9539 to
  # sum_all (80.68).
  ridge01)      echo "proj ${ALL7} 1 0.1" ;;
  ridge05)      echo "proj ${ALL7} 1 0.5" ;;
  # --- PICK: one level per class, chosen by ALIGNMENT instead of by group size ---
  # k* = argmax_k cos(mu_k, O). cascade also picks one level per class, but the deepest one whose group
  # is big enough -- a coverage rule, not an explanatory one. Same gate, same reason: a singleton group
  # has mu = O, would score cos = 1, and would zero the row.
  # Measured at gate 2: genus 4678 | family 1639 | order 575 | global 492 | kingdom 327 | class 245 |
  # phylum 186 -- 3464 classes prefer a coarser level than cascade would hand them.
  # zero rows 0, min row norm 1.561, cos-to-global 0.5631, top5conf 0.4170, cos 0.9534 to proj.
  pick)         echo "pick ${ALL7} 2 0.00000001" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for arm in ${ARMS}; do
  spec=$(arm_spec "${arm}") || { echo "unknown arm ${arm}"; exit 1; }
  set -- ${spec}; mode="$1"; lv="$2"; gate="$3"; ridge="$4"
  out="${OUT_ROOT}/inat2018/${arm}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  echo "=== [inat2018] ${arm}: mode=${mode} levels=${lv} gate=${gate} ridge=${ridge} (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE "${mode}" \
    PROMPT_CENTER_LEVEL "${lv}" PROMPT_CENTER_GENUS_MIN "${gate}" \
    PROMPT_CENTER_PROJ_RIDGE "${ridge}" \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read each log's '[PROMPT_CENTER proj] ...' line FIRST. It MUST say 0/8142 rows are ZERO;"
echo "    a nonzero count means a singleton level got into the span and the run is worthless."
echo "    Expected mean span dim: proj 6.56/7, proj_ms5 6.08/7, proj_nog 5.93/6, proj_nog_ms5 5.62/6."
echo "    Read it as a 2x2: (proj, proj_ms5) vs (proj_nog, proj_nog_ms5) is the genus factor,"
echo "    and (proj, proj_nog) vs (proj_ms5, proj_nog_ms5) is the gate factor. Main effects first,"
echo "    interaction only if both main effects clear the seed noise."
