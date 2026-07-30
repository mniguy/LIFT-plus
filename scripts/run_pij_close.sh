#!/bin/bash
#
# Closes the two statistical holes in the P/I/J argument (fixes #4 and #5).
#
# HOLE 1 (fix #4) -- P2 has no negative control under the intervention that defines it.
#   run_freeze_center.sh compared frozen {baseline, center} only. If freezing the classifier is
#   what reveals the latent init advantage, then freezing a J control (randdir: a random direction
#   of matched norm, which does NOT remove mu) must reveal NOTHING. Without this cell, "freezing
#   proves the centered init is better" is compatible with "freezing helps any perturbed init".
#
# HOLE 2 (fix #5) -- the whole causal core is single-seed.
#   frozen center gave Few +11.62 (IN) / +6.62 (PL) at n=1, and the three trainable J controls are
#   also n=1. Effect sizes are large relative to the baseline seed std (IN Few sigma=0.32), but a
#   reviewer asking "11pp from one seed?" has no answer in the current data.
#
# What this runs (IN/PL only -- iNat freeze is 3h/run and stays at n=1 with a stated caveat):
#   A) frozen  x {baseline, center, randdir} x seeds {0,1,2}      -> 18 runs
#   B) trainable J x {randdir, headonly, perclass_rand} x seeds {1,2}  -> 12 runs
#      (seed 0 already exists in output/center_control25 with identical settings and is pooled
#       by scripts/agg_pij.py, so it is not re-run here)
#
# Cost: ImageNet ~15 min/run, Places ~5 min/run.
#   A = 3 variants x 3 seeds x (15+5) = 3.0 h      B = 3 x 2 x (15+5) = 2.0 h      total ~5 h
# Split by dataset across two GPUs to roughly halve it:
#   GPU_ID=0 DATASETS="imagenet_lt" bash scripts/run_pij_close.sh &
#   GPU_ID=1 DATASETS="places_lt"   bash scripts/run_pij_close.sh &
#
#   Predictions (write these down before reading results):
#     frozen center  - frozen baseline : LARGE positive Few (reproduce ~+11.6 IN / ~+6.6 PL)
#     frozen randdir - frozen baseline : ~0 or negative Few   <- the cell that makes P2 selective
#     trainable J at 3 seeds           : all stay at or below baseline Few, gaps hold up
#
#   bash scripts/run_pij_close.sh
#   python scripts/agg_pij.py
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
SEEDS_FROZEN=${SEEDS_FROZEN:-"0 1 2"}
SEEDS_TRAINABLE=${SEEDS_TRAINABLE:-"1 2"}
FROZEN_VARIANTS=${FROZEN_VARIANTS:-"baseline center randdir"}
TRAINABLE_VARIANTS=${TRAINABLE_VARIANTS:-"randdir headonly perclass_rand"}
EPOCHS=${EPOCHS:-5}
STAGES=${STAGES:-"frozen trainable"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# byte-identical to run_freeze_center.sh / run_center_control.sh apart from FREEZE_CLASSIFIER
COMMON=(
  classifier_init semantic classifier_scale 25
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True num_epochs "${EPOCHS}" PEFT_WARMUP False
)
variant_args(){ case "$1" in
  baseline)      echo "PROMPT_CENTER False" ;;
  center)        echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  randdir)       echo "PROMPT_CENTER True PROMPT_CENTER_MODE randdir" ;;
  headonly)      echo "PROMPT_CENTER True PROMPT_CENTER_MODE headonly" ;;
  perclass_rand) echo "PROMPT_CENTER True PROMPT_CENTER_MODE perclass_rand" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

# Reuse the seed-0 frozen runs that already exist under a different name. freeze_center25 holds
# frozen {baseline, center} at seed 0 with byte-identical settings (verified: FREEZE_CLASSIFIER
# True, scale 25, 5 ep, seed 0), so re-running them would burn ~40 min for nothing.
# agg_pij.py already reads those legacy dirs for seed 0. randdir has no legacy run and IS needed
# at seed 0 -- that is the whole point of the new cell.
legacy_frozen(){  # $1=data $2=variant $3=seed -> echoes a legacy dir, or nothing
  [ "$3" = "0" ] || return 0
  case "$2" in baseline|center) echo "freeze_center25/$1/$2" ;; esac
}

run_one(){  # $1=data $2=variant $3=seed $4=outdir $5..=extra args
  local data="$1" v="$2" s="$3" out="$4"; shift 4
  completed "${out}" && { echo "  [skip] ${out}"; return; }
  local va; va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
  echo "=== [${data}] ${v} seed ${s} -> ${out} ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
    "${COMMON[@]}" "$@" ${va} seed "${s}" output_dir "${out}"
}

for data in ${DATASETS}; do
  for stage in ${STAGES}; do
    case "${stage}" in
      frozen)
        for v in ${FROZEN_VARIANTS}; do
          for s in ${SEEDS_FROZEN}; do
            leg=$(legacy_frozen "${data}" "${v}" "${s}")
            if [ -n "${leg}" ] && completed "${leg}"; then
              echo "  [reuse] ${data} frozen ${v} seed ${s} <- output/${leg}"
              continue
            fi
            run_one "${data}" "${v}" "${s}" "pij_frozen25/${data}/${v}_seed${s}" FREEZE_CLASSIFIER True
          done
        done ;;
      trainable)
        for v in ${TRAINABLE_VARIANTS}; do
          for s in ${SEEDS_TRAINABLE}; do
            run_one "${data}" "${v}" "${s}" "pij_control25/${data}/${v}_seed${s}"
          done
        done ;;
      *) echo "unknown stage ${stage}"; exit 1 ;;
    esac
  done
done
echo; echo "=== analyze: ${PYTHON} scripts/agg_pij.py ==="
