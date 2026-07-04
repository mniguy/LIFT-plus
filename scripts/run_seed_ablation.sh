#!/bin/bash
#
# Seed ablation (seeds 0..10) for THE warmup recipe = output/final_tte/verify_best.
#
# The recipe: hybrid caption init + KD (0.001) + InfoNCE (0.005), NO gating, with
# TTE, MDA, and PEFT image-warmup (1 epoch, lr 5e-4). This is the only run in the
# whole sweep whose ImageNet Few beat the LIFT+ baseline (73.87 -> 74.37, +0.5).
#
# NOTE: current config.py defaults have DRIFTED away from that run
#   classifier_init=semantic, tte=False, PEFT_WARMUP=False, PEFT_WARMUP_EPOCHS=2
# so every recipe knob below is set EXPLICITLY -- do not rely on defaults.
#
# A single-seed method sweep cannot separate a real gain from the ~0.8 run-noise
# band, so by default we ALSO run the paired baseline (semantic, no-warmup) at the
# SAME seeds. The paired mean+-std on Few is the number that decides salvage-or-drop.
# Set RUN_BASELINE=0 to run the method only, or RUN_METHOD=0 to run the baseline only.
#
#   bash scripts/run_seed_ablation.sh
#   SEEDS="0 1 2" DATASETS=imagenet_lt bash scripts/run_seed_ablation.sh
#   RUN_METHOD=0 bash scripts/run_seed_ablation.sh   # baseline only
#   python scripts/agg_seed_ablation.py            # aggregate when done
set -euo pipefail

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10"}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
RUN_METHOD=${RUN_METHOD:-1}      # set 0 to run only the baseline
RUN_BASELINE=${RUN_BASELINE:-1}  # set 0 to run only the method
OUT_ROOT=${OUT_ROOT:-"final_tte/seed_ablation"}

[ -f main.py ] || { echo "ERROR: run from the repo root (main.py not found)"; exit 1; }

# --- the warmup recipe: everything that departs from a plain 'lift+' run ---
METHOD_ARGS=(
  classifier_init hybrid
  TEXT_REG_LAMBDA 0.001
  INFONCE_LAMBDA 0.005
  PRIOR_REG_MODE fixed
  HYBRID_CAPTION_SOURCE wiki
  HYBRID_TOPK 8
  SIM_THRESHOLD 0.6
  mda True
  tte True
  num_epochs 5
  PEFT_WARMUP True
  PEFT_WARMUP_EPOCHS 1
  PEFT_WARMUP_LR 5e-4
  PEFT_WARMUP_IMAGE True
)

# --- paired baseline: LIFT+ (semantic init, no KD/InfoNCE, no warmup) ---
BASELINE_ARGS=(
  classifier_init semantic
  TEXT_REG_LAMBDA 0.0
  INFONCE_LAMBDA 0.0
  PRIOR_REG_MODE fixed
  mda True
  tte True
  num_epochs 5
  PEFT_WARMUP False
)

# already-finished run? (a log with a final '* Many:' line)
completed () { grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

run_one () {
  local data=$1 out=$2; shift 2
  if completed "$out"; then echo "  [skip] ${out} (already completed)"; return; fi
  echo "  [run ] ${out}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
    -d "${data}" -b clip_vit_b16 -m lift+ \
    "$@" \
    output_dir "${out}"
}

for data in ${DATASETS}; do
  for s in ${SEEDS}; do
    if [ "${RUN_METHOD}" = "1" ]; then
      echo "=== [${data}] seed=${s} : method (warmup) ==="
      run_one "${data}" "${OUT_ROOT}/${data}/method_seed${s}" "${METHOD_ARGS[@]}" seed "${s}"
    fi
    if [ "${RUN_BASELINE}" = "1" ]; then
      echo "=== [${data}] seed=${s} : baseline (LIFT+) ==="
      run_one "${data}" "${OUT_ROOT}/${data}/baseline_seed${s}" "${BASELINE_ARGS[@]}" seed "${s}"
    fi
  done
done

echo ""
echo "=== aggregate ==="
echo "  ${PYTHON} scripts/agg_seed_ablation.py --root output/${OUT_ROOT} --datasets ${DATASETS}"
