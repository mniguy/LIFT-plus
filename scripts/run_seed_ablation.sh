#!/bin/bash
#
# Seed ablation (seeds 0..10) for the shared method base (hybrid init + KD 0.001 +
# InfoNCE 0.005, TTE, MDA), in one of two variants via METHOD_VARIANT:
#   warmup : + PEFT image-warmup (ep1, lr 5e-4), NO gating   (= final_tte/verify_best)
#   gate   : + inverse-frequency rank gate (freq_inv), NO warmup
#            (best-Few gating variant from gate_controls; -> seed_ablation_gate/)
#            baselines are NOT re-run for gate (RUN_BASELINE=0); it reuses the
#            existing scale-30 baseline in output/seed_ablation/ for the paired test.
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
#   bash scripts/run_seed_ablation.sh                       # warmup variant + baseline
#   METHOD_VARIANT=gate bash scripts/run_seed_ablation.sh   # gate variant (reuses baseline)
#   SEEDS="0 1 2" DATASETS=imagenet_lt bash scripts/run_seed_ablation.sh
#   RUN_METHOD=0 bash scripts/run_seed_ablation.sh   # baseline only
#   python scripts/agg_seed_ablation.py            # aggregate when done
set -euo pipefail

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10"}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
METHOD_VARIANT=${METHOD_VARIANT:-warmup}  # warmup | gate
RUN_METHOD=${RUN_METHOD:-1}      # set 0 to run only the baseline

[ -f main.py ] || { echo "ERROR: run from the repo root (main.py not found)"; exit 1; }

# classifier_scale 30 is pinned everywhere: the whole seed_ablation family used 30,
# but the config default has since drifted to 25 (which is ~0.4 worse on Few).

# --- method recipe: shared base (hybrid init + KD + InfoNCE), variant differs ---
case "${METHOD_VARIANT}" in
  warmup)
    # PEFT image-warmup (ep1, lr 5e-4), NO gating.
    METHOD_ARGS=(
      classifier_init hybrid classifier_scale 30
      TEXT_REG_LAMBDA 0.001 INFONCE_LAMBDA 0.005
      PRIOR_REG_MODE fixed
      HYBRID_CAPTION_SOURCE wiki HYBRID_TOPK 8 SIM_THRESHOLD 0.6
      mda True tte True num_epochs 5
      PEFT_WARMUP True PEFT_WARMUP_EPOCHS 1 PEFT_WARMUP_LR 5e-4 PEFT_WARMUP_IMAGE True
    )
    DEFAULT_ROOT="final_tte/seed_ablation"
    DEFAULT_RUN_BASELINE=1
    ;;
  gate)
    # best-Few gating variant = inverse-frequency rank gate (freq_inv), NO warmup.
    METHOD_ARGS=(
      classifier_init hybrid classifier_scale 30
      TEXT_REG_LAMBDA 0.001 INFONCE_LAMBDA 0.005
      HYBRID_CAPTION_SOURCE wiki HYBRID_TOPK 8 SIM_THRESHOLD 0.6
      mda True tte True num_epochs 5
      PEFT_WARMUP False
      PRIOR_REG_MODE class_gate PRIOR_GATE_SOURCE frequency
      PRIOR_GATE_NORM rank PRIOR_GATE_INVERT True PRIOR_GATE_POWER 1.0
    )
    DEFAULT_ROOT="seed_ablation_gate"
    DEFAULT_RUN_BASELINE=0   # reuse existing scale-30 baseline in output/seed_ablation
    ;;
  *)
    echo "ERROR: unknown METHOD_VARIANT='${METHOD_VARIANT}' (use warmup|gate)"; exit 1 ;;
esac

OUT_ROOT=${OUT_ROOT:-$DEFAULT_ROOT}
RUN_BASELINE=${RUN_BASELINE:-$DEFAULT_RUN_BASELINE}  # set 0 to run only the method

# --- paired baseline: LIFT+ (semantic init, no KD/InfoNCE, no warmup, no gate) ---
BASELINE_ARGS=(
  classifier_init semantic classifier_scale 30
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
      echo "=== [${data}] seed=${s} : method (${METHOD_VARIANT}) ==="
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
if [ "${RUN_BASELINE}" = "1" ]; then
  echo "  ${PYTHON} scripts/agg_seed_ablation.py --root \"output/${OUT_ROOT}\" --datasets ${DATASETS}"
else
  # methods live in OUT_ROOT; pair against the existing scale-30 baseline root
  echo "  ${PYTHON} scripts/agg_seed_ablation.py --root \"output/${OUT_ROOT}\" \\"
  echo "      --baseline-root output/seed_ablation --datasets ${DATASETS}"
fi
