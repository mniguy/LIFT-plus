#!/bin/bash
#
# SHOULD-ADD (G) -- does centering survive a change of PEFT module?
#
# WHY. Every reported number uses AdaptFormer (configs/method/lift+.yaml sets v.adaptformer),
# yet draft_intro_method.tex Sec. "Preliminaries" says "adapted by lightweight modules
# (AdaptFormer/LoRA)" and draft_discussion.tex states a general design principle for "a small
# trainable adapter" under "a frozen encoder". Those are claims about a family; the evidence
# covers one member. draft_limitations.tex already concedes this.
#
#   Prediction of the paper's own mechanism: the effect is a property of the INITIALIZATION,
#   not of the adapter, so Delta-Few should stay positive under LoRA and Adapter. Under full
#   fine-tuning it should SHRINK toward zero, because FFT gives every class enough capacity to
#   leave its init (the same argument the paper uses for iNaturalist's neutral result).
#   That makes FFT a prediction, not just a robustness check -- report it either way.
#
# NOTE on FFT. -m fft unfreezes the whole visual encoder. It needs far more memory than PEFT
# and lr=0.02 (the PEFT default) is too large for it; set FFT_LR to something like 1e-3.
# It is therefore opt-in: VARIANTS="adaptformer lora adapter fft" to include it.
#
#   bash scripts/run_peft_variants.sh
#   VARIANTS="adaptformer lora adapter fft" FFT_LR=0.001 bash scripts/run_peft_variants.sh
#   python scripts/agg_runs.py output/peft_variants25 --sort path
#
# Cost (default): 3 PEFT x 2 arms x 3 seeds x 2 datasets = 36 runs @ 5 ep.
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"adaptformer lora adapter"}
ARMS=${ARMS:-"baseline center"}
SEEDS=${SEEDS:-"0 1 2"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}
FFT_LR=${FFT_LR:-0.001}
OUT_ROOT=${OUT_ROOT:-"peft_variants25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=( classifier_init semantic classifier_scale "${SCALE}" mda True tte True )

# method config + the overrides that switch the visual PEFT module off/on
peft_method(){ case "$1" in fft) echo "fft" ;; *) echo "lift+" ;; esac; }
peft_args(){ case "$1" in
  adaptformer) echo "" ;;                                     # lift+ default
  lora)        echo "v.adaptformer False v.lora True" ;;
  adapter)     echo "v.adaptformer False v.adapter True" ;;
  fft)         echo "lr ${FFT_LR}" ;;                          # -m fft already sets v.fft True
  *) return 1 ;; esac; }

arm_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for p in ${VARIANTS}; do
    m=$(peft_method "${p}")
    pa=$(peft_args "${p}") || { echo "unknown PEFT ${p}"; exit 1; }
    for a in ${ARMS}; do
      aa=$(arm_args "${a}") || { echo "unknown arm ${a}"; exit 1; }
      for s in ${SEEDS}; do
        out="${OUT_ROOT}/${data}/${p}_${a}_seed${s}"
        completed "${out}" && { echo "  [skip] ${out}"; continue; }
        echo "=== [${data}] ${p} / ${a} seed=${s} ==="
        CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
          -d "${data}" -b clip_vit_b16 -m "${m}" \
          "${BASE_ARGS[@]}" num_epochs "${EPOCHS}" ${pa} ${aa} \
          seed "${s}" output_dir "${out}"
      done
    done
  done
done
echo
echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    Report Delta-Few per PEFT module. Positive under LoRA/Adapter => the effect is a"
echo "    property of the init. Shrinking under FFT => confirms the capacity account."
