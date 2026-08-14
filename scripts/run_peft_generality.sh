#!/bin/bash
#
# PEFT generality: does prototype centering still help when the image-side adapter is not AdaptFormer?
#
# WHY THIS EXISTS. draft_limitations.tex states the gap outright: "All results use ... AdaptFormer
# (the LIFT+ recipe); we did not verify the effect persists under other PEFT choices (LoRA, plain
# adapters, full fine-tuning)." A contemporaneous CVPR paper on the same LIFT baseline (CUE) closes
# exactly this gap for its own method across five PEFT strategies, so a reviewer will ask.
#
# WHAT MAKES THIS A CLEAN TEST. Centering acts only on the TEXT prototypes that seed the classifier.
# The image-side PEFT module (AdaptFormer / LoRA / Adapter) is inserted inside the visual encoder and
# does not touch the text encoder, text_proj, or image_proj, all of which stay frozen. The
# initialization is therefore bit-identical across these three settings; the ONLY thing that changes
# is the mechanism by which the image encoder co-adapts to it. That isolates the question to "does
# the advantage survive a different adaptation mechanism", with no confound from a different init.
# The script verifies this rather than assuming it (see the init-hash check at the end).
#
# SCOPE. Run on ImageNet-LT and Places-LT only. Those are the two datasets where the method is
# claimed to work (Few +1.59 / +1.61 at 5 seeds). iNaturalist is a documented scope boundary where
# centering is neutral because the tail does not stay near its initialization (drift 0.853 vs the
# rule's 0.15 threshold), so a PEFT sweep there would test nothing about generality.
#
# BOTTLENECK DIMENSIONS are auto-derived by trainer.py from the class count, so they differ per
# dataset and per module by design (LoRA: 2^floor(log2(C/(12*4))), Adapter/AdaptFormer:
# 2^floor(log2(C/(12*2)))). ImageNet-LT gets LoRA dim 16 / Adapter dim 32; Places-LT gets 4 / 8.
#
# HOW TO READ THE RESULT. The claim under test is "the Few-shot gain survives a different adapter",
# NOT "LoRA beats AdaptFormer". classifier_scale=25 was tuned for the AdaptFormer recipe and is held
# fixed here so that baseline and centered arms differ in exactly one thing; absolute accuracy under
# LoRA/Adapter may therefore sit below the AdaptFormer numbers without that meaning anything. Compare
# WITHIN a PEFT block (center - baseline), never across blocks.
#
# Reference (AdaptFormer, 5 seeds, the numbers this must be read against):
#   ImageNet-LT  baseline 78.33 / 81.20 / 77.37 / 73.58   +center 78.51 / 81.06 / 77.42 / 75.17
#                delta    +0.18 / -0.14 / +0.05 / +1.59
#   Places-LT    baseline 52.14 / 51.45 / 52.94 / 51.62   +center 52.24 / 51.18 / 52.65 / 53.23
#                delta    +0.09 / -0.27 / -0.28 / +1.61
#   Single seed here, so read the SIGN and rough magnitude of Delta-Few, not a 0.1-level ranking.
#
# COST: 8 runs. ImageNet-LT ~13 min each, Places-LT ~7 min each -> about 80 minutes total.
#
#   bash scripts/run_peft_generality.sh
#   PEFTS="lora" bash scripts/run_peft_generality.sh          # LoRA only
#   DATASETS="imagenet_lt" bash scripts/run_peft_generality.sh
#   python scripts/agg_runs.py output/peft_generality25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
PEFTS=${PEFTS:-"lora adapter"}
ARMS=${ARMS:-"baseline center"}
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"peft_generality25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

# lift+ sets v.adaptformer True, so each alternative must switch it off explicitly.
peft_args(){ case "$1" in
  adaptformer) echo "v.adaptformer True" ;;
  lora)        echo "v.adaptformer False v.lora True" ;;
  adapter)     echo "v.adaptformer False v.adapter True" ;;
  *) return 1 ;; esac; }

arm_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for peft in ${PEFTS}; do
    pa=$(peft_args "${peft}") || { echo "unknown peft ${peft}"; exit 1; }
    for arm in ${ARMS}; do
      aa=$(arm_args "${arm}") || { echo "unknown arm ${arm}"; exit 1; }
      out="${OUT_ROOT}/${data}/${peft}_${arm}"
      completed "${out}" && { echo "  [skip] ${out}"; continue; }
      echo "=== [${data}] ${peft} / ${arm} (${EPOCHS} ep) ==="
      CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
        classifier_init semantic classifier_scale 25 mda True tte True \
        ${pa} ${aa} num_epochs "${EPOCHS}" seed "${SEED}" output_dir "${out}"
    done
  done
done

echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    Compare center-vs-baseline WITHIN each PEFT block. Delta-Few is the claim; the absolute"
echo "    numbers under LoRA/Adapter are not tuned and are not the point."
echo
echo "=== init-identity check (the premise of this experiment, verified not assumed) ==="
${PYTHON} - <<'PY' || true
import torch, os, glob, torch.nn.functional as F
root=os.environ.get("OUT_ROOT","peft_generality25")
for data in ("imagenet_lt","places_lt"):
    ws={}
    for d in sorted(glob.glob(f"output/{root}/{data}/*_center/ckpts/init/checkpoint.pth.tar")):
        ws[d.split("/")[-4].replace("_center","")]=torch.load(d,map_location="cpu",weights_only=False)["tuner"]["classifier.weight"].float()
    if len(ws)<2: continue
    ks=list(ws); base=ws[ks[0]]
    print(f"  {data}: centered init across PEFT modules ->", ", ".join(
        f"cos({ks[0]},{k})={F.cosine_similarity(base.flatten(),ws[k].flatten(),dim=0):.6f}" for k in ks[1:]))
    print("    (must be 1.000000: centering touches only the text side, so the init cannot depend on the image-side adapter)")
PY
