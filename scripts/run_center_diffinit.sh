#!/bin/bash
#
# PROMPT_CENTER_MODE=diff_init -- variant A': the lexical diff IS the initialization (2026-08-07).
#
# W0 = normalize( d - mean(d) ),  d_c = embed("a photo of a Genus species.") - embed("a photo of a species.")
#
# Every other lexical mode (genus_lex, cascade_lex) uses d only to build something to SUBTRACT from the
# real prototype. This one throws the prototype away and initializes from d itself, then centers it.
#
# ⚠️ PREREQUISITE BUG FIX (2026-08-07): genus_lex/cascade_lex previously unit-normalized ONLY the
# epithet side before subtracting, while X arrives as raw CLIP text features (norm ~8.6 on B/16 --
# compute_prompt_class_features' single-template path does not normalize). The subtraction was
# therefore ~20x too small: measured cos(diff, X) = 0.9968, i.e. "diff" was numerically just X and the
# lexical isolation was a no-op. This also explains why the completed cascade_lex run sat at
# cos 0.9896 to plain cascade. Both branches now use raw-scale epithet features (cos(diff,X) = 0.1700).
# THE EXISTING output/center_cascadelex25 RESULT (80.62) IS INVALID and should be discarded or re-run.
#
# MEASUREMENT-SPACE NOTE (found while checking the first run's log): trainer-side numbers and the
# offline table below live in DIFFERENT spaces. PEFT_Text.forward returns the EOT feature BEFORE
# text_projection (mean norm ~23), and init_classifier_weight applies text_proj/image_proj afterwards;
# the offline probe used vanilla clip.encode_text, which applies text_projection (mean norm ~8.6).
# Hence the log prints cos(diff,X)=0.38 where the offline probe measured 0.17 -- same operation, two
# spaces, NOT a bug. The offline ORDERING still transfers because group-mean subtraction and the diff
# are linear and commute with a linear projection: verified on the nested arms, where offline
# cos-to-global predictions matched the saved inits to ~0.003 (topdown3 0.7164 predicted / 0.7191
# actual).
#
# OFFLINE GEOMETRY (2026-08-07, real CLIP ViT-B/16 CPU encode of all 8142 iNat classnames,
# scripts/probe_diff_init.py; lower within-genus / top5-conf is better, top5-conf = mean cosine to
# each class's 5 nearest OTHER classes, which is iNat's actual confusion bottleneck):
#   init                     within-genus  within-family  all-pairs  top5-conf   |mu|   cos-to-glo
#   raw (baseline)               0.9442        0.7366       0.6899     0.9106   0.8306    0.572
#   global centering             0.8340        0.1673       0.0040     0.6679   0.0641    1.000
#   A  diff, uncentered          0.6903        0.2499       0.1660     0.6103   0.4076    0.481
#   A' diff, centered  <-- THIS  0.6236        0.1082       0.0044     0.5570   0.0676    0.560
#   B  X - diff (= epithet)      0.7869        0.7820       0.7783     0.9258   0.8822    0.219
#   B' epithet, centered         0.0290        0.0226       0.0010     0.6433   0.0328    0.368
# A' is the best thing measured on top5-conf (0.557 vs global's 0.668) AND on within-genus among the
# variants that keep any semantic content. Blends norm(Xc)+a*norm(Dc) were also measured for
# a in {0.25,0.5,1,2} and none beat A' (within-genus 0.81-0.89), so A' is the one worth a GPU slot.
#
# B' is the instructive trap and is NOT queued: its within-genus (0.029) is 20x better than A' yet its
# top5-conf is WORSE (0.643 vs 0.557) -- congeneric species got spread apart while each class's actual
# nearest neighbours stayed close, i.e. the arrangement lost its semantic organization rather than
# gaining discriminability. This reproduces the paper's epithet-only finding (tab:epithet: 0.026 /
# 0.024 / 0.001, matching the 0.0290 / 0.0226 / 0.0010 measured here).
#
# ⚠️ HONEST PRIOR -- reasons this may well lose, written before running:
#   1. It replaces the SEMANTIC target geometry ("what this species is") with a LEXICAL one ("what the
#      genus word contributes"). Under the paper's own mechanism the prototypes are the targets the
#      image encoder aligns features to, so the encoder now has a much larger remapping to learn.
#   2. cos-to-global 0.560 is far outside the 0.72-0.75 band where every arm that has actually won on
#      iNat sits (cascade 0.743, topdown3 0.719, g_topdown 0.732), and close to where the failed
#      zero-shot centering experiment lived.
#   3. top5-conf has NOT predicted accuracy in this project: in the single-level ladder, family had the
#      BEST top5-conf (0.483) yet only 80.60, while genus at 0.492 came last at 80.46. So the geometry
#      above justifies testing this, not expecting it to win.
#
# Reference anchors (iNat, 15 ep, seed 0, scale 25, mda+tte -- identical args to below):
#   baseline (breadth25/inat2018/baseline_15ep)  80.63  74.62  80.50  82.36
#   global   (breadth25/inat2018/center_15ep)    80.52  74.86  80.41  82.13
#   cascade  (center_local25/inat2018/cascade)   80.84  75.81  80.57  82.50   <- best so far
#   topdown3 (center_nested25/inat2018)          80.91  75.34  80.86  82.42
#   iNat seed noise (5ep/scale30 proxy): All ~0.06, Head ~0.74, Med ~0.16, Few ~0.23. Head gaps under
#   ~1.5 pts are not interpretable at one seed.
#
#   bash scripts/run_center_diffinit.sh
#   python scripts/agg_runs.py output/center_diffinit25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"center_diffinit25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

out="${OUT_ROOT}/inat2018/diff_init"
if completed "${out}"; then
  echo "  [skip] ${out}"
else
  echo "=== [inat2018] diff_init (A': centered lexical diff as W0) (${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE diff_init \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
fi
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    CHECK FIRST: the log's '[PROMPT_CENTER diff_init] ... cos(diff,X)=' should be ~0.38 and"
echo "    mean|X| ~23 (both measured on the first real run). Only a value near 1.0 means the epithet"
echo "    subtraction silently did nothing (the bug fixed 2026-08-07) -- kill the run then."
