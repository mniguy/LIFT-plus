#!/bin/bash
#
# NULL CONTROL: global-then-cascade vs plain cascade on iNaturalist.
#
# WHAT IT TESTS. Not a new method. The two initializations are provably the same operation:
# mean subtraction is linear, so removing the global centroid first cancels out of every group mean
# computed afterwards,
#     (X - mu_g) - [mu_group(X) - mu_g] = X - mu_group(X),
# and the 368 classes that reach the global fallback see a residual whose global mean is already zero.
# Verified offline on the real prototypes: per-class cosine 1.00000000 (min 0.99999964), max
# elementwise difference 2e-7, i.e. float32 rounding.
#
# WHY RUN IT ANYWAY. It is the cleanest possible measurement of the run-to-run floor on this dataset.
# The previous estimate came from bottomup3 vs bottomup3_gL, whose initializations were cosine 0.9990
# with 38 classes below 0.99 -- genuinely different vectors, so trajectory divergence there was not
# separable from a real (if tiny) operational difference. Here there is no operational difference at
# all, so whatever accuracy gap appears is attributable to nothing but run-to-run variation.
#
# TWO OUTCOMES, both worth having:
#   identical (or within ~0.02) => training is effectively deterministic at this precision, and the
#     0.2-0.4 pt spreads seen across the 30+ arms reflect real initialization differences after all,
#     which would sharpen how those sweeps should be read.
#   differs by ~0.2 pt          => a float32-rounding perturbation is enough to reroute training. That
#     is the strongest possible statement of the resolution limit, and it is what justifies reporting
#     the iNaturalist arm comparisons as a group-level effect only.
#
# Reference (same protocol, seed 0, 15 ep):
#   cascade (center_local25/inat2018/cascade)  80.84 / 75.81 / 80.57 / 82.50
#   Read the delta against that. Also compare the log's assignment line, which must match exactly:
#   genus=2279 family=4427 order=1068 global=368.
#
#   bash scripts/run_cascade_nullctrl.sh
#   python scripts/agg_runs.py output/global_cascade25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-1}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
INAT_EPOCHS=${INAT_EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-"global_cascade25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

ARMS=${ARMS:-"cascade_then_global"}
arm_args(){ case "$1" in
  global_then_cascade) echo "PROMPT_CENTER_CASCADE_GLOBAL_FIRST True" ;;
  cascade_then_global) echo "PROMPT_CENTER_CASCADE_GLOBAL_LAST True" ;;
  *) return 1 ;; esac; }

for arm in ${ARMS}; do
  aa=$(arm_args "${arm}") || { echo "unknown arm ${arm}"; exit 1; }
  out="${OUT_ROOT}/inat2018/${arm}"
  completed "${out}" && { echo "  [skip] ${out}"; continue; }
  echo "=== [inat2018] ${arm} (null control, ${INAT_EPOCHS} ep) ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d inat2018 -b clip_vit_b16 -m lift+ \
    classifier_init semantic classifier_scale 25 mda True tte True \
    PROMPT_CENTER True PROMPT_CENTER_MODE cascade \
    PROMPT_CENTER_CASCADE genus,family,order ${aa} \
    num_epochs "${INAT_EPOCHS}" seed "${SEED}" output_dir "${out}"
done
echo; echo "=== compare against plain cascade: 80.84 / 75.81 / 80.57 / 82.50 ==="
echo "    the log must show 'global centroid removed first' AND the same assignment counts"
echo "    (genus=2279 family=4427 order=1068 global=368); if either differs, the run is not the control."
${PYTHON} - <<'PY' || true
import torch, os, torch.nn.functional as F
root=os.environ.get('OUT_ROOT','cascade_nullctrl25')
b="output/center_local25/inat2018/cascade/ckpts/init/checkpoint.pth.tar"
W=lambda p: F.normalize(torch.load(p,map_location="cpu",weights_only=False)["tuner"]["classifier.weight"].float(),dim=-1)
for arm,exp in [("global_then_cascade","1.0000000"),("cascade_then_global","0.999937")]:
    a=f"output/{root}/inat2018/{arm}/ckpts/init/checkpoint.pth.tar"
    if os.path.exists(a) and os.path.exists(b):
        c=(W(a)*W(b)).sum(-1)
        print(f"    {arm:20s} vs cascade: per-class cosine mean={c.mean():.7f} min={c.min():.7f}  (expect {exp})")
PY
