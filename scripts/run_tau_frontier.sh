#!/bin/bash
#
# MUST-ADD (A) -- the head/tail trade-off frontier control.
#
# WHY. The headline claim is "Few +1.59, Many essentially unchanged" (tables_centering.tex,
# tab:main). A reviewer's first objection is that ANY strengthening of the logit adjustment
# buys Few at the cost of Many for free, so centering may just be a reparameterization of
# tau. This script traces that trade-off curve explicitly for BOTH arms.
#
#   Pass condition: at every tau, the centered arm's (Many, Few) point lies ABOVE the
#   baseline arm's curve -- i.e. centering is not on the baseline frontier, it moves it.
#   Fail condition: some baseline tau reproduces (Many -0.14, Few +1.59). Then the paper's
#   contribution collapses to "tune tau", and this must be reported.
#
# HOW. LA's tau is hard-coded to 1.0 in utils/losses.py (LogitAdjustedLoss), so we sweep it
# through VSLoss, which is exactly LA when gamma=0:
#     VS:  logit/Delta_j + iota_j,  Delta_j=(n_max/n_j)^gamma,  iota_j=tau*log(n_j/N)
#     LA:  logit + tau*log(n_j/N)
# gamma=0 => Delta_j=1 => identical. VS_TAU=1.0 therefore REPRODUCES the LA baseline and is
# included as a built-in sanity check: tau=1.00/baseline must match
# "seed_ablation 25" (IN 78.33+/-0.06 / Few 73.58+/-0.24; PL 52.15+/-0.10 / Few 51.62+/-0.33).
# Everything else (scale 25, MDA, TTE, 5 ep, semantic init) is identical to tab:main.
#
#   bash scripts/run_tau_frontier.sh
#   TAUS="1.00" SEEDS="0" bash scripts/run_tau_frontier.sh     # sanity check only (4 runs)
#   DATASETS=imagenet_lt bash scripts/run_tau_frontier.sh
#   python scripts/agg_runs.py output/tau_frontier25 --sort path
#
# Cost: 6 taus x 2 arms x 3 seeds x 2 datasets = 72 runs @ 5 ep.
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
TAUS=${TAUS:-"0.50 0.75 1.00 1.25 1.50 2.00"}
VARIANTS=${VARIANTS:-"baseline center"}
SEEDS=${SEEDS:-"0 1 2"}
SCALE=${SCALE:-25}
EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-"tau_frontier25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale "${SCALE}"
  loss_type VS VS_GAMMA 0.0
  mda True tte True
)
variant_args(){ case "$1" in
  baseline) echo "PROMPT_CENTER False" ;;
  center)   echo "PROMPT_CENTER True PROMPT_CENTER_MODE global" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for t in ${TAUS}; do
    tag="tau$(echo "${t}" | tr -d '.')"        # 0.75 -> tau075
    for v in ${VARIANTS}; do
      va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
      for s in ${SEEDS}; do
        out="${OUT_ROOT}/${data}/${v}_${tag}_seed${s}"
        completed "${out}" && { echo "  [skip] ${out}"; continue; }
        echo "=== [${data}] ${v} tau=${t} seed=${s} ==="
        CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py \
          -d "${data}" -b clip_vit_b16 -m lift+ \
          "${BASE_ARGS[@]}" VS_TAU "${t}" num_epochs "${EPOCHS}" ${va} \
          seed "${s}" output_dir "${out}"
      done
    done
  done
done
echo
echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    Plot (Many, Few) per arm. The claim survives only if the centered curve is"
echo "    strictly outside the baseline curve, not a point on it."
