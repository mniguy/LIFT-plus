#!/bin/bash
#
# #4 + #5 follow-up (2026-07-24 brainstorm): two changes landed in _center_prototypes
# (trainer.py) and need a GPU run to validate.
#
#   #4 normalize fix: the default single-template PROMPT_MODE fed _center_prototypes raw
#      (non-unit) CLIP text-encoder rows -- measured CoV~0.10 norm spread on real iNat names,
#      but direction change was small (cos(mu_old,mu_new)=0.9998, mean per-class centered-output
#      shift 0.058). Now X is row-normalized before any mean/cov. This changes EVERY mode's
#      result slightly, incl. plain "global" -- so re-running "global" here with identical
#      settings to the existing center_seeds25 anchor isolates the fix's real effect: diff
#      this run's global Few/Head/Med against the old center_seeds25 (5-seed) numbers.
#
#   #5 kappa mode (new): tail's rank-based rarity (1 - rank/(C-1)) gives Med only a middling,
#      hard-to-tune partial correction. kappa mode swaps in the same count-based shrink already
#      used by img_shrink: rarity_c = kappa/(n_c+kappa) (rare->1, common->0, kappa = the count at
#      which rarity=0.5). Sweep kappa to find a sweet spot between fewonly (kappa->0) and global
#      (kappa->inf) that keeps tail's Med preservation win while recovering more of global's Few.
#
#   Reference anchors already on record (pre-fix, for comparison):
#     baseline (center_seeds25):     IN 78.33/81.20/77.37/73.58   PL 52.15/51.45/52.94/51.62
#     global   (center_seeds25):     IN 78.51/81.01/77.46/75.17   PL 52.24/51.19/52.65/53.23
#     tail     (center_geom25):      IN 78.52/80.98/77.65/74.62   PL 52.43/51.50/52.88/53.08
#     fewonly  (center_control25):   IN 78.38/80.94/77.47/74.31   PL 52.35/51.54/52.58/53.32
#
#   bash scripts/run_center_kappa.sh
#   python scripts/agg_runs.py output/center_kappa25 --sort path
set -euo pipefail
GPU_ID=${GPU_ID:-0}; PYTHON=${PYTHON:-python}; SEED=${SEED:-0}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
VARIANTS=${VARIANTS:-"global tail kappa5 kappa20 kappa50"}
OUT_ROOT=${OUT_ROOT:-"center_kappa25"}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

BASE_ARGS=(
  classifier_init semantic classifier_scale 25 PROMPT_CENTER True
  TEXT_REG_LAMBDA 0.0 INFONCE_LAMBDA 0.0
  mda True tte True num_epochs 5 PEFT_WARMUP False
)
variant_args(){ case "$1" in
  global|tail) echo "PROMPT_CENTER_MODE $1" ;;
  kappa5)      echo "PROMPT_CENTER_MODE kappa PROMPT_CENTER_KAPPA 5" ;;
  kappa20)     echo "PROMPT_CENTER_MODE kappa PROMPT_CENTER_KAPPA 20" ;;
  kappa50)     echo "PROMPT_CENTER_MODE kappa PROMPT_CENTER_KAPPA 50" ;;
  *) return 1 ;; esac; }

completed(){ grep -lq "\* Many:" "./output/$1"/log-*.txt 2>/dev/null; }

for data in ${DATASETS}; do
  for v in ${VARIANTS}; do
    va=$(variant_args "$v") || { echo "unknown variant $v"; exit 1; }
    out="${OUT_ROOT}/${data}/${v}"
    completed "${out}" && { echo "  [skip] ${out}"; continue; }
    echo "=== [${data}] center-kappa ${v} ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} main.py -d "${data}" -b clip_vit_b16 -m lift+ \
      "${BASE_ARGS[@]}" ${va} seed "${SEED}" output_dir "${out}"
  done
done
echo; echo "=== tabulate: ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path ==="
echo "    read: 'global' here (post-fix) vs center_seeds25 'center' (pre-fix) = #4's real effect"
echo "    read: kappa5/20/50 vs tail vs fewonly on Med/Head preservation + Few recovery = #5's sweep"
