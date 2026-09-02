#!/bin/bash
#
# Multi-GPU fan-out for run_center_levelcode.sh (2026-08-28; re-split for 9 no-global arms 2026-08-31).
#
# THE NAME IS HISTORICAL. This no longer assumes four cards -- GPUS says which ones to use, and it
# defaults to "2 3": GPU 0 and GPU 1 are in use by someone else's train_linear_probe.py. Set
# GPUS="0 1 2 3" (and give each ARMS_GPU<n> a list) to go back to four.
#
# WHY NOT SHARE A BUSY CARD: measured 2026-08-31 on GPU 1 against a 2.6 GB neighbour -- memory was
# never the problem (17.2 GB of 49 GB) but SM contention took the arm from 0.29 to 0.72 s/batch,
# a 2.4x slowdown, ETA 4 h -> 10 h. It also slows the neighbour. Wait for a free card instead.
#
# The grid, what the codes mean, and how to read the numbers live in run_center_levelcode.sh --
# READ THAT HEADER FIRST. This file only decides WHICH ARM RUNS ON WHICH GPU. It shells out to
# that script once per GPU with GPU_ID and ARMS set, so there is exactly one copy of the
# experiment logic and this file never drifts from it on the science.
#
# TRACKING THE PARENT (last synced 2026-08-31):
#   - no S / PROMPT_CENTER_NESTED_S: each level subtracts the FULL mean, X <- X - mu_level(X)
#   - PROMPT_CENTER_GENUS_MIN 2: a group of one is SKIPPED at that level, not zeroed
#   - ARMS is 8 no-global codes: 12 34 56 135 246 123 456 123456
#   - RENORM defaults False, and the parent header now records that RENORM=True measured as a
#     dead knob (mean dAll +0.057 against a 0.082 noise floor). Do not spend GPUs re-testing it.
#   - output dirs are bare  c<code>  -- no _s..._rn... suffix
# If the parent's ARMS changes again, the per-GPU lists below must be re-split BY HAND; nothing
# here derives them automatically.
#
# RENORM AND THE OUTPUT PATH: the dir is bare c<code>, so RENORM is not encoded in it. The parent
# now takes SUFFIX for exactly this -- RENORM=True SUFFIX=_rnT writes c<code>_rnT beside the
# RENORM=False result instead of on top of it. See run_center_rn_noglobal_4gpu.sh for a set that
# uses it. (Repointing OUT_ROOT also works, but then the two halves cannot be tabulated together.)
#
# ============================ THE SPLIT ============================
# 8 arms, 2 GPUs (2 and 3 -- GPU 0 and 1 are taken), iNat2018 @ 15 epochs each, 4 arms per card.
# ~4 h per arm on a card of its own, so ~16 h for the set.
#
# The split is by ROUND: each GPU walks its list in order, so the k-th arm on both GPUs finishes
# at roughly the same wall-clock time.
#
#            GPU 2   | GPU 3          genus-reaching is marked (Y)
#   round 1     56(Y)|  12
#   round 2     34   | 246(Y)
#   round 3    456(Y)| 123
#   round 4    135   | 123456(Y)
#
#   GPU 2     56  34  456  135        (4 arms: 2 reach genus, 2 do not)
#   GPU 3     12 246  123  123456     (4 arms: 2 reach genus, 2 do not)
#
# COUNTERBALANCED ON PURPOSE, two ways at once. Whether the chain reaches genus is the only
# variable known to move All (parent header: 80.968 vs 80.707, t=5.69), so it must not be aliased
# with anything else:
#   - every ROUND is one genus-reaching arm against one that is not, so a set stopped early is
#     still interpretable;
#   - every GPU carries two of each, so a card-to-card difference cannot masquerade as the effect.
# The obvious lazy split (all genus arms on one card) would confound exactly the thing being
# measured. Do not "simplify" it back.
#
# Round 1 lands the sharpest pair in the grid: 56 against the already-finished 056. Those two
# differ only on the 463 classes family skips, so their gap is the whole point of the set.
#
# ============================ ONE SET AT A TIME ============================
# The guard below refuses to start if any run_center_levelcode.sh is already running. This is
# not hypothetical: on 2026-08-28 a second set was launched on top of one that was 22 min into
# epoch 2. Both slowed from 0.29 to 0.61 s/batch and the older set did not survive it. The
# parent's completed() cannot catch this -- it only knows about arms that already FINISHED.
# Verified firing by running this script while a matching process was alive.
#
# USAGE
#   bash scripts/run_center_levelcode_4gpu.sh              # launch every GPU in GPUS, background, wait
#   # from the host, detached -- survives your SSH dropping and this terminal closing:
#   docker exec -d -w /home/mingyu/mingyu/LIFT-plus mingyu bash -c \
#     'setsid nohup bash scripts/run_center_levelcode_4gpu.sh \
#        > output/center_levelcode25/_launch/launcher.log 2>&1 < /dev/null'
#   DRY_RUN=1 bash scripts/run_center_levelcode_4gpu.sh    # print the split, run nothing
#   FORCE=1   bash scripts/run_center_levelcode_4gpu.sh    # bypass the already-running guard
#
# Arms already finished (log contains "* Many:") are skipped by the parent, so re-running this
# after a crash resumes instead of redoing work.
#
#   tail -f output/center_levelcode25/_launch/launch_gpu*.log
#   python scripts/agg_runs.py output/center_levelcode25 --sort path
set -euo pipefail
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

if [ "${FORCE:-0}" != "1" ] && ps -eo args | grep -v grep | grep -q "run_center_levelcode\.sh"; then
  echo "ERROR: run_center_levelcode.sh is already running -- refusing to start a duplicate set."
  echo "       inspect:  ps -eo pid,pgid,etime,args | grep -E 'levelcode|main\.py' | grep -v grep"
  echo "       override: FORCE=1 bash scripts/run_center_levelcode_4gpu.sh"
  exit 1
fi

# THE ENV is the conda env "ltl" INSIDE the running "mingyu" docker container:
#     /home/mingyu/.conda/envs/ltl/bin/python   (torch 2.8.0+cu128, 4 GPUs, timm 1.0.25, sklearn 1.8.0)
# It lives under /home/mingyu, which is bind-mounted in, so it survives a container restart.
# Note the container login shell activates py310, which has NO torch.
#
# PYTHONNOUSERSITE=1 IS LOAD-BEARING -- do not drop it. /home/mingyu/.local/lib/python3.11 holds
# a stray user-site copy of clip/timm/yacs/sklearn/tensorboard at NEWER versions (timm 1.0.28,
# sklearn 1.9.0, tensorboard 2.21.0). User-site takes precedence over the env, so without this
# flag the runs silently import those instead of ltl's pinned versions.
PYTHON=${PYTHON:-$([ -x /home/mingyu/.conda/envs/ltl/bin/python ] \
  && echo /home/mingyu/.conda/envs/ltl/bin/python || echo python)}
export PYTHONNOUSERSITE=1
OUT_ROOT=${OUT_ROOT:-"center_levelcode25"}
DRY_RUN=${DRY_RUN:-0}

# Which cards to drive, and what each one runs. A GPU listed in GPUS with an empty ARMS_GPU<n>
# is skipped with a warning rather than crashing under `set -u`.
GPUS=${GPUS:-"2 3"}                       # GPU 0 and 1 are in use by someone else
ARMS_GPU0=${ARMS_GPU0:-""}
ARMS_GPU1=${ARMS_GPU1:-""}
ARMS_GPU2=${ARMS_GPU2:-"56 34 456 135"}
ARMS_GPU3=${ARMS_GPU3:-"12 246 123 123456"}

LOG_DIR="output/${OUT_ROOT}/_launch"
mkdir -p "${LOG_DIR}"

pids=(); pid_gpus=()
for g in ${GPUS}; do
  eval "arms=\${ARMS_GPU${g}:-}"
  if [ -z "${arms}" ]; then echo "GPU ${g}: no ARMS_GPU${g} set -- skipping"; continue; fi
  log="${LOG_DIR}/launch_gpu${g}.log"
  echo "GPU ${g}: ${arms}   -> ${log}"
  if [ "${DRY_RUN}" = "1" ]; then continue; fi
  GPU_ID="${g}" ARMS="${arms}" PYTHON="${PYTHON}" OUT_ROOT="${OUT_ROOT}" \
    bash scripts/run_center_levelcode.sh > "${log}" 2>&1 &
  pids+=("$!"); pid_gpus+=("${g}")     # keep the GPU id: array index != GPU number once GPUS != 0..3
done

[ "${DRY_RUN}" = "1" ] && { echo "(dry run, nothing launched)"; exit 0; }

echo; echo "launched ${#pids[@]} jobs: ${pids[*]}   (Ctrl-C kills this waiter, NOT the jobs)"
fail=0
for i in "${!pids[@]}"; do
  g="${pid_gpus[$i]}"
  wait "${pids[$i]}" || { echo "!! GPU ${g} exited nonzero -- see ${LOG_DIR}/launch_gpu${g}.log"; fail=1; }
done

echo; echo "=== all GPUs done (fail=${fail}) ==="
echo "  ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path"
exit "${fail}"
