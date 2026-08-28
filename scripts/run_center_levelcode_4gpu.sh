#!/bin/bash
#
# 4-GPU fan-out for run_center_levelcode.sh (2026-08-28).
#
# The grid, what the codes mean, and how to read the numbers live in run_center_levelcode.sh --
# READ THAT HEADER FIRST. This file only decides WHICH ARM RUNS ON WHICH GPU. It shells out to
# that script once per GPU with GPU_ID and ARMS set, so there is exactly one copy of the
# experiment logic and this file never drifts from it on the science.
#
# TRACKING THE PARENT (it changed on 2026-08-28 -- this launcher was updated to match):
#   - no S / PROMPT_CENTER_NESTED_S: each level subtracts the FULL mean, X <- X - mu_level(X)
#   - PROMPT_CENTER_GENUS_MIN 2: a group of one is SKIPPED at that level, not zeroed
#   - ARMS is 10 codes: 1 2 3 4 5 6 012 034 056 0135   (0246 / 0123 / 0123456 are gone)
#   - output dirs are now bare  c<code>  -- no _s..._rn... suffix
# If the parent's ARMS changes again, the four lists below must be re-split by hand; nothing
# here derives them automatically.
#
# GOTCHA created by the new bare c<code> path: RENORM is no longer encoded in the output dir,
# so a RENORM=True rerun lands on top of the RENORM=False result for the same code. Point
# OUT_ROOT somewhere else for such a rerun.
#
# ============================ THE SPLIT ============================
# 10 arms, 4 GPUs, iNat2018 @ 15 epochs each. SINGLE-DIGIT CODES GO FIRST -- the six one-level
# arms are the reference column the multi-level codes are read against, and under the new
# settings (full mean, GENUS_MIN 2) none of them has been measured yet, so nothing in the grid
# is interpretable until they land. The four multi-level codes are queued behind them.
#
# The split is by ROUND: each GPU walks its list in order, so the k-th arm on every GPU finishes
# at roughly the same wall-clock time.
#
#   round 1     2 |   1 |   4 |    3    <- singles
#   round 2     6 |   5 | 056 | 0135    <- last two singles; the 2-arm GPUs start their multi
#   round 3   012 | 034                 <- remaining multis
#
#   GPU 0     2  6  012     (3 arms)
#   GPU 1     1  5  034     (3 arms)
#   GPU 2     4  056        (2 arms)
#   GPU 3     3  0135       (2 arms)
#
# Round 1 (~6 h) gives four of the six singles. The reference column is complete after round 2,
# which is also when the first two multi-level codes finish. NOTE the 80.58-80.85 single-level
# table in the parent header came from mode=shrink at s=0.963 -- it is NOT a baseline for these
# full-mean runs, which is exactly why these six arms have to be rerun here.
#
# ============================ ONE SET AT A TIME ============================
# The guard below refuses to start if any run_center_levelcode.sh is already running. This is
# not hypothetical: on 2026-08-28 a second set was launched on top of one that was 22 min into
# epoch 2. Both slowed from 0.29 to 0.61 s/batch and the older set did not survive it. The
# parent's completed() cannot catch this -- it only knows about arms that already FINISHED.
# Verified firing by running this script while a matching process was alive.
#
# USAGE
#   bash scripts/run_center_levelcode_4gpu.sh              # launch all 4, background, wait
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

ARMS_GPU0=${ARMS_GPU0:-"2 6 012"}
ARMS_GPU1=${ARMS_GPU1:-"1 5 034"}
ARMS_GPU2=${ARMS_GPU2:-"4 056"}
ARMS_GPU3=${ARMS_GPU3:-"3 0135"}

LOG_DIR="output/${OUT_ROOT}/_launch"
mkdir -p "${LOG_DIR}"

pids=()
for g in 0 1 2 3; do
  eval "arms=\${ARMS_GPU${g}}"
  log="${LOG_DIR}/launch_gpu${g}.log"
  echo "GPU ${g}: ${arms}   -> ${log}"
  if [ "${DRY_RUN}" = "1" ]; then continue; fi
  GPU_ID="${g}" ARMS="${arms}" PYTHON="${PYTHON}" OUT_ROOT="${OUT_ROOT}" \
    bash scripts/run_center_levelcode.sh > "${log}" 2>&1 &
  pids+=("$!")
done

[ "${DRY_RUN}" = "1" ] && { echo "(dry run, nothing launched)"; exit 0; }

echo; echo "launched ${#pids[@]} jobs: ${pids[*]}   (Ctrl-C kills this waiter, NOT the jobs)"
fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "!! GPU ${i} exited nonzero -- see ${LOG_DIR}/launch_gpu${i}.log"; fail=1; }
done

echo; echo "=== all GPUs done (fail=${fail}) ==="
echo "  ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort path"
exit "${fail}"
