#!/bin/bash
#
# 4-GPU fan-out for run_center_granularity.sh (2026-08-31).
#
# The science, the arm syntax, and how to read the numbers live in run_center_granularity.sh --
# READ THAT HEADER FIRST. This file only decides WHICH ARM RUNS ON WHICH GPU, by shelling out to
# that script once per GPU, so the experiment logic exists in exactly one place.
#
# ============================ THE SPLIT ============================
# 10 arms, 4 GPUs, iNat2018 @ 15 epochs, ~4 h each => ~12 h for the set (3 rounds).
#
#            GPU 0 | GPU 1 | GPU 2 | GPU 3
#   round 1    s2  |  t06  |  s7   |  t05     <- BOTH anchor pairs, all four concurrent
#   round 2    s4  |  s15  |  s30  |  s64     <- the middle of the curve
#   round 3   s143 | s326  |   -   |   -      <- the coarse end
#
#   GPU 0    s2  s4  s143      (3 arms)
#   GPU 1   t06 s15  s326      (3 arms)
#   GPU 2    s7 s30            (2 arms)
#   GPU 3   t05 s64            (2 arms)
#
# ROUND 1 IS THE EXPERIMENT. s2 vs t06 (genus granularity) and s7 vs t05 (family granularity) are
# the two comparisons this set exists to make, and on four cards BOTH complete in the first ~4 h,
# concurrently. If the set has to be killed early, kill it after round 1 and nothing is lost.
# Rounds 2 and 3 only fill in the shape of the curve between and beyond those points.
#
# A PAIR NECESSARILY SPANS TWO CARDS -- it cannot be concurrent otherwise. That is the right
# trade: all four GPUs here are the same 49 GB model, whereas machine state (a neighbour job
# starting, thermals) varies over HOURS and would hit a sequential pair unequally. No card effect
# has ever shown up in this project; 21 runs of the digit-code grid were spread over four cards
# with the genus effect landing at t=5.69 regardless.
#
# ============================ TWO GUARDS, BOTH EARNED ============================
# 1. DUPLICATE SET: refuses to start while any granularity lane is alive.
# 2. BUSY GPU: refuses to start if a target card already has a compute process on it. This has
#    bitten twice. Measured 2026-08-31, one arm sharing a card with a 2.6 GB neighbour:
#    0.29 -> 0.72 s/batch, a 2.4x slowdown, ETA 4 h -> 10 h, and the neighbour slows too. Memory
#    is never the constraint (17 GB of 49 GB) -- SM contention is. Note this guard also catches
#    run_center_levelcode.sh still holding GPUs 2 and 3, which guard 1 knows nothing about.
# Both are bypassed with FORCE=1, and GPUS trims the set to the free cards, e.g.
#     GPUS="2 3" ARMS_GPU2="s2 s7 s30 s143 s326" ARMS_GPU3="t06 t05 s4 s15 s64" \
#       bash scripts/run_center_granularity_4gpu.sh
#
# USAGE
#   bash scripts/run_center_granularity_4gpu.sh            # launch all 4, background, wait
#   # detached -- survives your SSH dropping and this terminal closing:
#   setsid nohup bash scripts/run_center_granularity_4gpu.sh \
#     > output/center_gran25/_launch/launcher.log 2>&1 < /dev/null &
#   DRY_RUN=1 bash scripts/run_center_granularity_4gpu.sh  # print the split, check GPUs, run nothing
#   FORCE=1   bash scripts/run_center_granularity_4gpu.sh  # bypass both guards
#
# Arms already finished (log contains "* Many:") are skipped by the worker, so re-running this
# after a crash resumes instead of redoing work.
#
#   tail -f output/center_gran25/_launch/launch_gpu*.log
#   python scripts/agg_runs.py output/center_gran25 --sort all
set -euo pipefail
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
GPUS=${GPUS:-"0 1 2 3"}
ARMS_GPU0=${ARMS_GPU0:-"s2 s4 s143"}
ARMS_GPU1=${ARMS_GPU1:-"t06 s15 s326"}
ARMS_GPU2=${ARMS_GPU2:-"s7 s30"}
ARMS_GPU3=${ARMS_GPU3:-"t05 s64"}

# ---- guard 1: one set at a time -------------------------------------------------------------
# ANCHOR ON THE EXACT LANE COMMAND. A loose "run_center_granularity.sh" substring match also hits
# any process that merely NAMES the script -- an editor, a tail, a grep, a shell whose argv quotes
# it -- and refuses to start for no reason. Observed doing exactly that on 2026-08-31. The lanes
# are spawned literally as `bash scripts/run_center_granularity.sh` with no arguments, so pin the
# whole cmdline. Same fix, same reason, as the pgrep in ensure_slack_gpu_bot.sh.
lanes_alive(){ pgrep -fc "^bash scripts/run_center_granularity\.sh$" 2>/dev/null || true; }
if [ "${FORCE}" != "1" ] && [ "$(lanes_alive)" != "0" ] && [ -n "$(lanes_alive)" ]; then
  echo "ERROR: run_center_granularity.sh is already running -- refusing to start a duplicate set."
  echo "       inspect:  ps -eo pid,pgid,etime,args | grep -E 'granularity|main\.py' | grep -v grep"
  echo "       override: FORCE=1 bash scripts/run_center_granularity_4gpu.sh"
  exit 1
fi

# ---- guard 2: every target GPU must be idle --------------------------------------------------
# Map each compute PID to a GPU INDEX via the uuid table; --query-compute-apps reports gpu_uuid,
# not the index, and the two orders are not guaranteed to agree.
busy=""
uuids="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)"
apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null || true)"
for g in ${GPUS}; do
  uuid="$(echo "${uuids}" | awk -F', ' -v i="${g}" '$1==i {print $2}')"
  [ -z "${uuid}" ] && continue
  hit="$(echo "${apps}" | grep -F "${uuid}" || true)"
  if [ -n "${hit}" ]; then
    busy="${busy}${busy:+$'\n'}  GPU ${g}: $(echo "${hit}" | awk -F', ' '{printf "pid %s (%s) ", $2, $3}')"
  fi
done
if [ -n "${busy}" ]; then
  echo "WARNING: target GPU(s) already busy --"
  echo "${busy}"
  if [ "${FORCE}" != "1" ]; then
    echo "REFUSING to start: sharing a card measured 0.29 -> 0.72 s/batch (2.4x) on 2026-08-31,"
    echo "and it slows the other job too. Either wait, or run only the free cards, e.g."
    echo "    GPUS=\"2 3\" ARMS_GPU2=\"s2 s7 s30 s143 s326\" ARMS_GPU3=\"t06 t05 s4 s15 s64\" \\"
    echo "      bash scripts/run_center_granularity_4gpu.sh"
    echo "override: FORCE=1 bash scripts/run_center_granularity_4gpu.sh"
    exit 1
  fi
  echo "(FORCE=1 -- proceeding onto busy cards anyway)"
fi

# THE ENV is the conda env "ltl" inside the running "mingyu" docker container.
# PYTHONNOUSERSITE=1 IS LOAD-BEARING -- ~/.local/lib/python3.11 holds a stray user-site copy of
# clip/timm/yacs/sklearn at NEWER versions, and user-site outranks the env. sklearn matters here
# in particular: mode=cluster calls KMeans, so a version swap changes the partition itself.
PYTHON=${PYTHON:-$([ -x /home/mingyu/.conda/envs/ltl/bin/python ] \
  && echo /home/mingyu/.conda/envs/ltl/bin/python || echo python)}
export PYTHONNOUSERSITE=1
OUT_ROOT=${OUT_ROOT:-"center_gran25"}

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
    bash scripts/run_center_granularity.sh > "${log}" 2>&1 &
  pids+=("$!"); pid_gpus+=("${g}")     # array index != GPU number once GPUS != 0..3
done

[ "${DRY_RUN}" = "1" ] && { echo "(dry run, nothing launched)"; exit 0; }

echo; echo "launched ${#pids[@]} jobs: ${pids[*]}   (Ctrl-C kills this waiter, NOT the jobs)"
fail=0
for i in "${!pids[@]}"; do
  g="${pid_gpus[$i]}"
  wait "${pids[$i]}" || { echo "!! GPU ${g} exited nonzero -- see ${LOG_DIR}/launch_gpu${g}.log"; fail=1; }
done

echo; echo "=== all GPUs done (fail=${fail}) ==="
echo "  ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort all"
exit "${fail}"
