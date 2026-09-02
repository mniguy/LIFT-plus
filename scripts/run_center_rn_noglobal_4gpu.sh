#!/bin/bash
#
# 2-GPU fan-out: the NO-GLOBAL half of the renorm 2x2 (2026-09-01).
#
# Arms: 56 246 456 123456 with RENORM=True. Two sequential runs per card on GPUs 2 and 3;
# GPUs 0 and 1 are reserved for other users and must remain untouched.
# The worker is run_center_levelcode.sh -- READ ITS HEADER for the digit codes and the settings.
# This file only picks arms, sets RENORM/SUFFIX, and splits across GPUs.
#
# ============================ WHY THIS SET EXISTS ============================
# Under renorm=False the leading 0 is a PROVEN no-op. Each level's mean is taken on the running
# residual, so the global term cancels out of every later group mean; it can only survive for a
# class that the GENUS_MIN=2 gate SKIPS at the first level. Measured rows actually changed:
#       code       renorm=False   renorm=True
#       56              472           8142
#       246               5           8142
#       456              66           8142
#       123456            1           8142
# and the 8 measured pairs behaved accordingly -- mean dAll -0.06 with NO relationship to how
# many rows the 0 touched (r=+0.13; the two biggest drops were pairs where it touched ONE class).
# That is why "is global centering needed?" is still unanswered: under renorm=False the question
# is not even askable, because the operation barely differs.
#
# RENORM=True changes that. Row-renormalising after every level destroys the telescoping identity,
# so the global step no longer cancels -- it moves ALL 8142 rows. These four arms are therefore
# the FIRST setting in which global centering can actually do something, and the comparison
# against the already-finished c0X_rnT runs is the first real test of whether it does.
#
# ============================ THE 2x2 IT COMPLETES ============================
#                    renorm=False                    renorm=True
#   with 0     c056     80.93                   c056_rnT     80.99
#              c0246    81.09                   c0246_rnT    81.02
#              c0456    80.99                   c0456_rnT    80.87
#              c0123456 80.93                   c0123456_rnT 81.11
#   no 0       c56      80.90                   c56_rnT      <- this set
#              c246     81.07                   c246_rnT
#              c456     80.86                   c456_rnT
#              c123456  80.94                   c123456_rnT
#
# HOW TO READ IT. Noise floor 0.08 (sd of 8 near-replicate pairs 0.080; 13-run regression
# residual 0.082 -- two independent estimates agreeing).
#   - all four land within ~0.1 of their c0X_rnT twin  ->  global centering is worthless even
#     where it CAN act. That is a far stronger claim than the renorm=False result, and it retires
#     the leading 0 from the method for good.
#   - the twins separate  ->  global centering does contribute, but only once renorm stops the
#     telescoping from erasing it. Then the method needs BOTH, and renorm stops being a dead knob.
# Note the prior on renorm itself is that it does nothing: the with-global half measured
# +0.18 / +0.06 / -0.07, mean +0.057 against a 0.082 floor, and its rnF/rnT rankings
# anti-correlate (r=-0.28). Do not expect large numbers here.
#
# ============================ OUTPUT PATHS ============================
# SUFFIX=_rnT writes c<code>_rnT into the SAME OUT_ROOT, beside the RENORM=False result rather
# than on top of it, so agg_runs.py tabulates the whole 2x2 in one table. (The parent used to
# have no way to do this; repointing OUT_ROOT worked but split the halves across two tables.)
#
# ============================ GUARDS ============================
# 1. duplicate set: refuses while any levelcode lane is alive. The pattern is anchored to the
#    whole cmdline -- a loose substring match also hits an editor or a tail that merely names
#    the script, which was observed firing spuriously on 2026-08-31.
# 2. busy GPU: refuses if a target card already has a compute process. Sharing measured
#    0.29 -> 0.72 s/batch (2.4x) and slowed the neighbour too. GPUS trims the set to free cards.
# FORCE=1 bypasses both.
#
# USAGE
#   bash scripts/run_center_rn_noglobal_4gpu.sh            # all four, background, wait
#
#   # the real invocation -- detached, with the Slack watcher. RUN BOTH LINES:
#   setsid nohup bash scripts/run_center_rn_noglobal_4gpu.sh \
#     > output/center_levelcode25/_launch/launcher_rn.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/notify_rn2x2.sh \
#     > output/center_levelcode25/_launch/notify_rn.log 2>&1 < /dev/null &
#   # the watcher must start AFTER the launcher, or it sees 0 arms alive and fires immediately.
#   DRY_RUN=1 bash scripts/run_center_rn_noglobal_4gpu.sh
#   GPUS="2 3" ARMS_GPU2="56 246" ARMS_GPU3="456 123456" bash scripts/run_center_rn_noglobal_4gpu.sh
#
#   python scripts/agg_runs.py output/center_levelcode25 --sort all
set -euo pipefail
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
GPUS=${GPUS:-"2 3"}
ARMS_GPU2=${ARMS_GPU2:-"56 246"}
ARMS_GPU3=${ARMS_GPU3:-"456 123456"}

export RENORM=${RENORM:-True}
export SUFFIX=${SUFFIX:-_rnT}
OUT_ROOT=${OUT_ROOT:-"center_levelcode25"}

lanes_alive(){ pgrep -fc "^bash scripts/run_center_levelcode\.sh$" 2>/dev/null || true; }
n="$(lanes_alive)"
if [ "${FORCE}" != "1" ] && [ -n "${n}" ] && [ "${n}" != "0" ]; then
  echo "ERROR: ${n} run_center_levelcode.sh lane(s) already running -- refusing to start."
  echo "       inspect:  ps -eo pid,etime,args | grep -E 'levelcode|main\.py' | grep -v grep"
  echo "       override: FORCE=1 bash scripts/run_center_rn_noglobal_4gpu.sh"
  exit 1
fi

busy=""
uuids="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)"
apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null || true)"
for g in ${GPUS}; do
  uuid="$(echo "${uuids}" | awk -F', ' -v i="${g}" '$1==i {print $2}')"
  [ -z "${uuid}" ] && continue
  hit="$(echo "${apps}" | grep -F "${uuid}" || true)"
  [ -n "${hit}" ] && busy="${busy}${busy:+$'\n'}  GPU ${g}: $(echo "${hit}" | awk -F', ' '{printf "pid %s (%s) ", $2, $3}')"
done
if [ -n "${busy}" ]; then
  echo "WARNING: target GPU(s) already busy --"; echo "${busy}"
  if [ "${FORCE}" != "1" ]; then
    echo "REFUSING to start: sharing a card measured 0.29 -> 0.72 s/batch (2.4x), both jobs slowed."
    echo "Run only the free cards, e.g.  GPUS=\"1 2 3\" ARMS_GPU1=\"56 246\" bash \$0"
    echo "override: FORCE=1 bash scripts/run_center_rn_noglobal_4gpu.sh"
    exit 1
  fi
  echo "(FORCE=1 -- proceeding onto busy cards anyway)"
fi

# PYTHONNOUSERSITE=1 IS LOAD-BEARING -- ~/.local/lib/python3.11 shadows the ltl env with newer
# clip/timm/yacs/sklearn, and user-site outranks the env.
PYTHON=${PYTHON:-$([ -x /home/mingyu/.conda/envs/ltl/bin/python ] \
  && echo /home/mingyu/.conda/envs/ltl/bin/python || echo python)}
export PYTHONNOUSERSITE=1

LOG_DIR="output/${OUT_ROOT}/_launch"
mkdir -p "${LOG_DIR}"

pids=(); pid_gpus=()
for g in ${GPUS}; do
  eval "arms=\${ARMS_GPU${g}:-}"
  if [ -z "${arms}" ]; then echo "GPU ${g}: no ARMS_GPU${g} set -- skipping"; continue; fi
  log="${LOG_DIR}/launch_rn_gpu${g}.log"
  echo "GPU ${g}: ${arms}   (RENORM=${RENORM}, SUFFIX=${SUFFIX})   -> ${log}"
  if [ "${DRY_RUN}" = "1" ]; then continue; fi
  GPU_ID="${g}" ARMS="${arms}" PYTHON="${PYTHON}" OUT_ROOT="${OUT_ROOT}" \
    bash scripts/run_center_levelcode.sh > "${log}" 2>&1 &
  pids+=("$!"); pid_gpus+=("${g}")
done

[ "${DRY_RUN}" = "1" ] && { echo "(dry run, nothing launched)"; exit 0; }

echo; echo "launched ${#pids[@]} jobs: ${pids[*]}   (Ctrl-C kills this waiter, NOT the jobs)"
fail=0
for i in "${!pids[@]}"; do
  g="${pid_gpus[$i]}"
  wait "${pids[$i]}" || { echo "!! GPU ${g} exited nonzero -- see ${LOG_DIR}/launch_rn_gpu${g}.log"; fail=1; }
done
echo; echo "=== all GPUs done (fail=${fail}) ==="
echo "  ${PYTHON} scripts/agg_runs.py output/${OUT_ROOT} --sort all"
exit "${fail}"
