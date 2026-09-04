#!/bin/bash
#
# Seed replication for the four final-method candidates (2026-09-02).
#
# 8 runs: {0123456, 123456} x {renorm off, renorm on} x {seed 1, seed 2}. Seed 0 already exists
# for all four, so this brings every candidate to n=3. Worker is run_center_levelcode.sh --
# READ ITS HEADER for the digit codes and the fixed settings.
#
# ============================ WHY THIS IS THE PRIORITY ============================
# All 39 tracked runs are seed 0. The noise floor of 0.08 used throughout this project is an
# APPROXIMATION from near-replicate pairs (inits differing in 1 row of 8142) and a regression
# residual -- it is NOT seed variance. Real seeds also change dataloader order, augmentation, and
# PEFT parameter init, so the true sd could be larger. Nothing in the method choice is safe until
# this is measured.
#
# THIS SET ADDS SEEDS 1 AND 2 ONLY. Seed 0 already exists for all four arms under
# output/center_levelcode25/ and is deliberately left out of the summary -- merging the three is
# done by hand. With n=3 and sd 0.08 the readable comparisons are:
#     se of a 2-arm difference     0.065
#     0123456_norm - 123456 = 0.17 t = 2.6   marginal but readable
#     0123456 - 123456 = 0.01      hopeless, and expected: they differ in 1 row of 8142
#
# ============================ THE SPLIT ============================
# 4 GPUs, 8 runs as 2/2/2/2, ~4 h each => ~8 h for the set.
#
# A JOB IS "<seed>:<F|T>:<code>" -- seed, renorm off/on, digit code. One GPU walks its job list in
# order, invoking the worker once per job, because the worker takes a single SEED and RENORM per
# call.
#
#            GPU 0            GPU 1            GPU 2            GPU 3
#   round 1  1:F:123456       1:F:0123456      1:T:123456       1:T:0123456
#   round 2  2:F:0123456      2:F:123456       2:T:0123456      2:T:123456
#
# ROUND 1 IS ALL OF SEED 1. At roughly the halfway mark there is a complete four-arm replicate at
# one seed, which is exactly what answers "does the seed-0 ordering reproduce?"; seed 2 then
# confirms or breaks it. A set stopped after round 1 is still worth having, whereas a split that
# interleaved the seeds would leave both incomplete.
#
# THE ARM/CARD ASSIGNMENT IS SWAPPED BETWEEN ROUNDS on purpose: each arm runs its two seeds on two
# DIFFERENT cards (123456/F on GPU 0 then GPU 1, 0123456/F on GPU 1 then GPU 0, and likewise for
# the renorm pair). All four are the same RTX 6000 Ada and no card effect has ever shown up in this
# project, but the swap makes it impossible for one to alias with an arm, and it costs nothing.
#
# OUTPUT DIRS: SUFFIX carries both, so nothing collides with the seed-0 runs --
#     c0123456_s1   c123456_s1   c0123456_rnT_s1   c123456_rnT_s1   (and _s2)
# The seed-0 runs keep their bare names; docs/results.tsv has a seed column, so the four groups
# tabulate correctly without renaming anything that has already finished.
#
# NOT COVERED HERE: the uncentered baseline and global-only. Multi-seed results for both already
# exist in a separate repository, so they are deliberately out of scope for this set.
#
# ============================ GUARDS ============================
# 1. duplicate set -- refuses while any run_center_levelcode.sh lane is alive. Anchored to the
#    whole cmdline; a loose substring match also hits an editor or a tail naming the script.
# 2. busy GPU -- refuses if a target card already has a compute process. Sharing measured
#    0.29 -> 0.72 s/batch (2.4x) and slowed the neighbour too.
# FORCE=1 bypasses both. GPUS trims to free cards.
#
# USAGE
#   DRY_RUN=1 bash scripts/run_center_seeds.sh          # print the plan, run nothing
#   setsid nohup bash scripts/run_center_seeds.sh \
#     > output/seeds25/_launch/launcher.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/notify_seeds.sh \
#     > output/seeds25/_launch/notify.log 2>&1 < /dev/null &      # start AFTER the launcher
#
#   python scripts/dump_results.py && column -t -s$'\t' docs/results.tsv
set -euo pipefail
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
GPUS=${GPUS:-"0 1 2 3"}
OUT_ROOT=${OUT_ROOT:-"seeds25"}

# each entry is a job: "<seed>:<F|T>:<code>"   (F/T = PROMPT_CENTER_NESTED_RENORM False/True)
JOBS_GPU0=${JOBS_GPU0:-"1:F:123456  2:F:0123456"}
JOBS_GPU1=${JOBS_GPU1:-"1:F:0123456 2:F:123456"}
JOBS_GPU2=${JOBS_GPU2:-"1:T:123456  2:T:0123456"}
JOBS_GPU3=${JOBS_GPU3:-"1:T:0123456 2:T:123456"}

lanes_alive(){ pgrep -fc "^bash scripts/run_center_levelcode\.sh$" 2>/dev/null || true; }
n="$(lanes_alive)"
if [ "${FORCE}" != "1" ] && [ -n "${n}" ] && [ "${n}" != "0" ]; then
  echo "ERROR: ${n} run_center_levelcode.sh lane(s) already running -- refusing to start."
  echo "       override: FORCE=1 bash scripts/run_center_seeds.sh"
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
    echo "REFUSING to start: sharing a card measured 0.29 -> 0.72 s/batch (2.4x)."
    echo "Run only the free cards, e.g.  GPUS=\"2 3\" bash \$0"
    echo "override: FORCE=1 bash scripts/run_center_seeds.sh"
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

# Expand a job spec into the worker's environment. Kept as a function so the dry run prints
# exactly what the real run would do.
job_env(){                                   # $1 = "<seed>:<F|T>:<code>"
  local j="$1"
  job_seed="${j%%:*}"; local rest="${j#*:}"
  local rn="${rest%%:*}"; job_code="${rest#*:}"
  case "${job_seed}" in ''|*[!0-9]*) echo "ERROR: bad seed in job '${j}'" >&2; return 1;; esac
  case "${job_code}" in ''|*[!0-6]*) echo "ERROR: bad code in job '${j}'" >&2; return 1;; esac
  case "${rn}" in
    F) job_renorm=False; job_suffix="_s${job_seed}" ;;
    T) job_renorm=True;  job_suffix="_rnT_s${job_seed}" ;;
    *) echo "ERROR: job '${j}' renorm must be F or T (got '${rn}')" >&2; return 1 ;;
  esac
}

# Validate every job before launching anything -- a typo in the last list should not surface
# three rounds in.
total=0
for g in ${GPUS}; do
  eval "jobs=\${JOBS_GPU${g}:-}"
  for j in ${jobs}; do job_env "${j}" || exit 1; total=$((total+1)); done
done
echo "${total} job(s) across GPUs ${GPUS}"

pids=(); pid_gpus=()
for g in ${GPUS}; do
  eval "jobs=\${JOBS_GPU${g}:-}"
  if [ -z "${jobs}" ]; then echo "GPU ${g}: no JOBS_GPU${g} set -- skipping"; continue; fi
  log="${LOG_DIR}/launch_gpu${g}.log"
  echo "GPU ${g}:"
  for j in ${jobs}; do
    job_env "${j}"
    printf '    %-16s seed %s  renorm %-5s  -> c%s%s\n' \
      "${j}" "${job_seed}" "${job_renorm}" "${job_code}" "${job_suffix}"
  done
  echo "         -> ${log}"
  if [ "${DRY_RUN}" = "1" ]; then continue; fi
  (
    for j in ${jobs}; do
      job_env "${j}"
      GPU_ID="${g}" ARMS="${job_code}" SEED="${job_seed}" RENORM="${job_renorm}" \
        SUFFIX="${job_suffix}" PYTHON="${PYTHON}" OUT_ROOT="${OUT_ROOT}" \
        bash scripts/run_center_levelcode.sh || exit 1
    done
  ) > "${log}" 2>&1 &
  pids+=("$!"); pid_gpus+=("${g}")
done

[ "${DRY_RUN}" = "1" ] && { echo "(dry run, nothing launched)"; exit 0; }

echo; echo "launched ${#pids[@]} GPU lane(s): ${pids[*]}   (Ctrl-C kills this waiter, NOT the jobs)"
fail=0
for i in "${!pids[@]}"; do
  g="${pid_gpus[$i]}"
  wait "${pids[$i]}" || { echo "!! GPU ${g} exited nonzero -- see ${LOG_DIR}/launch_gpu${g}.log"; fail=1; }
done
echo; echo "=== all lanes done (fail=${fail}) ==="
echo "  ${PYTHON} scripts/dump_results.py"
exit "${fail}"
