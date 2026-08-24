#!/bin/bash
#
# Overnight batch for GPU 0. Four 15-epoch iNat runs, one after another.
#
# Deliberately NOT 'set -e': a run that dies must not take the remaining ones down with it. Every step
# is attempted, its exit status recorded, and a PASS/FAIL summary printed at the end. Steps whose
# output directory already contains a finished log are skipped by the underlying scripts.
#
#   nohup bash scripts/run_night0.sh > night0.log 2>&1 &
#   tail -f night0.log
#   python scripts/agg_runs.py output/center_ms2 output/center_proj output/center_cohesion25 --sort path
set -uo pipefail
GPU_ID=${GPU_ID:-0}
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

STEPS=(
  "ms2|g_topdown_ms2|scripts/run_center_ms2.sh"
  "cohesion|w_one|scripts/run_center_cohesion.sh"
  "cohesion|shrink_group|scripts/run_center_cohesion.sh"
  "cohesion|shrink_level|scripts/run_center_cohesion.sh"
)

declare -a RESULT
echo "### GPU ${GPU_ID} batch started $(date '+%F %T') -- ${#STEPS[@]} runs"
for step in "${STEPS[@]}"; do
  IFS='|' read -r group arm script <<< "${step}"
  echo; echo "############ [$(date '+%F %T')] ${group}/${arm} ############"
  GPU_ID="${GPU_ID}" ARMS="${arm}" bash "${script}"
  rc=$?
  RESULT+=("$( [ $rc -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)" )  ${group}/${arm}")
  echo "---- ${group}/${arm} -> $( [ $rc -eq 0 ] && echo PASS || echo "FAIL rc=$rc" )"
done

echo; echo "### GPU ${GPU_ID} batch finished $(date '+%F %T')"
printf '  %s\n' "${RESULT[@]}"
fails=$(printf '%s\n' "${RESULT[@]}" | grep -c FAIL || true)
echo "### ${fails} failed of ${#STEPS[@]}"
