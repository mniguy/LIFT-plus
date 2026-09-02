#!/bin/bash
#
# Wait for the RENORM=True arm set to finish, tabulate it AGAINST its RENORM=False twin,
# and post the comparison to Slack.
#
# WHY NOT notify_when_done.sh
#     That one polls `ps` for live run_center_levelcode.sh LANES. This set was launched as bare
#     main.py processes (the parent script is unrunnable -- it still carries merge conflict
#     markers at line 96), so its lane count is 0 from the start and it would fire instantly on
#     an empty result. This script watches the main.py processes themselves instead.
#
# WHAT IT WATCHES
#     Any main.py whose cmdline contains an output_dir ending in _rnT. That is the whole set and
#     nothing else -- the RENORM=False runs all finished before this was written, and a future
#     unrelated run cannot match unless it is deliberately named _rnT.
#
# WHY IT ALSO CHECKS THE LOGS
#     A process disappearing is not the same as a run succeeding. After the wait, each arm's log
#     is checked for "* Many:" and anything missing it is reported as CRASHED, so a dead job is
#     never silently summarized as a result.
#
# THE WEBHOOK IS A SECRET (same contract as notify_when_done.sh): $SLACK_WEBHOOK_URL, else
# ~/.config/lift_slack_webhook (chmod 600). With neither, the summary is still written to disk.
#
# USAGE (detached; survives SSH dropping and this terminal closing)
#   setsid nohup bash scripts/notify_rnT.sh \
#     > output/center_levelcode25/_launch/notify_rnT.log 2>&1 < /dev/null &
#
set -uo pipefail
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

OUT_ROOT=${OUT_ROOT:-center_levelcode25}
DATASET=${DATASET:-inat2018}
POLL=${POLL:-120}
ARMS=${ARMS:-"056 0246 0456 0123456"}
WEBHOOK_FILE=${WEBHOOK_FILE:-$HOME/.config/lift_slack_webhook}
PYTHON=${PYTHON:-$([ -x /home/mingyu/.conda/envs/ltl/bin/python ] \
  && echo /home/mingyu/.conda/envs/ltl/bin/python || echo python3)}
export PYTHONNOUSERSITE=1          # see run_center_levelcode_4gpu.sh -- ~/.local shadows the env
SUMMARY="output/${OUT_ROOT}/SUMMARY_rnT.txt"

# Count DISTINCT arms, not processes: each main.py forks ~11 dataloader workers that share
# its cmdline, so a plain grep -c reports 44 for a 4-arm set.
alive(){ ps -eo args | grep -o "[m]ain\.py.*output_dir [^ ]*_rnT" \
           | grep -o "output_dir [^ ]*" | sort -u | wc -l; }

echo "[$(date '+%F %T')] watching ${ARMS}; $(alive) alive; polling every ${POLL}s"
while [ "$(alive)" -gt 0 ]; do sleep "${POLL}"; done
echo "[$(date '+%F %T')] all _rnT processes exited -- tabulating"

{
  echo "LIFT+ ${OUT_ROOT} RENORM=True set -- finished $(date '+%F %T %Z') on $(hostname)"
  echo
  "${PYTHON}" - "output/${OUT_ROOT}/${DATASET}" ${ARMS} <<'PY'
import glob, os, re, sys
root, codes = sys.argv[1], sys.argv[2:]

def read(d):
    for f in reversed(sorted(glob.glob(os.path.join(d, "log-*.txt")))):
        t = open(f, errors="ignore").read()
        m = re.findall(r"\* Many:\s*([\d.]+)%\s*Med:\s*([\d.]+)%\s*Few:\s*([\d.]+)%", t)
        if not m:
            continue
        ov = re.findall(r"\* Overall accuracy:\s*([\d.]+)%", t)
        return (float(ov[-1]) if ov else float("nan"), *map(float, m[-1]))
    return None

print(f"{'code':<9}{'renorm':>8}{'All':>8}{'Head':>7}{'Med':>7}{'Few':>7}   {'dAll':>7}")
bad = []
for c in codes:
    off, on = read(f"{root}/c{c}"), read(f"{root}/c{c}_rnT")
    if on is None:
        bad.append(c)
    for lbl, r in (("False", off), ("True", on)):
        if r is None:
            print(f"c{c:<8}{lbl:>8}{'  -- not run --':>29}")
            continue
        d = f"{r[0]-off[0]:+7.2f}" if (lbl == "True" and off) else ""
        print(f"c{c:<8}{lbl:>8}{r[0]:>8.2f}{r[1]:>7.2f}{r[2]:>7.2f}{r[3]:>7.2f}   {d:>7}")
    print()
if bad:
    print("CRASHED / no result: " + " ".join(f"c{c}_rnT" for c in bad))
PY
} > "${SUMMARY}" 2>&1
cat "${SUMMARY}"

url="${SLACK_WEBHOOK_URL:-}"
if [ -z "${url}" ] && [ -r "${WEBHOOK_FILE}" ]; then url="$(tr -d '\r\n' < "${WEBHOOK_FILE}")"; fi
if [ -z "${url}" ]; then
  echo "[$(date '+%F %T')] no webhook configured -- summary written to ${SUMMARY}, nothing sent"
  exit 0
fi

payload="$("${PYTHON}" - "${SUMMARY}" <<'PY'
import json, sys
body = open(sys.argv[1], errors="ignore").read()[:3500]
print(json.dumps({"text": "```\n" + body + "\n```"}))
PY
)"
if curl -sS -f -X POST -H 'Content-type: application/json' --data "${payload}" "${url}" >/dev/null; then
  echo "[$(date '+%F %T')] posted to Slack"
else
  echo "[$(date '+%F %T')] Slack POST FAILED -- summary still at ${SUMMARY}"
  exit 1
fi
