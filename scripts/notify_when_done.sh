#!/bin/bash
#
# Wait for the current center_levelcode run to finish, tabulate, and (optionally) post to Slack.
#
# This is the "Slack as a status channel" design, NOT a control plane: nothing here lets Slack
# drive the machine. It only pushes a finished result outward. Claude Tag / cloud routines cannot
# do this job at all -- they run in an Anthropic cloud sandbox with no route to this server, and
# output/ is gitignored so results never reach GitHub either.
#
# It is a SEPARATE process on purpose. run_center_levelcode{,_4gpu}.sh are being executed by a
# live bash right now, and editing a running script can make bash misread it at its next byte
# offset. Nothing here touches those files.
#
# WHAT IT DOES
#   1. polls until no run_center_levelcode.sh lane is left alive
#   2. writes output/<OUT_ROOT>/SUMMARY.txt  (agg_runs.py, sorted by All)
#   3. POSTs that summary to a Slack incoming webhook, if one is configured
#
# THE WEBHOOK IS A SECRET -- it is a URL that lets anyone holding it post into the channel.
# Keep it OUT of the repo. This script looks for it in, in order:
#     $SLACK_WEBHOOK_URL
#     ~/.config/lift_slack_webhook      (chmod 600)
# With neither, it still writes SUMMARY.txt and says so. The lookup happens when the run
# FINISHES, so the file can be dropped in any time before then.
#
# Get a webhook: Slack -> your app -> Incoming Webhooks -> Add New Webhook to Workspace.
#     mkdir -p ~/.config && umask 077 && cat > ~/.config/lift_slack_webhook   # paste, then Ctrl-D
#
# USAGE (detached, survives your SSH dropping and this terminal closing)
#   docker exec -d -w /home/mingyu/mingyu/LIFT-plus mingyu bash -c \
#     'setsid nohup bash scripts/notify_when_done.sh > output/center_levelcode25/_launch/notify.log 2>&1 < /dev/null'
#
set -uo pipefail
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

OUT_ROOT=${OUT_ROOT:-center_levelcode25}
POLL=${POLL:-120}
WEBHOOK_FILE=${WEBHOOK_FILE:-$HOME/.config/lift_slack_webhook}
PYTHON=${PYTHON:-$([ -x /home/mingyu/.conda/envs/ltl/bin/python ] \
  && echo /home/mingyu/.conda/envs/ltl/bin/python || echo python3)}
export PYTHONNOUSERSITE=1          # see run_center_levelcode_4gpu.sh -- ~/.local shadows the env
SUMMARY="output/${OUT_ROOT}/SUMMARY.txt"

lanes(){ ps -eo args | grep -c "[r]un_center_levelcode\.sh"; }

echo "[$(date '+%F %T')] watching; ${OUT_ROOT}; $(lanes) lane(s) alive; polling every ${POLL}s"
while [ "$(lanes)" -gt 0 ]; do sleep "${POLL}"; done
echo "[$(date '+%F %T')] all lanes exited -- tabulating"

{
  echo "LIFT+ ${OUT_ROOT} -- finished $(date '+%F %T %Z') on $(hostname)"
  echo
  "${PYTHON}" scripts/agg_runs.py "output/${OUT_ROOT}" --sort all
} > "${SUMMARY}" 2>&1
cat "${SUMMARY}"

url="${SLACK_WEBHOOK_URL:-}"
if [ -z "${url}" ] && [ -r "${WEBHOOK_FILE}" ]; then url="$(tr -d '\r\n' < "${WEBHOOK_FILE}")"; fi
if [ -z "${url}" ]; then
  echo "[$(date '+%F %T')] no webhook configured -- summary written to ${SUMMARY}, nothing sent"
  exit 0
fi

# Build the JSON with a real encoder; a summary line could otherwise break hand-rolled quoting.
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
