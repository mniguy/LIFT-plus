#!/bin/bash
#
# Idempotent supervisor for the Slack GPU bot. Safe to run every few minutes from cron.
#
# WHY CRON AND NOT SYSTEMD
#     This account has no passwordless sudo, so a system unit in /etc/systemd/system is out.
#     A systemd --user unit would work while logged in, but `loginctl show-user mingyu` reports
#     Linger=no, meaning user services are killed when the last session ends -- exactly the case
#     we need to survive. Enabling linger needs root. cron @reboot has neither limitation.
#
# WHAT IT TOUCHES
#     ONLY the bot. It never starts, stops, or inspects the training runs. `docker start` on an
#     already-running container is a no-op, so the experiment is never disturbed.
#
# INSTALL (crontab -e), both lines:
#     @reboot            /home/mingyu/mingyu/LIFT-plus/scripts/ensure_slack_gpu_bot.sh
#     */5 * * * *        /home/mingyu/mingyu/LIFT-plus/scripts/ensure_slack_gpu_bot.sh
#   @reboot brings it back after a machine reboot; the */5 sweep brings it back if it crashes
#   or if the container was restarted by hand.
#
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin        # cron gives a minimal PATH

CONTAINER=${CONTAINER:-mingyu}
REPO=${REPO:-/home/mingyu/mingyu/LIFT-plus}
ENV_FILE=${ENV_FILE:-/home/mingyu/.config/slack_gpu_bot.env}
VENV_PY=${VENV_PY:-/home/mingyu/.venvs/slackgpu/bin/python}
LOG=${LOG:-/home/mingyu/.local/state/slack_gpu_bot_supervisor.log}
LOCK=/tmp/.ensure_slack_gpu_bot.lock

mkdir -p "$(dirname "${LOG}")"
say(){ echo "[$(date '+%F %T')] $*" >> "${LOG}"; }

# One at a time: the */5 sweep must never race @reboot into launching two bots.
exec 9>"${LOCK}" || exit 0
flock -n 9 || exit 0

if ! docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null | grep -q true; then
  say "container ${CONTAINER} not running -- starting"
  docker start "${CONTAINER}" >/dev/null 2>&1 || { say "docker start FAILED"; exit 1; }
  sleep 5
fi

# Is a bot already alive inside? Anchor on the REAL python process cmdline: a bare
# "slack_gpu_bot.py" pattern also matches the launching wrapper shell, so a leftover
# wrapper could report "alive" after the bot itself had died.
alive=$(docker exec "${CONTAINER}" bash -c 'pgrep -fc "^[^ ]*slackgpu/bin/python scripts/slack_gpu_bot\.py$" || true' 2>/dev/null | tr -d '[:space:]')
alive=${alive:-0}
if [ "${alive}" -gt 0 ]; then
  exit 0                                        # healthy: stay silent, do not spam the log
fi

say "bot not running (count=${alive}) -- launching"
docker exec -d -w "${REPO}" "${CONTAINER}" bash -c \
  "set -a; . ${ENV_FILE}; set +a; \
   PYTHONNOUSERSITE=1 setsid nohup ${VENV_PY} scripts/slack_gpu_bot.py \
     >> output/slack_gpu_bot.log 2>&1 < /dev/null"

sleep 4
again=$(docker exec "${CONTAINER}" bash -c 'pgrep -fc "^[^ ]*slackgpu/bin/python scripts/slack_gpu_bot\.py$" || true' 2>/dev/null | tr -d '[:space:]')
if [ "${again:-0}" -gt 0 ]; then say "bot started OK"; else say "bot FAILED to start -- see ${REPO}/output/slack_gpu_bot.log"; fi
