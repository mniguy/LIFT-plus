#!/bin/bash
#
# Wait for the no-global RENORM=True set to finish, print the completed renorm 2x2, and post it
# to Slack. Companion to run_center_rn_noglobal_4gpu.sh -- read that header for the science.
#
# WHAT IT WATCHES: main.py processes whose output_dir ends in _rnT plus the dedicated
# run_center_rn_noglobal_4gpu.sh launcher. The launcher check matters when two arms run
# sequentially per GPU: it keeps the watcher alive across the brief gap between those arms.
# It deliberately does NOT watch generic run_center_levelcode.sh lanes, which other sets share.
#
# WHY A PROCESS WATCH AND A LOG CHECK: a process disappearing is not a run succeeding. After the
# wait, each arm's log is checked for "* Many:" and anything missing it is reported as CRASHED,
# so a dead job is never silently summarised as a result.
#
# THE READING, so the message is interpretable without going back to the scripts:
#   Under renorm=False the leading 0 is a proven no-op (it changes 1-472 rows of 8142 and the 8
#   measured pairs showed no relationship between the drop and how many rows it touched).
#   Under renorm=True it changes all 8142 rows, so the no-0 vs with-0 column here is the first
#   real test of whether global centering contributes anything. Noise floor 0.08.
#
# THE WEBHOOK IS A SECRET: $SLACK_WEBHOOK_URL, else ~/.config/lift_slack_webhook (chmod 600).
# With neither, the summary is still written to disk.
#
# USAGE -- launch it right after the experiment, both detached:
#   setsid nohup bash scripts/run_center_rn_noglobal_4gpu.sh \
#     > output/center_levelcode25/_launch/launcher_rn.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/notify_rn2x2.sh \
#     > output/center_levelcode25/_launch/notify_rn.log 2>&1 < /dev/null &
#
set -uo pipefail
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

OUT_ROOT=${OUT_ROOT:-center_levelcode25}
DATASET=${DATASET:-inat2018}
POLL=${POLL:-120}
ARMS=${ARMS:-"56 246 456 123456"}
WEBHOOK_FILE=${WEBHOOK_FILE:-$HOME/.config/lift_slack_webhook}
PYTHON=${PYTHON:-$([ -x /home/mingyu/.conda/envs/ltl/bin/python ] \
  && echo /home/mingyu/.conda/envs/ltl/bin/python || echo python3)}
export PYTHONNOUSERSITE=1
SUMMARY="output/${OUT_ROOT}/SUMMARY_rn2x2.txt"

# Count DISTINCT arms, not processes: each main.py forks ~11 dataloader workers sharing its
# cmdline, so a plain grep -c reports ~44 for a 4-arm set.
alive(){ ps -eo args | grep -o "[m]ain\.py.*output_dir [^ ]*_rnT" \
           | grep -o "output_dir [^ ]*" | sort -u | wc -l; }
launcher_alive(){ pgrep -fc "^bash scripts/run_center_rn_noglobal_4gpu\.sh$" 2>/dev/null || true; }

echo "[$(date '+%F %T')] watching ${ARMS}; $(alive) arm(s) alive; polling every ${POLL}s"
while [ "$(alive)" -gt 0 ] || [ "$(launcher_alive)" -gt 0 ]; do sleep "${POLL}"; done
echo "[$(date '+%F %T')] all _rnT processes exited -- tabulating"

{
  echo "LIFT+ renorm 2x2 complete -- $(date '+%F %T %Z') on $(hostname)"
  echo "baseline 80.63 | noise floor 0.08"
  echo
  "${PYTHON}" - "output/${OUT_ROOT}/${DATASET}" ${ARMS} <<'PY'
import glob, os, re, sys, statistics as st
root, codes = sys.argv[1], sys.argv[2:]

def read(d):
    for f in reversed(sorted(glob.glob(os.path.join(root, d, "log-*.txt")))):
        t = open(f, errors="ignore").read()
        m = re.findall(r"\* Many:\s*([\d.]+)%\s*Med:\s*([\d.]+)%\s*Few:\s*([\d.]+)%", t)
        if not m:
            continue
        ov = re.findall(r"\* Overall accuracy:\s*([\d.]+)%", t)
        return (float(ov[-1]), *map(float, m[-1]))
    return None

# c0456 has no local log; its value comes from the tracked results table
FIX = {"c0456": (80.99, 75.53, 80.70, 82.79)}
def get(d):
    return FIX.get(d) or read(d)

print(f"{'code':<9}{'renorm=False':>26}{'renorm=True':>26}")
print(f"{'':<9}{'no 0':>12}{'with 0':>14}{'no 0':>12}{'with 0':>14}")
bad, deltas = [], []
for c in codes:
    cells = []
    for d in (f"c{c}", f"c0{c}", f"c{c}_rnT", f"c0{c}_rnT"):
        r = get(d)
        if d == f"c{c}_rnT" and r is None:
            bad.append(c)
        cells.append(f"{r[0]:.2f}" if r else "  --")
    print(f"c{c:<8}{cells[0]:>12}{cells[1]:>14}{cells[2]:>12}{cells[3]:>14}")
    a, b = get(f"c{c}_rnT"), get(f"c0{c}_rnT")
    if a and b:
        deltas.append(a[0] - b[0])

print()
if deltas:
    print("THE TEST -- under renorm the leading 0 moves all 8142 rows, so this column decides it:")
    for c, d in zip([c for c in codes if get(f"c{c}_rnT") and get(f"c0{c}_rnT")], deltas):
        print(f"  c{c}_rnT  minus  c0{c}_rnT   dAll {d:+.2f}")
    m = st.mean(deltas)
    sd = st.stdev(deltas) if len(deltas) > 1 else float("nan")
    print(f"  mean {m:+.3f}" + (f"   sd {sd:.3f}" if len(deltas) > 1 else ""))
    print("  |mean| < 0.08  ->  global centering is worthless even where it CAN act;")
    print("                     drop the leading 0 from the method for good.")
    print("  |mean| > 0.15  ->  global does contribute, but only once renorm stops the")
    print("                     telescoping from erasing it; the method then needs both.")
if bad:
    print("\nCRASHED / no result: " + " ".join(f"c{c}_rnT" for c in bad))

print("\nfull table for reference:")
for c in codes:
    for d in (f"c{c}", f"c0{c}", f"c{c}_rnT", f"c0{c}_rnT"):
        r = get(d)
        if r:
            print(f"  {d:<14}{r[0]:>7.2f}{r[1]:>7.2f}{r[2]:>7.2f}{r[3]:>7.2f}")
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
print(json.dumps({"text": "```\n" + open(sys.argv[1], errors="ignore").read()[:3500] + "\n```"}))
PY
)"
if curl -sS -f -X POST -H 'Content-type: application/json' --data "${payload}" "${url}" >/dev/null; then
  echo "[$(date '+%F %T')] posted to Slack"
else
  echo "[$(date '+%F %T')] Slack POST FAILED -- summary still at ${SUMMARY}"
  exit 1
fi
