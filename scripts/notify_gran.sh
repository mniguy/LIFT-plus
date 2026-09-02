#!/bin/bash
#
# Wait for the granularity set to finish, tabulate the k-means sweep against its taxonomy
# anchors, and post to Slack.
#
# WHAT THE TABLE IS FOR (the science lives in run_center_granularity.sh -- read that header):
#   s<N> is k-means at N classes per cluster, t0<L> is the same construction using taxonomy.
#   The two pairs that matter are printed first and separately:
#       s2  vs t06   genus granularity   -- granularity, or biology?
#       s7  vs t05   family granularity
#   Everything else fills in the granularity curve.
#
# THE FALLBACK COLUMN IS NOT DECORATION. A cluster below GENUS_MIN is replaced by the GLOBAL
# mean, so an arm with a high fallback rate is partly a global-centering rerun and its
# granularity label is a lie. Measured at launch: s2 33.6%, s7 1.6% -- against genus skipping
# 36.8% and family 5.7%, which is why the s2/t06 pair is a fair comparison and s7/t05 is
# slightly tilted (the taxonomy anchor skips more than its cluster counterpart).
#
# WATCHING: the lanes are spawned as `bash scripts/run_center_granularity.sh` with no arguments,
# so the pattern is anchored to the whole cmdline. A loose substring match also hits any process
# that merely names the script -- an editor, a tail, a grep -- and would fire instantly.
#
# THE WEBHOOK IS A SECRET: $SLACK_WEBHOOK_URL, else ~/.config/lift_slack_webhook (chmod 600).
# With neither, the summary is still written to disk.
#
# USAGE (detached)
#   setsid nohup bash scripts/notify_gran.sh \
#     > output/center_gran25/_launch/notify_gran.log 2>&1 < /dev/null &
#
set -uo pipefail
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

OUT_ROOT=${OUT_ROOT:-center_gran25}
DATASET=${DATASET:-inat2018}
POLL=${POLL:-120}
WEBHOOK_FILE=${WEBHOOK_FILE:-$HOME/.config/lift_slack_webhook}
PYTHON=${PYTHON:-$([ -x /home/mingyu/.conda/envs/ltl/bin/python ] \
  && echo /home/mingyu/.conda/envs/ltl/bin/python || echo python3)}
export PYTHONNOUSERSITE=1
SUMMARY="output/${OUT_ROOT}/SUMMARY_gran.txt"

lanes(){ pgrep -fc "^bash scripts/run_center_granularity\.sh$" 2>/dev/null || true; }

n="$(lanes)"; echo "[$(date '+%F %T')] watching ${OUT_ROOT}; ${n:-0} lane(s) alive; polling every ${POLL}s"
while n="$(lanes)"; [ -n "${n}" ] && [ "${n}" != "0" ]; do sleep "${POLL}"; done
echo "[$(date '+%F %T')] all lanes exited -- tabulating"

{
  echo "LIFT+ ${OUT_ROOT} granularity sweep -- finished $(date '+%F %T %Z') on $(hostname)"
  echo "baseline 80.63 | global-only 80.52 | best so far 81.11 | noise floor 0.08"
  echo
  "${PYTHON}" - "output/${OUT_ROOT}/${DATASET}" <<'PY'
import glob, os, re, sys
root = sys.argv[1]

def read(d):
    for f in reversed(sorted(glob.glob(os.path.join(root, d, "log-*.txt")))):
        t = open(f, errors="ignore").read()
        m = re.findall(r"\* Many:\s*([\d.]+)%\s*Med:\s*([\d.]+)%\s*Few:\s*([\d.]+)%", t)
        if not m:
            continue
        ov = re.findall(r"\* Overall accuracy:\s*([\d.]+)%", t)
        fb = re.search(r"->\s*(\d+)/(\d+) classes fell back", t)
        sk = None
        if fb:
            sk = 100.0 * int(fb.group(1)) / int(fb.group(2))
        return (float(ov[-1]), *map(float, m[-1])), sk
    return None, None

def line(lbl, d):
    r, sk = read(d)
    if r is None:
        return f"  {lbl:<16}{'-- no result --':>28}"
    s = f"{sk:5.1f}%" if sk is not None else "    -"
    return f"  {lbl:<16}{r[0]:>7.2f}{r[1]:>7.2f}{r[2]:>7.2f}{r[3]:>7.2f}   {s}"

hdr = f"  {'arm':<16}{'All':>7}{'Head':>7}{'Med':>7}{'Few':>7}   {'fallback':>6}"
print("THE TWO PAIRS")
print(hdr)
for a, b, what in (("s2", "t06", "genus granularity"), ("s7", "t05", "family granularity")):
    print(f"  -- {what}")
    print(line(f"{a}  (k-means)", a))
    print(line(f"{b} (taxonomy)", b))
    ra, _ = read(a); rb, _ = read(b)
    if ra and rb:
        d = ra[0] - rb[0]
        verdict = "granularity (taxonomy adds nothing)" if abs(d) < 0.1 else \
                  ("taxonomy AHEAD" if d < 0 else "k-means AHEAD")
        print(f"      dAll {d:+.2f}  ->  {verdict}   (noise floor 0.08)")
print()
print("GRANULARITY CURVE  (k-means only; NOTE the fallback column -- local-group coverage")
print("                    is NOT constant down the sweep, so the fine end is diluted)")
print(hdr)
for s in (2, 4, 7, 15, 30, 64, 143, 326):
    print(line(f"s{s}  (~{s}/cluster)", f"s{s}"))
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
