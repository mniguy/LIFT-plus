#!/bin/bash
#
# Wait for the seed-replication set to finish, compute the FIRST REAL seed variance this project
# has had, and post it to Slack. Companion to run_center_seeds.sh.
#
# SCOPE: seeds 1 and 2 only. Seed 0 already exists for all four arms under
# output/center_levelcode25/ and is deliberately NOT read here -- merging the three seeds is done
# by hand. The uncentered baseline and global-only live in a separate repository with their own
# seeds, so no delta against them is computed.
#
# WHAT TWO SEEDS CAN AND CANNOT SAY. An sd from n=2 has one degree of freedom, so it is not a
# usable variance estimate and this script does not pretend otherwise. What two seeds DO answer is
# reproducibility of the ORDERING: the seed-0 run put 0123456_norm 0.17 above 123456 and the other
# two arms level with it. If that pattern repeats at seed 1 and seed 2 it is real; if the sign
# flips between seeds it was noise. The summary therefore prints the per-seed deltas against
# 123456 rather than a pooled t-test.
#
# WATCHES main.py processes whose output_dir is under seeds25/, not the shared levelcode lanes.
# Counts DISTINCT arms: each main.py forks ~11 dataloader workers sharing its cmdline.
#
# A process exiting is not a run succeeding -- after the wait, each expected dir is checked for
# "* Many:" and anything missing is reported as CRASHED.
#
# THE WEBHOOK IS A SECRET: $SLACK_WEBHOOK_URL, else ~/.config/lift_slack_webhook (chmod 600).
#
# USAGE -- start AFTER the launcher, or it sees 0 arms alive and fires immediately:
#   setsid nohup bash scripts/notify_seeds.sh \
#     > output/seeds25/_launch/notify.log 2>&1 < /dev/null &
#
set -uo pipefail
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

OUT_ROOT=${OUT_ROOT:-seeds25}
POLL=${POLL:-120}
WEBHOOK_FILE=${WEBHOOK_FILE:-$HOME/.config/lift_slack_webhook}
PYTHON=${PYTHON:-$([ -x /home/mingyu/.conda/envs/ltl/bin/python ] \
  && echo /home/mingyu/.conda/envs/ltl/bin/python || echo python3)}
export PYTHONNOUSERSITE=1
SUMMARY="output/${OUT_ROOT}/SUMMARY_seeds.txt"

alive(){ ps -eo args | grep -o "[m]ain\.py.*output_dir ${OUT_ROOT}/[^ ]*" \
           | grep -o "output_dir [^ ]*" | sort -u | wc -l; }

echo "[$(date '+%F %T')] watching ${OUT_ROOT}; $(alive) arm(s) alive; polling every ${POLL}s"
while [ "$(alive)" -gt 0 ]; do sleep "${POLL}"; done
echo "[$(date '+%F %T')] all ${OUT_ROOT} processes exited -- tabulating"

"${PYTHON}" scripts/dump_results.py >/dev/null 2>&1 || true

{
  echo "LIFT+ seed replication -- finished $(date '+%F %T %Z') on $(hostname)"
  echo
  "${PYTHON}" - <<'PY'
import glob, os, re, statistics as st

ROOT = "output/seeds25/inat2018"
ARMS = [("123456", ""), ("0123456", ""), ("123456", "_rnT"), ("0123456", "_rnT")]
SEEDS = (1, 2)
REF = "123456"          # the chosen method; everything is reported relative to it

def read(d):
    for f in reversed(sorted(glob.glob(os.path.join(d, "log-*.txt")))):
        t = open(f, errors="ignore").read()
        if "* Many:" not in t:
            continue
        ov = re.findall(r"\* Overall accuracy:\s*([\d.]+)%", t)
        m = re.findall(r"\* Many:\s*([\d.]+)%\s*Med:\s*([\d.]+)%\s*Few:\s*([\d.]+)%", t)
        return (float(ov[-1]), *map(float, m[-1]))
    return None

def label(code, rn):
    return f"{code}{' norm' if rn else ''}"

data, bad = {}, []
for code, rn in ARMS:
    for s in SEEDS:
        d = f"{ROOT}/c{code}{rn}_s{s}"
        r = read(d)
        if r is None:
            bad.append(os.path.basename(d))
        data[(label(code, rn), s)] = r

print("All / Head / Med / Few")
print(f"{'arm':<15}" + "".join(f"{'seed '+str(s):>28}" for s in SEEDS))
for code, rn in ARMS:
    lb = label(code, rn)
    row = ""
    for s in SEEDS:
        r = data[(lb, s)]
        row += (f"{r[0]:>8.2f}{r[1]:>7.2f}{r[2]:>7.2f}{r[3]:>7.2f}" if r
                else f"{'-- no result --':>28}")
    print(f"{lb:<15}{row}")

print(f"\nAll, relative to {REF} within each seed  (seed 0 gave: "
      f"0123456 +0.01, 0123456 norm +0.17, 123456 norm -0.01)")
for code, rn in ARMS:
    lb = label(code, rn)
    if lb == REF:
        continue
    cells = []
    for s in SEEDS:
        a, b = data[(lb, s)], data[(REF, s)]
        cells.append(f"{a[0]-b[0]:+.2f}" if a and b else "  --")
    got = [data[(lb, s)][0] - data[(REF, s)][0] for s in SEEDS
           if data[(lb, s)] and data[(REF, s)]]
    note = ""
    if len(got) == len(SEEDS):
        note = ("   same sign both seeds" if got[0] * got[1] > 0
                else "   SIGN FLIPPED between seeds -> noise")
    print(f"  {lb:<15}" + "".join(f"{c:>9}" for c in cells) + note)

print(f"\nPer-arm mean over seeds {SEEDS} (n={len(SEEDS)}; no sd -- one degree of freedom):")
for code, rn in ARMS:
    lb = label(code, rn)
    v = [data[(lb, s)][0] for s in SEEDS if data[(lb, s)]]
    if v:
        print(f"  {lb:<15}{st.mean(v):>8.2f}   spread {max(v)-min(v):.2f}")

print("\nMerge with the seed-0 runs in output/center_levelcode25/ by hand for the n=3 figures.")
if bad:
    print("\nCRASHED / no result: " + " ".join(bad))
PY
} > "${SUMMARY}" 2>&1
cat "${SUMMARY}"

url="${SLACK_WEBHOOK_URL:-}"
if [ -z "${url}" ] && [ -r "${WEBHOOK_FILE}" ]; then url="$(tr -d '\r\n' < "${WEBHOOK_FILE}")"; fi
if [ -z "${url}" ]; then
  echo "[$(date '+%F %T')] no webhook configured -- summary at ${SUMMARY}, nothing sent"; exit 0
fi
payload="$("${PYTHON}" - "${SUMMARY}" <<'PY'
import json, sys
print(json.dumps({"text": "```\n" + open(sys.argv[1], errors="ignore").read()[:3500] + "\n```"}))
PY
)"
if curl -sS -f -X POST -H 'Content-type: application/json' --data "${payload}" "${url}" >/dev/null; then
  echo "[$(date '+%F %T')] posted to Slack"
else
  echo "[$(date '+%F %T')] Slack POST FAILED -- summary still at ${SUMMARY}"; exit 1
fi
