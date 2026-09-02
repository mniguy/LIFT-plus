#!/bin/bash
#
# Wait for the NO-GLOBAL arm set (12 34 56 135 246 123 456 123456) to finish, tabulate each arm
# AGAINST its with-global twin (c12 vs c012, ...), and post the comparison to Slack.
#
# WHAT THIS SET ACTUALLY MEASURES -- read before interpreting the table.
#     With NESTED_MEAN=recompute and renorm off, each level's mean is taken on the running
#     residual, so for any class COVERED at the first non-global level the leading global step
#     cancels exactly:
#         X1 = X - mean(X);   shift = mu_L(X1) = mu_L(X) - mean(X);   X1 - shift = X - mu_L(X)
#     (verified in trainer.py: src = X_out when mean_mode == "recompute").
#     So c0L... and cL... differ ONLY on classes SKIPPED at that level by the GENUS_MIN=2 gate --
#     a skipped class gets shift=0 and therefore KEEPS the global centering it would otherwise
#     have lost. The gap between twins is thus a direct readout of how much the singleton classes
#     matter, and the arms where nothing is skipped are near-exact replicates.
#     Skip counts at the FIRST level of each code: kingdom 1, phylum 5, class 9, order 64,
#     family 463, genus 3000 (of 8142).
#
# THAT MAKES THIS SET DOUBLE DUTY: a coverage probe AND the best noise-floor estimate available,
# since the near-replicate pairs are drawn independently. The summary prints both.
#
# WHY NOT notify_rnT.sh: that one watches bare main.py processes with _rnT output dirs. This set
# runs through run_center_levelcode.sh, so it watches that script's lanes instead -- the same
# detector notify_when_done.sh uses.
#
# THE WEBHOOK IS A SECRET: $SLACK_WEBHOOK_URL, else ~/.config/lift_slack_webhook (chmod 600).
# With neither, the summary is still written to disk.
#
# USAGE (detached; survives SSH dropping and this terminal closing)
#   setsid nohup bash scripts/notify_noglobal.sh \
#     > output/center_levelcode25/_launch/notify_noglobal.log 2>&1 < /dev/null &
#
set -uo pipefail
[ -f main.py ] || { echo "ERROR: run from repo root"; exit 1; }

OUT_ROOT=${OUT_ROOT:-center_levelcode25}
DATASET=${DATASET:-inat2018}
POLL=${POLL:-120}
ARMS=${ARMS:-"12 34 56 135 246 123 456 123456"}
WEBHOOK_FILE=${WEBHOOK_FILE:-$HOME/.config/lift_slack_webhook}
PYTHON=${PYTHON:-$([ -x /home/mingyu/.conda/envs/ltl/bin/python ] \
  && echo /home/mingyu/.conda/envs/ltl/bin/python || echo python3)}
export PYTHONNOUSERSITE=1          # see run_center_levelcode_4gpu.sh -- ~/.local shadows the env
SUMMARY="output/${OUT_ROOT}/SUMMARY_noglobal.txt"

lanes(){ ps -eo args | grep -c "[r]un_center_levelcode\.sh"; }

echo "[$(date '+%F %T')] watching ${ARMS}; $(lanes) lane(s) alive; polling every ${POLL}s"
while [ "$(lanes)" -gt 0 ]; do sleep "${POLL}"; done
echo "[$(date '+%F %T')] all lanes exited -- tabulating"

{
  echo "LIFT+ ${OUT_ROOT} no-global set -- finished $(date '+%F %T %Z') on $(hostname)"
  echo
  "${PYTHON}" - "output/${OUT_ROOT}/${DATASET}" ${ARMS} <<'PY'
import glob, os, re, sys, statistics as st
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

SKIP = {"1": ("kingdom", 1), "2": ("phylum", 5), "3": ("class", 9),
        "4": ("order", 64), "5": ("family", 463), "6": ("genus", 3000)}
print(f"{'code':<10}{'All':>7}{'Head':>7}{'Med':>7}{'Few':>7}   {'twin':>8}{'twinAll':>9}{'dAll':>7}   first-level skips")
deltas, bad = [], []
for c in codes:
    new, old = read(f"{root}/c{c}"), read(f"{root}/c0{c}")
    if new is None:
        bad.append(c); print(f"c{c:<9}{'  -- no result --':>28}"); continue
    lv, n = SKIP.get(c[0], ("?", 0))
    if old:
        d = new[0] - old[0]; deltas.append(d)
        tw, ta, ds = f"c0{c}", f"{old[0]:.2f}", f"{d:+.2f}"
    else:
        tw, ta, ds = f"c0{c}", "  --", "   --"
    print(f"c{c:<9}{new[0]:>7.2f}{new[1]:>7.2f}{new[2]:>7.2f}{new[3]:>7.2f}   "
          f"{tw:>8}{ta:>9}{ds:>7}   {lv} {n}/8142")
if deltas:
    print(f"\npaired |dAll|: mean {st.mean(map(abs, deltas)):.3f}  max {max(map(abs, deltas)):.3f}"
          + (f"  sd of dAll {st.stdev(deltas):.3f}" if len(deltas) > 1 else ""))
    print("These pairs are near-replicates by construction, so sd(dAll) IS a noise-floor estimate.")
if bad:
    print("CRASHED / no result: " + " ".join(f"c{c}" for c in bad))
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
