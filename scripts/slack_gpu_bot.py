#!/usr/bin/env python
"""Slack Socket Mode bot for the LIFT+ centering experiments: GPUs, runs, results, loss curves.

WHY SOCKET MODE
    An incoming webhook is outbound-only, so it cannot answer "show me the GPUs right now".
    Socket Mode fixes that WITHOUT opening an inbound port: this process dials OUT to Slack and
    Slack pushes events down that WebSocket. Nothing has to be reachable from the internet --
    which is what makes it usable on a GPU box behind a firewall. (Slash commands would also
    arrive over this same socket, if you ever add them; they do NOT require a public endpoint.)

THE SECURITY PROPERTY THIS FILE EXISTS TO KEEP
    There is no LLM here. It costs zero tokens however often it is used, and it cannot be talked
    into running something else. Every command is a FIXED argv list -- shell=False everywhere, no
    string interpolation. The only value that ever comes from a Slack message is an arm code, and
    that is matched against ^[0-6]{1,7}$ before it is used, so it can only ever name an arm.
    Contrast a general Claude-Code-in-Slack bridge: there, anyone who can post in the channel gets
    arbitrary command execution on this machine, plus a token bill per message.

COMMANDS
    !gpu                     per-GPU utilisation, memory, temperature
    !status                  arms training right now, with epoch/batch/eta
    !result                  agg_runs.py table of every finished arm
    !run <code>              start one arm on a free GPU   (e.g. !run 0246)
    !loss <code> [<code>…]   training-loss curves, uploaded as a PNG
    !help

ACCESS CONTROL (optional, on top of Slack's own channel membership)
    SLACK_GPU_ALLOW_CHANNELS   comma-separated channel IDs; empty = any channel it is in
    SLACK_GPU_ALLOW_USERS      comma-separated user IDs;    empty = anyone in that channel
    SLACK_GPU_ALLOW_RUN        "1" to enable !run (default off -- !run starts GPU jobs)

CONFIG
    SLACK_BOT_TOKEN   xoxb-…   required
    SLACK_APP_TOKEN   xapp-…   required, app-level token with connections:write
"""
import os
import re
import subprocess
import sys
import time

from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.environ.get("SLACK_GPU_OUT_ROOT", "center_levelcode25")
DATASET = os.environ.get("SLACK_GPU_DATASET", "inat2018")
LTL_PY = "/home/mingyu/.conda/envs/ltl/bin/python"

ALLOW_CHANNELS = {c for c in os.environ.get("SLACK_GPU_ALLOW_CHANNELS", "").replace(" ", "").split(",") if c}
ALLOW_USERS = {u for u in os.environ.get("SLACK_GPU_ALLOW_USERS", "").replace(" ", "").split(",") if u}
ALLOW_RUN = os.environ.get("SLACK_GPU_ALLOW_RUN", "0") == "1"

# The ONLY shape a Slack message may contribute. A code is a chain of taxonomy levels,
# 0=global 1=kingdom 2=phylum 3=class 4=order 5=family 6=genus.
ARM_RE = re.compile(r"^[0-6]{1,7}$")

SMI_GPU = ["nvidia-smi", "--format=csv,noheader,nounits",
           "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"]
SMI_APPS = ["nvidia-smi", "--format=csv,noheader,nounits", "--query-compute-apps=gpu_uuid,pid,used_memory"]
SMI_UUID = ["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu=index,uuid"]

BATCH_RE = re.compile(r"epoch \[(\d+)/(\d+)\] batch \[(\d+)/(\d+)\].*?eta ([\d:]+)")
DONE_RE = re.compile(r"\* Many:")


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def run(argv, timeout=60, cwd=REPO):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, shell=False, cwd=cwd)


def arm_dir(code):
    return os.path.join(REPO, "output", OUT_ROOT, DATASET, f"c{code}")


def newest_log(d):
    try:
        logs = sorted(f for f in os.listdir(d) if f.startswith("log-") and f.endswith(".txt"))
    except OSError:
        return None
    return os.path.join(d, logs[-1]) if logs else None


def running_arms():
    """{code: (epoch, total, batch, batches, eta)} for arms with a live main.py."""
    out = {}
    try:
        ps = run(["ps", "-eo", "args"], timeout=15).stdout
    except Exception:
        return out
    for line in ps.splitlines():
        if "main.py" not in line or "output_dir" not in line:
            continue
        m = re.search(r"output_dir\s+\S*?/c([0-6]{1,7})(?:\s|$)", line)
        if not m:
            continue
        code = m.group(1)
        if code in out:
            continue
        prog = None
        lg = newest_log(arm_dir(code))
        if lg:
            try:
                with open(lg, errors="ignore") as f:
                    tail = f.readlines()[-400:]
                for ln in reversed(tail):
                    b = BATCH_RE.search(ln)
                    if b:
                        prog = b.groups()
                        break
            except OSError:
                pass
        out[code] = prog
    return out


def arm_finished(code):
    lg = newest_log(arm_dir(code))
    if not lg:
        return False
    try:
        with open(lg, errors="ignore") as f:
            return bool(DONE_RE.search(f.read()))
    except OSError:
        return False


def free_gpus():
    """GPU indices with no compute processes on them."""
    try:
        idx2uuid = {}
        for row in run(SMI_UUID, timeout=15).stdout.strip().splitlines():
            p = [x.strip() for x in row.split(",")]
            if len(p) >= 2:
                idx2uuid[p[0]] = p[1]
        busy = {r.split(",")[0].strip() for r in run(SMI_APPS, timeout=15).stdout.strip().splitlines() if r.strip()}
    except Exception:
        return []
    return [i for i, u in sorted(idx2uuid.items(), key=lambda kv: int(kv[0])) if u not in busy]


# ----------------------------------------------------------------------------- commands
def cmd_gpu():
    try:
        r = run(SMI_GPU, timeout=25)
    except (OSError, subprocess.SubprocessError) as e:
        return f"nvidia-smi failed: {e}"
    if r.returncode != 0:
        return f"nvidia-smi exited {r.returncode}: {r.stderr.strip()[:300]}"
    nproc = len([p for p in run(SMI_APPS, timeout=25).stdout.strip().splitlines() if p.strip()])
    lines = [f"{'GPU':<4}{'util':>6}{'memory':>16}{'temp':>7}  name"]
    for row in r.stdout.strip().splitlines():
        p = [x.strip() for x in row.split(",")]
        if len(p) < 6:
            continue
        idx, name, util, used, total, temp = p[:6]
        try:
            pct = f"{100 * int(used) / int(total):.0f}%"
        except (ValueError, ZeroDivisionError):
            pct = "?"
        lines.append(f"{idx:<4}{util + '%':>6}{f'{used}/{total} MiB':>16}{temp + 'C':>7}  {name} ({pct})")
    lines.append(f"\n{nproc} compute process(es)   {time.strftime('%F %T %Z')}")
    return "\n".join(lines)


def cmd_status():
    live = running_arms()
    if not live:
        return f"no arm is training right now.   free GPUs: {free_gpus() or 'none'}"
    lines = [f"{len(live)} arm(s) training:"]
    for code, prog in sorted(live.items()):
        if prog:
            e, E, b, B, eta = prog
            lines.append(f"  c{code:<8} epoch {e}/{E}  batch {b}/{B}  eta {eta}")
        else:
            lines.append(f"  c{code:<8} (starting up)")
    lines.append(f"\nfree GPUs: {free_gpus() or 'none'}")
    return "\n".join(lines)


def cmd_result():
    try:
        r = run([LTL_PY, "scripts/agg_runs.py", os.path.join("output", OUT_ROOT), "--sort", "all"],
                timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return f"agg_runs failed: {e}"
    body = (r.stdout or r.stderr).strip()
    return body[:3400] if body else "no finished runs yet"


def cmd_run(code):
    if not ALLOW_RUN:
        return ("!run is disabled. It starts GPU jobs, so it is opt-in:\n"
                "set SLACK_GPU_ALLOW_RUN=1 in /home/mingyu/.config/slack_gpu_bot.env and restart the bot.")
    if not ARM_RE.match(code):
        return f"`{code[:20]}` is not a valid arm code (digits 0-6, 1-7 of them, e.g. 0246)"
    if code in running_arms():
        return f"c{code} is already training -- refusing to start a duplicate."
    if arm_finished(code):
        return f"c{code} already has a finished result. Use `!result`, or pick another code."
    gpus = free_gpus()
    if not gpus:
        return "every GPU is busy -- not starting anything. `!gpu` to see."
    gpu = gpus[0]

    os.makedirs(os.path.join(REPO, "output", OUT_ROOT, "_launch"), exist_ok=True)
    logpath = os.path.join(REPO, "output", OUT_ROOT, "_launch", f"slackrun_c{code}.log")
    env = dict(os.environ, GPU_ID=str(gpu), ARMS=code, OUT_ROOT=OUT_ROOT,
               PYTHON=LTL_PY, PYTHONNOUSERSITE="1")
    try:
        with open(logpath, "ab") as lf:
            subprocess.Popen(["bash", "scripts/run_center_levelcode.sh"], cwd=REPO, env=env,
                             stdout=lf, stderr=lf, stdin=subprocess.DEVNULL,
                             start_new_session=True)     # detached: outlives this bot
    except Exception as e:
        return f"failed to launch c{code}: {e}"
    return (f"started c{code} on GPU {gpu}\n"
            f"  log: output/{OUT_ROOT}/_launch/slackrun_c{code}.log\n"
            f"  `!status` to follow it; the finish notifier will post the table when everything ends.")


def cmd_loss(codes):
    bad = [c for c in codes if not ARM_RE.match(c)]
    if bad:
        return None, f"invalid arm code(s): {', '.join(b[:20] for b in bad)}"
    missing = [c for c in codes if not newest_log(arm_dir(c))]
    if missing:
        return None, f"no log yet for: {', '.join('c' + c for c in missing)}"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from plot_train_loss import parse, smooth      # reuse the repo's own parser, no duplicate regex

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for c in codes:
        x, y = parse(arm_dir(c))
        if len(y) == 0:
            continue
        ax.plot(x, smooth(y, 25), lw=1.6, label=f"c{c}  (end {y[-50:].mean():.3f})")
    ax.set_xlabel("training progress (epoch fraction)")
    ax.set_ylabel("train loss (smoothed, window 25)")
    ax.set_title(f"{OUT_ROOT} / {DATASET}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(REPO, "output", OUT_ROOT, f"_loss_{'_'.join(codes)}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, f"train loss: {', '.join('c' + c for c in codes)}"


HELP = ("`!gpu` GPUs   `!status` what's training   `!result` finished-run table\n"
        "`!run <code>` start an arm on a free GPU   `!loss <code> [<code>…]` loss curves\n"
        "codes are digits 0-6: 0=global 1=kingdom 2=phylum 3=class 4=order 5=family 6=genus")


def main():
    bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
    app_token = os.environ.get("SLACK_APP_TOKEN", "")
    if not bot_token.startswith("xoxb-") or not app_token.startswith("xapp-"):
        sys.exit("ERROR: SLACK_BOT_TOKEN (xoxb-…) and SLACK_APP_TOKEN (xapp-…) must both be set")

    web = WebClient(token=bot_token)
    sm = SocketModeClient(app_token=app_token, web_client=web)

    def on_request(client: SocketModeClient, req: SocketModeRequest):
        # Ack first, always. Slack redelivers anything not acked within 3s -> double posts.
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        if req.type != "events_api":
            return
        ev = req.payload.get("event", {}) or {}
        if ev.get("type") not in ("message", "app_mention"):
            return
        if ev.get("bot_id") or ev.get("subtype"):        # ignore bots/edits/joins -> no loops
            return

        text = re.sub(r"<@[UW][A-Z0-9]+>", "", ev.get("text") or "").strip()
        if not text.startswith("!"):
            return
        parts = text.split()
        cmd, args = parts[0].lower(), parts[1:]
        channel, user = ev.get("channel", ""), ev.get("user", "")
        if ALLOW_CHANNELS and channel not in ALLOW_CHANNELS:
            log(f"ignored: channel {channel} not allowed"); return
        if ALLOW_USERS and user not in ALLOW_USERS:
            log(f"ignored: user {user} not allowed"); return

        thread = ev.get("thread_ts")
        log(f"{cmd} {args} from user={user} channel={channel}")
        try:
            if cmd == "!gpu":
                web.chat_postMessage(channel=channel, thread_ts=thread, text="```\n" + cmd_gpu() + "\n```")
            elif cmd == "!status":
                web.chat_postMessage(channel=channel, thread_ts=thread, text="```\n" + cmd_status() + "\n```")
            elif cmd == "!result":
                web.chat_postMessage(channel=channel, thread_ts=thread, text="```\n" + cmd_result() + "\n```")
            elif cmd == "!run":
                msg = cmd_run(args[0]) if args else "usage: `!run <code>`  e.g. `!run 0246`"
                web.chat_postMessage(channel=channel, thread_ts=thread, text="```\n" + msg + "\n```")
            elif cmd == "!loss":
                if not args:
                    web.chat_postMessage(channel=channel, thread_ts=thread,
                                         text="usage: `!loss <code> [<code>…]`  e.g. `!loss 0246 0123`")
                    return
                path, note = cmd_loss(args[:6])
                if path:
                    web.files_upload_v2(channel=channel, thread_ts=thread, file=path,
                                        title=note, initial_comment=note)
                else:
                    web.chat_postMessage(channel=channel, thread_ts=thread, text=note)
            elif cmd in ("!help", "!commands"):
                web.chat_postMessage(channel=channel, thread_ts=thread, text=HELP)
        except Exception as e:
            log(f"handler failed: {e}")
            try:
                web.chat_postMessage(channel=channel, thread_ts=thread, text=f"error: `{str(e)[:300]}`")
            except Exception:
                pass

    sm.socket_mode_request_listeners.append(on_request)
    log(f"connecting; channels={ALLOW_CHANNELS or 'any'}; users={ALLOW_USERS or 'any'}; run={'ON' if ALLOW_RUN else 'OFF'}")
    sm.connect()
    log("connected -- waiting for events")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
