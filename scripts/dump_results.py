#!/usr/bin/env python
"""Dump every finished run under output/ to docs/results.tsv.

WHY THIS EXISTS
    output/ is gitignored (.gitignore: /output), so every measured number is invisible to
    anything that only has the repository -- a fresh session, a different agent, a collaborator,
    future-you on another machine. This writes the numbers, AND the configuration that produced
    them, into a tracked file so they survive without the logs.

    agg_runs.py already tabulates All/Head/Med/Few for eyeballing. This is different: it also
    records mode / levels / min_size / renorm / seed, because a row labelled only "c0246" is
    useless to a reader who does not already know the digit-code convention.

USAGE
    python scripts/dump_results.py                 # rewrite docs/results.tsv
    python scripts/dump_results.py --check         # exit 1 if the file is stale (for a hook)
    python scripts/dump_results.py --root output/center_gran25
"""
import argparse
import glob
import os
import re
import sys

FIELDS = ["run", "dataset", "All", "Head", "Med", "Few",
          "mode", "detail", "min_size", "renorm", "seed", "epochs", "fallback"]


def parse(log_path):
    txt = open(log_path, errors="ignore").read()
    hit = re.findall(r"\* Many:\s*([\d.]+)%\s*Med:\s*([\d.]+)%\s*Few:\s*([\d.]+)%", txt)
    if not hit:
        return None                      # unfinished or crashed run
    overall = re.findall(r"\* Overall accuracy:\s*([\d.]+)%", txt)
    head, med, few = hit[-1]

    def cfg(key, default="-"):
        m = re.search(rf"^{key}:\s*(.*)$", txt, re.M)
        return m.group(1).strip() if m else default

    mode = cfg("PROMPT_CENTER_MODE")
    if cfg("PROMPT_CENTER") != "True":
        mode = "none"                    # a plain baseline, whatever the stale MODE default says

    # "detail" is whatever identifies THIS arm inside its mode
    detail = "-"
    if mode == "nested":
        detail = cfg("PROMPT_CENTER_NESTED_LEVELS")
    elif mode == "cluster":
        size = cfg("PROMPT_CENTER_CLUSTER_SIZE", "0")
        detail = f"size={size}" if size not in ("0", "-") else f"k={cfg('PROMPT_CENTER_CLUSTER_K')}"
    elif mode in ("cascade", "cascade_lex"):
        detail = cfg("PROMPT_CENTER_CASCADE")
    elif mode in ("level", "level_keep", "blend", "shrink", "proj"):
        detail = cfg("PROMPT_CENTER_LEVEL")

    # k-means/genus classes that got the GLOBAL mean instead of a local one -- the number that
    # decides whether an arm's granularity label is honest
    fb = re.search(r"->\s*(\d+)/(\d+) classes fell back", txt)
    fallback = f"{100 * int(fb.group(1)) / int(fb.group(2)):.1f}%" if fb else "-"

    return {
        "All": overall[-1] if overall else "nan",
        "Head": head, "Med": med, "Few": few,
        "mode": mode, "detail": detail or "-",
        "min_size": cfg("PROMPT_CENTER_GENUS_MIN"),
        "renorm": cfg("PROMPT_CENTER_NESTED_RENORM"),
        "seed": cfg("seed"), "epochs": cfg("num_epochs"),
        "fallback": fallback,
    }


def collect(root):
    rows = []
    for d in sorted({os.path.dirname(f) for f in
                     glob.glob(os.path.join(root, "**", "log-*.txt"), recursive=True)}):
        logs = sorted(glob.glob(os.path.join(d, "log-*.txt")))
        rec = next((r for r in (parse(f) for f in reversed(logs)) if r), None)
        if rec is None:
            continue
        rel = os.path.relpath(d, root).split(os.sep)
        rec["dataset"] = rel[-2] if len(rel) > 1 else "-"
        rec["run"] = os.path.relpath(d, "output")
        rows.append(rec)
    return rows


def render(rows):
    out = ["\t".join(FIELDS)]
    for r in sorted(rows, key=lambda x: x["run"]):
        out.append("\t".join(str(r.get(f, "-")) for f in FIELDS))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--out", default="docs/results.tsv")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the file on disk is out of date")
    args = ap.parse_args()

    body = render(collect(args.root))
    if args.check:
        cur = open(args.out).read() if os.path.exists(args.out) else ""
        if cur != body:
            print(f"{args.out} is STALE -- run: python scripts/dump_results.py", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date")
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(body)
    print(f"wrote {args.out}  ({len(body.splitlines()) - 1} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
