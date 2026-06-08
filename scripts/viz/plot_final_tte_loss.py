#!/usr/bin/env python3
import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_RE = re.compile(
    r"epoch \[(?P<epoch>\d+)/(?P<epochs>\d+)\] "
    r"batch \[(?P<batch>\d+)/(?P<batches>\d+)\].*? "
    r"loss (?P<loss>[0-9.]+) \((?P<loss_avg>[0-9.]+)\)"
)
ACC_RE = re.compile(r"\* Overall accuracy: (?P<acc>[0-9.]+)%")


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name)


def parse_log(path):
    rows = []
    final_acc = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LOG_RE.search(line)
            if m:
                epoch = int(m.group("epoch"))
                batch = int(m.group("batch"))
                batches = int(m.group("batches"))
                rows.append(
                    {
                        "step": (epoch - 1) * batches + batch,
                        "epoch": epoch,
                        "batch": batch,
                        "batches": batches,
                        "loss": float(m.group("loss")),
                        "loss_avg": float(m.group("loss_avg")),
                    }
                )
                continue

            acc_m = ACC_RE.search(line)
            if acc_m:
                final_acc = float(acc_m.group("acc"))

    return rows, final_acc


def plot_series(series, title, out_path):
    plt.figure(figsize=(11, 6))
    for label, rows, final_acc in series:
        if not rows:
            continue
        xs = [r["step"] for r in rows]
        ys = [r["loss_avg"] for r in rows]
        suffix = "" if final_acc is None else f" ({final_acc:.2f}%)"
        plt.plot(xs, ys, linewidth=1.6, label=f"{label}{suffix}")

    plt.title(title)
    plt.xlabel("Training step")
    plt.ylabel("Smoothed training loss")
    plt.grid(True, alpha=0.25)
    if len(series) <= 12:
        plt.legend(fontsize=8)
    else:
        plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_individual(label, rows, final_acc, out_path):
    xs = [r["step"] for r in rows]
    raw = [r["loss"] for r in rows]
    avg = [r["loss_avg"] for r in rows]

    plt.figure(figsize=(11, 6))
    plt.plot(xs, raw, linewidth=0.8, alpha=0.35, label="loss")
    plt.plot(xs, avg, linewidth=1.8, label="smoothed loss")
    title = label if final_acc is None else f"{label} | final acc {final_acc:.2f}%"
    plt.title(title)
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output/final_tte")
    parser.add_argument("--out", default="output/final_tte/loss_plots")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    individual_dir = out_dir / "individual"
    by_dir_dir = out_dir / "by_directory"
    individual_dir.mkdir(parents=True, exist_ok=True)
    by_dir_dir.mkdir(parents=True, exist_ok=True)

    logs = sorted(root.rglob("log-*.txt"))
    parsed = []

    for path in logs:
        rows, final_acc = parse_log(path)
        if not rows:
            continue
        rel = path.relative_to(root)
        label = str(rel).replace("/log-", " / ").replace(".txt", "")
        parsed.append((path, label, rows, final_acc))

        plot_individual(
            label,
            rows,
            final_acc,
            individual_dir / f"{safe_name(str(rel.with_suffix('')))}.png",
        )

    groups = defaultdict(list)
    for path, label, rows, final_acc in parsed:
        run_dir = str(path.parent.relative_to(root))
        groups[run_dir].append((path.name.replace("log-", "").replace(".txt", ""), rows, final_acc))

    for run_dir, series in groups.items():
        plot_series(
            series,
            f"Training loss: {run_dir}",
            by_dir_dir / f"{safe_name(run_dir)}.png",
        )

    plot_series(
        [(label, rows, final_acc) for _, label, rows, final_acc in parsed],
        "Training loss: all output/final_tte logs",
        out_dir / "all_logs_loss.png",
    )

    csv_path = out_dir / "loss_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["log", "final_acc", "num_points", "first_loss_avg", "last_loss_avg"])
        for path, _, rows, final_acc in parsed:
            writer.writerow(
                [
                    str(path),
                    "" if final_acc is None else final_acc,
                    len(rows),
                    rows[0]["loss_avg"],
                    rows[-1]["loss_avg"],
                ]
            )

    print(f"Parsed logs: {len(parsed)}")
    print(f"Wrote: {out_dir / 'all_logs_loss.png'}")
    print(f"Wrote: {by_dir_dir}")
    print(f"Wrote: {individual_dir}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
