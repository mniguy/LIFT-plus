#!/usr/bin/env python
"""Generate paper-ready LaTeX for T2 (component ablation, 3-seed mean±std) and
T3 (gating @ strong reg), from output/. Writes output/paper/tables.tex and prints.

  python scripts/make_tables.py
"""
import glob
import os

import numpy as np


def splits(c):
    c = np.asarray(c)
    return (c > 100), ((c >= 20) & (c <= 100)), (c < 20)


def seed_stats(root, rung, h, m, fw):
    dirs = sorted(glob.glob(f"{root}/{rung}_seed*"))
    A = [np.load(os.path.join(d, "cls_accs.npy")).astype(float) for d in dirs
         if os.path.exists(os.path.join(d, "cls_accs.npy"))]
    if not A:
        return None
    V = np.stack([[a.mean(), a[h].mean(), a[m].mean(), a[fw].mean()] for a in A])
    return V.mean(0), (V.std(0, ddof=1) if len(A) > 1 else np.zeros(4)), len(A)


def split_row(d, h, m, fw):
    a = np.load(os.path.join(d, "cls_accs.npy")).astype(float)
    return np.array([a.mean(), a[h].mean(), a[m].mean(), a[fw].mean()])


def main():
    os.makedirs("output/paper", exist_ok=True)
    lines = []

    # ---- T2: component ablation (method_ablation, 3-seed) ----
    rungs = [("lift+", "LIFT+ (repro)"), ("hybrid", "\\ +hybrid init"),
             ("hybrid_kd", "\\ +KD"), ("full", "\\ +InfoNCE"),
             ("full_gate", "\\ +gating (HyGAT)")]
    lines.append("% ===== T2: component ablation (3-seed mean$\\pm$std) =====")
    lines.append("\\begin{tabular}{l cccc cccc}\\toprule")
    lines.append(" & \\multicolumn{4}{c}{ImageNet-LT} & \\multicolumn{4}{c}{Places-LT}\\\\")
    lines.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}")
    lines.append("Method & All & Many & Med & Few & All & Many & Med & Few\\\\\\midrule")
    cache = {}
    for ds in ["imagenet_lt", "places_lt"]:
        root = f"output/method_ablation/{ds}"
        num = np.load(f"{root}/lift+_seed0/cls_num_list.npy")
        cache[ds] = (root, splits(num))
    for rung, label in rungs:
        cells = []
        for ds in ["imagenet_lt", "places_lt"]:
            root, (h, m, fw) = cache[ds]
            r = seed_stats(root, rung, h, m, fw)
            if r is None:
                cells += ["--"] * 4
            else:
                mean, std, _ = r
                cells += [f"{mean[i]:.2f}\\tiny$\\pm${std[i]:.2f}" for i in range(4)]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    lines.append("\\bottomrule\\end{tabular}\n")

    # ---- T3: gating at strong reg (gate_collapse, single seed) ----
    variants = [("fixed", "fixed (no gate)"), ("agreement", "agreement"),
                ("freq", "frequency"), ("freq_inv", "frequency$^{-1}$"),
                ("shuffled", "shuffled (control)")]
    lines.append("% ===== T3: gating @ strong reg (InfoNCE $\\lambda$=0.05, $T$=0.001) =====")
    lines.append("\\begin{tabular}{l cccc cccc}\\toprule")
    lines.append(" & \\multicolumn{4}{c}{ImageNet-LT} & \\multicolumn{4}{c}{Places-LT}\\\\")
    lines.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}")
    lines.append("Gate & All & Head & Med & Few & All & Head & Med & Few\\\\\\midrule")
    for v, label in variants:
        cells = []
        for ds in ["imagenet_lt", "places_lt"]:
            root = f"output/gate_collapse/{ds}"
            num = np.load(f"{root}/fixed/cls_num_list.npy"); h, m, fw = splits(num)
            d = os.path.join(root, v)
            if os.path.exists(os.path.join(d, "cls_accs.npy")):
                r = split_row(d, h, m, fw); cells += [f"{r[i]:.2f}" for i in range(4)]
            else:
                cells += ["--"] * 4
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    lines.append("\\bottomrule\\end{tabular}")

    tex = "\n".join(lines)
    with open("output/paper/tables.tex", "w") as f:
        f.write(tex + "\n")
    print(tex)
    print("\n[save] output/paper/tables.tex")


if __name__ == "__main__":
    main()
