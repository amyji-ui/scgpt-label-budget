#!/usr/bin/env python
"""Draw the project flowchart and the result plots into plots/.

Run from the project root, after util/02_evaluate.py:
    .venv/bin/python util/07_plots.py

Outputs (PNG at 200 dpi, plus SVG for the report):
    plots/flowchart.*                     The pipeline from data to scoring.
    plots/budget_curve_macro_f1.*         Main result: macro F1 against labeled cells, one line per
                                          method, mean of seeds, with each seed's value marked.
    plots/budget_curve_accuracy.*         The same for accuracy.
    plots/seed_effect.*                   One panel per method: each seed's macro F1 at each budget.
    plots/train_time.*                    Wall-clock time per run (embedding + training + prediction).
    plots/train_time.csv                  The timing numbers, split by step and device.
    plots/results_by_budget.csv           The numbers behind the plots (table view).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS = Path("results/test_smartseq2")
OUT = Path("plots")

# Reference palette (dataviz skill), light mode, categorical slots in fixed order.
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

# Color follows the method, never its rank.
METHODS = {
    "scgpt_finetuned": ("scGPT fine-tuned", SLOTS[0]),
    "scgpt_frozen_logreg": ("scGPT frozen + LR", SLOTS[1]),
    "hvg_logreg": ("HVG + LR", SLOTS[2]),
    "harmony_logreg": ("Harmony + LR", SLOTS[3]),
    "scvi_logreg": ("scVI + LR", SLOTS[4]),
}
SECONDARY = {
    "scgpt_finetuned_logreg": "scGPT fine-tuned + LR\n(secondary check)",
    "scvi_no_smarter_logreg": "scVI without smarter\n(sensitivity run)",
}
BUDGET_ORDER = ["500", "1000", "5000", "all"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID, "xtick.color": INK_2, "ytick.color": INK_2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / name}.png/.svg")


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- flowchart

def box(ax, x, y, w, h, title, body, edge=INK_3, fill="#ffffff", title_color=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                linewidth=1.4, edgecolor=edge, facecolor=fill))
    ax.text(x, y + h / 2 - 0.2, title, ha="center", va="top", fontsize=9.5, weight="bold", color=title_color)
    ax.text(x, y + h / 2 - 0.52, body, ha="center", va="top", fontsize=8, color=INK_2, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color=INK_3, shrinkA=0, shrinkB=0))


def flowchart():
    fig, ax = plt.subplots(figsize=(13, 10.9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10.9)
    ax.axis("off")

    box(ax, 6.5, 9.65, 6.2, 1.15, "Data: Luecken et al. human pancreas",
        "16,382 cells × 19,093 genes · 9 technology batches · 14 cell types\n"
        "restricted to the 18,160 genes in the scGPT vocabulary")
    box(ax, 6.5, 7.95, 11.8, 1.55, "Split  (util/00_preprocessing.py)",
        "Test: whole held-out batch smartseq2 (2,394 cells, 13 types) · labels used only for scoring\n"
        "Training pool: other 8 batches (13,988 cells)\n"
        "Nested label budgets 500 · 1,000 · 5,000 · all, ≥5 cells per type, 10% validation (counts toward budget)\n"
        "Seeds 0 and 1 draw different label subsets; the same subsets are used by every method")
    arrow(ax, 6.5, 9.07, 6.5, 8.73)

    cols = [1.3, 3.9, 6.5, 9.1, 11.7]
    method_boxes = [
        ("scgpt_finetuned", "scGPT fine-tuned", "util/01_finetuning.py",
         "binned expression\n≤3,000 random genes/cell\n10 epochs, best val loss\nclassifier head predicts"),
        ("scgpt_frozen_logreg", "scGPT frozen", "util/04_scgpt_embed.py",
         "same input as fine-tuned\npretrained weights, no training\n512-d <cls> embedding"),
        ("hvg_logreg", "HVG", "util/03_logreg.py",
         "log-normalized\n2,000 HVGs chosen on\ntraining batches only"),
        ("harmony_logreg", "Harmony", "util/05_harmony_embed.py",
         "same 2,000 HVGs → scale\n→ 50 PCs → Harmony\n(batch = tech, all cells)"),
        ("scvi_logreg", "scVI", "util/06_scvi_embed.py",
         "counts fixed (UMI inverse,\nrounding) · same HVGs\n30-d latent, gene-batch disp."),
    ]
    for x, (key, title, script, body) in zip(cols, method_boxes):
        color = METHODS[key][1]
        box(ax, x, 5.55, 2.35, 2.05, title, f"{script}\n\n{body}", edge=color, title_color=INK)
        ax.add_patch(FancyBboxPatch((x - 1.175, 6.5), 2.35, 0.08, boxstyle="square,pad=0", color=color, lw=0))
        arrow(ax, x, 7.17, x, 6.6)

    box(ax, 7.8, 3.55, 9.4, 1.05, "Logistic regression  (util/03_logreg.py)",
        "same fixed settings for every budget: C = 0.01, α = 0 (no tuning)\n"
        "trained on the budget's labeled cells (fit + val)")
    for x in cols[1:]:
        arrow(ax, x, 4.52, x, 4.08)

    box(ax, 6.5, 1.95, 11.8, 1.15, "Evaluation  (util/02_evaluate.py, frozen before scoring)",
        "one scoring function for every run: macro F1 over the 13 types in the test batch (primary),\n"
        "accuracy, per-type F1 · predictions must cover exactly the test cells")
    arrow(ax, cols[0], 4.52, cols[0], 2.53)
    for x in cols[1:]:
        arrow(ax, x, 3.02, x, 2.53)

    box(ax, 6.5, 0.55, 5.0, 0.8, "Label-budget curves  (util/07_plots.py)", "macro F1 against labeled cells, one line per method")
    arrow(ax, 6.5, 1.37, 6.5, 0.95)

    ax.text(0.05, 10.85, "Pipeline: how many labels does scGPT need?", fontsize=13, weight="bold", va="top")
    save(fig, "flowchart")


# ---------------------------------------------------------------- results

def load():
    s = pd.read_csv(RESULTS / "metrics_summary.csv", dtype={"budget": str})
    s["n_labels"] = s["n_fit"] + s["n_val"]
    return s


def budget_curve(s, metric, ylabel, name, ylim):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    style_axes(ax)
    budgets = s.groupby("budget")["n_labels"].max().reindex(BUDGET_ORDER)
    for key, (label, color) in METHODS.items():
        m = s[s["method"] == key]
        if m.empty:
            continue
        g = m.groupby("budget")[metric].agg(["mean", "min", "max"]).reindex(BUDGET_ORDER)
        x = budgets.to_numpy()
        for xi, lo, hi in zip(x, g["min"], g["max"]):
            ax.plot([xi, xi], [lo, hi], color=color, linewidth=1.2, alpha=0.6, solid_capstyle="round")
        for seed, marker in ((0, "o"), (1, "s")):
            sv = m[m["seed"] == seed].set_index("budget")[metric].reindex(BUDGET_ORDER)
            ax.scatter(x, sv, marker=marker, s=14, facecolor=SURFACE, edgecolor=color, linewidth=1, zorder=3)
        ax.plot(x, g["mean"], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(budgets.to_numpy())
    ax.set_xticklabels([f"{int(n):,}" + ("\n(all)" if b == "all" else "") for b, n in budgets.items()])
    ax.minorticks_off()
    ax.set_xlim(budgets.iloc[0] / 1.3, budgets.iloc[-1] * 1.3)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Labeled cells (fit + validation), log scale")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} against label budget")
    # Method lines, then what the small markers mean (the mean line is the large marker).
    handles, labels = ax.get_legend_handles_labels()
    for marker, text in (("o", "seed 0"), ("s", "seed 1")):
        handles.append(plt.Line2D([], [], linestyle="none", marker=marker, markersize=5,
                                  markerfacecolor=SURFACE, markeredgecolor=INK_3))
        labels.append(text)
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    save(fig, name)


def seed_effect(s):
    panels = [(k, v[0]) for k, v in METHODS.items()] + [(k, v) for k, v in SECONDARY.items()]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), sharey=True)
    seed_colors = {0: SLOTS[0], 1: SLOTS[1]}
    budgets = s.groupby("budget")["n_labels"].max().reindex(BUDGET_ORDER)
    x = np.arange(len(BUDGET_ORDER))
    for ax, (key, title) in zip(axes.flat, panels):
        style_axes(ax)
        m = s[s["method"] == key]
        v = {seed: m[m["seed"] == seed].set_index("budget")["macro_f1"].reindex(BUDGET_ORDER) for seed in (0, 1)}
        dx = 0.09  # side by side, so equal values stay visible
        for xi, a, b in zip(x, v[0], v[1]):
            ax.plot([xi - dx, xi + dx], [a, b], color=INK_3, linewidth=1.2, zorder=1)
            ax.text(xi + 0.2, (a + b) / 2, f"{abs(a - b):.2f}", fontsize=7.5, color=INK_2, va="center")
        for seed, marker in ((0, "o"), (1, "s")):
            ax.scatter(x + (dx if seed else -dx), v[seed], s=46, marker=marker, color=seed_colors[seed], edgecolor=SURFACE,
                       linewidth=1.5, zorder=3, label=f"seed {seed}")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["500", "1k", "5k", "all"])
        ax.set_xlim(-0.4, len(x) - 0.4)
    last = axes.flat[-1]
    last.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    last.legend(handles, labels, loc="upper left", frameon=False, fontsize=10, title="Macro F1 per seed",
                title_fontproperties={"weight": "bold", "size": 10})
    last.text(0, 0.62, "Each seed draws a different label subset.\n"
              "For scVI and scGPT it also changes model\ninitialization and training randomness.\n\n"
              "Numbers beside each pair are the gap\nbetween seeds (macro F1).\n\n"
              "Seed 0 is drawn left, seed 1 right;\nthe grey line joins the pair.",
              transform=last.transAxes, fontsize=8.5, color=INK_2, va="top", linespacing=1.4)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro F1")
    for ax in axes[1, :3]:
        ax.set_xlabel("Labeled cells")
    fig.suptitle("How much does the seed change the result?", x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "seed_effect")


def table(s):
    names = {**{k: v[0] for k, v in METHODS.items()}, **{k: v.replace("\n", " ") for k, v in SECONDARY.items()}}
    t = (s[s["method"].isin(names)]
         .groupby(["method", "budget"])
         .agg(n_labels=("n_labels", "max"), macro_f1_mean=("macro_f1", "mean"), macro_f1_seed_min=("macro_f1", "min"),
              macro_f1_seed_max=("macro_f1", "max"), accuracy_mean=("accuracy", "mean"), n_seeds=("seed", "nunique"))
         .reset_index())
    t["method"] = t["method"].map(names)
    t["budget"] = pd.Categorical(t["budget"], BUDGET_ORDER, ordered=True)
    t = t.sort_values(["method", "budget"]).round(4)
    t.to_csv(OUT / "results_by_budget.csv", index=False)
    print(f"Wrote {OUT / 'results_by_budget.csv'}")


EMBED_DIR = Path("results/embeddings/processed")
# Label-free embedding step per method: (folder, device). Its cost is paid once per seed and shared
# by every budget, but each budget's run needs it, so it is added to every budget's total.
EMBED_STEPS = {"scgpt_frozen_logreg": ("scgpt_frozen", "GPU"), "harmony_logreg": ("harmony", "CPU"),
               "scvi_logreg": ("scvi", "GPU")}
DEVICE_LABEL = {
    "scgpt_finetuned": "GPU fine-tuning",
    "scgpt_frozen_logreg": "GPU embedding + CPU LR",
    "hvg_logreg": "CPU only",
    "harmony_logreg": "CPU only",
    "scvi_logreg": "GPU training + CPU LR",
}


def embed_seconds(folder, seed):
    info = json.loads((EMBED_DIR / folder / f"seed_{seed}.json").read_text())
    if folder == "harmony":
        # HVG selection and PCA are cached after the first seed, so seed 1 logs ~0 s for them;
        # charge every seed the first seed's PCA time.
        first = json.loads((EMBED_DIR / folder / "seed_0.json").read_text())
        return first["pca_seconds"] + info["harmony_seconds"]
    return info.get("embed_seconds", info.get("train_seconds"))


def timing_table(s):
    rows = []
    for _, r in s[s["method"].isin(METHODS)].iterrows():
        fit = r["train_seconds"] + r["test_inference_seconds"]
        step = EMBED_STEPS.get(r["method"])
        embed = embed_seconds(step[0], r["seed"]) if step else 0.0
        rows.append({"method": r["method"], "budget": r["budget"], "seed": r["seed"], "n_labels": r["n_labels"],
                     "embedding_seconds": embed, "embedding_device": step[1] if step else "",
                     "training_and_prediction_seconds": fit,
                     "training_device": "GPU" if r["method"] == "scgpt_finetuned" else "CPU",
                     "total_seconds": embed + fit})
    return pd.DataFrame(rows)


def train_time(s):
    t = timing_table(s)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    style_axes(ax)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8)
    budgets = s.groupby("budget")["n_labels"].max().reindex(BUDGET_ORDER)
    x = budgets.to_numpy()
    for key, (label, color) in METHODS.items():
        g = t[t["method"] == key].groupby("budget")["total_seconds"].mean().reindex(BUDGET_ORDER)
        gpu = key in ("scgpt_finetuned", "scgpt_frozen_logreg", "scvi_logreg")
        ax.plot(x, g, color=color, linewidth=2, linestyle="-" if gpu else (0, (4, 2)), marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{label}  ({DEVICE_LABEL[key]})", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n):,}" + ("\n(all)" if b == "all" else "") for b, n in budgets.items()])
    ticks = [1, 10, 60, 600, 3600, 3 * 3600]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["1 s", "10 s", "1 min", "10 min", "1 h", "3 h"])
    ax.minorticks_off()
    ax.set_xlim(x[0] / 1.3, x[-1] * 1.3)
    ax.set_ylim(0.3, 4 * 3600)
    ax.set_xlabel("Labeled cells (fit + validation), log scale")
    ax.set_ylabel("Wall-clock time per run, log scale")
    ax.set_title("Training time against label budget")
    # Plain lines in the legend, so solid versus dashed stays readable.
    handles = [plt.Line2D([], [], color=h.get_color(), linewidth=2, linestyle=h.get_linestyle())
               for h in ax.get_lines()]
    labels = [h.get_label() for h in ax.get_lines()]
    handles += [plt.Line2D([], [], color=INK_3, linewidth=2), plt.Line2D([], [], color=INK_3, linewidth=2, linestyle=(0, (4, 2)))]
    labels += ["solid: uses the GPU", "dashed: CPU only"]
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9,
              handlelength=3)
    save(fig, "train_time")

    out = (t.groupby(["method", "budget"])
           .agg(n_labels=("n_labels", "max"), embedding_seconds=("embedding_seconds", "mean"),
                embedding_device=("embedding_device", "first"),
                training_and_prediction_seconds=("training_and_prediction_seconds", "mean"),
                training_device=("training_device", "first"), total_seconds=("total_seconds", "mean"))
           .reset_index())
    out["budget"] = pd.Categorical(out["budget"], BUDGET_ORDER, ordered=True)
    out["method"] = out["method"].map({k: v[0] for k, v in METHODS.items()})
    out.sort_values(["method", "budget"]).round(1).to_csv(OUT / "train_time.csv", index=False)
    print(f"Wrote {OUT / 'train_time.csv'}")


def main():
    OUT.mkdir(exist_ok=True)
    flowchart()
    s = load()
    budget_curve(s, "macro_f1", "Macro F1", "budget_curve_macro_f1", (0.55, 1.0))
    budget_curve(s, "accuracy", "Accuracy", "budget_curve_accuracy", (0.75, 1.0))
    seed_effect(s)
    train_time(s)
    table(s)


if __name__ == "__main__":
    main()
