"""Plot HUM-L1000 case-study panels C and D."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ======================== modify here ========================
PROJECT_DIR = "E:/ChenClaw/ProbPred"
RATIO = "1_5"
RESULT_DIR = os.path.join(PROJECT_DIR, "case_study", "results", RATIO)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "figure", "case_study")
FIGURE_NAME = "hum_l1000_panels_C_D"
TOP_PERCENTAGES = [1, 5, 10]
# =============================================================


STRATEGIES = ["random", "cold_protein", "cold_probe"]
STRATEGY_LABELS = {"random": "Random", "cold_protein": "Cold\nprotein", "cold_probe": "Cold\nprobe"}
COLORS = {"random": "#4C78A8", "cold_protein": "#E45756", "cold_probe": "#59A14F"}


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_metrics():
    seed = pd.read_csv(os.path.join(RESULT_DIR, "seed_metrics.csv"))
    ensemble = pd.read_csv(os.path.join(RESULT_DIR, "ensemble_metrics.csv"))
    for strategy in STRATEGIES:
        if strategy not in set(seed["Strategy"]) or strategy not in set(ensemble["Strategy"]):
            raise ValueError(f"Missing metrics for strategy: {strategy}")
    return seed, ensemble


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def plot_metric(axis, seed, ensemble, metric, ylabel, ylim, baseline=None, value_format="{:.3f}"):
    x = np.arange(len(STRATEGIES))
    for index, strategy in enumerate(STRATEGIES):
        seed_values = seed.loc[seed["Strategy"] == strategy, metric].to_numpy(dtype=float)
        offsets = np.linspace(-0.12, 0.12, len(seed_values))
        axis.scatter(index + offsets, seed_values, s=24, color=COLORS[strategy], alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
        ensemble_value = float(ensemble.loc[ensemble["Strategy"] == strategy, metric].iloc[0])
        axis.scatter(index, ensemble_value, s=125, marker="*", color=COLORS[strategy], edgecolor="black", linewidth=0.6, zorder=4)
        axis.text(index, ensemble_value + (ylim[1] - ylim[0]) * 0.035, value_format.format(ensemble_value), ha="center", va="bottom", fontsize=7.5)
    if baseline is not None:
        axis.axhline(baseline, color="#777777", linestyle="--", linewidth=1, label="Random baseline")
    axis.set_xticks(x, [STRATEGY_LABELS[value] for value in STRATEGIES])
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)
    style_axis(axis)


def plot_panel_c(axis_auroc, axis_ap, seed, ensemble):
    plot_metric(axis_auroc, seed, ensemble, "AUROC", "AUROC", (0.50, 0.90), baseline=0.50)
    plot_metric(axis_ap, seed, ensemble, "AP_fold_over_baseline", "AP / random AP", (0, 9), baseline=1, value_format="{:.2f}×")
    axis_auroc.set_title("Global ranking")
    axis_ap.set_title("Positive-site enrichment")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#777777", markeredgecolor="white", markersize=5, label="Individual seed"),
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#777777", markeredgecolor="black", markersize=9, label="Five-model ensemble"),
        Line2D([0], [0], color="#777777", linestyle="--", linewidth=1, label="Random baseline"),
    ]
    axis_auroc.legend(handles=handles, frameon=False, loc="lower right")
    axis_auroc.text(-0.32, 1.13, "C", transform=axis_auroc.transAxes, fontsize=15, fontweight="bold", va="top")


def cumulative_recovery(strategy):
    path = os.path.join(RESULT_DIR, f"{strategy}_predictions.csv.gz")
    frame = pd.read_csv(path, usecols=["Label", "Score_mean"]).sort_values("Score_mean", ascending=False)
    labels = frame["Label"].to_numpy(dtype=np.int8)
    total_positive = int(labels.sum())
    if total_positive == 0:
        raise ValueError(f"No observed positives in {path}")
    cumulative = np.cumsum(labels)
    ranks = np.unique(np.geomspace(1, len(frame), 2500).astype(int))
    for percentage in TOP_PERCENTAGES:
        ranks = np.append(ranks, int(np.ceil(len(frame) * percentage / 100)))
    ranks = np.unique(np.clip(ranks, 1, len(frame)))
    screened = ranks / len(frame) * 100
    recall = cumulative[ranks - 1] / total_positive * 100
    marked = {}
    for percentage in TOP_PERCENTAGES:
        cutoff = int(np.ceil(len(frame) * percentage / 100))
        marked[percentage] = cumulative[cutoff - 1] / total_positive * 100
    return screened, recall, marked


def plot_panel_d(axis):
    for strategy in STRATEGIES:
        screened, recall, marked = cumulative_recovery(strategy)
        axis.plot(screened, recall, color=COLORS[strategy], linewidth=2, label=STRATEGY_LABELS[strategy].replace("\n", " "))
        axis.scatter(TOP_PERCENTAGES, [marked[value] for value in TOP_PERCENTAGES], s=24, color=COLORS[strategy], edgecolor="white", linewidth=0.4, zorder=3)
    axis.plot([0.01, 100], [0.01, 100], color="#777777", linestyle="--", linewidth=1, label="Random ranking")
    for percentage in TOP_PERCENTAGES:
        axis.axvline(percentage, color="#BDBDBD", linestyle=":", linewidth=0.8)
        axis.text(percentage, 2.5, f"Top {percentage}%", rotation=90, ha="right", va="bottom", fontsize=7, color="#666666")
    axis.set_xscale("log")
    axis.set_xlim(0.01, 100)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Candidate sites screened (%)")
    axis.set_ylabel("Observed positive sites recovered (%)")
    axis.set_title("Proteome-wide cumulative recovery")
    axis.legend(frameon=False, loc="upper left")
    style_axis(axis)
    axis.text(-0.18, 1.13, "D", transform=axis.transAxes, fontsize=15, fontweight="bold", va="top")


def main():
    configure_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seed, ensemble = load_metrics()
    figure = plt.figure(figsize=(12.0, 4.1))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.8], wspace=0.38)
    axis_auroc = figure.add_subplot(grid[0, 0])
    axis_ap = figure.add_subplot(grid[0, 1])
    axis_recovery = figure.add_subplot(grid[0, 2])
    plot_panel_c(axis_auroc, axis_ap, seed, ensemble)
    plot_panel_d(axis_recovery)
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.88)
    figure.savefig(os.path.join(OUTPUT_DIR, f"{FIGURE_NAME}.pdf"), bbox_inches="tight")
    figure.savefig(os.path.join(OUTPUT_DIR, f"{FIGURE_NAME}.png"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {os.path.join(OUTPUT_DIR, FIGURE_NAME)}.pdf/.png")


if __name__ == "__main__":
    main()
