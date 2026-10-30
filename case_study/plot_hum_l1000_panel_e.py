"""Plot Panel E: early proteome-wide recovery of observed HUM-L1000 sites."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================== modify here ========================
PROJECT_DIR = "E:/ChenClaw/ProbPred"
RATIO = "1_5"
RESULT_DIR = os.path.join(PROJECT_DIR, "case_study", "results", RATIO)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "figure", "case_study")
FIGURE_NAME = "hum_l1000_panel_E"
MAX_SCREENED = 10
TOP_PERCENTAGES = [1, 5, 10]
# =============================================================


STRATEGIES = ["random", "cold_protein", "cold_probe"]
LABELS = {"random": "Random split", "cold_protein": "Cold-protein split", "cold_probe": "Cold-probe split"}
COLORS = {"random": "#4C78A8", "cold_protein": "#E45756", "cold_probe": "#59A14F"}


def load_recovery_curve(strategy):
    path = os.path.join(RESULT_DIR, f"{strategy}_predictions.csv.gz")
    frame = pd.read_csv(path, usecols=["Label", "Score_mean"]).sort_values("Score_mean", ascending=False)
    labels = frame["Label"].to_numpy(dtype=np.int8)
    total_sites, total_positives = len(labels), int(labels.sum())
    if total_positives == 0:
        raise ValueError(f"No observed positives in {path}")
    cumulative = np.cumsum(labels)
    maximum_rank = int(np.ceil(total_sites * MAX_SCREENED / 100))
    ranks = np.unique(np.linspace(1, maximum_rank, 2500).astype(int))
    exact_ranks = [int(np.ceil(total_sites * value / 100)) for value in TOP_PERCENTAGES]
    ranks = np.unique(np.append(ranks, exact_ranks))
    screened = ranks / total_sites * 100
    recovered = cumulative[ranks - 1] / total_positives * 100
    marked = {value: cumulative[rank - 1] / total_positives * 100 for value, rank in zip(TOP_PERCENTAGES, exact_ranks)}
    return screened, recovered, marked


def configure_style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})


def main():
    configure_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    for strategy in STRATEGIES:
        screened, recovered, marked = load_recovery_curve(strategy)
        axis.plot(screened, recovered, color=COLORS[strategy], linewidth=2.2, label=LABELS[strategy])
        axis.scatter(TOP_PERCENTAGES, [marked[value] for value in TOP_PERCENTAGES], s=30, color=COLORS[strategy], edgecolor="white", linewidth=0.5, zorder=3)
    random_x = np.linspace(0, MAX_SCREENED, 500)
    axis.plot(random_x, random_x, color="#777777", linestyle="--", linewidth=1.2, label="Random ranking")
    for percentage in TOP_PERCENTAGES:
        axis.axvline(percentage, color="#BDBDBD", linestyle=":", linewidth=0.8)
        axis.text(percentage, 1.5, f"Top {percentage}%", rotation=90, ha="right", va="bottom", fontsize=7.5, color="#666666")
    axis.set_xlim(0, MAX_SCREENED)
    axis.set_ylim(0, 60)
    axis.set_xlabel("Candidate arginine sites screened (%)")
    axis.set_ylabel("Observed positive sites recovered (%)")
    axis.set_title("Proteome-wide cumulative recovery")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")
    axis.text(-0.13, 1.08, "E", transform=axis.transAxes, fontsize=15, fontweight="bold", va="top")
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.90)
    figure.savefig(os.path.join(OUTPUT_DIR, f"{FIGURE_NAME}.pdf"), bbox_inches="tight")
    figure.savefig(os.path.join(OUTPUT_DIR, f"{FIGURE_NAME}.png"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {os.path.join(OUTPUT_DIR, FIGURE_NAME)}.pdf/.png")


if __name__ == "__main__":
    main()
