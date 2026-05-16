"""
HAM-A Score Distribution Visualizer
====================================
Compares the distribution of Hamilton Anxiety Rating Scale (HAM-A)
parameters across three LLM-scored datasets:
  - Llama 3.1 8B Instruct
  - Mistral 7B
  - Qwen 2.5 7B Instruct

Generates multiple visualization panels saved as PNG files.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from math import pi

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "json_transcripts")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "visualizations_all_3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "Llama 3.1 8B": "_batch_hama_scores_llama3.json",
    "Mistral 7B": "_batch_hama_scores_mistarl_7b.json",
    "Qwen 2.5 7B": "_batch_hama_scores_qwen25.json",
}

HAMA_PARAMS = [
    "anxious_mood",
    "tension",
    "fears",
    "insomnia",
    "intellectual",
    "depressed_mood",
    "somatic_muscular",
    "somatic_sensory",
    "cardiovascular",
    "respiratory",
    "gastrointestinal",
    "genitourinary",
    "autonomic",
    "behavior_at_interview",
]

PARAM_LABELS = [p.replace("_", " ").title() for p in HAMA_PARAMS]

# Color palette – distinct, colorblind-friendly
MODEL_COLORS = {
    "Llama 3.1 8B": "#6366f1",   # indigo
    "Mistral 7B": "#f59e0b",      # amber
    "Qwen 2.5 7B": "#10b981",     # emerald
}

# ── Load data ──────────────────────────────────────────────────────────────
def load_data():
    """Load all three JSON files into a dict of DataFrames."""
    data = {}
    for model_name, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
        data[model_name] = df
        print(f"Loaded {model_name}: {len(df)} records")
    return data


# ── 1. Overlaid histograms for each parameter ─────────────────────────────
def plot_histograms(data):
    """Overlaid histograms (one subplot per HAM-A parameter)."""
    fig, axes = plt.subplots(4, 4, figsize=(22, 18))
    fig.suptitle(
        "HAM-A Parameter Score Distributions — Histogram Comparison",
        fontsize=20, fontweight="bold", y=0.98,
    )
    axes_flat = axes.flatten()

    for idx, param in enumerate(HAMA_PARAMS):
        ax = axes_flat[idx]
        for model_name, df in data.items():
            values = df[param].dropna().values
            ax.hist(
                values,
                bins=np.arange(-0.5, 5.5, 1),
                alpha=0.45,
                label=model_name,
                color=MODEL_COLORS[model_name],
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(PARAM_LABELS[idx], fontsize=12, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xticks(range(5))
        ax.legend(fontsize=7, loc="upper right")

    # Hide unused subplots
    for idx in range(len(HAMA_PARAMS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "1_histograms.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── 2. Box plots side-by-side ─────────────────────────────────────────────
def plot_boxplots(data):
    """Grouped box plots for each parameter."""
    # Build long-form DataFrame
    frames = []
    for model_name, df in data.items():
        melted = df[HAMA_PARAMS].melt(var_name="Parameter", value_name="Score")
        melted["Model"] = model_name
        frames.append(melted)
    long_df = pd.concat(frames, ignore_index=True)
    long_df["Parameter"] = long_df["Parameter"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(24, 8))
    sns.boxplot(
        data=long_df,
        x="Parameter",
        y="Score",
        hue="Model",
        palette=MODEL_COLORS,
        ax=ax,
        fliersize=2,
        linewidth=0.8,
    )
    ax.set_title(
        "HAM-A Parameter Score Distributions — Box Plot Comparison",
        fontsize=18, fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=13)
    ax.tick_params(axis="x", rotation=40, labelsize=10)
    ax.legend(fontsize=11, title="Model")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "2_boxplots.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── 3. Violin plots ───────────────────────────────────────────────────────
def plot_violins(data):
    """Violin plots for each parameter across models."""
    frames = []
    for model_name, df in data.items():
        melted = df[HAMA_PARAMS].melt(var_name="Parameter", value_name="Score")
        melted["Model"] = model_name
        frames.append(melted)
    long_df = pd.concat(frames, ignore_index=True)
    long_df["Parameter"] = long_df["Parameter"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(24, 8))
    sns.violinplot(
        data=long_df,
        x="Parameter",
        y="Score",
        hue="Model",
        palette=MODEL_COLORS,
        ax=ax,
        inner="quartile",
        linewidth=0.7,
        cut=0,
        density_norm="width",
    )
    ax.set_title(
        "HAM-A Parameter Score Distributions — Violin Plot Comparison",
        fontsize=18, fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=13)
    ax.tick_params(axis="x", rotation=40, labelsize=10)
    ax.legend(fontsize=11, title="Model")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "3_violins.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── 4. Mean-score heatmap ─────────────────────────────────────────────────
def plot_heatmap(data):
    """Heatmap showing mean score per parameter per model."""
    means = {}
    for model_name, df in data.items():
        means[model_name] = df[HAMA_PARAMS].mean()
    mean_df = pd.DataFrame(means, index=PARAM_LABELS)

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        mean_df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.8,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Mean Score"},
    )
    ax.set_title(
        "Mean HAM-A Scores by Model",
        fontsize=18, fontweight="bold", pad=15,
    )
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0, labelsize=11)
    ax.tick_params(axis="x", labelsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "4_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── 5. Radar / spider chart of mean scores ────────────────────────────────
def plot_radar(data):
    """Radar chart comparing mean scores across models."""
    means = {}
    for model_name, df in data.items():
        means[model_name] = df[HAMA_PARAMS].mean().values

    N = len(HAMA_PARAMS)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PARAM_LABELS, fontsize=9)

    for model_name, vals in means.items():
        values = list(vals) + [vals[0]]
        ax.plot(
            angles, values,
            linewidth=2, linestyle="-",
            label=model_name,
            color=MODEL_COLORS[model_name],
        )
        ax.fill(angles, values, alpha=0.15, color=MODEL_COLORS[model_name])

    ax.set_title(
        "Mean HAM-A Scores — Radar Comparison",
        fontsize=18, fontweight="bold", pad=25,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "5_radar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── 6. Stacked bar – score-level proportions ──────────────────────────────
def plot_stacked_bars(data):
    """Stacked bar chart showing proportion of each score level (0-4)."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
    fig.suptitle(
        "Score-Level Proportions per Parameter by Model",
        fontsize=20, fontweight="bold", y=1.02,
    )
    score_colors = ["#e5e7eb", "#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a5f"]

    for ax, (model_name, df) in zip(axes, data.items()):
        proportions = {}
        for param in HAMA_PARAMS:
            counts = df[param].value_counts(normalize=True).reindex(range(5), fill_value=0)
            proportions[param.replace("_", " ").title()] = counts.values

        prop_df = pd.DataFrame(proportions, index=range(5)).T

        prop_df.plot(
            kind="barh", stacked=True, ax=ax,
            color=score_colors, edgecolor="white", linewidth=0.5,
            legend=False,
        )
        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Proportion")
        ax.set_xlim(0, 1)

    # Shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=score_colors[i]) for i in range(5)
    ]
    fig.legend(
        handles, ["Score 0", "Score 1", "Score 2", "Score 3", "Score 4"],
        loc="lower center", ncol=5, fontsize=12, frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "6_stacked_bars.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── 7. Total score distribution ───────────────────────────────────────────
def plot_total_score(data):
    """KDE + histogram of total_score across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Histogram
    for model_name, df in data.items():
        ax1.hist(
            df["total_score"].dropna(),
            bins=30, alpha=0.45,
            label=model_name,
            color=MODEL_COLORS[model_name],
            edgecolor="white",
        )
    ax1.set_title("Total Score — Histogram", fontsize=15, fontweight="bold")
    ax1.set_xlabel("Total Score")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=11)

    # KDE
    for model_name, df in data.items():
        sns.kdeplot(
            df["total_score"].dropna(),
            ax=ax2,
            label=model_name,
            color=MODEL_COLORS[model_name],
            linewidth=2.5,
            fill=True, alpha=0.2,
        )
    ax2.set_title("Total Score — Density (KDE)", fontsize=15, fontweight="bold")
    ax2.set_xlabel("Total Score")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=11)

    fig.suptitle(
        "Total HAM-A Score Distribution Across Models",
        fontsize=18, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "7_total_score.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── 8. Summary statistics table ───────────────────────────────────────────
def plot_summary_table(data):
    """Visual table of descriptive statistics."""
    rows = []
    for model_name, df in data.items():
        for param in HAMA_PARAMS:
            s = df[param].dropna()
            rows.append({
                "Model": model_name,
                "Parameter": param.replace("_", " ").title(),
                "Mean": f"{s.mean():.2f}",
                "Std": f"{s.std():.2f}",
                "Median": f"{s.median():.1f}",
                "Max": int(s.max()),
                "Non-Zero %": f"{(s > 0).mean() * 100:.1f}%",
            })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"), index=False)
    print(f"  Saved -> {os.path.join(OUTPUT_DIR, 'summary_statistics.csv')}")


# ── 9. Non-zero percentage comparison (grouped bar) ───────────────────────
def plot_nonzero_pct(data):
    """Grouped bar chart of non-zero score percentages."""
    pct_data = {}
    for model_name, df in data.items():
        pct_data[model_name] = [(df[p] > 0).mean() * 100 for p in HAMA_PARAMS]

    x = np.arange(len(HAMA_PARAMS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(22, 8))
    for i, (model_name, pcts) in enumerate(pct_data.items()):
        ax.bar(
            x + i * width, pcts, width,
            label=model_name,
            color=MODEL_COLORS[model_name],
            edgecolor="white", linewidth=0.5,
        )

    ax.set_title(
        "Percentage of Non-Zero Scores per Parameter",
        fontsize=18, fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Non-Zero %", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(PARAM_LABELS, rotation=40, ha="right", fontsize=10)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="grey", linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "8_nonzero_pct.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  HAM-A Score Distribution Visualizer")
    print("=" * 60)
    data = load_data()
    print(f"\nGenerating visualizations -> {OUTPUT_DIR}\n")

    plot_histograms(data)
    plot_boxplots(data)
    plot_violins(data)
    plot_heatmap(data)
    plot_radar(data)
    plot_stacked_bars(data)
    plot_total_score(data)
    plot_summary_table(data)
    plot_nonzero_pct(data)

    print("\nAll visualizations saved successfully!")
    print(f"   Output folder: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
