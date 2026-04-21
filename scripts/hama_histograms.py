"""
HAM-A Batch Scores — Histograms
================================
Saves one histogram PNG per subscale + total score.
Output: analysis_output/histograms/
"""

import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "json_transcripts" / "_batch_hama_scores.json"
OUT_DIR   = BASE_DIR / "analysis_output" / "histograms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Design ─────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
GRID    = "#21262d"

PALETTE = [
    "#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff", "#ffa657",
    "#79c0ff", "#56d364", "#f78166", "#bc8cff", "#ffb77c",
    "#a5f3fc", "#bbf7d0", "#fecaca", "#e9d5ff",
]
TOTAL_COLOR = "#f0c040"

# ── Subscales ──────────────────────────────────────────────────────────────────
SUBSCALES = [
    "anxious_mood", "tension", "fears", "insomnia", "intellectual",
    "depressed_mood", "somatic_muscular", "somatic_sensory",
    "cardiovascular", "respiratory", "gastrointestinal",
    "genitourinary", "autonomic", "behavior_at_interview",
]
LABELS = [s.replace("_", " ").title() for s in SUBSCALES]

# ── Load data ──────────────────────────────────────────────────────────────────
with open(DATA_FILE, "r") as f:
    records = json.load(f)

n      = len(records)
totals = np.array([r["total_score"] for r in records])

print(f"Loaded {n} records  |  mean total = {totals.mean():.2f}  |  max = {totals.max()}")
print(f"Saving histograms to: {OUT_DIR}\n")


def make_histogram(values, title, xlabel, color, filename,
                   bins=None, is_total=False):
    """Save a single styled histogram."""
    values = np.array(values)
    mean_v = values.mean()
    med_v  = np.median(values)
    std_v  = values.std()

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=SUBTEXT, labelsize=10)
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)

    # Bins
    if bins is None:
        bins = np.arange(-0.5, values.max() + 1.5, 1)

    counts, edges, patches = ax.hist(
        values, bins=bins, color=color,
        edgecolor=BG, linewidth=0.5, alpha=0.85
    )

    # Colour bars by count intensity
    max_c = counts.max() if counts.max() > 0 else 1
    for patch, count in zip(patches, counts):
        alpha = 0.45 + 0.55 * (count / max_c)
        patch.set_alpha(alpha)

    # Mean & median lines
    ax.axvline(mean_v, color="#f0c040",  linewidth=1.8, linestyle="-",
               label=f"Mean  {mean_v:.2f}")
    ax.axvline(med_v,  color="#ff7b72",  linewidth=1.8, linestyle="--",
               label=f"Median  {med_v:.0f}")

    # Stats box
    stats_text = (f"N = {n}\n"
                  f"Mean   = {mean_v:.2f}\n"
                  f"Median = {int(med_v)}\n"
                  f"Std    = {std_v:.2f}\n"
                  f"Min    = {int(values.min())}\n"
                  f"Max    = {int(values.max())}")
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes, va="top", ha="right",
            fontsize=9, color=TEXT, family="monospace",
            bbox=dict(facecolor=PANEL, edgecolor=BORDER,
                      boxstyle="round,pad=0.5", alpha=0.9))

    ax.set_title(title, color=TEXT, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Number of Records", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.2, labelcolor=TEXT, facecolor=PANEL)

    # Non-zero annotation
    nonzero_pct = (values > 0).mean() * 100
    ax.text(0.03, 0.97, f"Non-zero scores: {nonzero_pct:.1f}%",
            transform=ax.transAxes, va="top", fontsize=9,
            color=SUBTEXT,
            bbox=dict(facecolor=PANEL, edgecolor=BORDER,
                      boxstyle="round,pad=0.4", alpha=0.85))

    path = OUT_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── Generate one histogram per subscale ───────────────────────────────────────
for i, (s, lbl, color) in enumerate(zip(SUBSCALES, LABELS, PALETTE)):
    vals = [r.get(s, 0) for r in records]
    make_histogram(
        values   = vals,
        title    = f"HAM-A — {lbl}",
        xlabel   = "Score  (0 = Absent  ·  4 = Very Severe)",
        color    = color,
        filename = f"{i+1:02d}_{s}.png",
        bins     = np.arange(-0.5, 5.5, 1),
    )

# ── Total score histogram ─────────────────────────────────────────────────────
make_histogram(
    values   = totals,
    title    = "HAM-A — Total Score Distribution",
    xlabel   = "Total Score  (sum of all 14 subscales)",
    color    = TOTAL_COLOR,
    filename = "15_total_score.png",
    is_total = True,
)

print(f"\nAll 15 histograms saved to: {OUT_DIR}")
