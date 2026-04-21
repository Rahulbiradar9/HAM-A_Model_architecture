"""
HAM-A Batch Score — Alternative Visualizations v2
===================================================
Saves each graph as a separate PNG file to analysis_output/,
using completely different chart types from visualize_hama.py.

Output files:
  1_lollipop.png
  2_ridgeline.png
  3_waffle.png
  4_treemap.png
  5_bump_rank.png
  6_diverging_bar.png
  7_polar_area.png
  8_hexbin.png
  9_step_ecdf.png
  10_bubble.png
"""

import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "json_transcripts" / "_batch_hama_scores.json"
OUT_DIR   = BASE_DIR / "analysis_output" / "separate_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Design ─────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
GRID    = "#21262d"

PALETTE = [
    "#58a6ff","#3fb950","#ff7b72","#d2a8ff","#ffa657",
    "#79c0ff","#56d364","#f78166","#bc8cff","#ffb77c",
    "#a5f3fc","#bbf7d0","#fecaca","#e9d5ff",
]

SEVERITY_COLORS = {
    "Mild / None": "#3fb950",
    "Moderate":    "#d2a8ff",
    "Severe":      "#ffa657",
    "Very Severe": "#ff7b72",
}

# ── Subscales ──────────────────────────────────────────────────────────────────
SUBSCALES = [
    "anxious_mood","tension","fears","insomnia","intellectual",
    "depressed_mood","somatic_muscular","somatic_sensory","cardiovascular",
    "respiratory","gastrointestinal","genitourinary","autonomic",
    "behavior_at_interview",
]
LABELS = [s.replace("_", " ").title() for s in SUBSCALES]

# ── Load ───────────────────────────────────────────────────────────────────────
with open(DATA_FILE, "r") as f:
    records = json.load(f)

totals = np.array([r["total_score"] for r in records])
n      = len(records)
means  = np.array([np.mean([r.get(s, 0) for r in records]) for s in SUBSCALES])
stds   = np.array([np.std ([r.get(s, 0) for r in records]) for s in SUBSCALES])

def severity_label(score):
    if score < 17: return "Mild / None"
    if score < 25: return "Moderate"
    if score < 30: return "Severe"
    return "Very Severe"

sev_counts = {"Mild / None": 0, "Moderate": 0, "Severe": 0, "Very Severe": 0}
for t in totals:
    sev_counts[severity_label(t)] += 1

def base_style(ax, title="", grid_axis="y"):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=SUBTEXT, labelsize=9)
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)
    if title:
        ax.set_title(title, color=TEXT, fontsize=14, fontweight="bold", pad=10)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, linestyle="--", alpha=0.8)
        ax.set_axisbelow(True)

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved -> {path}")

print(f"Loaded {n} records | mean={totals.mean():.2f} | max={totals.max()}")
print(f"Saving to: {OUT_DIR}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOLLIPOP — mean score per subscale (sorted)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_lollipop():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    base_style(ax, "Lollipop — Mean Score per Subscale (sorted, ± Std Dev)", grid_axis="x")
    ax.grid(axis="y", visible=False)

    idx = np.argsort(means)
    slabels  = [LABELS[i]   for i in idx]
    smeans   = means[idx]
    sstds    = stds[idx]
    scolors  = [PALETTE[i % len(PALETTE)] for i in idx]

    y = np.arange(len(SUBSCALES))
    ax.hlines(y, 0, smeans, colors=BORDER, linewidth=1.5, zorder=1)
    ax.scatter(smeans, y, color=scolors, s=120, zorder=3,
               edgecolors=BG, linewidths=0.9)
    ax.errorbar(smeans, y, xerr=sstds, fmt="none",
                ecolor=SUBTEXT, elinewidth=1, capsize=4, capthick=1, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(slabels, color=TEXT, fontsize=10)
    ax.set_xlabel("Mean Score (0 – 4)", fontsize=11)
    ax.set_xlim(-0.05, max(smeans + sstds) + 0.25)
    fig.suptitle(f"N = {n}  |  Overall subscale mean = {means.mean():.3f}",
                 color=SUBTEXT, fontsize=10, y=0.01)
    save(fig, "1_lollipop.png")

plot_lollipop()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RIDGELINE (Joy Plot) — score distribution per subscale
# ═══════════════════════════════════════════════════════════════════════════════
def plot_ridgeline():
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
    base_style(ax, "Ridgeline (Joy Plot) — Score Distribution per Subscale", grid_axis=None)

    cmap   = LinearSegmentedColormap.from_list("ridge", ["#58a6ff", "#3fb950", "#ff7b72"])
    gap    = 2.8
    max_sc = 4

    for si, s in enumerate(SUBSCALES):
        vals   = np.array([r.get(s, 0) for r in records], dtype=float)
        x_grid = np.linspace(-0.5, max_sc + 0.5, 300)
        bw     = 0.22
        kde    = np.zeros_like(x_grid)
        for v in vals:
            kde += np.exp(-0.5 * ((x_grid - v) / bw) ** 2)
        kde /= kde.max() + 1e-9

        base_y = si * gap
        colour = cmap(si / (len(SUBSCALES) - 1))
        ax.fill_between(x_grid, base_y, base_y + kde * gap * 0.88,
                        alpha=0.72, color=colour, linewidth=0)
        ax.plot(x_grid, base_y + kde * gap * 0.88,
                color=colour, linewidth=1, alpha=0.95)
        ax.axhline(base_y, color=BORDER, linewidth=0.5, alpha=0.5)

    ax.set_yticks([i * gap for i in range(len(SUBSCALES))])
    ax.set_yticklabels(LABELS, color=TEXT, fontsize=10)
    ax.set_xlabel("Score Value  (0 – 4)", fontsize=11)
    ax.set_xlim(-0.5, max_sc + 0.5)
    ax.set_ylim(-gap * 0.3, len(SUBSCALES) * gap)
    save(fig, "2_ridgeline.png")

plot_ridgeline()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WAFFLE CHART — severity category proportions
# ═══════════════════════════════════════════════════════════════════════════════
def plot_waffle():
    fig, ax = plt.subplots(figsize=(10, 9), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_title("Waffle Chart — Severity Category Proportions",
                 color=TEXT, fontsize=14, fontweight="bold", pad=10)
    ax.set_aspect("equal")

    COLS, ROWS = 10, 10
    total_sq   = COLS * ROWS
    squares    = []
    for cat, cnt in sev_counts.items():
        squares.extend([cat] * round(cnt / n * total_sq))
    while len(squares) < total_sq:
        squares.append(list(sev_counts.keys())[0])
    squares = squares[:total_sq]

    for idx_w, cat in enumerate(squares):
        row = idx_w // COLS
        col = idx_w  % COLS
        rect = mpatches.FancyBboxPatch(
            (col + 0.06, ROWS - row - 1 + 0.06), 0.82, 0.82,
            boxstyle="round,pad=0.06",
            facecolor=SEVERITY_COLORS[cat],
            edgecolor=PANEL, linewidth=2
        )
        ax.add_patch(rect)

    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.axis("off")

    patches = [mpatches.Patch(facecolor=SEVERITY_COLORS[k],
                               label=f"{k}  ({sev_counts[k]}  –  {sev_counts[k]/n*100:.1f}%)")
               for k in sev_counts]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.09), ncol=2,
              fontsize=10, framealpha=0.15, labelcolor=TEXT, facecolor=PANEL)
    fig.suptitle(f"N = {n} records  ·  Each square ≈ 1 %",
                 color=SUBTEXT, fontsize=10, y=0.01)
    save(fig, "3_waffle.png")

plot_waffle()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TREEMAP — subscale share of total symptom burden
# ═══════════════════════════════════════════════════════════════════════════════
def plot_treemap():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_title("Treemap — Subscale Share of Total Symptom Burden",
                 color=TEXT, fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")

    burden = np.maximum(means, 0.001)
    props  = burden / burden.sum()

    # Simple row-based squarify
    remaining = list(zip(props, LABELS,
                         [PALETTE[i % len(PALETTE)] for i in range(len(LABELS))]))
    cx, cy, cw, ch = 0.0, 0.0, 1.0, 1.0

    while remaining:
        row_props, row, row_sum = [], [], 0.0
        for i, (p, l, c) in enumerate(remaining):
            new_sum = row_sum + p
            if row:
                row_h = new_sum * ch / (cw + 1e-9)
                ar_new = max((p / new_sum * cw) / (row_h + 1e-9),
                             row_h / max(p / new_sum * cw, 1e-9))
                row_h_old = row_sum * ch / (cw + 1e-9)
                ar_old = max((row_props[-1] / row_sum * cw) / (row_h_old + 1e-9),
                             row_h_old / max(row_props[-1] / row_sum * cw, 1e-9))
                if ar_new > ar_old:
                    break
            row_props.append(p)
            row.append((p, l, c))
            row_sum = new_sum

        remaining = remaining[len(row):]
        row_h = row_sum * ch / (cw + 1e-9)
        rx = cx
        for (p, l, c) in row:
            rw = p / row_sum * cw if row_sum else 0
            rect = mpatches.FancyBboxPatch(
                (rx + 0.004, cy + 0.004), rw - 0.008, row_h - 0.008,
                boxstyle="round,pad=0.003",
                facecolor=c, edgecolor=PANEL, linewidth=2,
                transform=ax.transAxes, clip_on=True
            )
            ax.add_patch(rect)
            if rw > 0.035 and row_h > 0.04:
                fs = max(7, min(12, rw * 65))
                ax.text(rx + rw / 2, cy + row_h / 2,
                        f"{l}\n{p*100:.1f}%",
                        ha="center", va="center", fontsize=fs,
                        color=TEXT, fontweight="bold",
                        transform=ax.transAxes,
                        path_effects=[pe.withStroke(linewidth=2, foreground=PANEL)])
            rx += rw
        cy += row_h
        ch -= row_h

    save(fig, "4_treemap.png")

plot_treemap()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BUMP / RANK CHART — subscale rank across severity groups
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bump():
    groups     = ["Mild / None","Moderate","Severe","Very Severe"]
    group_recs = {g: [r for r in records if severity_label(r["total_score"]) == g]
                  for g in groups}

    rank_matrix = np.full((len(SUBSCALES), len(groups)), np.nan)
    for gi, g in enumerate(groups):
        grecs = group_recs[g]
        if grecs:
            gmeans = [np.mean([r.get(s, 0) for r in grecs]) for s in SUBSCALES]
            order  = np.argsort(gmeans)[::-1]
            for rank, si in enumerate(order):
                rank_matrix[si, gi] = rank + 1

    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
    base_style(ax, "Bump/Rank Chart — Subscale Importance Rank by Severity Group",
               grid_axis=None)

    x_ticks = np.arange(len(groups))
    for si in range(len(SUBSCALES)):
        ranks = rank_matrix[si]
        valid = ~np.isnan(ranks)
        if valid.sum() < 2:
            continue
        colour = PALETTE[si % len(PALETTE)]
        ax.plot(x_ticks[valid], ranks[valid], "-o",
                color=colour, linewidth=1.8, markersize=8,
                markeredgecolor=BG, markeredgewidth=0.6, alpha=0.9)
        last = np.where(valid)[0][-1]
        ax.text(x_ticks[last] + 0.08, ranks[last],
                LABELS[si], fontsize=8.5, color=colour, va="center")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["Mild / None", "Moderate", "Severe", "Very Severe"],
                       color=TEXT, fontsize=10)
    ax.invert_yaxis()
    ax.set_ylabel("Subscale Rank  (1 = highest mean score)", fontsize=11)
    ax.set_xlim(-0.3, len(groups) + 1.5)
    ax.set_ylim(len(SUBSCALES) + 0.5, 0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=SUBTEXT)
    save(fig, "5_bump_rank.png")

plot_bump()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DIVERGING BAR — subscale mean vs. overall mean
# ═══════════════════════════════════════════════════════════════════════════════
def plot_diverging():
    overall = means.mean()
    deltas  = means - overall
    idx     = np.argsort(deltas)
    slabels = [LABELS[i] for i in idx]
    sdeltas = deltas[idx]
    colors  = [PALETTE[0] if d >= 0 else PALETTE[2] for d in sdeltas]

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    base_style(ax, "Diverging Bar — Subscale Mean vs. Overall Mean Score", grid_axis="x")
    ax.grid(axis="y", visible=False)

    y = np.arange(len(SUBSCALES))
    ax.barh(y, sdeltas, color=colors, edgecolor=BG, height=0.65, alpha=0.87)
    ax.axvline(0, color=SUBTEXT, linewidth=1.5)

    ax.set_yticks(y)
    ax.set_yticklabels(slabels, color=TEXT, fontsize=10)
    ax.set_xlabel(f"Δ  from overall subscale mean  ({overall:.3f})", fontsize=11)

    for bar_y, d in zip(y, sdeltas):
        ha  = "left"  if d >= 0 else "right"
        pad = 0.002   if d >= 0 else -0.002
        ax.text(d + pad, bar_y, f"{d:+.3f}",
                va="center", ha=ha, fontsize=8, color=TEXT)

    pos_p = mpatches.Patch(color=PALETTE[0], label="Above overall mean")
    neg_p = mpatches.Patch(color=PALETTE[2], label="Below overall mean")
    ax.legend(handles=[pos_p, neg_p], fontsize=10,
              framealpha=0.15, labelcolor=TEXT, facecolor=PANEL)
    fig.suptitle(f"Overall mean across all subscales = {overall:.3f}",
                 color=SUBTEXT, fontsize=10, y=0.01)
    save(fig, "6_diverging_bar.png")

plot_diverging()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. POLAR AREA CHART — mean subscale scores
# ═══════════════════════════════════════════════════════════════════════════════
def plot_polar_area():
    fig, ax = plt.subplots(figsize=(11, 11), facecolor=BG,
                            subplot_kw=dict(projection="polar"))
    ax.set_facecolor(PANEL)
    ax.set_title("Polar Area Chart — Mean Score per Subscale",
                 color=TEXT, fontsize=14, fontweight="bold", pad=22)

    N      = len(SUBSCALES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width  = 2 * np.pi / N - 0.06

    bars = ax.bar(angles, means, width=width, bottom=0,
                  color=PALETTE[:N], alpha=0.78,
                  edgecolor=BG, linewidth=1)

    ax.set_xticks(angles)
    ax.set_xticklabels(LABELS, fontsize=8.5, color=TEXT)
    ax.set_yticklabels([])
    ax.spines["polar"].set_edgecolor(BORDER)
    ax.tick_params(colors=SUBTEXT)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Annotate max
    mi = np.argmax(means)
    ax.annotate(f"{LABELS[mi]}\n{means[mi]:.3f}",
                xy=(angles[mi], means[mi]),
                xytext=(angles[mi], means[mi] * 1.5),
                color=PALETTE[mi], fontsize=9, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=PALETTE[mi], lw=1))

    fig.suptitle(f"N = {n}  |  Max subscale: {LABELS[mi]} ({means[mi]:.3f})",
                 color=SUBTEXT, fontsize=10, y=0.01)
    save(fig, "7_polar_area.png")

plot_polar_area()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. HEXBIN DENSITY — insomnia vs depressed_mood
# ═══════════════════════════════════════════════════════════════════════════════
def plot_hexbin():
    rng   = np.random.default_rng(42)
    x_raw = np.array([r.get("insomnia",      0) for r in records], dtype=float)
    y_raw = np.array([r.get("depressed_mood", 0) for r in records], dtype=float)
    x_j   = x_raw + rng.uniform(-0.25, 0.25, n)
    y_j   = y_raw + rng.uniform(-0.25, 0.25, n)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    base_style(ax, "Hexbin Density — Insomnia vs Depressed Mood Scores", grid_axis=None)

    hb = ax.hexbin(x_j, y_j, gridsize=14, cmap="plasma",
                   linewidths=0.5, edgecolors=PANEL, mincnt=1)
    cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Record Count", color=SUBTEXT, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=SUBTEXT, labelcolor=SUBTEXT)

    ax.set_xlabel("Insomnia Score  (jittered ± 0.25)", fontsize=11)
    ax.set_ylabel("Depressed Mood Score  (jittered ± 0.25)", fontsize=11)

    corr = np.corrcoef(x_raw, y_raw)[0, 1]
    ax.text(0.04, 0.96, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, color=TEXT, fontsize=10,
            va="top", bbox=dict(facecolor=PANEL, edgecolor=BORDER, pad=4))
    save(fig, "8_hexbin.png")

plot_hexbin()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. STEP ECDF — empirical CDF of total scores
# ═══════════════════════════════════════════════════════════════════════════════
def plot_ecdf():
    sorted_t = np.sort(totals)
    ecdf_y   = np.arange(1, n + 1) / n * 100

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    base_style(ax, "Step ECDF — Empirical Cumulative Distribution of Total Scores")

    ax.step(sorted_t, ecdf_y, where="post",
            color=PALETTE[0], linewidth=2.5, label="Empirical CDF")
    ax.fill_between(sorted_t, ecdf_y, step="post",
                    alpha=0.13, color=PALETTE[0])

    thresholds = [(17, PALETTE[3], "Moderate ≥ 17"),
                  (25, PALETTE[4], "Severe ≥ 25"),
                  (30, PALETTE[2], "Very Severe ≥ 30")]
    for thresh, col, lbl in thresholds:
        ax.axvline(thresh, color=col, linewidth=1.2, linestyle=":",
                   label=f"{lbl}  ({(totals >= thresh).sum()} records)")
        ax.text(thresh + 0.3, 6, lbl, color=col, fontsize=8.5, va="bottom")

    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_xlabel("Total HAM-A Score", fontsize=11)
    ax.set_ylabel("Cumulative %", fontsize=11)
    ax.set_ylim(0, 107)
    ax.set_xlim(sorted_t.min() - 0.5)
    ax.legend(fontsize=9, framealpha=0.15, labelcolor=TEXT, facecolor=PANEL)
    fig.suptitle(f"N = {n}  |  Mean = {totals.mean():.2f}  |  Median = {int(np.median(totals))}",
                 color=SUBTEXT, fontsize=10, y=0.01)
    save(fig, "9_step_ecdf.png")

plot_ecdf()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. BUBBLE CHART — mean vs std, bubble size = % non-zero
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bubble():
    nonzero = np.array([(sum(1 for r in records if r.get(s, 0) > 0)) / n * 100
                        for s in SUBSCALES])
    sizes   = (nonzero / nonzero.max() * 1200) + 40

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    base_style(ax, "Bubble Chart — Mean vs Std Dev per Subscale\n"
                    "(bubble size = % of records with non-zero score)")

    sc = ax.scatter(means, stds, s=sizes,
                    c=PALETTE[:len(SUBSCALES)],
                    alpha=0.82, edgecolors=BG, linewidths=0.9, zorder=3)

    for i, lbl in enumerate(LABELS):
        ax.annotate(lbl, (means[i], stds[i]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8.5, color=TEXT, alpha=0.9)

    ax.set_xlabel("Mean Score", fontsize=11)
    ax.set_ylabel("Std Dev", fontsize=11)

    for pct in [10, 30, 60]:
        sz = (pct / nonzero.max() * 1200) + 40
        ax.scatter([], [], s=sz, color=SUBTEXT, alpha=0.55,
                   label=f"{pct}% non-zero", edgecolors=BG)
    ax.legend(title="Bubble size", title_fontsize=9,
              fontsize=9, framealpha=0.15, labelcolor=TEXT, facecolor=PANEL)
    fig.suptitle(f"N = {n}  |  Each bubble = one HAM-A subscale",
                 color=SUBTEXT, fontsize=10, y=0.01)
    save(fig, "10_bubble.png")

plot_bubble()


print(f"\nAll 10 graphs saved to: {OUT_DIR}")
