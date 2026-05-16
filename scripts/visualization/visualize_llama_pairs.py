import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "all_combo-pair" / "all_conversation_pairs_scored_llama.json"
OUTPUT_DIR = BASE_DIR / "analysis_output" / "llama_pairs_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HAMA_PARAMS = [
    "anxious_mood", "tension", "fears", "insomnia", "intellectual",
    "depressed_mood", "somatic_muscular", "somatic_sensory",
    "cardiovascular", "respiratory", "gastrointestinal",
    "genitourinary", "autonomic", "behavior_at_interview"
]
PARAM_LABELS = [p.replace("_", " ").title() for p in HAMA_PARAMS]

def main():
    print("Loading data...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records.")
    
    sns.set_theme(style="darkgrid")
    
    # 1. Total Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="total_score", bins=30, kde=True, color="#6366f1")
    plt.title("Total HAM-A Score Distribution (Llama 3.1)", fontsize=16, fontweight="bold")
    plt.xlabel("Total Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_total_score_distribution.png", dpi=200)
    plt.close()
    print("Saved 1_total_score_distribution.png")

    # 2. Box Plot of Parameters
    melted = df[HAMA_PARAMS].melt(var_name="Parameter", value_name="Score")
    melted["Parameter"] = melted["Parameter"].str.replace("_", " ").str.title()
    
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=melted, x="Parameter", y="Score", palette="viridis")
    plt.title("Score Distributions per HAM-A Parameter", fontsize=18, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("")
    plt.ylabel("Score (0-4)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_parameter_boxplots.png", dpi=200)
    plt.close()
    print("Saved 2_parameter_boxplots.png")

    # 3. Non-Zero Percentages
    non_zero_pct = [(df[p] > 0).mean() * 100 for p in HAMA_PARAMS]
    plt.figure(figsize=(14, 7))
    sns.barplot(x=PARAM_LABELS, y=non_zero_pct, palette="magma")
    plt.title("Percentage of Non-Zero Scores per Parameter", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    
    for i, v in enumerate(non_zero_pct):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_nonzero_percentages.png", dpi=200)
    plt.close()
    print("Saved 3_nonzero_percentages.png")
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr = df[HAMA_PARAMS].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, 
                xticklabels=PARAM_LABELS, yticklabels=PARAM_LABELS)
    plt.title("Correlation Matrix of HAM-A Parameters", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_correlation_heatmap.png", dpi=200)
    plt.close()
    print("Saved 4_correlation_heatmap.png")
    
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
