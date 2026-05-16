import pandas as pd

# Load the previously calculated grid
df = pd.read_csv('../data/weight_combinations_grid.csv')

# Define a scoring function to rank "Best to Worst"
# Criteria for "Best":
# 1. Low Standard Deviation (Stability)
# 2. Mean close to the "Robust Median" of all 3 models (~8.52)
# 3. Penalty for Max scores > 41 (Llama's extreme outlier values)

def calculate_penalty(row):
    # 1. Variance Penalty (directly use standard deviation)
    std_penalty = row['Std_Dev']
    
    # 2. Mean Deviation Penalty (how far from the optimal middle-ground 8.52)
    mean_penalty = abs(row['Mean'] - 8.52) * 2 # Weight this slightly higher
    
    # 3. Outlier Penalty
    outlier_penalty = max(0, row['Max'] - 41) * 0.5 # Penalize max scores over 41
    
    total_penalty = std_penalty + mean_penalty + outlier_penalty
    return total_penalty

df['Penalty_Score'] = df.apply(calculate_penalty, axis=1)

# Sort from Lowest Penalty (Best) to Highest Penalty (Worst)
df_ranked = df.sort_values(by='Penalty_Score', ascending=True)

# Add a Rank column
df_ranked.insert(0, 'Rank', range(1, len(df_ranked) + 1))

# Format the output for Markdown
df_ranked_rounded = df_ranked.round(2)
headers = df_ranked_rounded.columns.tolist()

with open('../docs/ranked_weight_combinations.md', 'w', encoding='utf-8') as f:
    f.write("# Ranked Weight Combinations (Best to Worst)\n\n")
    f.write("We ranked the 66 combinations based on a 'Penalty Score' (lower is better).\n")
    f.write("The criteria for a 'Best' score are:\n")
    f.write("1. **Stability**: Low Standard Deviation.\n")
    f.write("2. **Balanced Mean**: Mean score closest to the robust median of ~8.52.\n")
    f.write("3. **Outlier Control**: Heavy penalty for combinations with a Max score > 41 (mitigating Llama's hallucinations).\n\n")
    
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|" + "|".join(["---" for _ in headers]) + "|"
    f.write(header_line + "\n")
    f.write(separator_line + "\n")
    
    for _, row in df_ranked_rounded.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
        f.write(row_str + "\n")

print("Ranking complete. Saved to ../docs/ranked_weight_combinations.md")
