import json
import pandas as pd
import numpy as np
import os
import itertools

# Define the file paths
files = {
    'Qwen': '../data/m_chat16k_combined_scored.json',
    'Llama': '../data/m_chat16k_combined_scored_llama.json',
    'Mistral': '../data/m_chat16k_combined_scored_mistral.json'
}

data_dict = {}
for model_name, filename in files.items():
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            records = json.load(f)
            scores = {}
            for r in records:
                if 'score' in r and 'total_score' in r['score']:
                    scores[r['id']] = r['score']['total_score']
            data_dict[model_name] = scores

df = pd.DataFrame(data_dict)

results = []

# Generate all weight combinations in steps of 0.1
weights = [round(x * 0.1, 1) for x in range(11)]
for w_q in weights:
    for w_l in weights:
        w_m = round(1.0 - w_q - w_l, 1)
        if w_m >= 0.0 and w_m <= 1.0:
            # We found a valid combination
            series = df['Qwen'] * w_q + df['Llama'] * w_l + df['Mistral'] * w_m
            
            desc = series.describe()
            results.append({
                'W_Qwen': w_q,
                'W_Llama': w_l,
                'W_Mistral': w_m,
                'Mean': desc['mean'],
                'Std_Dev': desc['std'],
                'Min': desc['min'],
                '25%': desc['25%'],
                'Median': desc['50%'],
                '75%': desc['75%'],
                'Max': desc['max']
            })

df_results = pd.DataFrame(results)

# Sort the results by standard deviation (from lowest to highest) to see the most stable ones,
# or by mean to see the range. Let's sort by Mean descending.
df_results_sorted = df_results.sort_values(by='Mean', ascending=False)

# Save to CSV and Text
df_results_sorted.to_csv('../data/weight_combinations_grid.csv', index=False)

# Also create a nicely formatted markdown table for the top, middle, and bottom combinations
with open('../docs/different_weight_combinations.md', 'w', encoding='utf-8') as f:
    f.write("# Grid Search of All Weight Combinations (Increments of 0.1)\n\n")
    f.write("We tested all 66 combinations of weights in 10% increments that sum to 100%.\n")
    f.write("The table below shows the resulting distribution for each combination, ordered by the resulting Mean score.\n\n")
    
    # Format for markdown manually
    df_results_rounded = df_results_sorted.round(2)
    headers = df_results_rounded.columns.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|" + "|".join(["---" for _ in headers]) + "|"
    f.write(header_line + "\n")
    f.write(separator_line + "\n")
    
    for _, row in df_results_rounded.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
        f.write(row_str + "\n")

print("Completed grid search. Results saved to ../docs/different_weight_combinations.md and ../data/weight_combinations_grid.csv")
