import json
import pandas as pd
import numpy as np
import os

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

# Define some weighting schemes to see how the distribution changes.
# The user mentioned: "u can give zero to anyone the data present"
weight_schemes = {
    'Equal_Weights (33% each)': {'Qwen': 1/3, 'Llama': 1/3, 'Mistral': 1/3},
    'Qwen_Llama_Only (50% Q, 50% L, 0% M)': {'Qwen': 0.5, 'Llama': 0.5, 'Mistral': 0.0},
    'Qwen_Mistral_Only (50% Q, 0% L, 50% M)': {'Qwen': 0.5, 'Llama': 0.0, 'Mistral': 0.5},
    'Llama_Mistral_Only (0% Q, 50% L, 50% M)': {'Qwen': 0.0, 'Llama': 0.5, 'Mistral': 0.5},
    'Heavy_Qwen (60% Q, 20% L, 20% M)': {'Qwen': 0.6, 'Llama': 0.2, 'Mistral': 0.2},
    'Heavy_Llama (20% Q, 60% L, 20% M)': {'Qwen': 0.2, 'Llama': 0.6, 'Mistral': 0.2},
    'Heavy_Mistral (20% Q, 20% L, 60% M)': {'Qwen': 0.2, 'Llama': 0.2, 'Mistral': 0.6}
}

for name, weights in weight_schemes.items():
    df[name] = df['Qwen'] * weights['Qwen'] + df['Llama'] * weights['Llama'] + df['Mistral'] * weights['Mistral']

# Also calculate the Median (a robust unweighted measure)
df['Median_Score'] = df[['Qwen', 'Llama', 'Mistral']].median(axis=1)

with open('../docs/weighted_distribution_changes.txt', 'w', encoding='utf-8') as f:
    f.write("=== HAM-A Total Score Distribution Under Different Weights ===\n\n")
    
    f.write("1. ORIGINAL MODELS\n")
    f.write("-" * 50 + "\n")
    f.write(df[['Qwen', 'Llama', 'Mistral']].describe().to_string())
    f.write("\n\n")
    
    f.write("2. COMBINATIONS INVOLVING ZERO WEIGHTS (Excluding one model)\n")
    f.write("-" * 50 + "\n")
    cols_zero = [c for c in df.columns if 'Only' in c]
    f.write(df[cols_zero].describe().to_string())
    f.write("\n\n")
    
    f.write("3. EQUAL AND HEAVY WEIGHTING COMBINATIONS\n")
    f.write("-" * 50 + "\n")
    cols_heavy = ['Equal_Weights (33% each)'] + [c for c in df.columns if 'Heavy' in c]
    f.write(df[cols_heavy].describe().to_string())
    f.write("\n\n")
    
    f.write("4. MEDIAN SCORE (Robust ensemble method)\n")
    f.write("-" * 50 + "\n")
    f.write(df[['Median_Score']].describe().to_string())
    f.write("\n")

print("Analysis complete. Saved to ../docs/weighted_distribution_changes.txt")
