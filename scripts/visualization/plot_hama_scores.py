import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

data_path = 'M_chat16k/m_chat16k_combined_scored.json'
print(f"Loading data from {data_path}...")
with open(data_path, 'r') as f:
    data = json.load(f)

# Extract scores
records = []
for item in data:
    if 'score' in item and isinstance(item['score'], dict):
        scores = item['score']
        # Extract total score
        total_score = scores.get('total_score', None)
        if total_score is None:
            total_score = sum([v for k, v in scores.items() if k != 'total_score' and isinstance(v, (int, float))])
        
        record = scores.copy()
        record['total_score'] = total_score
        records.append(record)

df = pd.DataFrame(records)

# Create an output directory for plots
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

print("Generating Distribution of HAM-A Total Scores plot...")
# 1. Histogram of Total Scores
plt.figure(figsize=(10, 6))
sns.histplot(df['total_score'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of HAM-A Total Scores')
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig(f'{output_dir}/total_score_distribution.png')
plt.close()

print("Generating Average HAM-A Sub-Scores plot...")
# 2. Average of each sub-score
sub_score_columns = [col for col in df.columns if col != 'total_score']
avg_sub_scores = df[sub_score_columns].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=avg_sub_scores.values, y=avg_sub_scores.index, palette='viridis')
plt.title('Average HAM-A Sub-Scores')
plt.xlabel('Average Score')
plt.ylabel('Symptom')
plt.grid(axis='x', alpha=0.75)
plt.tight_layout()
plt.savefig(f'{output_dir}/average_sub_scores.png')
plt.close()

print(f"Plots saved successfully to the '{output_dir}/' directory.")
