import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the file
filename = '../data/m_chat16k_combined_scored_weighted_rank1.json'
with open(filename, 'r', encoding='utf-8') as f:
    records = json.load(f)

# Extract scores
data = []
for r in records:
    if 'score' in r:
        score = r['score']
        if 'total_score' in score:
            data.append(score)

df = pd.DataFrame(data)

# 1. Total Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='total_score', kde=True, bins=20, color='royalblue')
plt.title('HAM-A Total Score Distribution (Rank 1 Weighted Combination)')
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.savefig('../visualizations/rank1_total_score_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Average Category Scores
category_cols = [c for c in df.columns if c != 'total_score']
mean_scores = df[category_cols].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=mean_scores.index, y=mean_scores.values, palette='viridis')
plt.title('Average HAM-A Category Scores (Rank 1 Weighted Combination)')
plt.xlabel('Category')
plt.ylabel('Average Score')
plt.xticks(rotation=45, ha='right')
plt.savefig('../visualizations/rank1_category_means.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graphs generated successfully.")
