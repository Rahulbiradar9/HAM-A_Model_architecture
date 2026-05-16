import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Define the file paths
files = {
    'Qwen': '../data/m_chat16k_combined_scored.json',
    'Llama': '../data/m_chat16k_combined_scored_llama.json',
    'Mistral': '../data/m_chat16k_combined_scored_mistral.json'
}

data = []

for model_name, filename in files.items():
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            records = json.load(f)
            for record in records:
                if 'score' in record:
                    score = record['score']
                    if 'total_score' in score:
                        row = {'Model': model_name, 'Total Score': score['total_score']}
                        for k, v in score.items():
                            if k != 'total_score':
                                row[k] = v
                        data.append(row)
    else:
        print(f"File {filename} not found.")

df = pd.DataFrame(data)

# 1. Total Score Distribution Comparison (Density Plot / Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Total Score', hue='Model', kde=True, bins=20, alpha=0.5, palette='Set2')
plt.title('HAM-A Total Score Distribution Comparison Across Models')
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.savefig('../visualizations/model_comparison_total_score_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Boxplot for Total Score Comparison
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Model', y='Total Score', palette='Set2')
plt.title('HAM-A Total Score Boxplot Comparison')
plt.savefig('../visualizations/model_comparison_total_score_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Category Score Means Comparison (Bar Plot)
category_cols = [c for c in df.columns if c not in ['Model', 'Total Score']]
df_melted = df.melt(id_vars=['Model'], value_vars=category_cols, var_name='Category', value_name='Score')

plt.figure(figsize=(14, 8))
sns.barplot(data=df_melted, x='Category', y='Score', hue='Model', errorbar=None, palette='Set2')
plt.title('Average HAM-A Category Scores Comparison Across Models')
plt.xlabel('Category')
plt.ylabel('Average Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model')
plt.savefig('../visualizations/model_comparison_category_means.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graphs generated successfully.")
