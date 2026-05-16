import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def load_all_scores(filepath, model_name):
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for item in data:
        if 'score' in item:
            record = item['score'].copy()
            record['Model'] = model_name
            records.append(record)
    return records

def main():
    base_dir = r"d:\Rahul_Intern\convo_model\M_chat16k"
    file1 = os.path.join(base_dir, "m_chat16k_combined_scored.json")
    file2 = os.path.join(base_dir, "m_chat16k_combined_scored_llama.json")
    
    records_qwen = load_all_scores(file1, 'Qwen')
    records_llama = load_all_scores(file2, 'LLaMA')
    
    df = pd.DataFrame(records_qwen + records_llama)
    
    # Exclude total_score to just compare the sub-categories
    categories = [c for c in df.columns if c not in ['total_score', 'Model']]
    
    # Melt the dataframe for seaborn
    df_melted = df.melt(id_vars=['Model'], value_vars=categories, var_name='Category', value_name='Score')
    
    # 1. Bar plot of means
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_melted, x='Category', y='Score', hue='Model', ci=None)
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Score per Category: Qwen vs LLaMA')
    plt.tight_layout()
    bar_output = os.path.join(base_dir, "category_means_comparison.png")
    plt.savefig(bar_output)
    print(f"Mean bar chart saved to {bar_output}")
    
    # 2. Boxplot of distributions
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df_melted, x='Category', y='Score', hue='Model')
    plt.xticks(rotation=45, ha='right')
    plt.title('Score Distribution per Category: Qwen vs LLaMA')
    plt.tight_layout()
    box_output = os.path.join(base_dir, "category_boxplot_comparison.png")
    plt.savefig(box_output)
    print(f"Boxplot saved to {box_output}")

if __name__ == "__main__":
    main()
