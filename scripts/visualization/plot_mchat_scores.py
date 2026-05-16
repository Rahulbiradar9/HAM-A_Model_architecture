import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style
sns.set_theme(style="whitegrid")

# Load the JSON file
file_path = r"d:\Rahul_Intern\convo_model\M_chat16k\m_chat16k_combined_scored.json"
print(f"Loading {file_path}...")
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} records.")

# Extract scores
total_scores = []
category_scores = {
    "anxious_mood": [], "tension": [], "fears": [], "insomnia": [],
    "intellectual": [], "depressed_mood": [], "somatic_muscular": [],
    "somatic_sensory": [], "cardiovascular": [], "respiratory": [],
    "gastrointestinal": [], "genitourinary": [], "autonomic": [],
    "behavior_at_interview": []
}

for item in data:
    score_dict = item.get("score", {})
    if "total_score" in score_dict:
        total_scores.append(score_dict["total_score"])
    
    for cat in category_scores.keys():
        if cat in score_dict:
            category_scores[cat].append(score_dict[cat])

# 1. Plot Histogram of Total Scores
plt.figure(figsize=(10, 6))
sns.histplot(total_scores, bins=range(0, 57, 2), kde=True, color="skyblue")
plt.title("Distribution of HAM-A Total Scores", fontsize=16)
plt.xlabel("Total Score", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.tight_layout()
plt.savefig(r"d:\Rahul_Intern\convo_model\M_chat16k\total_score_distribution.png", dpi=300)
plt.close()

# 2. Plot Bar Chart of Average Category Scores
plt.figure(figsize=(12, 8))
categories = list(category_scores.keys())
means = [np.mean(category_scores[cat]) if category_scores[cat] else 0 for cat in categories]

sns.barplot(x=means, y=categories, palette="viridis")
plt.title("Average Score per HAM-A Category", fontsize=16)
plt.xlabel("Average Score", fontsize=14)
plt.ylabel("Category", fontsize=14)
plt.tight_layout()
plt.savefig(r"d:\Rahul_Intern\convo_model\M_chat16k\average_category_scores.png", dpi=300)
plt.close()

print("Plots generated successfully!")
