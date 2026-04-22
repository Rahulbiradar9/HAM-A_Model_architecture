import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    file_path = os.path.join("after_scoring", "_batch_hama_scores_weighted_40_60.json")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    SUBSCALES = [
        "anxious_mood", "tension", "fears", "insomnia", "intellectual",
        "depressed_mood", "somatic_muscular", "somatic_sensory",
        "cardiovascular", "respiratory", "gastrointestinal",
        "genitourinary", "autonomic", "behavior_at_interview",
    ]

    # Calculate detections (any score > 0)
    detections = {subscale: 0 for subscale in SUBSCALES}
    total_records = len(data)

    for record in data:
        for subscale in SUBSCALES:
            # Check if the score is greater than 0
            if record.get(subscale, 0) > 0:
                detections[subscale] += 1

    # Calculate percentages
    percentages = {k: (v / total_records) * 100 for k, v in detections.items()}

    # Sort parameters by highest detection to lowest
    sorted_params = sorted(percentages.items(), key=lambda item: item[1], reverse=True)
    labels = [x[0].replace("_", " ").title() for x in sorted_params]
    values = [x[1] for x in sorted_params]

    # Create output directory
    out_dir = os.path.join("after_scoring", "visuals")
    os.makedirs(out_dir, exist_ok=True)

    # Setup the plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=values, y=labels, palette="magma")
    
    plt.title(f"Parameter Detection Rates (Llama 40%, Mistral 60%) - N={total_records}", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Detection Percentage (%)", fontsize=12)
    plt.ylabel("HAM-A Subscales", fontsize=12)
    plt.xlim(0, 100)

    # Add numeric labels to the end of each bar
    for index, value in enumerate(values):
        ax.text(value + 0.8, index, f'{value:.1f}%', va='center', fontsize=11, fontweight='bold')

    # Improve layout
    plt.tight_layout()

    # Save to file
    out_file = os.path.join(out_dir, "parameter_detection_40_60.png")
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Graph successfully generated and saved to: {out_file}")

if __name__ == "__main__":
    main()
