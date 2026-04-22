import json
import os

def main():
    folder = "after_scoring"
    files = {
        "mistral": os.path.join(folder, "_batch_hama_scores_mistarl_7b.json"),
        "llama": os.path.join(folder, "_batch_hama_scores_llama3.json")
    }

    weights = {
        "mistral": 0.60,
        "llama": 0.40
    }

    # Load all datasets
    data = {}
    print("Loading score files...")
    for model_name, path in files.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found!")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data[model_name] = json.load(f)
            print(f"  {model_name}: {len(data[model_name])} records")

    # Group records by filename
    by_filename = {}
    for model_name, records in data.items():
        for r in records:
            fname = r.get("filename")
            if not fname:
                continue
            if fname not in by_filename:
                by_filename[fname] = {}
            by_filename[fname][model_name] = r

    SUBSCALES = [
        "anxious_mood", "tension", "fears", "insomnia", "intellectual",
        "depressed_mood", "somatic_muscular", "somatic_sensory",
        "cardiovascular", "respiratory", "gastrointestinal",
        "genitourinary", "autonomic", "behavior_at_interview"
    ]

    results = []

    for fname, model_scores in by_filename.items():
        # Get metadata from the first available model
        first_model = list(model_scores.keys())[0]
        merged = {
            "filename": fname,
            "word_count": model_scores[first_model].get("word_count", 0),
            "was_truncated": model_scores[first_model].get("was_truncated", False)
        }
        
        # Calculate available weights in case one model failed on a given transcript
        available_weight = sum([weights[m] for m in model_scores.keys()])
        
        total_score = 0.0
        for subscale in SUBSCALES:
            weighted_val = 0.0
            for m in model_scores.keys():
                val = model_scores[m].get(subscale, 0)
                weighted_val += val * (weights[m] / available_weight)
            
            # Cast to integer for standard HAM-A scoring
            merged[subscale] = int(round(weighted_val))
            total_score += merged[subscale]
        
        merged["total_score"] = int(total_score)
        results.append(merged)

    # Sort results to be clean and ordered
    results.sort(key=lambda x: x["filename"])

    output_path = os.path.join(folder, "_batch_hama_scores_weighted_40_60.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nSuccessfully wrote {len(results)} merged records to: {output_path}")

if __name__ == "__main__":
    main()
