import os
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the fine-tuned model and tokenizer
MODEL_DIR = r"d:\Rahul_Intern\convo_model\M_chat16k\mental_roberta_training\results\final_model"

print(f"Loading model from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # Set model to evaluation mode

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

CATEGORIES = [
    "anxious_mood", "tension", "fears", "insomnia", "intellectual",
    "depressed_mood", "somatic_muscular", "somatic_sensory",
    "cardiovascular", "respiratory", "gastrointestinal",
    "genitourinary", "autonomic", "behavior_at_interview"
]

SEVERITY_LABELS = {
    0: "not_present",
    1: "mild",
    2: "moderate",
    3: "severe",
    4: "very_severe"
}

def get_final_severity(total_score):
    if total_score <= 17:
        return "mild"
    elif total_score <= 24:
        return "mild_moderate"
    elif total_score <= 30:
        return "moderate_severe"
    else:
        return "severe"

def predict_ham_a(input_text, output_text=""):
    """
    Takes a patient conversation, runs it through the fine-tuned RoBERTa model,
    and returns the exact HAM-A clinical JSON format.
    """
    # Format the input exactly as it was during training
    text = f"Patient: {input_text}\nCounselor: {output_text}"
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # The model outputs raw regression values for each of the 14 categories
    predictions = outputs.logits.squeeze().cpu().numpy()
    
    # Clip between 0 and 4, then round to nearest integer
    predictions = np.clip(predictions, 0, 4)
    rounded_preds = np.round(predictions).astype(int)
    
    result_json = {}
    total_score = 0
    
    for i, category in enumerate(CATEGORIES):
        score = rounded_preds[i]
        total_score += score
        result_json[category] = SEVERITY_LABELS[score]
        
    result_json["total_ham_a_score"] = int(total_score)
    result_json["final_severity"] = get_final_severity(total_score)
    
    return json.dumps(result_json, indent=2)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("READY FOR INFERENCE")
    print("="*50 + "\n")
    
    # Sample Test Case 1: High Anxiety
    sample_patient_1 = "I can't sleep at all. My heart races every time I try to lie down. I'm constantly terrified that something awful is going to happen to my family and I just sit there trembling and sweating."
    sample_counselor_1 = "It sounds like you are experiencing an overwhelming amount of panic and physical tension."
    
    # Sample Test Case 2: No Anxiety
    sample_patient_2 = "I've been feeling pretty good lately. Work is going well and I've been sleeping nicely."
    sample_counselor_2 = "That's wonderful to hear that you're in a stable place right now."

    print("--- Test Case 1 (High Anxiety Expected) ---")
    print(predict_ham_a(sample_patient_1, sample_counselor_1))
    print("\n")
    
    print("--- Test Case 2 (Low Anxiety Expected) ---")
    print(predict_ham_a(sample_patient_2, sample_counselor_2))
    print("\n")
