import os
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

CATEGORIES = [
    "anxious_mood", "tension", "fears", "insomnia", "intellectual",
    "depressed_mood", "somatic_muscular", "somatic_sensory",
    "cardiovascular", "respiratory", "gastrointestinal",
    "genitourinary", "autonomic", "behavior_at_interview"
]

def map_speaker(speaker):
    speaker = speaker.lower()
    if "participant" in speaker or "patient" in speaker or "user" in speaker:
        return "Patient"
    else:
        return "Counselor"

def get_transcript_chunks(transcript, tokenizer, max_tokens=450, overlap_turns=2):
    """
    Groups turns into text chunks that fit under max_tokens.
    Includes a sliding window overlap of `overlap_turns`.
    """
    chunks = []
    
    i = 0
    while i < len(transcript):
        current_chunk_turns = []
        current_text = ""
        j = i
        
        while j < len(transcript):
            turn = transcript[j]
            speaker = map_speaker(turn.get("speaker", "Unknown"))
            text = turn.get("value", "")
            formatted_turn = f"{speaker}: {text}"
            
            # Trial combination
            trial_turns = current_chunk_turns + [formatted_turn]
            trial_text = "\n".join(trial_turns)
            
            tokens = tokenizer.encode(trial_text, add_special_tokens=True)
            if len(tokens) > max_tokens:
                # If even the single turn is too long, we must take it and break
                if len(current_chunk_turns) == 0:
                    current_chunk_turns.append(formatted_turn)
                    j += 1
                break
            else:
                current_chunk_turns.append(formatted_turn)
                j += 1
                
        # Add the completed chunk
        if current_chunk_turns:
            chunks.append("\n".join(current_chunk_turns))
            
        # Move forward, creating an overlap. 
        # `j` is the index of the next turn to process.
        # We start the next chunk `overlap_turns` before `j`.
        next_i = j - overlap_turns
        
        # Ensure we always make forward progress
        if next_i <= i:
            next_i = i + 1
            
        i = next_i

    return chunks

def predict_chunk(text, model, tokenizer, device):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get predictions and clip to [0, 4] range
    predictions = outputs.logits.squeeze().cpu().numpy()
    predictions = np.clip(predictions, 0, 4)
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Run Mental-RoBERTa inference on a JSON transcript")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON transcript")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save output JSON")
    parser.add_argument("--model", "-m", type=str, default=r"d:\Rahul_Intern\convo_model\M_chat16k\mental_roberta_training\results\final_model", help="Path to fine-tuned model")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()
    
    print(f"Loading transcript from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
        
    chunks = get_transcript_chunks(transcript, tokenizer)
    print(f"Split transcript into {len(chunks)} chunks for processing.")
    
    all_predictions = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Scoring chunks")):
        preds = predict_chunk(chunk, model, tokenizer, device)
        all_predictions.append(preds)
        
    all_predictions = np.array(all_predictions)
    
    # Aggregate scores (Calculate Average and Max)
    avg_scores = np.mean(all_predictions, axis=0)
    max_scores = np.max(all_predictions, axis=0)
    
    # Format the output
    results = {
        "metadata": {
            "source_file": args.input,
            "num_chunks": len(chunks),
            "aggregation_note": "Scores are calculated across multiple chunks due to 512 token limit."
        },
        "scores_average": {},
        "scores_max": {},
        "chunk_scores": []
    }
    
    total_avg = 0
    total_max = 0
    for i, cat in enumerate(CATEGORIES):
        val_avg = float(avg_scores[i])
        val_max = float(max_scores[i])
        results["scores_average"][cat] = val_avg
        results["scores_max"][cat] = val_max
        total_avg += val_avg
        total_max += val_max
        
    results["scores_average"]["total_score"] = total_avg
    results["scores_max"]["total_score"] = total_max
    
    # Store individual chunk scores for detailed analysis
    for i, chunk_preds in enumerate(all_predictions):
        chunk_dict = {"chunk_index": i, "scores": {}}
        for j, cat in enumerate(CATEGORIES):
            chunk_dict["scores"][cat] = float(chunk_preds[j])
        results["chunk_scores"].append(chunk_dict)
    
    print("\n" + "="*50)
    print(f"RESULTS FOR: {os.path.basename(args.input)}")
    print("="*50)
    print(f"{'Category':<25} | {'Average':<8} | {'Max':<8}")
    print("-" * 50)
    for cat in CATEGORIES:
        print(f"{cat:<25} | {results['scores_average'][cat]:.2f}     | {results['scores_max'][cat]:.2f}")
    print("-" * 50)
    print(f"{'TOTAL SCORE':<25} | {total_avg:.2f}     | {total_max:.2f}")
    print("="*50)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
