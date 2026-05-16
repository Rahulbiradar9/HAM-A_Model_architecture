import torch
import numpy as np
from transformers import AutoTokenizer
from model import HAMARegressor
import argparse

HAMA_PARAMS = [
    "anxious_mood", "tension", "fears", "insomnia", "intellectual",
    "depressed_mood", "somatic_muscular", "somatic_sensory",
    "cardiovascular", "respiratory", "gastrointestinal",
    "genitourinary", "autonomic", "behavior_at_interview"
]

class HAMAPredictor:
    def __init__(self, model_path="checkpoints/longformer/best_model.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = HAMARegressor().to(self.device).float()

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print(f"Successfully loaded model weights from {model_path}")
            if missing:
                print(f"Note: {len(missing)} keys were missing (expected if head was not saved).")
        except FileNotFoundError:
            print(f"Warning: {model_path} not found. Model will output random predictions (untrained).")

    def predict(self, transcript_text):
        """
        Takes purely isolated input string and extracts HAM-A predictions.
        """
        encoding = self.tokenizer(
            transcript_text,
            add_special_tokens=True,
            truncation=True,
            max_length=4096,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            
        predictions = outputs.cpu().numpy()[0]
        
        predictions = np.clip(predictions, 0, 4)
        predictions = np.round(predictions).astype(int)
        
        result = {}
        total_score = 0
        for i, param in enumerate(HAMA_PARAMS):
            result[param] = int(predictions[i])
            total_score += int(predictions[i])
            
        result["total_score"] = int(total_score)
        
        return result

if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser(description="Run inference on explicit raw text or direct JSON transcript structures")
    parser.add_argument("--text", type=str, help="Patient raw transcript string to analyze")
    parser.add_argument("--file", type=str, help="Path to raw JSON transcript array inherently isolating Participant arrays")
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        print("Error: You must definitively provide either --text physically or point to an explicit --file securely!")
        exit(1)
        
    predictor = HAMAPredictor()
    
    if args.file:
        print(f"Loading transcript: {args.file}")
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            extracted = []
            
            # Auto-detect format
            # Format A (training data): top-level list [{"speaker": ..., "value": ...}]
            if isinstance(data, list):
                for turn in data:
                    speaker = turn.get("speaker", "").lower().strip()
                    if "participant" in speaker or "patient" in speaker:
                        extracted.append(turn.get("value", ""))
            
            # Format B (new test files): {"conversation": [{"role": ..., "text": ...}]}
            elif isinstance(data, dict):
                turns = data.get("conversation", data.get("turns", []))
                for turn in turns:
                    role = turn.get("role", turn.get("speaker", "")).lower().strip()
                    if "patient" in role or "participant" in role:
                        text = turn.get("text", turn.get("value", ""))
                        extracted.append(text)
            
            if not extracted:
                print("Warning: No patient/participant speech found. Using full transcript.")
                # Fallback: extract all text
                if isinstance(data, list):
                    extracted = [t.get("value", t.get("text", "")) for t in data]
                else:
                    turns = data.get("conversation", [])
                    extracted = [t.get("text", "") for t in turns]
            
            raw_text = " ".join(extracted).strip()
            print(f"Extracted {len(extracted)} patient turns. Running inference...")
            scores = predictor.predict(raw_text)
            
        except Exception as e:
            print(f"Error loading transcript file: {e}")
            exit(1)
            
    else:
        scores = predictor.predict(args.text)
    
    print("\n--- Predicted HAM-A Scores ---")
    for key, val in scores.items():
        if key != "total_score":
            print(f"{key.replace('_', ' ').title():<25}: {val}")
    print("-" * 30)
    print(f"{'Total Score':<25}: {scores['total_score']}")
