import torch
from torch.utils.data import DataLoader
from dataset import HAMADataset
from model import HAMARegressor
from utils import calculate_metrics
import numpy as np

def main():
    transcript_dir = "../json_transcripts"
    labels_path = "../after_scoring/_batch_hama_scores_weighted_60_40.json"
    
    # Must enforce checking specific explicitly generated paths structurally
    checkpoint_path = "checkpoints/best_model_epoch_8.pt" 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading test split dynamically mapping batch structural constraints...")
    val_dataset = HAMADataset(transcript_dir, labels_path, split='val')
    # Structurally limiting data scale bounds
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    model = HAMARegressor().to(device).float()
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Explicit architectural weights fully matched natively!")
    except Exception as e:
        print(f"Warning mapping structure natively: {e}")
        
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"\nEvaluating thoroughly referencing natively over {len(val_dataset)} isolated explicit targets...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(input_ids, attention_mask)
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    metrics = calculate_metrics(all_preds, all_labels)
    
    print("\n--- Detailed Mathematical Precision Arrays ---")
    print(f"Overall Mean Absolute Error (MAE): {metrics['overall_mae']:.4f}")
    print(f"Exact Match Accuracy: {metrics['exact_match_acc']:.2f}%")
    print(f"+/- 1 Score Accuracy: {metrics['plus_minus_one_acc']:.2f}%")
    print(f"Macro F1 Score: {metrics['f1_score']:.2f}%")
    print("\nPer-Label Validation Distribution Variance (MAE):")
    
    HAMA_PARAMS = [
        "anxious_mood", "tension", "fears", "insomnia", "intellectual",
        "depressed_mood", "somatic_muscular", "somatic_sensory",
        "cardiovascular", "respiratory", "gastrointestinal",
        "genitourinary", "autonomic", "behavior_at_interview"
    ]
    for param, err in zip(HAMA_PARAMS, metrics['mae_per_label']):
        print(f"  {param.replace('_', ' ').title():<25}: {err:.4f}")

if __name__ == "__main__":
    main()
