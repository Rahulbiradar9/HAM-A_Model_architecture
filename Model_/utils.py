import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

# Weights requested by user mapping to the 14 indices
LABEL_WEIGHTS = [1.0, 1.0, 1.2, 1.2, 1.2, 1.1, 2.5, 3.0, 4.0, 4.0, 4.5, 4.5, 4.0, 1.2]

def get_weighted_mse_loss(predictions, targets, device):
    """
    Computes custom MSE Loss natively merging Base weighted logic and dynamic Rare Label Boosting.
    """
    weights = torch.tensor(LABEL_WEIGHTS, dtype=torch.float).to(device)
    
    # Calculate standard squared error framework
    squared_err = (predictions - targets) ** 2
    
    # 1. Base normalization mechanism spanning specific label-multipliers
    base_loss = (squared_err * weights).sum() / weights.sum()
    
    # 2. Rare Label Boost mechanism penalizing non-zero ground-truths heavier algorithmically
    mask = (targets > 0).float()
    extra_loss = (squared_err * mask).mean()
    
    # Synthesize total loss trajectory mathematically
    final_loss = base_loss + 0.3 * extra_loss
    
    return final_loss

def calculate_metrics(predictions, targets):
    """
    Calculates detailed clinical and analytical metrics mathematically.
    predictions: numpy array of shape (N, 14)
    targets: numpy array of shape (N, 14)
    """
    # Force strict clinical classification boundaries natively
    rounded_preds = np.round(np.clip(predictions, 0, 4)).astype(int)
    targets = np.round(targets).astype(int)
    
    # Standard numerical offsets
    mae_per_label = np.mean(np.abs(rounded_preds - targets), axis=0)
    overall_mae = np.mean(mae_per_label)
    
    # Evaluate clinical absolute precision matches 
    exact_match = np.mean(rounded_preds == targets) * 100
    
    # Evaluate adjacent proximity clinical bounds (+/- 1 accuracy variance)
    abs_diff = np.abs(rounded_preds - targets)
    plus_minus_one = np.mean(abs_diff <= 1) * 100
    
    # Calculate F1, Precision, Recall tracking over all 14 parameters flattenedly
    flat_preds = rounded_preds.flatten()
    flat_targets = targets.flatten()
    
    global_f1 = f1_score(flat_targets, flat_preds, average='macro', zero_division=0) * 100
    global_prec = precision_score(flat_targets, flat_preds, average='macro', zero_division=0) * 100
    global_rec = recall_score(flat_targets, flat_preds, average='macro', zero_division=0) * 100
    
    return {
        "overall_mae": float(overall_mae),
        "mae_per_label": mae_per_label.tolist(),
        "exact_match_acc": float(exact_match),
        "plus_minus_one_acc": float(plus_minus_one),
        "f1_score": float(global_f1),
        "precision": float(global_prec),
        "recall": float(global_rec)
    }
