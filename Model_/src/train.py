import os
import sys
import json
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import HAMADataset
from model import HAMARegressor
from utils import get_weighted_mse_loss, calculate_metrics

# Reduce CUDA memory fragmentation (helps with the pure-PyTorch Mamba-2 kernel)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ------------------------------------------------------------------
# Tee Logger — prints to console AND writes to log file simultaneously
# ------------------------------------------------------------------
class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        
    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()


def main():
    transcript_dir  = "../json_transcripts"
    labels_path     = "../after_scoring/_batch_hama_scores_weighted_60_40.json"
    batch_size      = 1       # Keep at 1 — Mamba-2 pure-PyTorch SSD is very memory-heavy
    accum_steps     = 8       # Gradient accumulation: effective batch = 1 × 8 = 8
    epochs          = 50
    learning_rate   = 2e-5
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Log file setup
    # ------------------------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/longformer", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = f"logs/train_{timestamp}_bs{batch_size}x{accum_steps}_ep{epochs}.txt"
    tee       = TeeLogger(log_path)
    sys.stdout = tee

    print(f"Logging to : {log_path}")
    print(f"Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device     : {device}")
    print(f"Batch size : {batch_size}  |  Grad accum steps: {accum_steps}  "
          f"(effective batch = {batch_size * accum_steps})")
    print(f"Epochs     : {epochs}  |  LR: {learning_rate}")
    print(f"AMP        : enabled (float16 forward, float32 params)")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print("\nInitializing datasets...")
    train_dataset = HAMADataset(transcript_dir, labels_path, split="train")
    val_dataset   = HAMADataset(transcript_dir, labels_path, split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\nInitializing Pre-trained Longformer-base (4096 context) backbone...")
    model     = HAMARegressor().to(device).float()
    optimizer = AdamW([
        {'params': model.longformer.parameters(), 'lr': 2e-5},
        {'params': model.regression_head.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = GradScaler()   # AMP gradient scaler

    best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch + 1} / {epochs} ========")

        # ---- Train ----
        model.train()
        total_loss   = 0.0
        optimizer.zero_grad()
        train_loop   = tqdm(train_loader, desc="Training", file=tee.terminal)

        for step, batch in enumerate(train_loop):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)

            # AMP forward pass — runs in float16, saves ~50% VRAM
            with autocast():
                predictions = model(input_ids, attention_mask)
                loss = get_weighted_mse_loss(predictions, labels, device)
                loss = loss / accum_steps   # normalize for accumulation

            scaler.scale(loss).backward()

            # Only update weights every accum_steps steps
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            train_loop.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Free unused cache between train and val
        torch.cuda.empty_cache()

        # ---- Validate ----
        model.eval()
        all_preds  = []
        all_labels = []
        val_loss_total = 0.0
        val_loop   = tqdm(val_loader, desc="Validation", file=tee.terminal)

        with torch.no_grad():
            for batch in val_loop:
                input_ids      = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels         = batch["labels"].to(device, non_blocking=True)

                with autocast():
                    predictions = model(input_ids, attention_mask)
                    loss = get_weighted_mse_loss(predictions, labels, device)

                val_loss_total += loss.item()
                all_preds.append(predictions.float().cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss_total / len(val_loader)
        torch.cuda.empty_cache()

        all_preds  = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        metrics = calculate_metrics(all_preds, all_labels)
        print(f"Validation Loss            : {avg_val_loss:.4f}")
        print(f"Validation Rounded MAE     : {metrics['overall_mae']:.4f}")
        print(f"Exact Match Accuracy       : {metrics['exact_match_acc']:.2f}%")
        print(f"+/- 1 Accuracy             : {metrics['plus_minus_one_acc']:.2f}%")
        print(f"Macro F1 Score             : {metrics['f1_score']:.2f}%")
        print(f"Precision                  : {metrics['precision']:.2f}%")
        print(f"Recall                     : {metrics['recall']:.2f}%")
        print(f"Per-Label MAE              : {metrics['mae_per_label']}")

        # ---- Checkpoint ----
        if avg_val_loss < best_val_loss:
            best_val_loss   = avg_val_loss
            checkpoint_name = f"checkpoints/longformer/longformer_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_name)
            torch.save(model.state_dict(), "checkpoints/longformer/best_model.pt")

            with open("checkpoints/longformer/best_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            print(f"--> New best model saved: {checkpoint_name}  (Val Loss: {best_val_loss:.4f})")

        scheduler.step()
        print(f"    LR: {scheduler.get_last_lr()[0]:.2e}")

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tee.close()
    sys.stdout = tee.terminal


if __name__ == "__main__":
    main()
