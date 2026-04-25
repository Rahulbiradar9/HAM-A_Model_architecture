import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerConfig


class HAMARegressor(nn.Module):
    """
    HAM-A regression model using pre-trained Longformer as backbone.
    
    Longformer processes up to 4096 tokens efficiently using sparse local attention
    and global attention on the <s> token.

    Input  : token ids + attention mask  — shape (Batch, SeqLen)
    Output : 14 HAM-A parameter scores   — shape (Batch, 14)
    """

    def __init__(
        self,
        num_outputs: int = 14,
    ):
        super(HAMARegressor, self).__init__()

        # Load pre-trained Longformer-base
        print("Initializing Pre-trained Longformer-base (allenai/longformer-base-4096)...")
        self.longformer = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        
        # Enable gradient checkpointing to save VRAM on RTX 4090
        self.longformer.gradient_checkpointing_enable()

        self.hidden_size = self.longformer.config.hidden_size  # 768

        # Regression head: 768 → 256 → 14 HAM-A scores
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids     : (Batch, SeqLen) — tokenized transcript
            attention_mask: (Batch, SeqLen) — 1 for real tokens, 0 for padding
        Returns:
            predictions   : (Batch, 14) — raw regression scores
        """
        # Global attention mask for Longformer:
        # We set global attention (2) on the first token (<s>)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        
        # Mean Pooling over all tokens (ignoring padding)
        hidden_state = outputs.last_hidden_state  # (B, L, H)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask  # (B, 768)

        # Cast to match head dtype (safe for mixed precision)
        pooled = pooled.to(next(self.regression_head.parameters()).dtype)

        # Output bounded exactly between 0 and 4.0
        return self.regression_head(pooled) * 4.0  # (B, 14)
