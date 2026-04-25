import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

HAMA_PARAMS = [
    "anxious_mood", "tension", "fears", "insomnia", "intellectual",
    "depressed_mood", "somatic_muscular", "somatic_sensory",
    "cardiovascular", "respiratory", "gastrointestinal",
    "genitourinary", "autonomic", "behavior_at_interview"
]


class HAMADataset(Dataset):
    """
    Dataset for HAM-A regression using Longformer backbone.
    """

    def __init__(
        self,
        transcript_dir: str,
        labels_path: str,
        tokenizer_name: str = "allenai/longformer-base-4096",
        max_length: int = 4096,
        split: str = "train",
        split_ratio: float = 0.8,
    ):
        self.transcript_dir = transcript_dir
        self.max_length = max_length
        self.split = split

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        with open(labels_path, "r", encoding="utf-8") as f:
            all_labels = json.load(f)

        all_labels.sort(key=lambda x: x.get("filename", ""))

        valid_files = []
        for label_data in all_labels:
            fname = label_data.get("filename")
            if not fname:
                continue

            t_path = os.path.join(transcript_dir, fname)
            if not os.path.exists(t_path):
                continue

            scores = [label_data.get(p, 0) for p in HAMA_PARAMS]
            valid_files.append({
                "filename": fname,
                "path": t_path,
                "scores": scores,
            })

        split_idx = int(len(valid_files) * split_ratio)
        if split == "train":
            self.samples = valid_files[:split_idx]
        elif split == "val":
            self.samples = valid_files[split_idx:]

        print(f"Loaded {len(self.samples)} samples for '{split}' split.")

    def __len__(self) -> int:
        return len(self.samples)

    def extract_participant_text(self, filepath: str) -> str:
        """Extract only the participant's speech turns from a transcript JSON."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                dialogue = json.load(f)
        except Exception:
            return ""

        extracted = []
        for turn in dialogue:
            speaker = turn.get("speaker", "").lower().strip()
            if "participant" in speaker:
                extracted.append(turn.get("value", ""))

        return " ".join(extracted).strip()

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = self.extract_participant_text(sample["path"])

        if not text:
            text = "empty transcript"

        # Single-pass tokenization — no chunking, no stride
        # Mamba-2 handles long sequences natively in O(n)
        outputs = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            # squeeze(0): (1, SeqLen) → (SeqLen,); DataLoader stacks to (B, SeqLen)
            "input_ids": outputs["input_ids"].squeeze(0),
            "attention_mask": outputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["scores"], dtype=torch.float),
            "filename": sample["filename"],
        }
