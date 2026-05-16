import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Configuration
MODEL_NAME = "mental/mental-roberta-base"
DATA_FILE = r"d:\Rahul_Intern\convo_model\M_chat16k\m_chat16k_combined_scored_weighted_rank1.json"
OUTPUT_DIR = r"d:\Rahul_Intern\convo_model\M_chat16k\mental_roberta_training\results"

# 14 HAM-A Categories
CATEGORIES = [
    "anxious_mood", "tension", "fears", "insomnia", "intellectual",
    "depressed_mood", "somatic_muscular", "somatic_sensory",
    "cardiovascular", "respiratory", "gastrointestinal",
    "genitourinary", "autonomic", "behavior_at_interview"
]

print(f"Loading pre-trained model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set up regression model for 14 outputs
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(CATEGORIES),
    problem_type="regression"
)

print(f"Loading data from: {DATA_FILE}")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

# Prepare data
for item in data:
    # Combine input and output text as the conversation
    text = f"Patient: {item.get('input_text', '')}\nCounselor: {item.get('output_text', '')}"
    
    score_dict = item.get("score", {})
    
    # Extract the 14 integer scores (0-4)
    item_labels = []
    valid = True
    for cat in CATEGORIES:
        if cat in score_dict:
            item_labels.append(float(score_dict[cat]))
        else:
            valid = False
            break
            
    if valid:
        texts.append(text)
        labels.append(item_labels)

print(f"Total valid conversations: {len(texts)}")

# Split into train/val (90% / 10%)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Create HuggingFace Datasets
train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Custom compute metrics function for regression (Mean Squared Error & Mean Absolute Error)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Clip predictions to valid range [0, 4]
    predictions = np.clip(predictions, 0, 4)
    # Mean Squared Error
    mse = np.mean((predictions - labels) ** 2)
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - labels))
    # Accuracy when rounded to nearest integer
    rounded_preds = np.round(predictions)
    accuracy = np.mean(rounded_preds == labels)
    return {"mse": mse, "mae": mae, "rounded_accuracy": accuracy}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch size = 32
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    greater_is_better=False,        # Lower MSE is better
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Saving final optimized model...")
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

print(f"Training complete! Model saved to {os.path.join(OUTPUT_DIR, 'final_model')}")
