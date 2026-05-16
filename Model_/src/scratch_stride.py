import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

long_text = "hello " * 3000

encoding = tokenizer(
    long_text,
    add_special_tokens=True,
    truncation=True,
    max_length=1024,
    stride=256,
    padding='max_length',
    return_overflowing_tokens=True,
    return_tensors='pt'
)

print("Keys:", encoding.keys())
print("input_ids shape:", encoding['input_ids'].shape)
print("attention_mask shape:", encoding['attention_mask'].shape)
print("Number of chunks:", len(encoding['input_ids']))
