"""
Combine 6 batch output files into one final scored JSON.
Run this after all 6 batch processes finish.

Usage:
    python scripts\combine_batches.py
"""

import json
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
BATCHES_DIR = BASE_DIR / "all_combo-pair" / "batches"
OUTPUT_FILE = BASE_DIR / "all_combo-pair" / "all_conversation_pairs_scored_tinyllama.json"
NUM_BATCHES = 6

combined = []
total_scored = 0
total_errors = 0

print("=" * 60)
print("  Combining TinyLLaMA batch outputs")
print("=" * 60)

for b in range(1, NUM_BATCHES + 1):
    path = BATCHES_DIR / f"batch_{b}_of_{NUM_BATCHES}.json"
    if not path.exists():
        print(f"  [WARNING] Missing: {path.name} -- skipping")
        continue

    with open(path, "r", encoding="utf-8") as f:
        batch_data = json.load(f)

    scored  = sum(1 for x in batch_data if "total_score" in x)
    errors  = sum(1 for x in batch_data if "total_score" not in x)
    total_scored += scored
    total_errors += errors

    combined.extend(batch_data)
    print(f"  Batch {b}: {len(batch_data)} pairs | {scored} scored | {errors} unscored")

print(f"\n  Total pairs   : {len(combined)}")
print(f"  Total scored  : {total_scored}")
print(f"  Total unscored: {total_errors}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=4)

print(f"\n  Saved -> {OUTPUT_FILE}")
print("=" * 60)
