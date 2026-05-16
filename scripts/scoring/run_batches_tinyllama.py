"""
Run 6 parallel TinyLLaMA scoring processes.
Each process scores a ~1777-pair slice of all_conversation_pairs.json.
All output to separate batch files — combine with combine_batches.py after.

Usage:
    python scripts\run_batches_tinyllama.py
"""

import subprocess
import sys
import math
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "all_combo-pair" / "all_conversation_pairs.json"
OUT_DIR    = BASE_DIR / "all_combo-pair" / "batches"
SCRIPT     = Path(__file__).resolve().parent / "score_pairs_tinyllama.py"
NUM_BATCHES = 6
TOTAL_PAIRS = 10664

OUT_DIR.mkdir(parents=True, exist_ok=True)

batch_size = math.ceil(TOTAL_PAIRS / NUM_BATCHES)
processes  = []

print(f"Launching {NUM_BATCHES} scoring processes...")
print(f"Each batch: ~{batch_size} pairs\n")

for b in range(NUM_BATCHES):
    start = b * batch_size
    end   = min(start + batch_size, TOTAL_PAIRS)
    out   = OUT_DIR / f"batch_{b+1}_of_{NUM_BATCHES}.json"

    cmd = [
        sys.executable, str(SCRIPT),
        "--input",     str(INPUT_FILE),
        "--output",    str(out),
        "--start-idx", str(start),
        "--end-idx",   str(end),
        "--batch-size","50",
    ]
    print(f"  Batch {b+1}: idx {start}-{end} -> {out.name}")
    proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    processes.append((b + 1, proc))

print(f"\nAll {NUM_BATCHES} processes launched.")
print("Each opens in its own terminal window.")
print("Output files saved to:", OUT_DIR)
print("\nWhen all done, run:")
print("  python scripts\\combine_batches.py")
