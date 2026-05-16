"""
Convert Interview_Data_6K.csv and Synthetic_Data_10K.csv from the M_chat16k/data folder
into a single combined JSON file (m_chat16k_combined.json) in the same folder.

Output JSON structure:
[
    {
        "id": 0,
        "source": "Interview_Data_6K",
        "instruction": "...",
        "input": "...",
        "output": "..."
    },
    ...
]
"""

import json
import os
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
FOLDER      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FILE1       = os.path.join(FOLDER, "Interview_Data_6K.csv")
FILE2       = os.path.join(FOLDER, "Synthetic_Data_10K.csv")
OUTPUT_JSON = os.path.join(FOLDER, "m_chat16k_combined.json")

# ── Load CSVs ──────────────────────────────────────────────────────────────────
print(f"[1/4] Reading {os.path.basename(FILE1)} ...")
df1 = pd.read_csv(FILE1)
df1["source"] = "Interview_Data_6K"
print(f"      -> {len(df1):,} rows | columns: {list(df1.columns)}")

print(f"[2/4] Reading {os.path.basename(FILE2)} ...")
df2 = pd.read_csv(FILE2)
df2["source"] = "Synthetic_Data_10K"
print(f"      -> {len(df2):,} rows | columns: {list(df2.columns)}")

# ── Combine ────────────────────────────────────────────────────────────────────
print("[3/4] Combining and converting to JSON ...")
combined = pd.concat([df1, df2], ignore_index=True)

# Fill any NaN values with empty strings to keep JSON clean
combined = combined.fillna("")

# Build list of dicts with a global id and source tag
records = []
for idx, row in combined.iterrows():
    records.append({
        "id":          int(idx),
        "source":      row["source"],
        "instruction": str(row["instruction"]).strip(),
        "input":       str(row["input"]).strip(),
        "output":      str(row["output"]).strip(),
    })

# ── Write JSON ─────────────────────────────────────────────────────────────────
print(f"[4/4] Writing -> {OUTPUT_JSON}")
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

size_mb = os.path.getsize(OUTPUT_JSON) / (1024 * 1024)
print(f"\nDone! {len(records):,} records saved to:")
print(f"   {OUTPUT_JSON}")
print(f"   File size : {size_mb:.1f} MB")
print(f"\n   Breakdown:")
print(f"   - Interview_Data_6K : {len(df1):,} records")
print(f"   - Synthetic_Data_10K: {len(df2):,} records")
print(f"   - Total             : {len(records):,} records")
