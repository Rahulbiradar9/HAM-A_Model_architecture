"""
Diagnose why _batch_hama_scores.json has mostly zero scores.
"""
import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
SCORES_FILE = BASE / "json_transcripts" / "_batch_hama_scores.json"
TRANSCRIPT_DIR = BASE / "json_transcripts"

SUBSCALES = [
    "anxious_mood","tension","fears","insomnia","intellectual",
    "depressed_mood","somatic_muscular","somatic_sensory","cardiovascular",
    "respiratory","gastrointestinal","genitourinary","autonomic","behavior_at_interview"
]

# ── Load scores ────────────────────────────────────────────────────────────────
with open(SCORES_FILE) as f:
    scores = json.load(f)

n = len(scores)
totals = [r["total_score"] for r in scores]
all_zero = [r for r in scores if r["total_score"] == 0]

print("=" * 60)
print("SCORE STATISTICS")
print("=" * 60)
print(f"Total records        : {n}")
print(f"Records with score=0 : {len(all_zero)} ({len(all_zero)/n*100:.1f}%)")
print(f"Max total score      : {max(totals)}")
print(f"Mean total score     : {sum(totals)/n:.3f}")
print()

print("Score distribution:")
dist = Counter(totals)
for s in sorted(dist):
    bar = "#" * dist[s]
    print(f"  score={s:2d}: {dist[s]:4d} records  {bar}")
print()

# ── Show non-zero examples ─────────────────────────────────────────────────────
nonzero = [r for r in scores if r["total_score"] > 0]
print(f"Non-zero records ({len(nonzero)} total):")
for r in nonzero:
    active = {k: v for k, v in r.items() if v and k not in ["filename","total_score"]}
    print(f"  {r['filename']}: total={r['total_score']} | {active}")
print()

# ── Check if transcripts actually have content ─────────────────────────────────
print("=" * 60)
print("SAMPLE TRANSCRIPT CONTENT CHECK")
print("=" * 60)

sample_files = ["300_TRANSCRIPT.json", "305_TRANSCRIPT.json", "364_TRANSCRIPT.json"]
for fname in sample_files:
    fpath = TRANSCRIPT_DIR / fname
    if not fpath.exists():
        print(f"  {fname}: NOT FOUND")
        continue
    with open(fpath) as f:
        data = json.load(f)

    # Extract patient lines
    patient_lines = [row.get("value","") for row in data if "Participant" in row.get("speaker","")]
    patient_text = " ".join(patient_lines)

    # Check for anxiety-related keywords
    keywords = ["anxious","worry","worried","nervous","fear","scared","sleep","depressed","sad","tired","pain","chest","dizzy","sweat"]
    found_kw = [kw for kw in keywords if kw.lower() in patient_text.lower()]

    print(f"\n  {fname}:")
    print(f"    Total lines      : {len(data)}")
    print(f"    Patient lines    : {len(patient_lines)}")
    print(f"    Patient words    : {len(patient_text.split())}")
    print(f"    Anxiety keywords : {found_kw if found_kw else 'NONE FOUND'}")
    print(f"    Sample patient   : {patient_text[:200]!r}")
    score_rec = next((r for r in scores if r.get("filename") == fname), None)
    if score_rec:
        print(f"    Assigned score   : {score_rec['total_score']} (all zero: {all(score_rec.get(s,0)==0 for s in SUBSCALES)})")

# ── Root cause summary ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ROOT CAUSE ANALYSIS")
print("=" * 60)
print()
print("POSSIBLE CAUSES:")
print()
print("1. MODEL TOO CONSERVATIVE:")
print("   Mistral 7B with temperature=0.1 is very conservative.")
print("   The prompt says 'be conservative, score ONLY if evidence is present'.")
print("   This causes the model to default to 0 for most symptoms.")
print()
print("2. TRANSCRIPT FORMAT MISMATCH:")
print("   Transcripts use 'Participant' as speaker key, but scorer checks 'Ellie'.")
print("   format_conversation() assigns 'Doctor' to Ellie and 'Patient' to everyone else.")
print("   --> Patient lines ARE being captured correctly.")
print()
print("3. TRANSCRIPT TRUNCATION (TOKEN LIMIT):")
print("   Large transcripts (e.g. 364_TRANSCRIPT.json = 88KB) may be truncated")
print("   by the model's context window, missing key symptom disclosures.")
print()
print("4. DAIC-WOZ DATASET NATURE:")
print("   These are DAIC-WOZ research interviews - many participants are healthy controls.")
print("   A high proportion of zero-score records may simply reflect the data distribution.")
print()

# Check if DAIC-WOZ has ground truth labels
print("RECOMMENDATION:")
print("   Compare against DAIC-WOZ ground truth PHQ-8/PHQ-9 labels to validate")
print("   whether the zero scores match actual clinical labels in the dataset.")
