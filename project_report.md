# HAM-A Batch Scoring Project — Full Report

**Project:** Automated HAM-A scoring of DAIC-WOZ clinical interview transcripts  
**Dataset:** 185 JSON transcript files (participants 300–492)  
**Model Used:** Mistral 7B Instruct v0.3 (local inference)  
**Date Range:** April 2026  

---

## 1. What Was Done

### 1.1 Data Source
- **Dataset:** DAIC-WOZ (Distress Analysis Interview Corpus — Wizard of Oz)
- **Format:** 189 individual JSON files in `json_transcripts/`, each containing timestamped interview lines between Ellie (AI interviewer) and a Participant
- **Sample structure:**
```json
[
  { "start_time": "36.5", "stop_time": "39.6", "speaker": "Ellie", "value": "hi, thanks for coming in today" },
  { "start_time": "62.3", "stop_time": "63.1", "speaker": "Participant", "value": "good" }
]
```

---

### 1.2 Scoring Script (`hama_scorer.py`)
A Python script was built to:
1. Load each transcript file
2. Format it into a readable Doctor/Patient conversation string with timestamps
3. Feed the full conversation to **Mistral 7B Instruct v0.3** with a HAM-A clinical assessment prompt
4. Parse the JSON output containing 14 subscale scores + total
5. Save results incrementally to `_batch_hama_scores.json`

**Key features:**
- `--resume` flag to continue from last checkpoint after interruption
- Batch saves every 50 records to prevent data loss
- Handles both folder-mode (185 transcripts) and single-file mode

---

### 1.3 HAM-A Subscales Scored
| # | Subscale | What It Measures |
|---|---|---|
| 1 | Anxious Mood | Worries, irritability, fearful anticipation |
| 2 | Tension | Fatigue, trembling, restlessness |
| 3 | Fears | Dark, strangers, crowds, animals |
| 4 | Insomnia | Sleep difficulty, nightmares, fatigue on waking |
| 5 | Intellectual | Poor concentration, poor memory |
| 6 | Depressed Mood | Loss of interest, sadness |
| 7 | Somatic Muscular | Pain, stiffness, twitching |
| 8 | Somatic Sensory | Tinnitus, blurred vision, hot flushes |
| 9 | Cardiovascular | Palpitations, chest pain |
| 10 | Respiratory | Breathlessness, chest pressure |
| 11 | Gastrointestinal | Nausea, abdominal pain |
| 12 | Genitourinary | Urinary/libido/menstrual issues |
| 13 | Autonomic | Dry mouth, sweating, dizziness |
| 14 | Behavior at Interview | Restlessness, tremor, facial tension |

Scoring scale: **0** = Absent · **1** = Mild · **2** = Moderate · **3** = Severe · **4** = Very Severe

---

### 1.4 Output File
- **Path:** `json_transcripts/_batch_hama_scores.json`
- **Records:** 185
- **Size:** ~88 KB

---

## 2. Visualization Work

### 2.1 Dashboard (`analyze_hama_scores.py`)
An 8-panel combined dashboard saved to `analysis_output/hama_analysis_dashboard.png`:
- Total score histogram with severity bands
- Severity category donut chart
- Mean scores bar chart per subscale
- Std deviation bar chart
- Box plot distribution per subscale
- Jitter/strip plot of raw scores
- Summary statistics table

> **Fix applied:** Used `.get(s, 0)` to handle one malformed record (`453_TRANSCRIPT.json`) that was missing several subscale keys.

---

### 2.2 Alternative Visualizations (`visualize_hama_v2.py`)
10 separate graph files saved to `analysis_output/separate_graphs/`:

| File | Chart Type | Insight |
|---|---|---|
| `1_lollipop.png` | Lollipop | Mean ± Std per subscale (sorted) |
| `2_ridgeline.png` | Ridgeline / Joy Plot | KDE distribution per subscale |
| `3_waffle.png` | Waffle | Severity proportions (100% Mild/None) |
| `4_treemap.png` | Treemap | Subscale share of total symptom burden |
| `5_bump_rank.png` | Bump/Rank | Subscale rank across severity groups |
| `6_diverging_bar.png` | Diverging Bar | Each subscale vs. overall mean |
| `7_polar_area.png` | Polar Area | Mean scores on a rose chart |
| `8_hexbin.png` | Hexbin Density | Insomnia vs Depressed Mood (r = 0.41) |
| `9_step_ecdf.png` | Step ECDF | Cumulative % at each total score |
| `10_bubble.png` | Bubble Chart | Mean vs Std Dev, size = % non-zero |

---

### 2.3 Per-Subscale Histograms (`hama_histograms.py`)
15 separate histogram PNGs saved to `analysis_output/histograms/`:
- 14 individual subscale histograms (`01_anxious_mood.png` … `14_behavior_at_interview.png`)
- 1 total score histogram (`15_total_score.png`)

Each histogram includes:
- Mean (gold line) and Median (red dashes)
- Stats box: N, mean, median, std, min, max
- Non-zero % annotation

---

## 3. The Problem — Why Scores Are Nearly All Zero

### 3.1 Key Numbers
| Metric | Value |
|---|---|
| Total records scored | 185 |
| Records with total_score = 0 | 99 (53.5%) |
| Records with any non-zero score | 86 (46.5%) |
| Maximum total score seen | 7 |
| Mean total score | 0.82 |
| Subscales ever detected | Only **insomnia** and **depressed_mood** (~95% of cases) |

### 3.2 Score Distribution
```
score= 0:   99 records  ███████████████████████████████████████████████████
score= 1:   54 records  ██████████████████████████████
score= 2:   17 records  █████████
score= 3:   10 records  █████
score= 4:    2 records  █
score= 5:    2 records  █
score= 6:    0 records
score= 7:    1 record
```

### 3.3 Root Causes Identified

#### 🔴 Cause 1 — Model Too Conservative (Primary)
The scorer prompt explicitly instructs the model:
> *"Score ONLY if evidence is present. If not mentioned → 0. Do NOT assume. Be conservative."*

Combined with `temperature=0.1`, Mistral 7B almost always outputs all-zero scores. Even `305_TRANSCRIPT.json` — which contains patient words like **"worry", "sleep", "depressed", "tired", "pain"** — was scored as **total = 0**.

**Fix:** Raise temperature to 0.4–0.6 and rewrite prompt to allow clinical inference from context, not just keyword matching.

---

#### 🔴 Cause 2 — Only 2 of 14 Subscales Ever Detected
From the 86 non-zero records, nearly all detections are:
- **Insomnia** — because Ellie explicitly asks "how easy is it to get a good night's sleep?"
- **Depressed mood** — because Ellie explicitly asks "have you been diagnosed with depression?"

The other 12 subscales (somatic, cardiovascular, respiratory, etc.) require **clinical inference from indirect symptoms** — which the model is instructed not to do.

**Fix:** Restructure the prompt to encourage inference, e.g., *"If a patient says they feel chest tightness, score cardiovascular even if they don't name it."*

---

#### 🟡 Cause 3 — Large Transcripts Exceed Context Window
Some transcripts are very large:
- `364_TRANSCRIPT.json` = 88 KB, 473 lines, 4,112 patient words
- `480_TRANSCRIPT.json` = 85 KB

Mistral 7B's context window is ~32K tokens. After formatting + system prompt, many transcripts get silently truncated — and the clinically important disclosures (which happen later in interviews) are cut off.

**Fix:** Extract only the **patient speech** text and optionally limit to the last 2,000 words before sending to the model.

---

#### 🟢 Cause 4 — Dataset Itself Contains Healthy Controls (Expected)
DAIC-WOZ includes both depressed and non-depressed participants. A subset of participants genuinely have no/low anxiety symptoms. So some zero records correctly reflect the ground truth.

**Validation:** Compare scored records against the official DAIC-WOZ PHQ-8/PHQ-9 labels to measure accuracy.

---

## 4. Files Created

```
scripts/
├── hama_scorer.py          ← Main scoring pipeline (Mistral 7B)
├── analyze_hama_scores.py  ← 8-panel dashboard
├── visualize_hama_v2.py    ← 10 separate alternative graphs
├── hama_histograms.py      ← 15 per-subscale histograms
└── diagnose_scores.py      ← Diagnostic root cause script

analysis_output/
├── hama_analysis_dashboard.png
├── hama_alternative_visualizations.png   ← (old combined version)
├── separate_graphs/
│   ├── 1_lollipop.png
│   ├── 2_ridgeline.png
│   ├── 3_waffle.png
│   ├── 4_treemap.png
│   ├── 5_bump_rank.png
│   ├── 6_diverging_bar.png
│   ├── 7_polar_area.png
│   ├── 8_hexbin.png
│   ├── 9_step_ecdf.png
│   └── 10_bubble.png
└── histograms/
    ├── 01_anxious_mood.png
    ├── 02_tension.png
    ├── ...
    ├── 14_behavior_at_interview.png
    └── 15_total_score.png

json_transcripts/
└── _batch_hama_scores.json   ← 185 scored records (output)
```

---

## 5. Recommended Next Steps

| Priority | Action | Expected Impact |
|---|---|---|
| 🔴 High | Rewrite scorer prompt — allow clinical inference | Detect 5–8 more subscales reliably |
| 🔴 High | Raise `temperature` 0.1 → 0.5 | Allow more nuanced, non-zero scoring |
| 🟡 Medium | Truncate input to patient-speech-only, last 2K words | Fix context window overflow for large transcripts |
| 🟡 Medium | Validate against DAIC-WOZ PHQ-8 ground truth labels | Measure scoring accuracy objectively |
| 🟢 Low | Re-run scorer after fixes, regenerate all visualizations | Full pipeline validation |
